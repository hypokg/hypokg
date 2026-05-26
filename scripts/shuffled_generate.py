"""
HypoKG — Shuffled Path Control: Generation
Permutes intermediate nodes in all 550 paths while holding source and
terminal endpoints fixed, then regenerates C2 and C3 hypotheses.

Step 1 (run once): python shuffled_generate.py --step shuffle
Step 2 (API models): python shuffled_generate.py --step generate --model gpt4o
Step 2 (HF models):  python shuffled_generate.py --step generate --model biomistral_7b
                     python shuffled_generate.py --step generate --model medgemma_27b
"""

import os, json, random, time, gc, argparse
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

from datetime import datetime, timezone

HYPOKG_FOLDER    = os.environ.get("HYPOKG_FOLDER", "./")
PATHS_FILE       = f"{HYPOKG_FOLDER}/data/paths/hypokg_550_selected.jsonl"
SHUFFLED_DIR     = f"{HYPOKG_FOLDER}/shuffled_control"
SHUFFLED_PATHS   = f"{SHUFFLED_DIR}/shuffled_paths.jsonl"
SHUFFLED_GEN_DIR = f"{SHUFFLED_DIR}/generations"
os.makedirs(SHUFFLED_GEN_DIR, exist_ok=True)

RANDOM_SEED = 42

API_MODELS = {
    "gpt4o": {
        "display": "GPT-4o",
        "backend": "openai",
        "model_id": "gpt-4o",
        "rate_limit_s": 0.3,
    },
    "claude_sonnet": {
        "display": "Claude Sonnet 4.6",
        "backend": "claude",
        "model_id": "claude-sonnet-4-20250514",
        "rate_limit_s": 0.3,
    },
    "llama_70b": {
        "display": "Llama-3.3-70B",
        "backend": "together",
        "model_id": "meta-llama/Llama-3.3-70B-Instruct-Turbo",
        "rate_limit_s": 0.3,
    },
    "qwen3_235b": {
        "display": "Qwen3-235B",
        "backend": "together",
        "model_id": "Qwen/Qwen3-235B-A22B-Instruct-2507-tput",
        "rate_limit_s": 0.5,
    },
}

HF_MODELS = {"biomistral_7b", "medgemma_27b"}


# ── Step 1: Shuffle paths ─────────────────────────────────────────

def shuffle_intermediates(path_text, seed):
    random.seed(seed)
    edges = [e.strip() for e in path_text.strip().split("\n") if e.strip()]
    if len(edges) <= 2:
        return path_text, False
    first, last, middle = edges[0], edges[-1], edges[1:-1]
    if len(middle) <= 1:
        return path_text, False
    shuffled = middle.copy()
    for _ in range(20):
        random.shuffle(shuffled)
        if shuffled != middle:
            break
    return "\n".join([first] + shuffled + [last]), True


def create_shuffled_paths():
    random.seed(RANDOM_SEED)
    paths = [json.loads(l) for l in open(PATHS_FILE) if l.strip()]
    print(f"Loaded {len(paths)} paths")

    skipped = 0
    with open(SHUFFLED_PATHS, "w") as out:
        for i, path in enumerate(paths):
            shuffled_text, was_shuffled = shuffle_intermediates(
                path["path_text"], seed=RANDOM_SEED + i)
            if not was_shuffled:
                skipped += 1
            record = {**path,
                      "path_text_original": path["path_text"],
                      "path_text":          shuffled_text,
                      "was_shuffled":       was_shuffled,
                      "shuffle_seed":       RANDOM_SEED + i}
            out.write(json.dumps(record) + "\n")

    shuffled = len(paths) - skipped
    print(f"Saved {SHUFFLED_PATHS}")
    print(f"  Shuffled: {shuffled} | Too short: {skipped}")

    # Show one example
    records = [json.loads(l) for l in open(SHUFFLED_PATHS) if l.strip()]
    for r in records:
        if r["was_shuffled"]:
            print(f"\nExample: {r['hypokg_id']}")
            print("ORIGINAL:")
            for line in r["path_text_original"].split("\n"):
                print(f"  {line}")
            print("SHUFFLED:")
            for line in r["path_text"].split("\n"):
                print(f"  {line}")
            break


# ── Prompts ───────────────────────────────────────────────────────

def make_prompt(path_text, condition):
    if condition == 2:
        return f"""You are an expert biochemist. The following knowledge graph path describes a multi-step mechanistic relationship between biological entities:

{path_text}

Not every edge in this path represents an equally strong biological claim. Treat the path as partial mechanistic evidence of varying reliability.

Your task is mechanistic compression.
The path is not a script to retell. Instead, treat it as mechanistic evidence from which a biologically meaningful hypothesis may emerge.

Instructions:
1. Identify the single most biologically specific and meaningful mechanistic transition in the path.
2. Explicitly state why this transition is more biologically meaningful than the adjacent edges you chose to ignore.
3. Prioritize mechanistic coherence over graph completeness. Ignore weak, generic, or incidental edges.
4. Do not narrate the full path step-by-step.
5. Synthesize one focused scientific hypothesis around that specific mechanistic substructure.
6. State one falsifiable experimental prediction with a specific measurable output.
7. Name the specific entities involved — do not generalize to enzyme families.
8. The hypothesis should sound like a concise scientific claim, not a graph traversal.

Begin with "We hypothesize that" and limit to 150 words."""

    elif condition == 3:
        return f"""You are an expert biochemist. The following knowledge graph path describes a possible mechanistic relationship between biological entities:

{path_text}

Not every edge in this path represents an equally strong biological claim. Treat the path as partial mechanistic evidence of varying reliability.

Your task is mechanistic interpretation.
Reason about the path before generating the final hypothesis:
- Which transitions appear biologically meaningful?
- Which steps are weak, generic, or uncertain?
- Where does the path suggest a true mechanistic bottleneck, regulatory switch, or cross-domain interaction?

Do not assume every edge represents a direct causal relationship.
Use the graph as mechanistic guidance, not as a deterministic script.

Format your response as:
RATIONALE: [brief mechanistic interpretation, max 80 words]
HYPOTHESIS: We hypothesize that [concise mechanistic hypothesis including the falsifiable prediction, max 150 words]

Do not add any additional sections beyond RATIONALE and HYPOTHESIS."""


# ── API generation ────────────────────────────────────────────────

def run_api(model_key):
    import openai, anthropic
    OPENAI_API_KEY   = os.environ.get("OPENAI_API_KEY", "")
    ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
    TOGETHER_API_KEY  = os.environ.get("TOGETHER_API_KEY", "")

    oai_client      = openai.OpenAI(api_key=OPENAI_API_KEY)
    ant_client      = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
    together_client = openai.OpenAI(api_key=TOGETHER_API_KEY,
                                    base_url="https://api.together.xyz/v1")

    cfg = API_MODELS[model_key]

    shuffled_paths = [json.loads(l) for l in open(SHUFFLED_PATHS) if l.strip()]
    shuffled_paths = [p for p in shuffled_paths if p["was_shuffled"]]
    print(f"Shuffled paths: {len(shuffled_paths)}")

    out_file = f"{SHUFFLED_GEN_DIR}/{model_key}_shuffled.jsonl"
    done = set()
    if os.path.exists(out_file):
        for line in open(out_file):
            r = json.loads(line.strip())
            done.add((r["hypokg_id"], r["condition"]))

    todo = [(p, c) for p in shuffled_paths for c in [2, 3]
            if (p["hypokg_id"], c) not in done]
    print(f"{cfg['display']} — {len(todo)} remaining / {len(shuffled_paths)*2} total")

    def generate(prompt):
        t0 = time.time()
        if cfg["backend"] == "openai":
            resp = oai_client.chat.completions.create(
                model=cfg["model_id"], max_tokens=512,
                messages=[{"role": "user", "content": prompt}])
            return {"text": resp.choices[0].message.content,
                    "input_tok": resp.usage.prompt_tokens,
                    "output_tok": resp.usage.completion_tokens,
                    "latency_s": round(time.time()-t0, 2)}
        elif cfg["backend"] == "claude":
            resp = ant_client.messages.create(
                model=cfg["model_id"], max_tokens=512,
                messages=[{"role": "user", "content": prompt}])
            return {"text": resp.content[0].text,
                    "input_tok": resp.usage.input_tokens,
                    "output_tok": resp.usage.output_tokens,
                    "latency_s": round(time.time()-t0, 2)}
        elif cfg["backend"] == "together":
            resp = together_client.chat.completions.create(
                model=cfg["model_id"], max_tokens=512,
                messages=[{"role": "user", "content": prompt}])
            return {"text": resp.choices[0].message.content,
                    "input_tok": resp.usage.prompt_tokens,
                    "output_tok": resp.usage.completion_tokens,
                    "latency_s": round(time.time()-t0, 2)}

    for i, (path, condition) in enumerate(todo):
        hid    = path["hypokg_id"]
        source = (path["path_text"].split("\n")[0]
                  .split("[enzyme_kinetics]")[0].strip()
                  .split(",")[0].strip())
        try:
            result = generate(make_prompt(path["path_text"], condition))
            out = {
                "hypokg_id":          hid,
                "path_id":            path.get("path_id", hid),
                "tier_calibrated":    path["tier_calibrated"],
                "crossing_count":     path["crossing_count"],
                "start_node":         path["start_node"],
                "end_node":           path["end_node"],
                "in_gold_subset":     path["in_gold_subset"],
                "in_verified_subset": path["in_verified_subset"],
                "model_key":          model_key,
                "model_display":      cfg["display"],
                "condition":          condition,
                "path_type":          "shuffled",
                "prompt_version":     "v2_shuffled",
                "hypothesis_text":    result["text"],
                "input_tokens":       result["input_tok"],
                "output_tokens":      result["output_tok"],
                "latency_s":          result["latency_s"],
                "generated_at":       datetime.now(timezone.utc).isoformat(),
            }
            with open(out_file, "a") as f:
                f.write(json.dumps(out) + "\n")
            done.add((hid, condition))
            print(f"  [{i+1}/{len(todo)}] {hid} C{condition} "
                  f"src={source[:12]:<12} "
                  f"{result['input_tok']}in/{result['output_tok']}out "
                  f"{result['latency_s']}s")
        except Exception as e:
            print(f"  ERROR {hid} C{condition}: {e}")
            time.sleep(3)
        time.sleep(cfg["rate_limit_s"])


# ── HF generation ─────────────────────────────────────────────────

def run_hf(model_key):
    import torch
    from transformers import (AutoTokenizer, AutoModelForCausalLM,
                               AutoProcessor, AutoModelForImageTextToText,
                               BitsAndBytesConfig)
    from huggingface_hub import login

    HF_TOKEN = os.environ.get("HF_TOKEN", "")
    login(token=HF_TOKEN)

    gc.collect()
    torch.cuda.empty_cache()

    shuffled_paths = [json.loads(l) for l in open(SHUFFLED_PATHS) if l.strip()]
    shuffled_paths = [p for p in shuffled_paths if p["was_shuffled"]]

    out_file = f"{SHUFFLED_GEN_DIR}/{model_key}_shuffled.jsonl"
    done = set()
    if os.path.exists(out_file):
        for line in open(out_file):
            r = json.loads(line.strip())
            done.add((r["hypokg_id"], r["condition"]))

    todo = [(p, c) for p in shuffled_paths for c in [2, 3]
            if (p["hypokg_id"], c) not in done]

    if model_key == "biomistral_7b":
        display  = "BioMistral-7B"
        model_id = "BioMistral/BioMistral-7B"
        tokenizer = AutoTokenizer.from_pretrained(
            model_id, token=HF_TOKEN, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left"
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            quantization_config=BitsAndBytesConfig(load_in_8bit=True),
            device_map="auto", token=HF_TOKEN, trust_remote_code=True)
        model.eval()

        def generate(prompt):
            t0 = time.time()
            try:
                input_text = tokenizer.apply_chat_template(
                    [{"role": "user", "content": prompt}],
                    tokenize=False, add_generation_prompt=True)
            except Exception:
                input_text = prompt
            inputs = tokenizer(input_text, return_tensors="pt",
                               truncation=True, max_length=2048).to(model.device)
            input_len = inputs["input_ids"].shape[1]
            with torch.no_grad():
                outputs = model.generate(
                    **inputs, max_new_tokens=512, do_sample=False,
                    temperature=None, top_p=None,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id)
            text = tokenizer.decode(outputs[0][input_len:],
                                    skip_special_tokens=True).strip()
            return {"text": text, "input_tok": input_len,
                    "output_tok": len(outputs[0]) - input_len,
                    "latency_s": round(time.time()-t0, 2)}

    elif model_key == "medgemma_27b":
        display  = "MedGemma-27B"
        model_id = "google/medgemma-27b-it"
        tokenizer = AutoTokenizer.from_pretrained(
            model_id, token=HF_TOKEN, use_fast=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            quantization_config=BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=False,
                bnb_4bit_quant_type="nf4"),
            device_map="balanced_low_0",
            low_cpu_mem_usage=True,
            offload_folder="offload",
            offload_state_dict=True,
            token=HF_TOKEN)
        model.eval()

        def generate(prompt):
            t0 = time.time()
            messages = [
                {"role": "system", "content": "You are an expert biochemist."},
                {"role": "user",   "content": prompt},
            ]
            try:
                input_text = tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True)
            except Exception:
                input_text = prompt
            inputs = tokenizer(input_text, return_tensors="pt",
                               truncation=True, max_length=2048).to(model.device)
            input_len = inputs["input_ids"].shape[1]
            with torch.inference_mode():
                outputs = model.generate(
                    **inputs, max_new_tokens=512, do_sample=False,
                    use_cache=True, pad_token_id=tokenizer.eos_token_id)
            text = tokenizer.decode(outputs[0][input_len:],
                                    skip_special_tokens=True).strip()
            return {"text": text, "input_tok": input_len,
                    "output_tok": len(outputs[0]) - input_len,
                    "latency_s": round(time.time()-t0, 2)}

    print(f"\n{display} shuffled — {len(todo)} remaining / {len(shuffled_paths)*2} total")

    for i, (path, condition) in enumerate(todo):
        hid    = path["hypokg_id"]
        source = (path["path_text"].split("\n")[0]
                  .split("[enzyme_kinetics]")[0].strip()
                  .split(",")[0].strip())
        term   = path["end_node"].replace("disease:", "").replace("_", " ").strip()
        try:
            result = generate(make_prompt(path["path_text"], condition))
            out = {
                "hypokg_id":          hid,
                "path_id":            path.get("path_id", hid),
                "tier_calibrated":    path["tier_calibrated"],
                "crossing_count":     path["crossing_count"],
                "start_node":         path["start_node"],
                "end_node":           path["end_node"],
                "in_gold_subset":     path["in_gold_subset"],
                "in_verified_subset": path["in_verified_subset"],
                "model_key":          model_key,
                "model_display":      display,
                "condition":          condition,
                "path_type":          "shuffled",
                "prompt_version":     "v2_shuffled",
                "hypothesis_text":    result["text"],
                "input_tokens":       result["input_tok"],
                "output_tokens":      result["output_tok"],
                "latency_s":          result["latency_s"],
                "generated_at":       datetime.now(timezone.utc).isoformat(),
            }
            with open(out_file, "a") as f:
                f.write(json.dumps(out) + "\n")
            if (i + 1) % 25 == 0:
                gc.collect()
                torch.cuda.empty_cache()
            print(f"  [{i+1}/{len(todo)}] {hid} C{condition} "
                  f"src={source[:12]:<12} term={term[:20]:<20} "
                  f"{result['input_tok']}in/{result['output_tok']}out "
                  f"{result['latency_s']}s")
        except Exception as e:
            print(f"  ERROR {hid} C{condition}: {e}")
            gc.collect()
            torch.cuda.empty_cache()

    del model, tokenizer
    gc.collect()
    torch.cuda.empty_cache()

    records = [json.loads(l) for l in open(out_file) if l.strip()]
    expected = len(shuffled_paths) * 2
    status = "✅" if len(records) == expected else f"⚠️  {len(records)}/{expected}"
    print(f"\n{status} {display} shuffled complete")


# ── Entry point ───────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--step", required=True,
                        choices=["shuffle", "generate"])
    parser.add_argument("--model", choices=list(API_MODELS.keys()) + list(HF_MODELS))
    args = parser.parse_args()

    if args.step == "shuffle":
        create_shuffled_paths()
    elif args.step == "generate":
        if args.model is None:
            parser.error("--model required for --step generate")
        if args.model in HF_MODELS:
            run_hf(args.model)
        else:
            run_api(args.model)
