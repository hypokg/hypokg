"""
HypoKG — HuggingFace Model Generation
BioMistral-7B and MedGemma-27B
Requires A100 GPU (80GB) for MedGemma. BioMistral runs on T4.
Run one model at a time — GPU memory does not allow both simultaneously.
"""

import os
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

import json, time, gc, argparse
from datetime import datetime, timezone
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoProcessor,
    AutoModelForImageTextToText,
    BitsAndBytesConfig,
)
from huggingface_hub import login

HF_TOKEN      = os.environ.get("HF_TOKEN", "")
HYPOKG_FOLDER = os.environ.get("HYPOKG_FOLDER", "./")
PATHS_FILE    = f"{HYPOKG_FOLDER}/data/paths/hypokg_550_selected.jsonl"
GEN_FOLDER    = f"{HYPOKG_FOLDER}/generations"
os.makedirs(GEN_FOLDER, exist_ok=True)

login(token=HF_TOKEN)


def get_source_entity(record):
    return (
        record["path_text"].split("\n")[0]
        .split("[enzyme_kinetics]")[0].strip()
        .split(",")[0].strip()
    )


def get_terminal_name(record):
    return (
        record["end_node"]
        .replace("disease:", "")
        .replace("_", " ")
        .strip()
    )


def make_prompt(record, condition):
    source = get_source_entity(record)

    if condition == 1:
        return f"""You are an expert biochemist specializing in metabolic disease mechanisms.

Source enzyme: {source}
Target domain: disease mechanism

Generate a testable scientific hypothesis connecting this enzyme to a disease mechanism.
Your hypothesis must:
1. Name the specific enzyme — do not generalize to enzyme families.
2. Propose a specific mechanistic connection to a disease.
3. Make a specific, falsifiable prediction: what experimental result would confirm it, and what would refute it.
4. Be grounded in known biochemistry.

Begin with "We hypothesize that" and limit to 150 words."""

    elif condition == 2:
        return f"""You are an expert biochemist. The following knowledge graph path describes a multi-step mechanistic relationship between biological entities:

{record["path_text"]}

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

{record["path_text"]}

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

    elif condition == 4:
        terminal = get_terminal_name(record)
        return f"""You are an expert biochemist specializing in metabolic disease mechanisms.

Source enzyme: {source}
Target disease: {terminal}

Generate a testable scientific hypothesis proposing a specific mechanistic connection between this source enzyme and this target disease.

Your hypothesis must:
1. Name the specific source enzyme — do not generalize to enzyme families.
2. Propose a specific mechanistic pathway connecting this enzyme to the target disease.
3. Make a specific, falsifiable prediction.
4. Be grounded in known biochemistry.
5. Connect specifically to {terminal} — do not connect to a different disease.

Begin with "We hypothesize that" and limit to 150 words."""


def load_biomistral():
    model_id = "BioMistral/BioMistral-7B"
    tokenizer = AutoTokenizer.from_pretrained(
        model_id, token=HF_TOKEN, trust_remote_code=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=BitsAndBytesConfig(load_in_8bit=True),
        device_map="auto",
        token=HF_TOKEN,
        trust_remote_code=True,
    )
    model.eval()
    print(f"BioMistral loaded — GPU: {torch.cuda.memory_allocated()/1024**3:.1f} GB")
    return model, tokenizer


def load_medgemma():
    model_id = "google/medgemma-27b-it"
    processor = AutoProcessor.from_pretrained(model_id, token=HF_TOKEN)
    model = AutoModelForImageTextToText.from_pretrained(
        model_id,
        quantization_config=BitsAndBytesConfig(load_in_8bit=True),
        device_map="auto",
        token=HF_TOKEN,
        torch_dtype=torch.bfloat16,
    )
    model.eval()
    print(f"MedGemma loaded — GPU: {torch.cuda.memory_allocated()/1024**3:.1f} GB")
    return model, processor


def generate_causal(prompt, tokenizer, model):
    t0 = time.time()
    try:
        input_text = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False, add_generation_prompt=True
        )
    except Exception:
        input_text = prompt

    inputs = tokenizer(
        input_text, return_tensors="pt",
        truncation=True, max_length=2048
    ).to(model.device)
    input_len = inputs["input_ids"].shape[1]

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=512,
            do_sample=False,
            temperature=None,
            top_p=None,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    text = tokenizer.decode(outputs[0][input_len:], skip_special_tokens=True).strip()
    return {"text": text, "input_tok": input_len,
            "output_tok": len(outputs[0]) - input_len,
            "latency_s": round(time.time() - t0, 2)}


def generate_medgemma(prompt, processor, model):
    t0 = time.time()
    messages = [
        {"role": "system", "content": [{"type": "text", "text": "You are an expert biochemist."}]},
        {"role": "user",   "content": [{"type": "text", "text": prompt}]},
    ]
    inputs = processor.apply_chat_template(
        messages, add_generation_prompt=True,
        tokenize=True, return_dict=True, return_tensors="pt"
    ).to(model.device, dtype=torch.bfloat16)
    input_len = inputs["input_ids"].shape[-1]

    with torch.inference_mode():
        outputs = model.generate(**inputs, max_new_tokens=512, do_sample=False)

    text = processor.decode(outputs[0][input_len:], skip_special_tokens=True).strip()
    return {"text": text, "input_tok": input_len,
            "output_tok": len(outputs[0]) - input_len,
            "latency_s": round(time.time() - t0, 2)}


def checkpoint_path(model_key, condition):
    suffix = "c4" if condition == 4 else "v2"
    return f"{GEN_FOLDER}/{model_key}_generations_{suffix}.jsonl"


def load_done(model_key, condition):
    done = set()
    fpath = checkpoint_path(model_key, condition)
    if os.path.exists(fpath):
        with open(fpath) as f:
            for line in f:
                r = json.loads(line.strip())
                done.add((r["hypokg_id"], r["condition"]))
    return done


def run(model_key, conditions):
    paths = []
    with open(PATHS_FILE) as f:
        for line in f:
            paths.append(json.loads(line.strip()))

    if model_key == "biomistral_7b":
        model, tok_or_proc = load_biomistral()
        gen_fn = lambda p: generate_causal(p, tok_or_proc, model)
        display = "BioMistral-7B"
    elif model_key == "medgemma_27b":
        model, tok_or_proc = load_medgemma()
        gen_fn = lambda p: generate_medgemma(p, tok_or_proc, model)
        display = "MedGemma-27B"

    for condition in conditions:
        done  = load_done(model_key, condition)
        todo  = [(p, condition) for p in paths
                 if (p["hypokg_id"], condition) not in done]
        out_f = checkpoint_path(model_key, condition)

        print(f"\n{display} C{condition} — {len(todo)} remaining / {len(paths)} total")

        for i, (record, cond) in enumerate(todo):
            prompt = make_prompt(record, cond)
            try:
                result = gen_fn(prompt)
                out = {
                    "hypokg_id":          record["hypokg_id"],
                    "path_id":            record["path_id"],
                    "tier_calibrated":    record["tier_calibrated"],
                    "crossing_count":     record["crossing_count"],
                    "start_node":         record["start_node"],
                    "end_node":           record["end_node"],
                    "in_gold_subset":     record["in_gold_subset"],
                    "in_verified_subset": record["in_verified_subset"],
                    "model_key":          model_key,
                    "model_display":      display,
                    "condition":          cond,
                    "prompt_version":     "v2",
                    "hypothesis_text":    result["text"],
                    "input_tokens":       result["input_tok"],
                    "output_tokens":      result["output_tok"],
                    "latency_s":          result["latency_s"],
                    "generated_at":       datetime.now(timezone.utc).isoformat(),
                }
                with open(out_f, "a") as f:
                    f.write(json.dumps(out) + "\n")

                if (i + 1) % 25 == 0:
                    gc.collect()
                    torch.cuda.empty_cache()

                print(f"  [{i+1}/{len(todo)}] {record['hypokg_id']} C{cond} "
                      f"src={get_source_entity(record)[:12]:<12} "
                      f"{result['input_tok']}in/{result['output_tok']}out "
                      f"{result['latency_s']}s")

            except Exception as e:
                print(f"  ERROR {record['hypokg_id']} C{cond}: {e}")
                gc.collect()
                torch.cuda.empty_cache()

    del model, tok_or_proc
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=["biomistral_7b", "medgemma_27b"])
    parser.add_argument("--conditions", nargs="+", type=int, default=[1, 2, 3])
    args = parser.parse_args()
    run(args.model, args.conditions)
