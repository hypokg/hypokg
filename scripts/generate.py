import json, os, time, argparse
from datetime import datetime, timezone
import openai
import anthropic

HYPOKG_FOLDER = os.environ.get("HYPOKG_FOLDER", "./")
PATHS_FILE    = f"{HYPOKG_FOLDER}/data/paths/hypokg_550_selected.jsonl"
GEN_FOLDER    = f"{HYPOKG_FOLDER}/generations"
os.makedirs(GEN_FOLDER, exist_ok=True)

OPENAI_API_KEY    = os.environ.get("OPENAI_API_KEY", "")
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
TOGETHER_API_KEY  = os.environ.get("TOGETHER_API_KEY", "")

oai_client      = openai.OpenAI(api_key=OPENAI_API_KEY)
ant_client      = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
together_client = openai.OpenAI(
    api_key=TOGETHER_API_KEY,
    base_url="https://api.together.xyz/v1"
)


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
6. State one falsifiable experimental prediction with a specific measurable output (e.g., enzyme activity, metabolite concentration, assay readout, or cellular phenotype).
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
- Which localized mechanistic theme in the path appears most biologically meaningful, even if the full path may not represent a direct causal chain?

Do not assume every edge represents a direct causal relationship.
Do not feel pressure to connect the source entity to the terminal disease if a stronger, more localized biological claim exists within the path.
Use the graph as mechanistic guidance, not as a deterministic script.

Then generate:
1. A short mechanistic rationale identifying the most biologically meaningful substructure in the path and why you selected it over other edges.
2. One coherent, falsifiable scientific hypothesis grounded in that strongest mechanistic theme.
3. Include a specific measurable prediction inside the hypothesis.

Format your response as:
RATIONALE: [brief mechanistic interpretation, max 80 words]
HYPOTHESIS: We hypothesize that [concise mechanistic hypothesis including the falsifiable prediction, max 150 words]

Do not add any additional sections beyond RATIONALE and HYPOTHESIS."""

    elif condition == 4:
        terminal = get_terminal_name(record)
        return f"""You are an expert biochemist specializing in metabolic disease mechanisms.

Source enzyme: {source}
Target disease: {terminal}

Your task is to generate a testable scientific hypothesis proposing a specific mechanistic connection between this source enzyme and this target disease.

Your hypothesis must:
1. Name the specific source enzyme — do not generalize to enzyme families.
2. Propose a specific mechanistic pathway or biological process connecting this enzyme to the target disease.
3. Make a specific, falsifiable prediction: what experimental result would confirm it, and what would refute it.
4. Be grounded in known biochemistry.
5. Connect specifically to {terminal} — do not connect to a different disease.

Begin with "We hypothesize that" and limit to 150 words."""


def generate_openai(prompt, model_id):
    t0 = time.time()
    resp = oai_client.chat.completions.create(
        model=model_id,
        max_tokens=512,
        messages=[{"role": "user", "content": prompt}]
    )
    return {
        "text":       resp.choices[0].message.content,
        "input_tok":  resp.usage.prompt_tokens,
        "output_tok": resp.usage.completion_tokens,
        "latency_s":  round(time.time() - t0, 2),
    }


def generate_claude(prompt, model_id):
    t0 = time.time()
    resp = ant_client.messages.create(
        model=model_id,
        max_tokens=512,
        messages=[{"role": "user", "content": prompt}]
    )
    return {
        "text":       resp.content[0].text,
        "input_tok":  resp.usage.input_tokens,
        "output_tok": resp.usage.output_tokens,
        "latency_s":  round(time.time() - t0, 2),
    }


def generate_together(prompt, model_id):
    t0 = time.time()
    resp = together_client.chat.completions.create(
        model=model_id,
        max_tokens=512,
        messages=[{"role": "user", "content": prompt}]
    )
    return {
        "text":       resp.choices[0].message.content,
        "input_tok":  resp.usage.prompt_tokens,
        "output_tok": resp.usage.completion_tokens,
        "latency_s":  round(time.time() - t0, 2),
    }


MODELS = {
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
    cfg = MODELS[model_key]

    paths = []
    with open(PATHS_FILE) as f:
        for line in f:
            paths.append(json.loads(line.strip()))

    for condition in conditions:
        done  = load_done(model_key, condition)
        todo  = [(p, condition) for p in paths
                 if (p["hypokg_id"], condition) not in done]
        total = len(paths)
        out_f = checkpoint_path(model_key, condition)

        print(f"\n{cfg['display']} C{condition} — {len(todo)} remaining / {total} total")

        for i, (record, cond) in enumerate(todo):
            prompt = make_prompt(record, cond)
            try:
                if cfg["backend"] == "openai":
                    result = generate_openai(prompt, cfg["model_id"])
                elif cfg["backend"] == "claude":
                    result = generate_claude(prompt, cfg["model_id"])
                elif cfg["backend"] == "together":
                    result = generate_together(prompt, cfg["model_id"])

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
                    "model_display":      cfg["display"],
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

                print(f"  [{i+1}/{len(todo)}] {record['hypokg_id']} C{cond} "
                      f"{result['input_tok']}in/{result['output_tok']}out "
                      f"{result['latency_s']}s")

            except Exception as e:
                print(f"  ERROR {record['hypokg_id']} C{cond}: {e}")
                time.sleep(3)

            time.sleep(cfg["rate_limit_s"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=list(MODELS.keys()))
    parser.add_argument("--conditions", nargs="+", type=int, default=[1, 2, 3])
    args = parser.parse_args()

    run(args.model, args.conditions)
