"""
HypoKG — Shuffled Path Control: Judge + Analysis
Judges all shuffled hypotheses with GPT-4o (primary) and Gemini (secondary),
then compares scores against original paths to confirm genuine path utilization.

Usage:
  python shuffled_judge.py --step judge
  python shuffled_judge.py --step analyze
"""

import json, os, time, re, argparse
from datetime import datetime, timezone
from collections import defaultdict
import openai
from google import genai
from google.genai import types
from scipy import stats
import numpy as np

HYPOKG_FOLDER    = os.environ.get("HYPOKG_FOLDER", "./")
SHUFFLED_DIR     = f"{HYPOKG_FOLDER}/shuffled_control"
SHUFFLED_GEN_DIR = f"{SHUFFLED_DIR}/generations"
SHUFFLED_PATHS   = f"{SHUFFLED_DIR}/shuffled_paths.jsonl"
ORIG_JUDGE_FILE  = f"{HYPOKG_FOLDER}/judge/judge_full_gpt4o.jsonl"

SHUF_JUDGE_GPT4O  = f"{SHUFFLED_DIR}/shuffled_judge_gpt4o.jsonl"
SHUF_JUDGE_GEMINI = f"{SHUFFLED_DIR}/shuffled_judge_gemini.jsonl"
ANALYSIS_FILE     = f"{SHUFFLED_DIR}/shuffled_analysis.txt"

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")

oai_client    = openai.OpenAI(api_key=OPENAI_API_KEY)
gemini_client = genai.Client(api_key=GEMINI_API_KEY)

MODELS = ["claude_sonnet", "gpt4o", "llama_70b",
          "qwen3_235b", "biomistral_7b", "medgemma_27b"]

JUDGE_SYSTEM_PROMPT = """You are an expert scientific evaluator assessing the quality of AI-generated biochemical hypotheses.

You will be given:
1. A source enzyme/gene
2. A target disease
3. A knowledge graph path connecting the source to the disease
4. A generated hypothesis
5. A context line stating what evidence was available to the generator

Your task is to score the hypothesis on five criteria using a 1-5 scale.
Return ONLY a valid JSON object with your scores and reasoning. Nothing else.

SCORING SCALE:
1 = Poor
2 = Weak
3 = Acceptable / mixed
4 = Strong
5 = Excellent

THE FIVE CRITERIA:

CRITERION 1 — Path/Task Relevance (1-5)
Does the hypothesis address the assigned source-to-terminal biological problem?
- Score 5: Directly addresses source gene and terminal disease through a coherent biological mechanism
- Score 4: Addresses source and terminal with minor gaps
- Score 3: Partially addresses the path
- Score 2: Weakly connected
- Score 1: Addresses a completely different disease or biological question

CRITERION 2 — Mechanistic Specificity and Consistency (1-5)
Does the hypothesis propose a specific, biologically coherent mechanism?
- Score 5: Names specific enzyme, substrate, product, cell type, and mechanism
- Score 4: Specific mechanism with minor gaps
- Score 3: Some specificity but relies on pathway-level language
- Score 2: Vague mechanism
- Score 1: Biochemically incoherent

CRITERION 3 — Experimental Testability and Operationalization (1-5)
Can the hypothesis be tested with a concrete perturbation, model system, assay, and measurable outcome?
- Score 5: Names perturbation, model system, measurable outcome, and falsification
- Score 4: Clear experimental design with one missing element
- Score 3: Testable in principle but missing key details
- Score 2: Vague prediction
- Score 1: No testable prediction

CRITERION 4 — Non-Trivial Novelty / Knowledge Advancement (1-5)
Does the hypothesis generate a new scientific question beyond restating textbook knowledge?
- Score 5: Genuinely new mechanistic connection
- Score 4: Extends known biology in a non-obvious direction
- Score 3: Has some novel element but mostly builds on well-known mechanisms
- Score 2: Mostly restates established biology
- Score 1: Directly restates a textbook mechanism

CRITERION 5 — Evidence-Proportionality and Full-Path Use (1-5)
Does the hypothesis use the KG path intelligently?
CRITICAL: If generator was shown the full KG path, score based on how intelligently
it used the path evidence. NOTE: This path has been shuffled — intermediate nodes
are in randomized order. A hypothesis that ignores the shuffled structure and invents
its own mechanism should score low on this criterion.
- Score 5: Uses both strong and upstream path elements intelligently
- Score 4: Uses strong path elements well
- Score 3: Uses only the most obvious path element
- Score 2: Overclaims beyond what the evidence supports
- Score 1: Completely ignores the path when one was shown

Return ONLY this JSON structure:
{
  "path_task_relevance": <1-5>,
  "mechanistic_specificity": <1-5>,
  "experimental_testability": <1-5>,
  "nontrivial_novelty": <1-5>,
  "evidence_proportionality": <1-5>,
  "total_score": <sum 5-25>,
  "reasoning": "<2-3 sentences>"
}"""


def parse_scores(text):
    text = re.sub(r"```json\s*", "", text.strip())
    text = re.sub(r"```\s*", "", text).strip()
    jm = re.search(r"\{.*\}", text, re.DOTALL)
    if not jm:
        return None
    try:
        scores = json.loads(jm.group())
        required = ["path_task_relevance", "mechanistic_specificity",
                    "experimental_testability", "nontrivial_novelty",
                    "evidence_proportionality"]
        for field in required:
            if field not in scores or not (1 <= int(scores[field]) <= 5):
                return None
            scores[field] = int(scores[field])
        scores["total_score"] = sum(scores[f] for f in required)
        return scores
    except Exception:
        return None


def call_gpt4o(prompt, retries=3):
    for attempt in range(retries):
        try:
            t0   = time.time()
            resp = oai_client.chat.completions.create(
                model="gpt-4o", max_tokens=400, temperature=0,
                messages=[{"role": "system", "content": JUDGE_SYSTEM_PROMPT},
                          {"role": "user",   "content": prompt}])
            sc = parse_scores(resp.choices[0].message.content)
            if sc is None:
                time.sleep(2)
                continue
            return {"scores": sc,
                    "in_tok": resp.usage.prompt_tokens,
                    "out_tok": resp.usage.completion_tokens,
                    "latency": round(time.time()-t0, 2)}
        except Exception as e:
            print(f"    gpt4o error {attempt+1}: {e}")
            time.sleep(3)
    return None


def call_gemini(prompt, retries=3):
    for attempt in range(retries):
        try:
            t0   = time.time()
            resp = gemini_client.models.generate_content(
                model="gemini-2.5-flash",
                contents=prompt,
                config=types.GenerateContentConfig(
                    system_instruction=JUDGE_SYSTEM_PROMPT,
                    max_output_tokens=2048, temperature=0,
                    thinking_config=types.ThinkingConfig(thinking_budget=0)))
            sc = parse_scores(resp.text)
            if sc is None:
                print(f"    gemini parse fail {attempt+1}: {repr(resp.text[:80])}")
                time.sleep(2)
                continue
            return {"scores": sc, "in_tok": 0,
                    "out_tok": len(resp.text.split()),
                    "latency": round(time.time()-t0, 2)}
        except Exception as e:
            print(f"    gemini error {attempt+1}: {e}")
            time.sleep(3)
    return None


def make_judge_input(hyp, path_text):
    condition = hyp["condition"]
    if condition in (2, 3):
        context = ("Evidence available to generator: full knowledge graph path "
                   "shown below. NOTE: This path has been SHUFFLED — "
                   "intermediate nodes are in randomized order.")
    else:
        context = f"Evidence available to generator: condition {condition}."

    source = (path_text.split("\n")[0]
              .split("[enzyme_kinetics]")[0].strip()
              .split(",")[0].strip())
    terminal = hyp["end_node"].replace("disease:", "").replace("_", " ").strip()

    return (f"Source enzyme/gene: {source}\n"
            f"Target disease: {terminal}\n"
            f"{context}\n\n"
            f"Knowledge graph path:\n{path_text}\n\n"
            f"Generated hypothesis:\n{hyp['hypothesis_text']}\n\n"
            f"Score this hypothesis on the five criteria. Return only the JSON object.")


def load_done(fpath):
    done = set()
    if os.path.exists(fpath):
        for line in open(fpath):
            r = json.loads(line.strip())
            done.add((r["hypokg_id"], r["model_key"], r["condition"]))
    return done


def run_judge():
    shuffled_lookup = {}
    for line in open(SHUFFLED_PATHS):
        r = json.loads(line.strip())
        shuffled_lookup[r["hypokg_id"]] = r

    hypotheses = []
    for model_key in MODELS:
        fpath = f"{SHUFFLED_GEN_DIR}/{model_key}_shuffled.jsonl"
        if not os.path.exists(fpath):
            print(f"  missing: {fpath}")
            continue
        for line in open(fpath):
            hypotheses.append(json.loads(line.strip()))

    print(f"Shuffled hypotheses to judge: {len(hypotheses):,}")

    done_gpt4o  = load_done(SHUF_JUDGE_GPT4O)
    done_gemini = load_done(SHUF_JUDGE_GEMINI)
    errors = 0

    for i, hyp in enumerate(hypotheses):
        hid       = hyp["hypokg_id"]
        model_key = hyp["model_key"]
        condition = hyp["condition"]
        key       = (hid, model_key, condition)

        path = shuffled_lookup.get(hid)
        if path is None:
            continue

        prompt = make_judge_input(hyp, path["path_text"])

        source   = (path["path_text"].split("\n")[0]
                    .split("[enzyme_kinetics]")[0].strip()
                    .split(",")[0].strip())
        terminal = path["end_node"].replace("disease:", "").replace("_", " ").strip()

        for judge_fn, done_set, out_file, judge_name in [
            (call_gpt4o,  done_gpt4o,  SHUF_JUDGE_GPT4O,  "gpt4o"),
            (call_gemini, done_gemini, SHUF_JUDGE_GEMINI, "gemini"),
        ]:
            if key in done_set:
                continue
            result = judge_fn(prompt)
            if result is None:
                errors += 1
                print(f"  ❌ {hid} {model_key} C{condition} {judge_name}")
                continue
            record = {
                "hypokg_id":               hid,
                "model_key":               model_key,
                "model_display":           hyp.get("model_display", model_key),
                "condition":               condition,
                "path_type":               "shuffled",
                "tier_calibrated":         hyp["tier_calibrated"],
                "crossing_count":          hyp["crossing_count"],
                "in_gold_subset":          hyp["in_gold_subset"],
                "in_verified_subset":      hyp["in_verified_subset"],
                "source_symbol":           source,
                "terminal_name":           terminal,
                "judge":                   judge_name,
                "judge_prompt_version":    "v2_fixed_criterion5",
                "path_task_relevance":     result["scores"]["path_task_relevance"],
                "mechanistic_specificity": result["scores"]["mechanistic_specificity"],
                "experimental_testability":result["scores"]["experimental_testability"],
                "nontrivial_novelty":      result["scores"]["nontrivial_novelty"],
                "evidence_proportionality":result["scores"]["evidence_proportionality"],
                "total_score":             result["scores"]["total_score"],
                "reasoning":               result["scores"].get("reasoning", ""),
                "judge_input_tokens":      result["in_tok"],
                "judge_output_tokens":     result["out_tok"],
                "judge_latency_s":         result["latency"],
                "scored_at":               datetime.now(timezone.utc).isoformat(),
            }
            with open(out_file, "a") as f:
                f.write(json.dumps(record) + "\n")
            done_set.add(key)

        if (i + 1) % 100 == 0:
            print(f"  [{i+1}/{len(hypotheses)}] "
                  f"gpt4o={len(done_gpt4o)} gemini={len(done_gemini)} errors={errors}")
        time.sleep(0.3)

    print(f"\nJudging complete. Errors: {errors}")
    print(f"  GPT-4o:  {len(done_gpt4o):,}")
    print(f"  Gemini:  {len(done_gemini):,}")


# ── Analysis ──────────────────────────────────────────────────────

def run_analysis():
    orig_scores = {}
    for line in open(ORIG_JUDGE_FILE):
        r = json.loads(line.strip())
        if r["condition"] in (2, 3):
            orig_scores[(r["hypokg_id"], r["model_key"], r["condition"])] = r

    shuf_scores = {}
    for line in open(SHUF_JUDGE_GPT4O):
        r = json.loads(line.strip())
        shuf_scores[(r["hypokg_id"], r["model_key"], r["condition"])] = r

    common = set(orig_scores) & set(shuf_scores)
    print(f"Original scores (C2+C3): {len(orig_scores):,}")
    print(f"Shuffled scores:         {len(shuf_scores):,}")
    print(f"Matched pairs:           {len(common):,}")

    def mean(vals):
        return round(sum(vals)/len(vals), 3) if vals else 0.0

    METRICS = [
        ("total_score",             "Total Score"),
        ("evidence_proportionality","Evidence Proportionality"),
        ("path_task_relevance",     "Path/Task Relevance"),
        ("mechanistic_specificity", "Mechanistic Specificity"),
        ("nontrivial_novelty",      "Novelty"),
    ]

    print(f"\n{'─'*72}")
    print(f"{'Metric':<28} {'Cond':<5} {'Orig':>7} {'Shuf':>7} {'Drop':>7} {'p':>10} {'sig':>5}")
    print(f"{'─'*72}")

    results = {}
    for metric, label in METRICS:
        for cond in [2, 3]:
            keys = [k for k in common if k[2] == cond]
            orig_vals = [orig_scores[k][metric] for k in keys]
            shuf_vals = [shuf_scores[k][metric] for k in keys]
            _, p = stats.wilcoxon(orig_vals, shuf_vals, alternative="greater")
            drop = round(mean(orig_vals) - mean(shuf_vals), 3)
            sig  = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"
            results[(metric, cond)] = {
                "orig": mean(orig_vals), "shuf": mean(shuf_vals),
                "drop": drop, "p": p, "n": len(keys)
            }
            print(f"  {label:<26} C{cond:<4} "
                  f"{mean(orig_vals):>7.3f} {mean(shuf_vals):>7.3f} "
                  f"{drop:>+7.3f} {p:>10.4f} {sig:>5}")
        print()

    print(f"\n{'─'*72}")
    print("EVIDENCE PROPORTIONALITY DROP BY MODEL")
    print(f"{'─'*72}")
    for model_key in MODELS:
        for cond in [2, 3]:
            keys = [k for k in common if k[1] == model_key and k[2] == cond]
            if not keys:
                continue
            orig_vals = [orig_scores[k]["evidence_proportionality"] for k in keys]
            shuf_vals = [shuf_scores[k]["evidence_proportionality"] for k in keys]
            try:
                _, p = stats.wilcoxon(orig_vals, shuf_vals, alternative="greater")
            except Exception:
                p = 1.0
            drop = round(mean(orig_vals) - mean(shuf_vals), 3)
            sig  = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"
            print(f"  {model_key:<20} C{cond}  "
                  f"orig={mean(orig_vals):.3f} shuf={mean(shuf_vals):.3f} "
                  f"drop={drop:+.3f} {sig}  n={len(keys)}")

    r = results.get(("evidence_proportionality", 2), {})
    summary = (
        f"Shuffled Path Control\n"
        f"N matched pairs per condition: {r.get('n', 0)}\n\n"
        f"Evidence proportionality:\n"
        f"  C2: {results[('evidence_proportionality',2)]['orig']:.3f} → "
        f"{results[('evidence_proportionality',2)]['shuf']:.3f} "
        f"(drop={results[('evidence_proportionality',2)]['drop']:+.3f}, "
        f"p={results[('evidence_proportionality',2)]['p']:.6f})\n"
        f"  C3: {results[('evidence_proportionality',3)]['orig']:.3f} → "
        f"{results[('evidence_proportionality',3)]['shuf']:.3f} "
        f"(drop={results[('evidence_proportionality',3)]['drop']:+.3f}, "
        f"p={results[('evidence_proportionality',3)]['p']:.6f})\n"
    )
    print(f"\n{summary}")
    with open(ANALYSIS_FILE, "w") as f:
        f.write(summary)
    print(f"Saved: {ANALYSIS_FILE}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--step", required=True, choices=["judge", "analyze"])
    args = parser.parse_args()

    if args.step == "judge":
        run_judge()
    elif args.step == "analyze":
        run_analysis()
