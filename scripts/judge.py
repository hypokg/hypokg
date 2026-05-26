import json, os, time, re
from datetime import datetime, timezone
import openai
from google import genai
from google.genai import types

HYPOKG_FOLDER = os.environ.get("HYPOKG_FOLDER", "./")
PATHS_FILE    = f"{HYPOKG_FOLDER}/data/paths/hypokg_550_selected.jsonl"
GEN_FOLDER    = f"{HYPOKG_FOLDER}/generations"
JUDGE_FOLDER  = f"{HYPOKG_FOLDER}/judge"
os.makedirs(JUDGE_FOLDER, exist_ok=True)

OPENAI_API_KEY  = os.environ.get("OPENAI_API_KEY", "")
TOGETHER_API_KEY = os.environ.get("TOGETHER_API_KEY", "")
GEMINI_API_KEY  = os.environ.get("GEMINI_API_KEY", "")

oai_client    = openai.OpenAI(api_key=OPENAI_API_KEY)
qwen_client   = openai.OpenAI(api_key=TOGETHER_API_KEY, base_url="https://api.together.xyz/v1")
gemini_client = genai.Client(api_key=GEMINI_API_KEY)

GENERATION_FILES = {
    "claude_sonnet": f"{GEN_FOLDER}/claude_sonnet_generations_v2.jsonl",
    "gpt4o":         f"{GEN_FOLDER}/gpt4o_generations_v2.jsonl",
    "llama_70b":     f"{GEN_FOLDER}/llama_70b_generations_v2.jsonl",
    "qwen3_235b":    f"{GEN_FOLDER}/qwen3_235b_generations_v2.jsonl",
    "biomistral_7b": f"{GEN_FOLDER}/biomistral_7b_generations_v2.jsonl",
    "medgemma_27b":  f"{GEN_FOLDER}/medgemma_27b_generations_v2.jsonl",
}

C4_FILES = {k: v.replace("_v2.jsonl", "_c4.jsonl") for k, v in GENERATION_FILES.items()}

MODEL_ORDER = list(GENERATION_FILES.keys())

# GPT-4o cannot judge its own outputs — Qwen3 steps in as primary
JUDGE_ROUTING = {
    "claude_sonnet": ("gpt4o",  "gemini"),
    "gpt4o":         ("qwen3",  "gemini"),
    "llama_70b":     ("gpt4o",  "gemini"),
    "qwen3_235b":    ("gpt4o",  "gemini"),
    "biomistral_7b": ("gpt4o",  "gemini"),
    "medgemma_27b":  ("gpt4o",  "gemini"),
}

JUDGE_FILES = {
    "gpt4o":  f"{JUDGE_FOLDER}/judge_full_gpt4o.jsonl",
    "qwen3":  f"{JUDGE_FOLDER}/judge_full_qwen3.jsonl",
    "gemini": f"{JUDGE_FOLDER}/judge_full_gemini.jsonl",
}

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
Does it connect to the correct terminal disease through the given path rather than drifting to a different disease or biological question?
- Score 5: Directly addresses source gene and terminal disease through a coherent biological mechanism
- Score 4: Addresses source and terminal with minor gaps in the connection
- Score 3: Partially addresses the path — connects to terminal but mechanism is vague, or focuses only on intermediate nodes
- Score 2: Weakly connected — mentions the disease but mechanism does not involve the source gene meaningfully
- Score 1: Addresses a completely different disease or biological question

CRITERION 2 — Mechanistic Specificity and Consistency (1-5)
Does the hypothesis propose a specific, biologically coherent mechanism involving named enzymes, metabolites, reactions, cell types, or pathway logic? Is the mechanism consistent with known biochemistry?
- Score 5: Names specific enzyme, substrate, product, cell type, and mechanism — all biochemically accurate and supported
- Score 4: Specific mechanism with minor biochemical gaps or simplifications
- Score 3: Some specificity but relies on pathway-level language, has one inaccuracy, or uses biochemical terminology without explaining the actual reaction steps
- Score 2: Vague mechanism — names entities but does not explain how they connect, or uses technical-sounding language without mechanistic substance
- Score 1: Biochemically incoherent, contradicts known biochemistry, or uses only generic language

NOTE: Do not reward biochemical-sounding mechanisms if the intermediate reactions are vague, unsupported, or biologically questionable. The mechanistic chain must be biochemically justified step by step.

CRITERION 3 — Experimental Testability and Operationalization (1-5)
Can the hypothesis be tested with a concrete perturbation, model system, assay, and measurable outcome?
Is there an explicit falsification condition?
When quantitative thresholds or fold-changes are provided, are they biologically justified rather than unsupported precise numbers?
- Score 5: Names perturbation, model system, measurable outcome, and falsification — quantitative claims are justified or appropriately hedged
- Score 4: Clear experimental design with one missing element, or includes numbers framed as approximate predictions
- Score 3: Testable in principle but missing key details — or includes unjustified precise numbers stated as fact
- Score 2: Vague prediction — "may affect" or "could influence" without specifics
- Score 1: No testable prediction, untestable, or purely observational without experiment

CRITERION 4 — Non-Trivial Novelty / Knowledge Advancement (1-5)
Does the hypothesis generate a new scientific question beyond restating textbook knowledge?
Does it propose a new relationship, mechanism, modifier, dose-response, or angle not already established?
- Score 5: Proposes a genuinely new mechanistic connection or modifier not established in existing literature
- Score 4: Extends known biology in a non-obvious direction — one credible step beyond established knowledge
- Score 3: Has some novel element but mostly builds on well-known mechanisms
- Score 2: Mostly restates established biology — the core claim is already known
- Score 1: Directly restates a textbook disease mechanism with no new prediction

CRITERION 5 — Evidence-Proportionality and Full-Path Use (1-5)
Does the hypothesis use the KG path intelligently — preserving strong edges, treating weak edges cautiously, attempting to extract value from upstream elements?
Does the strength of the claim match the evidence provided to the generator?

CRITICAL RULE FOR SCORING THIS CRITERION:
- If the generator was shown the full KG path: score based on how intelligently it used the path evidence
- If the generator was NOT shown the KG path (endpoints only or source only): a high score requires that the proposed mechanistic bridge is either (a) explicitly framed as speculative and biologically plausible, or (b) directly supported by well-established published biochemistry. A hypothesis that invents a plausible-sounding but unsupported bridge, without acknowledging uncertainty, should score at most 3 regardless of fluency.

- Score 5: Uses both strong and upstream path elements intelligently — OR if no path shown, bridge is grounded in established biochemistry and explicitly proportionate to evidence
- Score 4: Uses strong path elements well — OR if no path shown, mechanism is plausible but missing explicit uncertainty framing
- Score 3: Uses only the most obvious path element — OR if no path shown, reasonable but unsupported bridge stated with overconfidence
- Score 2: Overclaims beyond evidence, or largely ignores the path when one was provided
- Score 1: Completely ignores path when shown — OR if no path shown, proposes a biochemically implausible mechanism

ANCHOR EXAMPLES:

--- ANCHOR 1: PATH-IRRELEVANT BUT FLUENT [Scores: 1, 2, 4, 3, 1] ---
Context: Generator was shown source enzyme only. Path was NOT shown.
Source: GSR (glutathione reductase)
Terminal: Aminoacylase-1 deficiency
Path: GSR → Glutathione → CHAC2/GCG1 → Gaucher disease → GUCY1A2 → MHC class I deficiency 2 → ABHD14A-ACY1 → Aminoacylase-1 deficiency
Hypothesis: We hypothesize that glutathione reductase deficiency in retinal pigment epithelial cells contributes to age-related macular degeneration through impaired glutathione regeneration, leading to lipofuscin accumulation and complement activation. Reduced GSR activity causes elevated GSSG and decreased GSH. This promotes A2E formation and inflammasome activation. RPE cells with siRNA-mediated GSR knockdown will show increased GSSG/GSH ratios, elevated A2E fluorescence, and increased IL-1β secretion, falsified if knockdown shows no lipofuscin increase.
Scores: {"path_task_relevance": 1, "mechanistic_specificity": 2, "experimental_testability": 4, "nontrivial_novelty": 3, "evidence_proportionality": 1}
Why: Connects GSR to AMD not Aminoacylase-1 deficiency. Evidence proportionality 1 because generator had no path and invented a completely different disease target.

--- ANCHOR 2: CORRECT BUT TEXTBOOK/TRIVIAL [Scores: 4, 4, 2, 1, 3] ---
Context: Generator was shown the full KG path.
Source: CDO1 (cysteine dioxygenase 1)
Terminal: Primary Hyperoxaluria Type 1
Path: CDO1 → Cyanoamino acid metabolism → AGXT → Primary Hyperoxaluria Type 1
Hypothesis: We hypothesize that loss-of-function mutations in AGXT disrupt glyoxylate metabolism in hepatocytes, leading to pathological accumulation of oxalate and causing Primary Hyperoxaluria Type 1. AGXT directly catalyzes transamination of glyoxylate to glycine, preventing oxalate overproduction. The upstream CDO1 link is metabolically tangential. Patients with biallelic AGXT mutations will exhibit reduced hepatic AGXT enzyme activity (less than 15% of wild-type) and elevated urinary oxalate (greater than 0.7 mmol/day), measurable via tandem mass spectrometry.
Scores: {"path_task_relevance": 4, "mechanistic_specificity": 4, "experimental_testability": 2, "nontrivial_novelty": 1, "evidence_proportionality": 3}
Why: Correct terminal and mechanism. Restates known AGXT-PH1 textbook biology. Uses strongest path edge but discards CDO1.

--- ANCHOR 3: STRONG BUT IMPERFECT [Scores: 4, 4, 4, 3, 3] ---
Context: Generator was shown the full KG path.
Source: CDO1 (cysteine dioxygenase 1)
Terminal: Primary Hyperoxaluria Type 1
Path: CDO1 → Cyanoamino acid metabolism → AGXT → Primary Hyperoxaluria Type 1
Hypothesis: RATIONALE: The strongest mechanistic connection is between AGXT enzyme dysfunction and Primary Hyperoxaluria Type 1. AGXT catalyzes alanine-glyoxylate aminotransferase activity and its deficiency directly causes oxalate accumulation. The cyanoamino acid connection appears weaker.
HYPOTHESIS: We hypothesize that reduced AGXT enzyme activity creates a metabolic bottleneck in glyoxylate detoxification, leading to increased glyoxylate-to-oxalate conversion. Cells with AGXT knockdown will show a dose-dependent increase in oxalate production with levels increasing 2 to 5 fold at 50% AGXT activity reduction, measurable by oxalate secretion assays.
Scores: {"path_task_relevance": 4, "mechanistic_specificity": 4, "experimental_testability": 4, "nontrivial_novelty": 3, "evidence_proportionality": 3}
Why: Correct target. Testable with dose-response but cell type unspecified. Drops CDO1 upstream entirely.

--- ANCHOR 4: ENDPOINT-ONLY, PLAUSIBLE BUT OVERCONFIDENT [Scores: 4, 3, 3, 3, 2] ---
Context: Generator was shown source enzyme and target disease only. The KG path was NOT shown.
Source: CDO1 (cysteine dioxygenase 1)
Terminal: Primary Hyperoxaluria Type 1
Hypothesis: We hypothesize that CDO1 deficiency leads to cysteine accumulation, which is then shunted through alternative metabolic pathways to produce glyoxylate, thereby exacerbating oxalate overproduction in Primary Hyperoxaluria Type 1 patients who already carry AGXT mutations. CDO1 knockdown in AGXT-deficient hepatocytes will show higher urinary oxalate than AGXT knockdown alone.
Scores: {"path_task_relevance": 4, "mechanistic_specificity": 3, "experimental_testability": 3, "nontrivial_novelty": 3, "evidence_proportionality": 2}
Why: Correctly connects CDO1 to PH1. Mechanism plausible but stated with overconfidence — no path was shown so bridge should be framed as speculative.

--- ANCHOR 5: EXCELLENT [Scores: 5, 5, 5, 5, 5] ---
Context: Generator was shown the full KG path.
Source: CDO1 (cysteine dioxygenase 1)
Terminal: Primary Hyperoxaluria Type 1
Path: CDO1 → Cyanoamino acid metabolism → AGXT → Primary Hyperoxaluria Type 1
Hypothesis: We hypothesize that CDO1 activity may modulate the severity of AGXT-deficient Primary Hyperoxaluria Type 1 by influencing cysteine-derived metabolite competition for shared peroxisomal processing capacity and pyridoxal phosphate cofactor availability. When CDO1 is impaired alongside AGXT deficiency, accumulated cysteine metabolites may reduce effective PLP availability or alter peroxisomal metabolic load, potentially amplifying oxalate overproduction beyond what AGXT deficiency alone would produce. We predict that CDO1 knockdown in AGXT-deficient human hepatocytes will produce greater oxalate accumulation than AGXT knockdown alone, measurable by LC-MS oxalate quantification, and that pyridoxine supplementation will partially rescue this amplified phenotype, supporting a PLP-sensitive modifier mechanism.
Scores: {"path_task_relevance": 5, "mechanistic_specificity": 5, "experimental_testability": 5, "nontrivial_novelty": 5, "evidence_proportionality": 5}
Why: Novel CDO1-as-modifier mechanism. Biochemically precise. Language appropriately hedged. Uses both CDO1 and AGXT intelligently.

NOW SCORE THE FOLLOWING HYPOTHESIS:

Return ONLY this JSON structure with no other text:
{
  "path_task_relevance": <1-5>,
  "mechanistic_specificity": <1-5>,
  "experimental_testability": <1-5>,
  "nontrivial_novelty": <1-5>,
  "evidence_proportionality": <1-5>,
  "total_score": <sum, 5-25>,
  "reasoning": "<2-3 sentences explaining the scores>"
}"""


def parse_source(path_text):
    return (path_text.split("\n")[0]
            .split("[enzyme_kinetics]")[0].strip()
            .split(",")[0].strip())


def parse_terminal(end_node):
    return end_node.replace("disease:", "").replace("_", " ").strip()


def make_judge_input(hyp):
    condition = hyp["condition"]
    if condition == 1:
        context = "Evidence available to generator: source enzyme name only. The KG path and terminal disease were NOT shown."
    elif condition in (2, 3):
        context = "Evidence available to generator: full knowledge graph path shown below."
    elif condition == 4:
        context = "Evidence available to generator: source enzyme name and target disease name only. The KG path was NOT shown."
    else:
        context = f"Evidence available to generator: condition {condition}."

    return (f"Source enzyme/gene: {hyp['source_symbol']}\n"
            f"Target disease: {hyp['terminal_name']}\n"
            f"{context}\n\n"
            f"Knowledge graph path:\n{hyp['path_text']}\n\n"
            f"Generated hypothesis:\n{hyp['hypothesis_text']}\n\n"
            f"Score this hypothesis on the five criteria. Return only the JSON object.")


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
                          {"role": "user",   "content": prompt}]
            )
            raw = resp.choices[0].message.content
            sc  = parse_scores(raw)
            if sc is None:
                time.sleep(2)
                continue
            return {"scores": sc, "in_tok": resp.usage.prompt_tokens,
                    "out_tok": resp.usage.completion_tokens,
                    "latency": round(time.time() - t0, 2)}
        except Exception as e:
            print(f"    gpt4o error {attempt+1}: {e}")
            time.sleep(3)
    return None


def call_qwen(prompt, retries=3):
    for attempt in range(retries):
        try:
            t0   = time.time()
            resp = qwen_client.chat.completions.create(
                model="Qwen/Qwen3-235B-A22B-Instruct-2507-tput",
                max_tokens=400, temperature=0,
                messages=[{"role": "system", "content": JUDGE_SYSTEM_PROMPT},
                          {"role": "user",   "content": prompt}]
            )
            raw = resp.choices[0].message.content
            sc  = parse_scores(raw)
            if sc is None:
                time.sleep(2)
                continue
            return {"scores": sc, "in_tok": resp.usage.prompt_tokens,
                    "out_tok": resp.usage.completion_tokens,
                    "latency": round(time.time() - t0, 2)}
        except Exception as e:
            print(f"    qwen error {attempt+1}: {e}")
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
                    max_output_tokens=2048,
                    temperature=0,
                    thinking_config=types.ThinkingConfig(thinking_budget=0),
                )
            )
            raw = resp.text
            sc  = parse_scores(raw)
            if sc is None:
                print(f"    gemini parse fail {attempt+1}: {repr(raw[:80])}")
                time.sleep(2)
                continue
            return {"scores": sc, "in_tok": 0,
                    "out_tok": len(raw.split()),
                    "latency": round(time.time() - t0, 2)}
        except Exception as e:
            print(f"    gemini error {attempt+1}: {e}")
            time.sleep(3)
    return None


JUDGE_FNS = {"gpt4o": call_gpt4o, "qwen3": call_qwen, "gemini": call_gemini}


def load_scored(filepath):
    done = set()
    if os.path.exists(filepath):
        with open(filepath) as f:
            for line in f:
                r = json.loads(line.strip())
                done.add((r["hypokg_id"], r["model_key"], r["condition"]))
    return done


def load_hypotheses():
    paths_dict = {}
    with open(PATHS_FILE) as f:
        for line in f:
            r = json.loads(line.strip())
            paths_dict[r["hypokg_id"]] = r

    hypotheses = []
    for model_key in MODEL_ORDER:
        for fpath in [GENERATION_FILES[model_key], C4_FILES[model_key]]:
            if not os.path.exists(fpath):
                continue
            with open(fpath) as f:
                for line in f:
                    r = json.loads(line.strip())
                    hid = r["hypokg_id"]
                    if hid not in paths_dict:
                        continue
                    p = paths_dict[hid]
                    r["path_text"]      = p["path_text"]
                    r["source_symbol"]  = parse_source(p["path_text"])
                    r["terminal_name"]  = parse_terminal(p["end_node"])
                    r["tier_calibrated"]    = p["tier_calibrated"]
                    r["crossing_count"]     = p["crossing_count"]
                    r["in_gold_subset"]     = p["in_gold_subset"]
                    r["in_verified_subset"] = p["in_verified_subset"]
                    hypotheses.append(r)

    return hypotheses


def run():
    hypotheses = load_hypotheses()
    print(f"Loaded {len(hypotheses):,} hypotheses")

    done_sets = {j: load_scored(JUDGE_FILES[j]) for j in JUDGE_FILES}
    errors = 0

    for i, hyp in enumerate(hypotheses):
        model_key = hyp["model_key"]
        primary, secondary = JUDGE_ROUTING.get(model_key, ("gpt4o", "gemini"))
        prompt = make_judge_input(hyp)
        key    = (hyp["hypokg_id"], model_key, hyp["condition"])

        for judge_name in (primary, secondary):
            if key in done_sets[judge_name]:
                continue

            result = JUDGE_FNS[judge_name](prompt)
            if result is None:
                errors += 1
                print(f"  ❌ {hyp['hypokg_id']} {model_key} C{hyp['condition']} {judge_name}")
                continue

            record = {
                "hypokg_id":              hyp["hypokg_id"],
                "model_key":              model_key,
                "model_display":          hyp.get("model_display", model_key),
                "condition":              hyp["condition"],
                "tier_calibrated":        hyp["tier_calibrated"],
                "crossing_count":         hyp["crossing_count"],
                "in_gold_subset":         hyp["in_gold_subset"],
                "in_verified_subset":     hyp["in_verified_subset"],
                "source_symbol":          hyp["source_symbol"],
                "terminal_name":          hyp["terminal_name"],
                "judge":                  judge_name,
                "judge_prompt_version":   "v2_fixed_criterion5",
                "path_task_relevance":    result["scores"]["path_task_relevance"],
                "mechanistic_specificity":result["scores"]["mechanistic_specificity"],
                "experimental_testability":result["scores"]["experimental_testability"],
                "nontrivial_novelty":     result["scores"]["nontrivial_novelty"],
                "evidence_proportionality":result["scores"]["evidence_proportionality"],
                "total_score":            result["scores"]["total_score"],
                "reasoning":              result["scores"].get("reasoning", ""),
                "judge_input_tokens":     result["in_tok"],
                "judge_output_tokens":    result["out_tok"],
                "judge_latency_s":        result["latency"],
                "scored_at":              datetime.now(timezone.utc).isoformat(),
            }

            with open(JUDGE_FILES[judge_name], "a") as f:
                f.write(json.dumps(record) + "\n")
            done_sets[judge_name].add(key)

        if (i + 1) % 200 == 0:
            totals = {j: len(done_sets[j]) for j in JUDGE_FILES}
            print(f"  [{i+1}/{len(hypotheses)}] "
                  f"gpt4o={totals['gpt4o']} qwen3={totals['qwen3']} "
                  f"gemini={totals['gemini']} errors={errors}")

        time.sleep(0.3)

    print(f"\nDone. Errors: {errors}")
    for judge_name, fpath in JUDGE_FILES.items():
        n = len(load_scored(fpath))
        print(f"  {judge_name}: {n:,} scored → {fpath}")


if __name__ == "__main__":
    run()
