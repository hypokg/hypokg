# HypoKG: Evidence-Disciplined Biomedical Hypothesis Generation Beyond Endpoint Knowledge

Code and data for the paper *HypoKG: Evidence-Disciplined Biomedical Hypothesis Generation Beyond Endpoint Knowledge*.

## What this is

HypoKG tests whether knowledge graph (KG) evidence changes how LLMs reason when generating biomedical hypotheses, or whether models simply exploit endpoint knowledge to produce plausible-sounding guesses. We build a benchmark from an integrated KEGG–Rhea–UniProt graph, sample 550 multi-hop enzyme-to-disease paths, and generate 13,200 hypotheses from six LLMs under four controlled conditions. The central finding: endpoint-only prompting achieves the highest aggregate scores, but full KG paths uniquely improve evidence proportionality — the degree to which mechanistic claims are grounded in the provided path evidence rather than freely invented.

## Repo structure

```
hypokg/
├── data/
│   ├── paths/
│   │   └── hypokg_550_selected.jsonl     # 550 benchmark paths
│   └── prevalence/
│       └── prevalence_summary.jsonl      # PubMed co-citation labels
├── generations/
│   └── {model}_generations_v2.jsonl      # C1/C2/C3 hypotheses per model
│   └── {model}_generations_c4.jsonl      # C4 hypotheses per model
├── judge/
│   ├── judge_full_gpt4o.jsonl            # primary judge scores
│   ├── judge_full_gemini.jsonl           # secondary judge scores
│   └── judge_full_qwen3.jsonl            # primary judge for GPT-4o outputs
├── shuffled_control/
│   ├── shuffled_paths.jsonl              # permuted intermediate nodes
│   ├── shuffled_judge_gpt4o.jsonl
│   ├── shuffled_judge_gemini.jsonl
│   └── generations/
│       └── {model}_shuffled.jsonl
└── scripts/
    ├── generate.py                       # C1–C4 generation, API models
    ├── generate_hf.py                    # C1–C4 generation, HF models
    ├── judge.py                          # cross-judge scoring
    ├── shuffled_generate.py              # shuffle paths + generate
    ├── shuffled_judge.py                 # judge + analyze shuffled hypotheses
    └── analysis.py                       # reproduce all paper tables
```

## Reproducing the results

### Setup

```bash
pip install -r requirements.txt
export HYPOKG_FOLDER=/path/to/this/repo
```

Set API keys:

```bash
export OPENAI_API_KEY=...
export ANTHROPIC_API_KEY=...
export TOGETHER_API_KEY=...
export GEMINI_API_KEY=...
export HF_TOKEN=...          # for BioMistral and MedGemma
```

### Generate hypotheses

API models (GPT-4o, Claude, Llama, Qwen):

```bash
python scripts/generate.py --model gpt4o --conditions 1 2 3
python scripts/generate.py --model claude_sonnet --conditions 1 2 3
python scripts/generate.py --model llama_70b --conditions 1 2 3
python scripts/generate.py --model qwen3_235b --conditions 1 2 3

# C4 (endpoint-only) — all models
python scripts/generate.py --model gpt4o --conditions 4
```

HuggingFace models — requires GPU (both models run on A100 80GB):

```bash
python scripts/generate_hf.py --model biomistral_7b --conditions 1 2 3
python scripts/generate_hf.py --model medgemma_27b --conditions 1 2 3
```

All generation scripts are checkpointed and safe to restart.

### Judge

```bash
python scripts/judge.py
```

Cross-judging scheme: GPT-4o judges all models except GPT-4o itself; Qwen3-235B judges GPT-4o outputs; Gemini 2.5 Flash serves as secondary judge for all models. Claude was excluded after a 14% content filter refusal rate on biochemical disease hypotheses.

### Shuffled path control

```bash
# Step 1: create shuffled paths (run once)
python scripts/shuffled_generate.py --step shuffle

# Step 2: generate from shuffled paths
python scripts/shuffled_generate.py --step generate --model gpt4o
python scripts/shuffled_generate.py --step generate --model biomistral_7b

# Step 3: judge shuffled hypotheses
python scripts/shuffled_judge.py --step judge

# Step 4: run analysis
python scripts/shuffled_judge.py --step analyze
```

### Reproduce paper tables

```bash
python scripts/analysis.py           # prints all tables to stdout
python scripts/analysis.py --save    # also writes CSVs to analysis/
```

This reproduces Tables 3, 4, 6, 7, 8, and 9 from the paper. Table 5 (human expert validation) was collected manually from three PhD-level domain experts and is reported directly in the paper.

## Data format

**hypokg_550_selected.jsonl** — one path per line:

```json
{
  "hypokg_id": "HKG_0000",
  "path_text": "GLS2 [enzyme_kinetics] --maplink--> Butanoate metabolism [pathway_link]\n...",
  "crossing_count": 4,
  "path_length": 5,
  "domains_traversed": ["enzyme_kinetics", "pathway_link", "metabolic_pathway", "disease_mechanism"],
  "start_domain": "enzyme_kinetics",
  "end_domain": "disease_mechanism",
  "start_node": "hsa:27165",
  "end_node": "disease:Meconium_ileus",
  "tier_calibrated": "T1"
}
```

**prevalence_summary.jsonl** — one path per line:

```json
{
  "hypokg_id": "HKG_0000",
  "prevalence_label_final": "Novel",
  "pubmed_count_final": 0
}
```

**Generation files** — one hypothesis per line:

```json
{
  "hypokg_id": "HKG_0000",
  "model_key": "gpt4o",
  "condition": 2,
  "hypothesis_text": "We hypothesize that ...",
  "input_tokens": 312,
  "output_tokens": 148,
  "latency_s": 1.43
}
```

**Judge files** — one scored hypothesis per line:

```json
{
  "hypokg_id": "HKG_0000",
  "model_key": "gpt4o",
  "condition": 2,
  "judge": "gpt4o",
  "path_task_relevance": 4,
  "mechanistic_specificity": 3,
  "experimental_testability": 4,
  "nontrivial_novelty": 3,
  "evidence_proportionality": 4,
  "total_score": 18,
  "reasoning": "..."
}
```

## Models

| Key | Display name | Access |
|---|---|---|
| `claude_sonnet` | Claude Sonnet 4.6 | Anthropic API |
| `gpt4o` | GPT-4o | OpenAI API |
| `llama_70b` | Llama-3.3-70B | Together AI |
| `qwen3_235b` | Qwen3-235B | Together AI |
| `biomistral_7b` | BioMistral-7B | HuggingFace |
| `medgemma_27b` | MedGemma-27B | HuggingFace |


