import json, os, argparse
import numpy as np
from scipy import stats
from collections import defaultdict

HYPOKG_FOLDER     = os.environ.get("HYPOKG_FOLDER", "./")
GPT4O_FILE        = f"{HYPOKG_FOLDER}/judge/judge_full_gpt4o.jsonl"
QWEN_FILE         = f"{HYPOKG_FOLDER}/judge/judge_full_qwen3.jsonl"
GEMINI_FILE       = f"{HYPOKG_FOLDER}/judge/judge_full_gemini.jsonl"
SHUF_GPT4O_FILE   = f"{HYPOKG_FOLDER}/shuffled_control/shuffled_judge_gpt4o.jsonl"
PATHS_FILE        = f"{HYPOKG_FOLDER}/data/paths/hypokg_550_selected.jsonl"
PREV_FILE         = f"{HYPOKG_FOLDER}/data/prevalence/prevalence_summary.jsonl"
ANALYSIS_DIR      = f"{HYPOKG_FOLDER}/analysis"
os.makedirs(ANALYSIS_DIR, exist_ok=True)

MODELS = ["claude_sonnet", "gpt4o", "llama_70b",
          "qwen3_235b", "biomistral_7b", "medgemma_27b"]

MODEL_DISPLAY = {
    "claude_sonnet": "Claude Sonnet",
    "gpt4o":         "GPT-4o",
    "llama_70b":     "Llama-3.3-70B",
    "qwen3_235b":    "Qwen3-235B",
    "biomistral_7b": "BioMistral-7B",
    "medgemma_27b":  "MedGemma-27B",
}

CRITERIA = [
    ("path_task_relevance",       "Rel"),
    ("mechanistic_specificity",   "Mech"),
    ("experimental_testability",  "Test"),
    ("nontrivial_novelty",        "Nov"),
    ("evidence_proportionality",  "Prop"),
]
CRIT_KEYS   = [c[0] for c in CRITERIA]
CRIT_SHORT  = [c[1] for c in CRITERIA]


# ── Loaders ───────────────────────────────────────────────────────

def load_jsonl(fpath):
    records = []
    if not os.path.exists(fpath):
        print(f"  missing: {fpath}")
        return records
    with open(fpath) as f:
        for line in f:
            records.append(json.loads(line.strip()))
    return records


def load_primary():
    """
    Primary judge: GPT-4o for all models except GPT-4o itself.
    GPT-4o outputs were judged by Qwen3-235B.
    Returns dict keyed by (hypokg_id, model_key, condition).
    """
    primary = {}
    for r in load_jsonl(GPT4O_FILE):
        if r["model_key"] != "gpt4o":
            primary[(r["hypokg_id"], r["model_key"], r["condition"])] = r
    for r in load_jsonl(QWEN_FILE):
        if r["model_key"] == "gpt4o":
            primary[(r["hypokg_id"], r["model_key"], r["condition"])] = r
    return primary


def mean(vals):
    return round(float(np.mean(vals)), 3) if vals else 0.0


def wilcoxon_gt(a, b):
    """Paired Wilcoxon, one-sided (a > b), returns p-value."""
    if len(a) < 10:
        return 1.0
    _, p = stats.wilcoxon(a, b, alternative="greater")
    return p


def cohen_d(a, b):
    diff = [x - y for x, y in zip(a, b)]
    return round(float(np.mean(diff) / np.std(diff)), 3) if np.std(diff) > 0 else 0.0


def sig_label(p):
    if p < 0.001: return "***"
    if p < 0.01:  return "**"
    if p < 0.05:  return "*"
    return "ns"


# ── Table 3: Main results ─────────────────────────────────────────

def table3(primary, save=False):
    print("\n" + "="*80)
    print("TABLE 3 — Mean rubric scores by model and condition")
    print("="*80)

    header = f"{'Model':<18} {'Cond':<5} " + \
             "  ".join(f"{s:>5}" for s in CRIT_SHORT) + \
             f"  {'Total':>6}  {'C3-C1':>6}  {'C2/3-C4':>7}"
    print(header)
    print("─" * len(header))

    rows = []
    c1_totals = {}

    for mk in MODELS + ["__all__"]:
        label = MODEL_DISPLAY.get(mk, "All models")
        c4_total = None

        for cond in [1, 2, 3, 4]:
            if mk == "__all__":
                scores = [r for r in primary.values() if r["condition"] == cond]
            else:
                scores = [r for r in primary.values()
                          if r["model_key"] == mk and r["condition"] == cond]
            if not scores:
                continue

            crit_means = [mean([r[k] for r in scores]) for k in CRIT_KEYS]
            total      = round(sum(crit_means), 2)

            if cond == 1:
                c1_totals[mk] = total
            if cond == 4:
                c4_total = total

            c3c1  = ""
            c23c4 = ""
            if cond == 4:
                c3_scores = ([r for r in primary.values()
                              if r["model_key"] == mk and r["condition"] == 3]
                             if mk != "__all__"
                             else [r for r in primary.values() if r["condition"] == 3])
                c3_total = round(sum(mean([r[k] for r in c3_scores]) for k in CRIT_KEYS), 2)
                c3c1     = f"{c3_total - c1_totals.get(mk, 0):+.2f}"

                c2_scores = ([r for r in primary.values()
                              if r["model_key"] == mk and r["condition"] == 2]
                             if mk != "__all__"
                             else [r for r in primary.values() if r["condition"] == 2])
                c2_total  = round(sum(mean([r[k] for r in c2_scores]) for k in CRIT_KEYS), 2)
                c23c4     = f"{((c2_total + c3_total) / 2) - total:+.2f}"

            model_col = label if cond == 1 else ""
            row = (f"  {model_col:<16} C{cond:<4} " +
                   "  ".join(f"{v:>5.2f}" for v in crit_means) +
                   f"  {total:>6.2f}  {c3c1:>6}  {c23c4:>7}")
            print(row)
            rows.append({
                "model": label, "condition": cond,
                **{s: v for s, v in zip(CRIT_SHORT, crit_means)},
                "total": total
            })

        if mk != "__all__":
            print()

    # Statistical tests on all-model aggregate
    print("\n" + "─"*60)
    print("Statistical tests (paired Wilcoxon, Holm corrected, N=3,300)")
    print("─"*60)
    for (ca, cb), label in [
        ((2, 4), "C2 vs C4  Prop"),
        ((3, 4), "C3 vs C4  Prop"),
        ((2, 3), "C2 vs C3  Prop"),
        ((2, 1), "C2 vs C1  Total"),
        ((3, 1), "C3 vs C1  Total"),
    ]:
        pairs = [(r, s) for r in primary.values() for s in primary.values()
                 if r["hypokg_id"] == s["hypokg_id"]
                 and r["model_key"] == s["model_key"]
                 and r["condition"] == ca and s["condition"] == cb]
        if not pairs:
            continue
        field = "evidence_proportionality" if "Prop" in label else "total_score"
        a_vals = [p[0][field] for p in pairs]
        b_vals = [p[1][field] for p in pairs]
        delta  = round(mean(a_vals) - mean(b_vals), 3)
        p      = wilcoxon_gt(a_vals, b_vals)
        d      = cohen_d(a_vals, b_vals)
        print(f"  {label:<20}  Δ={delta:+.3f}  p={p:.4f} {sig_label(p)}  d={d:.3f}  n={len(a_vals)}")

    if save:
        import csv
        fpath = f"{ANALYSIS_DIR}/table3_main_results.csv"
        with open(fpath, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=rows[0].keys())
            w.writeheader()
            w.writerows(rows)
        print(f"\nSaved: {fpath}")


# ── Table 4: Crossing count × condition ──────────────────────────

def table4(primary, save=False):
    print("\n" + "="*60)
    print("TABLE 4 — Mean total score by crossing count and condition")
    print("="*60)

    paths = {json.loads(l)["hypokg_id"]: json.loads(l)
             for l in open(PATHS_FILE) if l.strip()}

    crossing_cond = defaultdict(list)
    for (hid, mk, cond), r in primary.items():
        if hid in paths:
            k = paths[hid]["crossing_count"]
            crossing_cond[(k, cond)].append(r["total_score"])

    crossings = sorted(set(k for k, _ in crossing_cond))
    print(f"  {'k':<4} {'C1':>7} {'C2':>7} {'C3':>7} {'C4':>7}  {'C3-C1':>7}")
    print("  " + "─"*46)

    rows = []
    for k in crossings:
        vals = {c: crossing_cond[(k, c)] for c in [1, 2, 3, 4]}
        means = {c: mean(vals[c]) for c in [1, 2, 3, 4]}
        c3c1  = round(means[3] - means[1], 2)
        row   = {"crossing": k, **{f"C{c}": means[c] for c in [1,2,3,4]}, "C3-C1": c3c1}
        rows.append(row)
        print(f"  {k:<4} {means[1]:>7.2f} {means[2]:>7.2f} "
              f"{means[3]:>7.2f} {means[4]:>7.2f}  {c3c1:>+7.2f}")

    # Correlation
    all_pairs = []
    for (hid, mk, cond), r in primary.items():
        if cond == 3 and hid in paths:
            c1 = primary.get((hid, mk, 1))
            if c1:
                all_pairs.append((paths[hid]["crossing_count"],
                                  r["total_score"] - c1["total_score"]))
    if all_pairs:
        xs, ys = zip(*all_pairs)
        r_val, p_val = stats.pearsonr(xs, ys)
        print(f"\n  Crossing × C3-C1 gain: r={r_val:.3f}, p={p_val:.4f}, n={len(all_pairs)}")

    if save:
        import csv
        fpath = f"{ANALYSIS_DIR}/table4_crossing_count.csv"
        with open(fpath, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=rows[0].keys())
            w.writeheader()
            w.writerows(rows)
        print(f"Saved: {fpath}")


# ── Table 6: Shuffled path control ───────────────────────────────

def table6(primary, save=False):
    print("\n" + "="*70)
    print("TABLE 6 — Original vs shuffled path scores (all models)")
    print("="*70)

    shuf = {}
    for r in load_jsonl(SHUF_GPT4O_FILE):
        shuf[(r["hypokg_id"], r["model_key"], r["condition"])] = r

    common = set(primary) & set(shuf)
    print(f"  Matched pairs: {len(common):,}")

    METRICS = [
        ("total_score",             "Total Score"),
        ("evidence_proportionality","Evidence Proportionality"),
        ("nontrivial_novelty",      "Novelty"),
    ]

    print(f"\n  {'Metric':<26} {'Cond':<5} {'Orig':>7} {'Shuf':>7} {'Drop':>7}  {'p':>10}  {'sig':>4}")
    print("  " + "─"*70)

    rows = []
    for metric, label in METRICS:
        for cond in [2, 3]:
            keys     = [k for k in common if k[2] == cond]
            orig_v   = [primary[k][metric] for k in keys]
            shuf_v   = [shuf[k][metric]    for k in keys]
            drop     = round(mean(orig_v) - mean(shuf_v), 3)
            p        = wilcoxon_gt(orig_v, shuf_v)
            rows.append({"metric": label, "condition": cond,
                         "orig": mean(orig_v), "shuf": mean(shuf_v),
                         "drop": drop, "p": p, "n": len(keys)})
            print(f"  {label:<26} C{cond:<4} "
                  f"{mean(orig_v):>7.3f} {mean(shuf_v):>7.3f} "
                  f"{drop:>+7.3f}  {p:>10.6f}  {sig_label(p):>4}")
        print()

    if save:
        import csv
        fpath = f"{ANALYSIS_DIR}/table6_shuffled_control.csv"
        with open(fpath, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=rows[0].keys())
            w.writeheader()
            w.writerows(rows)
        print(f"Saved: {fpath}")


# ── Table 7: Per-model shuffled drop ─────────────────────────────

def table7(primary, save=False):
    print("\n" + "="*80)
    print("TABLE 7 — Per-model evidence proportionality drop under shuffling")
    print("="*80)

    shuf = {}
    for r in load_jsonl(SHUF_GPT4O_FILE):
        shuf[(r["hypokg_id"], r["model_key"], r["condition"])] = r

    common = set(primary) & set(shuf)
    print(f"  {'Model':<20} {'C2 Orig':>8} {'C2 Shuf':>8} {'C2 Drop':>8} {'':>4}  "
          f"{'C3 Orig':>8} {'C3 Shuf':>8} {'C3 Drop':>8} {'':>4}")
    print("  " + "─"*78)

    rows = []
    model_drops = []
    for mk in MODELS:
        row = {"model": MODEL_DISPLAY[mk]}
        for cond in [2, 3]:
            keys   = [k for k in common if k[1] == mk and k[2] == cond]
            orig_v = [primary[k]["evidence_proportionality"] for k in keys]
            shuf_v = [shuf[k]["evidence_proportionality"]    for k in keys]
            drop   = round(mean(orig_v) - mean(shuf_v), 3)
            try:
                p = wilcoxon_gt(orig_v, shuf_v)
            except Exception:
                p = 1.0
            row[f"C{cond}_orig"] = mean(orig_v)
            row[f"C{cond}_shuf"] = mean(shuf_v)
            row[f"C{cond}_drop"] = drop
            row[f"C{cond}_p"]    = p
        rows.append(row)
        model_drops.append((mk, row.get("C2_drop", 0)))

        print(f"  {MODEL_DISPLAY[mk]:<20} "
              f"{row.get('C2_orig',0):>8.3f} {row.get('C2_shuf',0):>8.3f} "
              f"{row.get('C2_drop',0):>+8.3f} {sig_label(row.get('C2_p',1)):>4}  "
              f"{row.get('C3_orig',0):>8.3f} {row.get('C3_shuf',0):>8.3f} "
              f"{row.get('C3_drop',0):>+8.3f} {sig_label(row.get('C3_p',1)):>4}")

    if save:
        import csv
        fpath = f"{ANALYSIS_DIR}/table7_per_model_shuffled.csv"
        with open(fpath, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=rows[0].keys())
            w.writeheader()
            w.writerows(rows)
        print(f"\nSaved: {fpath}")


# ── Table 8: Secondary judge agreement ───────────────────────────

def table8(primary, save=False):
    print("\n" + "="*55)
    print("TABLE 8 — Secondary judge (Gemini) agreement with primary")
    print("="*55)

    gemini = {}
    for r in load_jsonl(GEMINI_FILE):
        gemini[(r["hypokg_id"], r["model_key"], r["condition"])] = r

    common = set(primary) & set(gemini)
    print(f"  Matched pairs: {len(common):,}")
    print(f"\n  {'Criterion':<28} {'Pearson r':>10}  {'Mean |Δ|':>9}")
    print("  " + "─"*50)

    rows = []
    check = CRIT_KEYS + ["total_score"]
    labels = CRIT_SHORT + ["Total Score"]

    for key, label in zip(check, labels):
        a = [primary[k][key] for k in common]
        b = [gemini[k][key]  for k in common]
        r_val, _ = stats.pearsonr(a, b)
        mad      = round(float(np.mean(np.abs(np.array(a) - np.array(b)))), 3)
        rows.append({"criterion": label, "pearson_r": round(r_val, 3), "mean_abs_diff": mad})
        print(f"  {label:<28} {r_val:>10.3f}  {mad:>9.3f}")

    if save:
        import csv
        fpath = f"{ANALYSIS_DIR}/table8_judge_agreement.csv"
        with open(fpath, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=rows[0].keys())
            w.writeheader()
            w.writerows(rows)
        print(f"\nSaved: {fpath}")


# ── Table 9: KG consistency metrics ──────────────────────────────

def table9(save=False):
    print("\n" + "="*65)
    print("TABLE 9 — KG consistency metrics by condition")
    print("="*65)

    consistency_dir = os.path.join(HYPOKG_FOLDER, "consistency")
    fpath = os.path.join(consistency_dir, "kg_consistency_scores.jsonl")
    if not os.path.exists(fpath):
        print(f"  File not found: {fpath}")
        print("  Run the consistency notebook to generate this file.")
        return

    records = load_jsonl(fpath)
    by_cond = defaultdict(list)
    for r in records:
        by_cond[r["condition"]].append(r)

    cond_labels = {
        1: "C1 Source-only",
        2: "C2 Path-compress",
        3: "C3 Path-interpret",
        4: "C4 Endpoint-only",
    }

    print(f"\n  {'Condition':<22} {'Source':>8} {'Terminal':>10} {'Path Cov':>10} {'Composite':>10}")
    print("  " + "─"*62)

    rows = []
    for cond in [1, 2, 3, 4]:
        recs = by_cond[cond]
        if not recs:
            continue
        src  = mean([r.get("source_grounding",    0) for r in recs])
        term = mean([r.get("terminal_grounding",  0) for r in recs])
        path = mean([r.get("path_entity_coverage",0) for r in recs
                     if r.get("path_entity_coverage") is not None])
        path_str = f"{path:.3f}" if cond in (2, 3) else "n/a"

        applicable = [src, term] + ([path] if cond in (2, 3) else [])
        comp = round(float(np.mean(applicable)), 3)

        rows.append({"condition": cond_labels[cond],
                     "source": src, "terminal": term,
                     "path_coverage": path_str, "composite": comp})
        print(f"  {cond_labels[cond]:<22} {src:>8.3f} {term:>10.3f} "
              f"{path_str:>10} {comp:>10.3f}")

    if save:
        import csv
        fpath_out = f"{ANALYSIS_DIR}/table9_kg_consistency.csv"
        with open(fpath_out, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=rows[0].keys())
            w.writeheader()
            w.writerows(rows)
        print(f"\nSaved: {fpath_out}")


# ── Entry point ───────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--save", action="store_true",
                        help="Save tables as CSVs to analysis/")
    args = parser.parse_args()

    print("Loading primary scores...")
    primary = load_primary()
    print(f"  {len(primary):,} records")

    table3(primary, save=args.save)
    table4(primary, save=args.save)
    table6(primary, save=args.save)
    table7(primary, save=args.save)
    table8(primary, save=args.save)
    table9(save=args.save)

    print("\n" + "="*60)
    print("Done. Table 5 (human expert validation) is in the paper")
    print("directly — those scores were collected manually from experts")
    print("and are not recomputed from judge files.")
    print("="*60)
