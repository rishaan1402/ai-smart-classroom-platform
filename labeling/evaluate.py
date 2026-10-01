"""
Evaluate pipeline predictions against human labels.

Per lesson (and pooled over --lesson repeats):
  - labelled tasks, excluding cant_tell / not_a_student (reported separately:
    not_a_student is a detection/tracking error rate)
  - accuracy and macro-F1 over states; per-state precision / recall / F1 / support
  - confusion matrix (rows = label, columns = prediction)
  - engaged-vs-not accuracy: the binary call the class engagement curve
    actually depends on
  - inter-labeller agreement (Cohen's kappa) where tasks have ≥ 2 labellers
  - resolution breakdown: the same metrics for small / medium / large person
    boxes (back / middle / front rows), the bias check available without
    demographic data

With several labellers, the reference label is the majority (ties → first labeller).

Predictions come from the task manifest (frozen at export). --repredict
instead looks each task up in the lesson's current tokens.parquet by (track,
time), to score a changed pipeline against the same labels. This only works
while detections are reused from the cache, so track ids are unchanged.

Usage:
  .venv/bin/python -m labeling.evaluate --lesson cctv_securus_2mp_audio [--lesson cctv_altima] \\
      [--labeller alice] [--repredict] [--out docs/eval_report.md]
"""

import argparse
import json
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from .tasks import DATA_ROOT, load_labels, load_tasks

STATES = ["attentive", "writing", "hand_raised", "head_down", "turned_away", "interacting", "phone", "out_of_seat"]
ENGAGED = {"attentive", "writing", "hand_raised"}
EXCLUDED = {"cant_tell", "not_a_student"}


def current_predictions(lesson: str, outputs: Path = Path("outputs"), tolerance_s: float = 0.2) -> dict[str, str]:
    """task id → state from the lesson's current tokens.parquet, matched on track id and time."""
    tasks = load_tasks(lesson)
    tokens = pd.read_parquet(outputs / lesson / "tokens.parquet", columns=["track_id", "timestamp_s", "state"])
    by_track = {tid: g for tid, g in tokens.groupby("track_id")}
    out = {}
    for t in tasks["tasks"]:
        g = by_track.get(t["track_id"])
        if g is None:
            continue
        nearest = (g["timestamp_s"] - t["t"]).abs()
        if nearest.min() <= tolerance_s:
            out[t["id"]] = g.loc[nearest.idxmin(), "state"]
    return out


def joined(
    lessons: list[str], root: Path = DATA_ROOT, labeller: Optional[str] = None, repredict: bool = False,
) -> pd.DataFrame:
    """One row per labelled task: prediction, reference label, labellers, box size."""
    rows = []
    for lesson in lessons:
        tasks = load_tasks(lesson, root)
        if tasks is None:
            continue
        fresh = current_predictions(lesson) if repredict else {}
        labels = load_labels(lesson, root)
        if labeller:
            labels = labels[labels["labeller"] == labeller]
        by_task = labels.groupby("task_id")
        for t in tasks["tasks"]:
            if t["id"] not in by_task.groups:
                continue
            g = by_task.get_group(t["id"])
            counts = g["label"].value_counts()
            top = counts[counts == counts.max()].index
            reference = g["label"].iloc[0] if len(top) > 1 else top[0]
            rows.append({
                "lesson": lesson, "task_id": t["id"], "predicted": fresh.get(t["id"], t["predicted"]), "label": reference,
                "n_labellers": g["labeller"].nunique(), "box_height_px": t["box_height_px"],
                "labels": dict(zip(g["labeller"], g["label"])),
            })
    return pd.DataFrame(rows)


def per_class(df: pd.DataFrame) -> pd.DataFrame:
    out = []
    for s in STATES:
        tp = int(((df["predicted"] == s) & (df["label"] == s)).sum())
        pred = int((df["predicted"] == s).sum())
        true = int((df["label"] == s).sum())
        p = tp / pred if pred else np.nan
        r = tp / true if true else np.nan
        f1 = 2 * p * r / (p + r) if pred and true and (p + r) else np.nan
        out.append({"state": s, "precision": p, "recall": r, "f1": f1, "support": true, "predicted": pred})
    return pd.DataFrame(out)


def cohen_kappa(a: list[str], b: list[str]) -> float:
    cats = sorted(set(a) | set(b))
    idx = {c: i for i, c in enumerate(cats)}
    m = np.zeros((len(cats), len(cats)))
    for x, y in zip(a, b):
        m[idx[x], idx[y]] += 1
    n = m.sum()
    po = np.trace(m) / n
    pe = (m.sum(axis=0) * m.sum(axis=1)).sum() / n ** 2
    return float((po - pe) / (1 - pe)) if pe < 1 else 1.0


def agreement(df: pd.DataFrame) -> Optional[dict]:
    multi = df[df["n_labellers"] >= 2]
    if multi.empty:
        return None
    pairs = []
    for labels in multi["labels"]:
        names = sorted(labels)
        pairs.append((labels[names[0]], labels[names[1]]))
    a, b = zip(*pairs)
    return {"tasks": len(pairs), "raw_agreement": float(np.mean([x == y for x, y in pairs])),
            "kappa": cohen_kappa(list(a), list(b))}


def metrics(df: pd.DataFrame) -> dict:
    scored = df[~df["label"].isin(EXCLUDED)]
    if scored.empty:
        return {"n": 0}
    pc = per_class(scored)
    return {
        "n": int(len(scored)),
        "accuracy": float((scored["predicted"] == scored["label"]).mean()),
        "macro_f1": float(pc["f1"].dropna().mean()) if pc["f1"].notna().any() else float("nan"),
        "engaged_accuracy": float((scored["predicted"].isin(ENGAGED) == scored["label"].isin(ENGAGED)).mean()),
        "label_engaged_share": float(scored["label"].isin(ENGAGED).mean()),
        "predicted_engaged_share": float(scored["predicted"].isin(ENGAGED).mean()),
    }


def by_resolution(df: pd.DataFrame) -> pd.DataFrame:
    scored = df[~df["label"].isin(EXCLUDED)].copy()
    if len(scored) < 9:
        return pd.DataFrame()
    scored["size"] = pd.qcut(scored["box_height_px"], 3, labels=["small (back)", "medium", "large (front)"])
    rows = []
    for size, g in scored.groupby("size", observed=True):
        m = metrics(g)
        rows.append({"boxes": str(size), "px": f"{g['box_height_px'].min():.0f}–{g['box_height_px'].max():.0f}",
                     "n": m["n"], "accuracy": m["accuracy"], "engaged_accuracy": m["engaged_accuracy"]})
    return pd.DataFrame(rows)


def report(
    lessons: list[str], root: Path = DATA_ROOT, labeller: Optional[str] = None, repredict: bool = False,
) -> tuple[str, dict]:
    df = joined(lessons, root, labeller, repredict)
    if df.empty:
        return "No labels yet.\n", {}
    m = metrics(df)
    pc = per_class(df[~df["label"].isin(EXCLUDED)])
    labellers = sorted({name for labels in df["labels"] for name in labels})
    excluded = df["label"].value_counts().reindex(sorted(EXCLUDED), fill_value=0).to_dict()

    lines = [
        f"# Evaluation: {', '.join(lessons)}",
        "",
        f"- Labelled tasks: {len(df)} (scored {m['n']}; excluded {excluded})",
        f"- Labellers: {', '.join(labellers)}",
        f"- **State accuracy {m['accuracy']:.0%}**, macro-F1 {m['macro_f1']:.2f}",
        f"- **Engaged-vs-not accuracy {m['engaged_accuracy']:.0%}**; engaged share by labels "
        f"{m['label_engaged_share']:.0%} vs predicted {m['predicted_engaged_share']:.0%}",
        f"- Detection errors (not_a_student): {excluded.get('not_a_student', 0)} of {len(df)}",
    ]
    agree = agreement(df)
    if agree:
        lines.append(f"- Inter-labeller agreement on {agree['tasks']} tasks: {agree['raw_agreement']:.0%} "
                     f"(Cohen's κ {agree['kappa']:.2f})")
    lines += ["", "## Per state", "", "| state | precision | recall | F1 | support | predicted |", "|---|---|---|---|---|---|"]
    for r in pc.itertuples():
        f = lambda v: "—" if pd.isna(v) else f"{v:.2f}"
        lines.append(f"| {r.state} | {f(r.precision)} | {f(r.recall)} | {f(r.f1)} | {r.support} | {r.predicted} |")

    scored = df[~df["label"].isin(EXCLUDED)]
    present = [s for s in STATES if s in set(scored["label"]) | set(scored["predicted"])]
    cm = pd.crosstab(scored["label"], scored["predicted"]).reindex(index=present, columns=present, fill_value=0)
    lines += ["", "## Confusion matrix (rows = label, columns = prediction)", "",
              "| label \\ pred | " + " | ".join(present) + " |", "|---" * (len(present) + 1) + "|"]
    for s in present:
        lines.append(f"| {s} | " + " | ".join(str(int(cm.loc[s, c])) for c in present) + " |")

    res = by_resolution(df)
    if not res.empty:
        lines += ["", "## By person size (resolution bias check)", "",
                  "| boxes | height px | n | accuracy | engaged accuracy |", "|---|---|---|---|---|"]
        for r in res.itertuples():
            lines.append(f"| {r.boxes} | {r.px} | {r.n} | {r.accuracy:.0%} | {r.engaged_accuracy:.0%} |")

    payload = {"lessons": lessons, "labellers": labellers, "metrics": m, "agreement": agree,
               "per_class": json.loads(pc.to_json(orient="records")),
               "by_resolution": json.loads(res.to_json(orient="records")) if not res.empty else []}
    return "\n".join(lines) + "\n", payload


def main(argv: Optional[list[str]] = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--lesson", action="append", required=True)
    parser.add_argument("--labeller", default=None, help="Only this labeller's labels")
    parser.add_argument("--repredict", action="store_true", help="Score the current tokens.parquet, not export-time predictions")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv)

    text, payload = report(args.lesson, labeller=args.labeller, repredict=args.repredict)
    print(text)
    if args.out:
        args.out.write_text(text)
        args.out.with_suffix(".json").write_text(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
