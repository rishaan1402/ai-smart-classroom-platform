"""
Lesson report: topics × teacher audio × class engagement, joined on time.

Inputs (from earlier steps):
  topics/topics.json, topics/questions_llm.json     step 6 (topics.segment)
  audio/transcript.json, audio/teacher_metrics.json step 5 (audio.pipeline)
  class_engagement.csv                              step 3 (classroom.pipeline), optional

Per topic: duration, teaching method, pace, questions by type, and, when an
engagement curve is given, mean engaged share, the dominant disengaged state,
and the difference from the lesson average. Aggregated by method it answers
"which ways of teaching hold this class's attention", which is the input to
curriculum and coaching reports.

Video and audio clocks may differ; --engagement-offset shifts the engagement
curve (seconds to add to engagement time to reach audio time).

Usage:
  .venv/bin/python -m analytics.lesson --lesson outputs/nptel_iitb_pharmacognosy_l1 \\
      [--engagement outputs/<video>/class_engagement.csv --engagement-offset 0]
"""

import argparse
import json
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from audio.asr import Transcript

DISENGAGED_COLUMNS = ["share_head_down", "share_turned_away", "share_interacting", "share_phone", "share_out_of_seat"]


def topic_table(
    topics: list[dict],
    transcript: Transcript,
    questions: list[dict],
    engagement: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    word_times = np.array([w.start for s in transcript.segments for w in s.words])
    q_df = pd.DataFrame(questions) if questions else pd.DataFrame(columns=["time_s", "type"])
    # Prefer the wait-time-corrected type (topics.segment.add_wait_times) when present
    type_col = "effective_type" if "effective_type" in q_df else "type"
    lesson_mean = float(engagement["engaged_share"].mean()) if engagement is not None and len(engagement) else None

    rows = []
    for t in topics:
        start, end = t["start_s"], t["end_s"]
        minutes = max(1e-6, (end - start) / 60)
        in_topic = q_df[(q_df["time_s"] >= start) & (q_df["time_s"] < end)]
        row = {
            "start_min": round(start / 60, 1),
            "end_min": round(end / 60, 1),
            "minutes": round(minutes, 1),
            "title": t["title"],
            "method": t["method"],
            "summary": t.get("summary", ""),
            "words_per_min": round(float(((word_times >= start) & (word_times < end)).sum()) / minutes, 1),
            "questions": len(in_topic),
            **{f"q_{k}": int((in_topic[type_col] == k).sum()) for k in ["open", "closed", "rhetorical", "check_understanding"]},
        }
        if engagement is not None:
            window = engagement[(engagement["second"] >= start) & (engagement["second"] < end)]
            if len(window):
                row["engaged_share"] = round(float(window["engaged_share"].mean()), 3)
                row["vs_lesson_avg"] = round(row["engaged_share"] - lesson_mean, 3)
                present = [c for c in DISENGAGED_COLUMNS if c in window]
                if present:
                    means = window[present].mean()
                    row["main_disengagement"] = means.idxmax().removeprefix("share_") if means.max() > 0 else None
            else:
                row["engaged_share"] = None
        rows.append(row)
    return pd.DataFrame(rows)


def method_summary(table: pd.DataFrame) -> pd.DataFrame:
    agg = {"minutes": "sum", "questions": "sum", "title": "count"}
    if "engaged_share" in table:
        # Time-weighted engagement per method
        table = table.assign(_eng_min=table["engaged_share"].fillna(0) * table["minutes"],
                             _eng_w=table["minutes"].where(table["engaged_share"].notna(), 0))
        agg.update({"_eng_min": "sum", "_eng_w": "sum"})
    out = table.groupby("method").agg(agg).rename(columns={"title": "segments"})
    if "_eng_min" in out:
        out["engaged_share"] = (out["_eng_min"] / out["_eng_w"].replace(0, np.nan)).round(3)
        out = out.drop(columns=["_eng_min", "_eng_w"])
    return out.sort_values("minutes", ascending=False).reset_index()


def to_markdown(name: str, metrics: dict, table: pd.DataFrame, methods: pd.DataFrame, has_engagement: bool) -> str:
    lines = [
        f"# Lesson report: {name}",
        "",
        f"- Duration {metrics['duration_s'] / 60:.0f} min · teacher talk {metrics['talk_ratio']:.0%} · "
        f"{metrics['words_per_minute']:.0f} words/min · language {metrics.get('language', '?')}",
        f"- Questions (LLM-classified, wait-time corrected): {int(table['questions'].sum())} "
        f"(open {int(table['q_open'].sum())}, closed {int(table['q_closed'].sum())}, "
        f"rhetorical {int(table['q_rhetorical'].sum())}, checks {int(table['q_check_understanding'].sum())})",
        "- Engagement: " + ("joined from the classroom video" if has_engagement else "no classroom video for this lesson"),
        "",
        "## Topics",
        "",
    ]
    cols = ["start_min", "end_min", "title", "method", "words_per_min", "questions"]
    if has_engagement:
        cols += ["engaged_share", "vs_lesson_avg", "main_disengagement"]
    header = "| " + " | ".join(cols) + " |"
    lines += [header, "|" + "---|" * len(cols)]
    for _, r in table.iterrows():
        lines.append("| " + " | ".join("" if pd.isna(r.get(c)) else str(r.get(c)) for c in cols) + " |")
    lines += ["", "## Time by teaching method", "", "| " + " | ".join(methods.columns) + " |",
              "|" + "---|" * len(methods.columns)]
    for _, r in methods.iterrows():
        lines.append("| " + " | ".join(str(v) for v in r.values) + " |")
    return "\n".join(lines) + "\n"


def main(argv: Optional[list[str]] = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--lesson", type=Path, required=True, help="outputs/<lesson> dir with audio/ and topics/")
    parser.add_argument("--engagement", type=Path, default=None, help="class_engagement.csv from classroom.pipeline")
    parser.add_argument("--engagement-offset", type=float, default=0.0)
    args = parser.parse_args(argv)

    transcript = Transcript.from_dict(json.loads((args.lesson / "audio" / "transcript.json").read_text()))
    metrics = json.loads((args.lesson / "audio" / "teacher_metrics.json").read_text())
    topics = json.loads((args.lesson / "topics" / "topics.json").read_text())
    questions = json.loads((args.lesson / "topics" / "questions_llm.json").read_text())

    engagement = None
    if args.engagement:
        engagement = pd.read_csv(args.engagement)
        engagement["second"] = engagement["second"] + args.engagement_offset

    table = topic_table(topics, transcript, questions, engagement)
    methods = method_summary(table)
    table.to_csv(args.lesson / "lesson_topics.csv", index=False)
    methods.to_csv(args.lesson / "lesson_methods.csv", index=False)
    report = to_markdown(args.lesson.name, metrics, table, methods, engagement is not None)
    (args.lesson / "lesson_report.md").write_text(report, encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
