"""
Offline audio pipeline.

  --role teacher (lapel mic): VAD → [enhance] → ASR → teaching metrics
  --role room    (room mic):  per-second noise / chatter curve only

Outputs in --out (default outputs/<stem>/audio):
  teacher: transcript.json, transcript.txt, speech_segments.json,
           teacher_metrics.json, questions.json, minute_timeline.csv
  room:    noise_curve.csv, room_summary.json

Usage:
  .venv/bin/python -m audio.pipeline --input footage/lectures/nptel_hindi_nutrition_intro.mp4 --language hi
  .venv/bin/python -m audio.pipeline --input footage/cctv_securus_2mp_audio.mp4 --role room
"""

import argparse
import json
import logging
from dataclasses import asdict
from pathlib import Path
from typing import Optional

from .io import SAMPLE_RATE, load_audio

logger = logging.getLogger(__name__)


def run_teacher(
    audio, out: Path, model_size: str, language: Optional[str], prompt: Optional[str], do_enhance: bool,
    reuse_transcript: bool = False,
):
    from .asr import Transcript, load_model, transcribe
    from .teacher import compute_metrics, find_questions, minute_timeline
    from .vad import speech_segments

    if do_enhance:
        from .enhance import enhance
        audio = enhance(audio)

    duration_s = len(audio) / SAMPLE_RATE
    speech = speech_segments(audio)
    logger.info(f"VAD: {sum(s.duration for s in speech):.0f}s speech in {duration_s:.0f}s")

    cached = out / "transcript.json"
    if reuse_transcript and cached.exists():
        transcript = Transcript.from_dict(json.loads(cached.read_text()))
        logger.info(f"Reusing {cached}")
    else:
        transcript = transcribe(load_model(model_size), audio, model_size, language=language, initial_prompt=prompt)
    logger.info(
        f"ASR: {len(transcript.segments)} segments, language={transcript.language} "
        f"({transcript.language_probability:.2f}), {transcript.decode_seconds:.0f}s decode "
        f"= {transcript.decode_seconds / duration_s:.2f}× real time"
    )

    metrics = compute_metrics(transcript, speech, duration_s)
    (out / "transcript.json").write_text(json.dumps(transcript.to_dict(), ensure_ascii=False, indent=1))
    (out / "transcript.txt").write_text(
        "\n".join(f"[{s.start:7.1f}–{s.end:7.1f}] {s.text}" for s in transcript.segments), encoding="utf-8",
    )
    (out / "speech_segments.json").write_text(json.dumps([asdict(s) for s in speech]))
    (out / "questions.json").write_text(
        json.dumps([asdict(q) for q in find_questions(transcript)], ensure_ascii=False, indent=1)
    )
    minute_timeline(transcript, speech, duration_s).to_csv(out / "minute_timeline.csv", index=False)
    summary = {
        **metrics.to_dict(),
        "language": transcript.language,
        "language_probability": round(transcript.language_probability, 3),
        "model": transcript.model,
        "enhanced": do_enhance,
        "asr_realtime_factor": round(transcript.decode_seconds / duration_s, 3),
    }
    (out / "teacher_metrics.json").write_text(json.dumps(summary, indent=2))
    logger.info(json.dumps(summary, indent=2))


def run_room(audio, out: Path):
    from .room import noise_curve

    curve = noise_curve(audio)
    curve.to_csv(out / "noise_curve.csv", index=False)
    summary = {
        "seconds": len(curve),
        "median_rms_db": round(float(curve["rms_db"].median()), 1),
        "spikes": int(curve["spike"].sum()),
        "noisy_seconds": int(curve["noisy"].sum()),
        "noisy_share": round(float(curve["noisy"].mean()), 3),
    }
    (out / "room_summary.json").write_text(json.dumps(summary, indent=2))
    logger.info(json.dumps(summary, indent=2))


def main(argv: Optional[list[str]] = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", type=Path, required=True, help="Audio or video file")
    parser.add_argument("--role", choices=["teacher", "room"], default="teacher")
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--model", default="large-v3", help="Whisper size: small, medium, large-v3, …")
    parser.add_argument("--language", default=None, help="ISO code (en, hi); default auto-detect")
    parser.add_argument("--prompt", default=None, help="Vocabulary hint, e.g. course / subject name")
    parser.add_argument("--enhance", action="store_true", help="Spectral-gating noise reduction before ASR")
    parser.add_argument("--reuse-transcript", action="store_true", help="Recompute metrics from saved transcript.json")
    parser.add_argument("--start", type=float, default=0.0)
    parser.add_argument("--duration", type=float, default=None)
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
    out = args.out or Path("outputs") / args.input.stem / "audio"
    out.mkdir(parents=True, exist_ok=True)

    audio = load_audio(args.input, args.start, args.duration)
    logger.info(f"Loaded {len(audio) / SAMPLE_RATE:.0f}s from {args.input.name}")
    if args.role == "teacher":
        run_teacher(audio, out, args.model, args.language, args.prompt, args.enhance, args.reuse_transcript)
    else:
        run_room(audio, out)


if __name__ == "__main__":
    main()
