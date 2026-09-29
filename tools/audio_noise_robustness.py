"""
How much does classroom noise hurt teacher transcription, and does enhancement help?

Takes a clean lecture excerpt, mixes in real classroom room-mic noise (looped)
at several SNRs, and transcribes each mix with and without spectral-gating
enhancement. The score is word error rate against the *clean* transcript from
the same model: it measures degradation caused by noise, not absolute
accuracy (there is no human reference transcript yet).

Rough mapping to mic placement (speech level at the mic vs. class noise):
  lapel mic ≈ 20+ dB SNR, desk/boundary mic ≈ 10 dB, ceiling/CCTV mic ≈ 0–5 dB.

Usage:
  .venv/bin/python tools/audio_noise_robustness.py \
      --speech footage/lectures/nptel_iitb_pharmacognosy_l1.mp4 --start 120 --duration 180 \
      --noise footage/cctv_securus_2mp_audio.mp4 --model small --language en
"""

import argparse
import json
import sys
from pathlib import Path

import jiwer
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from audio.asr import load_model, transcribe  # noqa: E402
from audio.enhance import enhance  # noqa: E402
from audio.io import load_audio, save_wav  # noqa: E402

SNRS_DB = [20, 10, 5, 0]


def rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(x ** 2)) + 1e-12)


def mix(speech: np.ndarray, noise: np.ndarray, snr_db: float) -> np.ndarray:
    """Add looped noise scaled so speech RMS / noise RMS = snr_db; peak-normalize if clipping."""
    reps = int(np.ceil(len(speech) / len(noise)))
    noise = np.tile(noise, reps)[: len(speech)]
    scaled = noise * (rms(speech) / rms(noise)) / (10 ** (snr_db / 20))
    out = speech + scaled
    peak = np.abs(out).max()
    return (out / peak * 0.95 if peak > 1 else out).astype(np.float32)


def normalize(text: str) -> str:
    return jiwer.RemovePunctuation()(jiwer.ToLowerCase()(text)).strip()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--speech", type=Path, required=True)
    parser.add_argument("--noise", type=Path, required=True)
    parser.add_argument("--start", type=float, default=0.0)
    parser.add_argument("--duration", type=float, default=180.0)
    parser.add_argument("--model", default="small")
    parser.add_argument("--language", default=None)
    parser.add_argument("--out", type=Path, default=Path("outputs/audio_robustness"))
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    speech = load_audio(args.speech, args.start, args.duration)
    noise = load_audio(args.noise)
    model = load_model(args.model)

    def run(audio: np.ndarray) -> str:
        return normalize(transcribe(model, audio, args.model, language=args.language).text)

    reference = run(speech)
    (args.out / "reference.txt").write_text(reference)
    rows = [{"condition": "clean", "snr_db": None, "enhanced": False, "wer": 0.0, "words": len(reference.split())}]
    print(f"clean: {len(reference.split())} words")

    for snr in SNRS_DB:
        noisy = mix(speech, noise, snr)
        save_wav(args.out / f"mix_{snr}db.wav", noisy)
        for enhanced in (False, True):
            hyp = run(enhance(noisy) if enhanced else noisy)
            wer = jiwer.wer(reference, hyp) if hyp else 1.0
            rows.append({
                "condition": f"{snr} dB" + (" + enhance" if enhanced else ""),
                "snr_db": snr, "enhanced": enhanced, "wer": round(wer, 3), "words": len(hyp.split()),
            })
            print(f"{rows[-1]['condition']:>16}: WER vs clean {wer:6.1%}  ({len(hyp.split())} words)")

    (args.out / "results.json").write_text(json.dumps({
        "speech": str(args.speech), "noise": str(args.noise), "start": args.start,
        "duration": args.duration, "model": args.model, "results": rows,
    }, indent=2))


if __name__ == "__main__":
    main()
