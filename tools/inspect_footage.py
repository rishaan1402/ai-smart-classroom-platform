"""
Footage inspection — step 1 of the classroom prototype.

For each video, reports what the pipeline has to cope with:
  - container facts: resolution, fps, duration, audio channels
  - vision: people per frame, person-box height distribution (back-row
    resolution), face-keypoint visibility, from sampled frames
  - audio: loudness, noise floor, share of speech-like frames

Writes footage/_inspect/<name>.md, a contact sheet <name>.jpg with pose
overlays, and a combined footage/_inspect/REPORT.md.

Anonymous only: no face crops, embeddings or identities are produced.

Usage:
  .venv/bin/python tools/inspect_footage.py footage/*.mp4 [--samples 24]
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sentinel_engine.device import detect_device  # noqa: E402

OUT_DIR = Path("footage/_inspect")
MODEL_PATH = "models/yolo11n-pose.pt"

# COCO keypoints: 0 nose, 1/2 eyes, 3/4 ears
FACE_KPS = [0, 1, 2]
KP_VISIBLE = 0.5

DIGITAL_SILENCE_DB = -90.0

# Person-box heights in pixels; rough usability bands for per-student analysis
HEIGHT_BANDS = [(0, 40, "too small"), (40, 80, "marginal"), (80, 10_000, "usable")]


def probe(path: Path) -> dict:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-print_format", "json", "-show_streams", "-show_format", str(path)],
        capture_output=True, text=True, check=True,
    )
    info = json.loads(out.stdout)
    video = next((s for s in info["streams"] if s["codec_type"] == "video"), {})
    audio = next((s for s in info["streams"] if s["codec_type"] == "audio"), None)
    num, den = (video.get("avg_frame_rate") or "0/1").split("/")
    return {
        "width": video.get("width"),
        "height": video.get("height"),
        "fps": round(float(num) / float(den), 2) if float(den) else 0.0,
        "duration_s": round(float(info["format"].get("duration", 0)), 1),
        "audio": None if audio is None else {
            "codec": audio.get("codec_name"),
            "channels": audio.get("channels"),
            "sample_rate": audio.get("sample_rate"),
        },
    }


def sample_frames(path: Path, n: int) -> list[tuple[float, np.ndarray]]:
    cap = cv2.VideoCapture(str(path))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    frames = []
    # Skip the first/last 3% (intros, logos, fades)
    for idx in np.linspace(total * 0.03, total * 0.97, n).astype(int):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ok, frame = cap.read()
        if ok:
            frames.append((idx / fps, frame))
    cap.release()
    return frames


def analyze_vision(model, frames, device: str) -> tuple[dict, list[np.ndarray]]:
    counts, heights, face_visible, annotated = [], [], [], []
    for _, frame in frames:
        res = model.predict(frame, device=device, conf=0.25, imgsz=1280, verbose=False)[0]
        n = 0 if res.boxes is None else len(res.boxes)
        counts.append(n)
        if n:
            boxes = res.boxes.xyxy.cpu().numpy()
            heights.extend((boxes[:, 3] - boxes[:, 1]).tolist())
            if res.keypoints is not None and res.keypoints.conf is not None:
                conf = res.keypoints.conf.cpu().numpy()
                face_visible.extend((conf[:, FACE_KPS] > KP_VISIBLE).all(axis=1).tolist())
        annotated.append(res.plot(labels=False, conf=False))

    heights_arr = np.array(heights) if heights else np.array([0.0])
    bands = {
        label: round(float(((heights_arr >= lo) & (heights_arr < hi)).mean()) * 100, 1)
        for lo, hi, label in HEIGHT_BANDS
    }
    return {
        "people_per_frame": {
            "median": float(np.median(counts)) if counts else 0.0,
            "max": int(max(counts)) if counts else 0,
        },
        "box_height_px": {
            "p10": round(float(np.percentile(heights_arr, 10)), 1),
            "median": round(float(np.median(heights_arr)), 1),
            "p90": round(float(np.percentile(heights_arr, 90)), 1),
        },
        "box_height_bands_pct": bands,
        "face_keypoints_visible_pct": round(float(np.mean(face_visible)) * 100, 1) if face_visible else 0.0,
    }, annotated


def analyze_audio(path: Path) -> dict | None:
    # 16 kHz mono float PCM straight from ffmpeg
    proc = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", str(path), "-vn", "-ac", "1", "-ar", "16000", "-f", "f32le", "-"],
        capture_output=True,
    )
    pcm = np.frombuffer(proc.stdout, dtype=np.float32)
    if pcm.size < 16000:
        return None

    frame = 400  # 25 ms
    usable = pcm[: pcm.size // frame * frame].reshape(-1, frame)
    rms_db = 20 * np.log10(np.sqrt((usable ** 2).mean(axis=1)) + 1e-9)

    # Speech-band energy share (300–3400 Hz) as a cheap speech-likeness proxy
    spec = np.abs(np.fft.rfft(usable * np.hanning(frame), axis=1)) ** 2
    freqs = np.fft.rfftfreq(frame, 1 / 16000)
    band = (freqs >= 300) & (freqs <= 3400)
    speech_ratio = spec[:, band].sum(axis=1) / (spec.sum(axis=1) + 1e-12)

    # Digital silence (muted track, edited gaps) would drag the noise floor to -inf
    silent = rms_db < DIGITAL_SILENCE_DB
    if silent.all():
        return {"digital_silence_pct": 100.0}
    live_db, live_ratio = rms_db[~silent], speech_ratio[~silent]

    noise_floor = float(np.percentile(live_db, 10))
    active = live_db > noise_floor + 10
    speech_like = active & (live_ratio > 0.6)
    return {
        "digital_silence_pct": round(float(silent.mean()) * 100, 1),
        "mean_db": round(float(live_db.mean()), 1),
        "noise_floor_db": round(noise_floor, 1),
        "peak_db": round(float(np.percentile(live_db, 99)), 1),
        "snr_estimate_db": round(float(np.percentile(live_db, 90)) - noise_floor, 1),
        "active_pct": round(float(active.mean()) * 100, 1),
        "speech_like_pct": round(float(speech_like.mean()) * 100, 1),
    }


def contact_sheet(images: list[np.ndarray], cols: int = 4, width: int = 480) -> np.ndarray:
    tiles = [cv2.resize(im, (width, int(im.shape[0] * width / im.shape[1]))) for im in images]
    h = max(t.shape[0] for t in tiles)
    tiles = [cv2.copyMakeBorder(t, 0, h - t.shape[0], 0, 0, cv2.BORDER_CONSTANT) for t in tiles]
    while len(tiles) % cols:
        tiles.append(np.zeros_like(tiles[0]))
    rows = [np.hstack(tiles[i:i + cols]) for i in range(0, len(tiles), cols)]
    return np.vstack(rows)


def to_markdown(name: str, meta: dict, vision: dict, audio: dict | None) -> str:
    lines = [
        f"## {name}",
        f"- Video: {meta['width']}x{meta['height']} @ {meta['fps']} fps, {meta['duration_s']} s",
        f"- Audio stream: {meta['audio'] or 'none'}",
        f"- People per frame: median {vision['people_per_frame']['median']}, max {vision['people_per_frame']['max']}",
        f"- Person box height px: p10 {vision['box_height_px']['p10']}, median {vision['box_height_px']['median']}, "
        f"p90 {vision['box_height_px']['p90']}",
        f"- Box height bands: {vision['box_height_bands_pct']}",
        f"- Face keypoints visible (nose+eyes): {vision['face_keypoints_visible_pct']}%",
    ]
    if audio and "mean_db" not in audio:
        lines.append("- Audio: track is digitally silent")
    elif audio:
        lines.append(f"- Audio digital silence: {audio['digital_silence_pct']}%")
        lines.append(
            f"- Audio: mean {audio['mean_db']} dBFS, noise floor {audio['noise_floor_db']}, "
            f"SNR≈{audio['snr_estimate_db']} dB, active {audio['active_pct']}%, speech-like {audio['speech_like_pct']}%"
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("videos", nargs="+", type=Path)
    parser.add_argument("--samples", type=int, default=24)
    args = parser.parse_args()

    from ultralytics import YOLO

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    device = detect_device()
    model = YOLO(MODEL_PATH)

    report, summary = ["# Footage inspection\n"], {}
    for path in args.videos:
        print(f"→ {path.name}")
        meta = probe(path)
        vision, annotated = analyze_vision(model, sample_frames(path, args.samples), device)
        audio = analyze_audio(path) if meta["audio"] else None

        md = to_markdown(path.stem, meta, vision, audio)
        (OUT_DIR / f"{path.stem}.md").write_text(md)
        cv2.imwrite(str(OUT_DIR / f"{path.stem}.jpg"), contact_sheet(annotated[:12]))
        report.append(md)
        summary[path.stem] = {"meta": meta, "vision": vision, "audio": audio}

    (OUT_DIR / "REPORT.md").write_text("\n".join(report))
    (OUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"Wrote {OUT_DIR}/REPORT.md")


if __name__ == "__main__":
    main()
