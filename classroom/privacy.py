"""
Privacy operations: enroll, purge, list — all consent-checked and audit-logged.

  enroll  Add a student's face embeddings from photos. Refused unless the
          consent registry allows the purpose.
  purge   Remove a student everywhere: gallery embeddings, the student_id
          in every output (tokens, seat timelines, seat identities), and mark
          consent withdrawn. Seat-level anonymous rows are kept; they no
          longer point to the student.
  list    Show consent status and enrollment per student.

Usage:
  .venv/bin/python -m classroom.privacy enroll --student S123 --photos roster/S123/
  .venv/bin/python -m classroom.privacy purge --student S123
  .venv/bin/python -m classroom.privacy list
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

import cv2
import pandas as pd

from .consent import AuditLog, ConsentRegistry
from .identity import Gallery, enroll_from_images

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp"}


def enroll(
    registry: ConsentRegistry, gallery_path: Path, audit: AuditLog, student_id: str, photos: Path,
    purpose: str = "engagement", embedder=None,
) -> int:
    if not registry.allows(student_id, purpose):
        audit.record("enroll_refused", student_id, purpose=purpose, reason="no consent")
        raise PermissionError(f"{student_id}: consent does not allow '{purpose}'; not enrolled")
    if embedder is None:
        from .faces import OpenCVFaceEmbedder
        embedder = OpenCVFaceEmbedder()

    images = [cv2.imread(str(p)) for p in sorted(photos.iterdir()) if p.suffix.lower() in IMAGE_EXTS]
    images = [im for im in images if im is not None]
    gallery = Gallery.load(gallery_path, dim=embedder.dim)
    used = enroll_from_images(gallery, embedder, student_id, images)
    if used:
        gallery.save(gallery_path)
    audit.record("enroll", student_id, purpose=purpose, photos=len(images), embeddings=used)
    return used


def purge(
    registry: ConsentRegistry, gallery_path: Path, audit: AuditLog, student_id: str, outputs: Path,
) -> dict:
    """Remove every trace of the student's identity. Returns what was changed."""
    changed = {"gallery": False, "files": []}

    if gallery_path.exists():
        gallery = Gallery.load(gallery_path)
        if gallery.remove(student_id):
            gallery.save(gallery_path)
            changed["gallery"] = True

    for parquet in list(outputs.glob("*/tokens.parquet")) + list(outputs.glob("*/seat_timeline.parquet")):
        df = pd.read_parquet(parquet)
        if "student_id" in df and (df["student_id"] == student_id).any():
            df.loc[df["student_id"] == student_id, "student_id"] = None
            df.to_parquet(parquet, index=False)
            changed["files"].append(str(parquet))

    for ident in outputs.glob("*/seat_identity.json"):
        data = json.loads(ident.read_text())
        touched = False
        for seat in data.values():
            if seat.get("student_id") == student_id:
                seat.update(student_id=None, source="purged", confidence=0.0)
                touched = True
            if student_id in seat.get("candidates", {}):
                seat["candidates"].pop(student_id)
                touched = True
        if touched:
            ident.write_text(json.dumps(data, indent=2))
            changed["files"].append(str(ident))

    registry.withdraw(student_id)
    registry.save()
    audit.record("purge", student_id, gallery_removed=changed["gallery"], files=len(changed["files"]))
    return changed


def main(argv: Optional[list[str]] = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=["enroll", "purge", "list"])
    parser.add_argument("--student")
    parser.add_argument("--photos", type=Path)
    parser.add_argument("--purpose", default="engagement")
    parser.add_argument("--consent", type=Path, default=Path("data/consent/registry.csv"))
    parser.add_argument("--gallery", type=Path, default=Path("data/faces/gallery.enc"))
    parser.add_argument("--audit", type=Path, default=Path("data/audit/audit.jsonl"))
    parser.add_argument("--outputs", type=Path, default=Path("outputs"))
    args = parser.parse_args(argv)

    registry = ConsentRegistry(args.consent)
    audit = AuditLog(args.audit)

    if args.command == "list":
        enrolled = Gallery.load(args.gallery).students if args.gallery.exists() else set()
        for sid in sorted({*enrolled, *(r.student_id for r in registry.records())}):
            r = registry.get(sid)
            status = f"{r.status:9} {';'.join(r.purposes) or '-':18}" if r else "no record          "
            print(f"{sid:12} {status} {'enrolled' if sid in enrolled else ''}")
        audit.record("list")
        return

    if not args.student:
        sys.exit("--student is required")
    if args.command == "enroll":
        if not args.photos:
            sys.exit("--photos is required")
        try:
            used = enroll(registry, args.gallery, audit, args.student, args.photos, args.purpose)
        except PermissionError as e:
            sys.exit(str(e))
        print(f"{args.student}: {used} embeddings enrolled")
    else:
        changed = purge(registry, args.gallery, audit, args.student, args.outputs)
        print(f"{args.student}: purged (gallery: {changed['gallery']}, files: {len(changed['files'])}); consent withdrawn")


if __name__ == "__main__":
    main()
