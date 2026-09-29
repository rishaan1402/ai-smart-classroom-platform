"""
CLI entry point for SentinelCore engine.

Usage:
    python sentinel.py --config config/app_config.yaml
    python -m sentinel_engine --config config/app_config.yaml
"""

import argparse
import logging
import sys


def _request_camera_permission_macos() -> None:
    """
    On macOS, AVFoundation requires camera access to be granted from the
    main thread. Open and immediately release a capture to trigger the
    system permission dialog before spawning background threads.
    """
    import platform
    if platform.system() != "Darwin":
        return
    try:
        import cv2
        cap = cv2.VideoCapture(0)
        cap.release()
    except Exception:
        pass


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="sentinel-engine",
        description="SentinelCore AI surveillance engine",
    )
    parser.add_argument(
        "--config",
        default="config/app_config.yaml",
        help="Path to YAML config file (default: config/app_config.yaml)",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level (default: INFO)",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    _request_camera_permission_macos()

    from .orchestrator import SentinelCore

    core = SentinelCore(config_path=args.config)
    core.run()


if __name__ == "__main__":
    main()
