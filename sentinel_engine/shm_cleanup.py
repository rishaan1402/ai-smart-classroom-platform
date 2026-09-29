"""
ShmLeakChecker — cleans orphaned shared memory segments on startup.

When a camera worker process is killed with SIGKILL (OOM killer, docker kill),
signal handlers don't run and /dev/shm/cam_*_frame segments are never unlinked.

This runs at orchestrator startup BEFORE allocating any new shared memory,
so it's safe to unlink all stale segments from previous crashed runs.
"""

import glob
import logging
import os

logger = logging.getLogger(__name__)


def clean_stale_shm(prefix: str = "cam_") -> int:
    """
    Scan /dev/shm for stale sentinel shared memory segments and unlink them.

    Args:
        prefix: Filename prefix to match (default: "cam_")

    Returns:
        Number of segments cleaned up.
    """
    pattern = f"/dev/shm/{prefix}*"
    cleaned = 0

    for path in glob.glob(pattern):
        name = os.path.basename(path)
        try:
            from multiprocessing import shared_memory
            shm = shared_memory.SharedMemory(name=name, create=False)
            shm.close()
            shm.unlink()
            logger.warning(f"Cleaned up orphaned shared memory segment: {name}")
            cleaned += 1
        except FileNotFoundError:
            # Already cleaned up
            pass
        except Exception as e:
            # Try raw unlink as fallback
            try:
                os.unlink(path)
                logger.warning(f"Cleaned up orphaned shm (raw unlink): {name}")
                cleaned += 1
            except Exception as e2:
                logger.error(f"Failed to clean up {name}: {e} / {e2}")

    if cleaned > 0:
        logger.info(f"ShmLeakChecker: cleaned {cleaned} orphaned segments")
    else:
        logger.debug("ShmLeakChecker: no orphaned segments found")

    return cleaned
