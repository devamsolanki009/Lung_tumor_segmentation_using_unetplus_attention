"""gpu_utils.py — GPU detection and configuration for TensorFlow.

Call setup_gpu(config) once at the very start of any training/evaluation
script to ensure correct memory management and mixed precision.

Optimised for:  NVIDIA RTX 4060  (8 GB VRAM, Ada Lovelace architecture)
TF version:     2.12+
"""

from __future__ import annotations

import logging

import tensorflow as tf

logger = logging.getLogger(__name__)


def setup_gpu(config: dict | None = None) -> dict:
    """Configure GPU memory growth and mixed precision.

    Parameters
    ----------
    config : dict, optional
        Full project config dict. If None, sensible defaults are used:
        ``memory_growth=True``, ``mixed_precision=True``.

    Returns
    -------
    dict
        A summary of what was configured, e.g.::

            {
                "gpu_count": 1,
                "gpu_name": "NVIDIA GeForce RTX 4060",
                "memory_growth": True,
                "mixed_precision": True,
                "policy": "mixed_float16",
            }
    """
    gpu_cfg = (config or {}).get("gpu", {})
    use_memory_growth: bool = gpu_cfg.get("memory_growth", True)
    use_mixed_precision: bool = gpu_cfg.get("mixed_precision", True)

    gpus = tf.config.list_physical_devices("GPU")
    report: dict = {"gpu_count": len(gpus)}

    if not gpus:
        logger.warning(
            "No GPU detected — training will run on CPU. "
            "Install CUDA 11.8+ and cuDNN 8.6+ for RTX 4060 support."
        )
        report.update({"gpu_name": "None (CPU only)", "memory_growth": False,
                        "mixed_precision": False, "policy": "float32"})
        return report

    # ---- Memory growth -------------------------------------------------
    # Prevents TF from grabbing ALL VRAM on startup (important with 8 GB).
    if use_memory_growth:
        for gpu in gpus:
            try:
                tf.config.experimental.set_memory_growth(gpu, True)
                logger.info("Memory growth enabled for: %s", gpu.name)
            except RuntimeError as exc:
                # Already initialised — harmless
                logger.debug("Memory growth not set (runtime already started): %s", exc)

    # ---- GPU name -------------------------------------------------------
    try:
        gpu_details = tf.config.experimental.get_device_details(gpus[0])
        gpu_name = gpu_details.get("device_name", gpus[0].name)
    except Exception:  # noqa: BLE001
        gpu_name = gpus[0].name

    report["gpu_name"] = gpu_name
    report["memory_growth"] = use_memory_growth

    # ---- Mixed precision (float16) -------------------------------------
    # On RTX 4060 (Ada Lovelace), Tensor Cores deliver ~2× throughput
    # with float16 compute compared to float32.  The model weights and
    # the final sigmoid output are kept in float32 automatically by Keras.
    if use_mixed_precision:
        policy = tf.keras.mixed_precision.Policy("mixed_float16")
        tf.keras.mixed_precision.set_global_policy(policy)
        logger.info("Mixed precision enabled: policy = mixed_float16")
        report["mixed_precision"] = True
        report["policy"] = "mixed_float16"
    else:
        report["mixed_precision"] = False
        report["policy"] = "float32"

    # ---- Summary --------------------------------------------------------
    logger.info("=" * 55)
    logger.info("  GPU Configuration")
    logger.info("  GPU count       : %d", len(gpus))
    logger.info("  GPU name        : %s", gpu_name)
    logger.info("  Memory growth   : %s", use_memory_growth)
    logger.info("  Mixed precision : %s  (%s)", use_mixed_precision, report["policy"])
    logger.info("=" * 55)

    return report


def log_gpu_memory() -> None:
    """Log current GPU memory usage (requires TF 2.5+)."""
    try:
        for gpu in tf.config.list_physical_devices("GPU"):
            info = tf.config.experimental.get_memory_info(gpu.name.replace("physical_device:", ""))
            used_mb = info["current"] / 1024**2
            peak_mb = info["peak"] / 1024**2
            logger.info(
                "GPU memory — current: %.0f MB | peak: %.0f MB", used_mb, peak_mb
            )
    except Exception as exc:  # noqa: BLE001
        logger.debug("Could not read GPU memory info: %s", exc)
