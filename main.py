"""main.py — Master entry point for the Lung Tumor Segmentation pipeline.

Usage
-----
    python main.py --preprocess
    python main.py --train
    python main.py --evaluate
    python main.py --all
    python main.py --all --config path/to/config.yaml
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Ensure the package root is importable
_REPO_ROOT = Path(__file__).resolve().parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("main")


def _run_preprocess(config_path: str | None) -> None:
    from lung_tumor_segmentation.scripts.run_preprocessing import main as _main
    logger.info("▶  Starting PREPROCESSING stage …")
    _main(config_path=config_path)
    logger.info("✔  PREPROCESSING complete.\n")


def _run_train(config_path: str | None) -> None:
    from lung_tumor_segmentation.scripts.run_training import main as _main
    logger.info("▶  Starting TRAINING stage …")
    _main(config_path=config_path)
    logger.info("✔  TRAINING complete.\n")


def _run_evaluate(config_path: str | None) -> None:
    from lung_tumor_segmentation.scripts.run_evaluation import main as _main
    logger.info("▶  Starting EVALUATION stage …")
    _main(config_path=config_path)
    logger.info("✔  EVALUATION complete.\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Lung Tumor Segmentation — Master Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --all                     # run full pipeline
  python main.py --preprocess              # preprocess only
  python main.py --train                   # train only (needs preprocessed data)
  python main.py --evaluate                # evaluate only (needs trained model)
  python main.py --all --config cfg.yaml   # with custom config
        """,
    )
    parser.add_argument("--preprocess", action="store_true", help="Run preprocessing stage")
    parser.add_argument("--train", action="store_true", help="Run training stage")
    parser.add_argument("--evaluate", action="store_true", help="Run evaluation stage")
    parser.add_argument("--all", action="store_true", help="Run all stages in sequence")
    parser.add_argument("--config", type=str, default=None, help="Path to config.yaml")

    args = parser.parse_args()

    if not any([args.preprocess, args.train, args.evaluate, args.all]):
        parser.print_help()
        sys.exit(0)

    run_pre = args.preprocess or args.all
    run_train = args.train or args.all
    run_eval = args.evaluate or args.all

    logger.info("=" * 60)
    logger.info("  Lung Tumor Segmentation Pipeline")
    logger.info("  Stages: preprocess=%s | train=%s | evaluate=%s", run_pre, run_train, run_eval)
    logger.info("=" * 60)

    if run_pre:
        _run_preprocess(args.config)
    if run_train:
        _run_train(args.config)
    if run_eval:
        _run_evaluate(args.config)

    logger.info("All requested stages finished.")


if __name__ == "__main__":
    main()
