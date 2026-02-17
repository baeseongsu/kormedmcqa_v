"""
Medical QA Evaluation Script for API-based Models

Evaluate API-based models (OpenAI, Gemini, etc.) on medical QA benchmarks.
Supports KorMedMCQA (text-only), KorMedMCQA-V (multimodal), and KorMedMCQA-Mixed datasets.

Usage:
    # Evaluate with OpenAI GPT-5 Mini
    python scripts/eval.py --model-name gpt-5-mini-2025-08-07 --dataset kormedmcqa_v --subset doctor --split test_full

    # Evaluate with Google Gemini 3 Flash (via OpenAI-compatible endpoint)
    python scripts/eval.py \
        --model-name gemini-3-flash-preview \
        --dataset kormedmcqa_v \
        --subset doctor \
        --split test_full \
        --base-url https://generativelanguage.googleapis.com/v1beta/openai/ \
        --api-key $GEMINI_API_KEY
"""

import argparse
import logging
import os
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

from kormedeval.datasets import get_dataset
from kormedeval.evaluation.evaluator import evaluate_dataset
from kormedeval.utils import save_evaluation_result

load_dotenv()

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Evaluate API-based models on Medical QA"
    )

    parser.add_argument(
        "--model-name",
        type=str,
        required=True,
        help="Model name for API (e.g., gpt-5-mini-2025-08-07, gemini-3-flash-preview)",
    )

    parser.add_argument(
        "--dataset",
        type=str,
        default="kormedmcqa_v",
        choices=["kormedmcqa", "kormedmcqa_v", "kormedmcqa_mixed"],
        help="Dataset to evaluate (default: kormedmcqa_v)",
    )

    parser.add_argument(
        "--subset",
        type=str,
        default="doctor",
        help="Dataset subset to evaluate (default: doctor)",
    )

    parser.add_argument(
        "--split",
        type=str,
        default="test",
        choices=["train", "dev", "test", "test_full"],
        help="Dataset split to use (default: test)",
    )

    parser.add_argument(
        "--api-key",
        type=str,
        default=None,
        help="API key (default: auto-detected from model name — GEMINI_API_KEY for Gemini, OPENAI_API_KEY otherwise)",
    )

    parser.add_argument(
        "--base-url",
        type=str,
        default=None,
        help="API base URL (default: auto-detected from model name)",
    )

    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Generation temperature (default: 0.7)",
    )

    parser.add_argument(
        "--max-tokens",
        type=int,
        default=8192,
        help="Maximum tokens to generate (default: 8192)",
    )

    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Maximum number of samples to evaluate (for debugging)",
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="./results",
        help="Output directory for results (default: ./results)",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for evaluation (default: 42)",
    )

    parser.add_argument(
        "--reasoning-effort",
        type=str,
        default=None,
        help="Reasoning effort parameter for supported models (e.g., 'medium')",
    )

    return parser.parse_args()


def main():
    """Main function."""
    args = parse_args()

    dataset_config = {
        "name": args.dataset,
        "subset": args.subset,
        "split": args.split,
        "with_image": args.dataset in ["kormedmcqa_v", "kormedmcqa_mixed"],
    }

    dataset = get_dataset(args.dataset, dataset_config)
    dataset_info = dataset.get_dataset_info()

    is_gemini = "gemini" in args.model_name.lower()

    api_key = args.api_key
    if not api_key:
        if is_gemini:
            api_key = os.getenv("GEMINI_API_KEY")
        else:
            api_key = os.getenv("OPENAI_API_KEY")

    if not api_key:
        env_var = "GEMINI_API_KEY" if is_gemini else "OPENAI_API_KEY"
        raise ValueError(
            f"API key not provided. Use --api-key or set {env_var} environment variable."
        )

    base_url = args.base_url
    if not base_url:
        if is_gemini:
            base_url = "https://generativelanguage.googleapis.com/v1beta/openai/"
        else:
            base_url = "https://api.openai.com/v1"

    client = OpenAI(api_key=api_key, base_url=base_url)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    if args.output_dir and args.output_dir != "./results":
        output_dir = Path(args.output_dir) / "results"
    else:
        exp_name = f"{args.dataset}_{args.subset}_{timestamp}"
        output_dir = Path("experiments") / f"exp_{exp_name}" / "results"

    output_dir.mkdir(parents=True, exist_ok=True)

    # Determine if model is VLM based on dataset
    is_vlm = args.dataset in ["kormedmcqa_v", "kormedmcqa_mixed"]

    logger.info("=" * 80)
    logger.info("Medical QA Evaluation (API-based)")
    logger.info("=" * 80)
    logger.info(f"Dataset: {dataset_info['name']}")
    logger.info(f"Subset: {args.subset}")
    logger.info(f"Split: {args.split}")
    logger.info(f"Choices: {', '.join(dataset.get_choices())}")
    logger.info(f"Model: {args.model_name}")
    logger.info(f"Base URL: {base_url}")
    logger.info(f"Temperature: {args.temperature}")
    logger.info(f"Max tokens: {args.max_tokens}")
    logger.info("=" * 80)

    result = evaluate_dataset(
        client=client,
        model_name=args.model_name,
        dataset=dataset,
        max_samples=args.max_samples,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        is_vlm=is_vlm,
        reasoning_effort=args.reasoning_effort,
    )

    model_config_name = args.model_name.replace("/", "_")
    results_file = (
        output_dir
        / f"{model_config_name}_{args.dataset}_{args.subset}_{args.split}.json"
    )

    config_for_save = {
        "dataset": args.dataset,
        "subset": args.subset,
        "split": args.split,
        "temperature": args.temperature,
        "max_tokens": args.max_tokens,
        "base_url": base_url,
        "seed": args.seed,
    }

    save_evaluation_result(
        result=result,
        model_path=args.model_name,
        model_config_name=model_config_name,
        dataset_info=dataset_info,
        config=config_for_save,
        output_path=results_file,
    )

    logger.info(f"\nResults saved to: {output_dir}")
    logger.info("\n" + "=" * 80)
    logger.info("Evaluation Summary")
    logger.info("=" * 80)
    logger.info(f"Dataset: {dataset_info['name']}/{result['subset']}")
    logger.info(
        f"Accuracy: {result['accuracy']:.4f} ({result['correct']}/{result['evaluated_samples']})"
    )
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
