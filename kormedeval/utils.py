"""
Common utility functions for evaluation.
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

logger = logging.getLogger(__name__)


def extract_thinking_and_summary(text: str) -> tuple[str, str]:
    """Extract thinking content and clean summary from model output.

    Automatically detects and removes thinking blocks with various tag patterns.

    Supported patterns (checked in order):
        - <think>...</think>
        - ...thinking......thinking....
        - [THINK]...[/THINK]

    Args:
        text: Model generated text

    Returns:
        tuple of (thinking_content, clean_text)
        - thinking_content: extracted thinking text (empty if no tags)
        - clean_text: text with thinking blocks removed (original if no tags)
    """
    if not text:
        return "", ""

    thinking_patterns = [
        ("\u25c1think\u25b7", "\u25c1/think\u25b7"),
        ("...thinking...", "....thinking...."),
        ("[THINK]", "[/THINK]"),
    ]

    for bot, eot in thinking_patterns:
        if bot in text and eot in text:
            bot_idx = text.index(bot)
            eot_idx = text.index(eot)
            thinking = text[bot_idx + len(bot) : eot_idx].strip()
            clean = text[eot_idx + len(eot) :].strip()
            return thinking, clean

    return "", text


def save_evaluation_result(
    result: Dict[str, Any],
    model_path: str,
    model_config_name: str,
    dataset_info: Dict[str, Any],
    config: Dict[str, Any],
    output_path: Path,
) -> None:
    """
    Save evaluation result to JSON file.

    Args:
        result: Evaluation result dictionary
        model_path: Model path/name
        model_config_name: Model config name for filename
        dataset_info: Dataset information dictionary
        config: Configuration dictionary
        output_path: Path to save result file
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    summary = {
        "model": model_path,
        "model_config_name": model_config_name,
        "timestamp": timestamp,
        "dataset_info": dataset_info,
        "config": config,
        "overall_accuracy": result["accuracy"],
        "total_correct": result["correct"],
        "total_evaluated": result["evaluated_samples"],
        "predictions": result["predictions"],
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    logger.info(f"Saved evaluation result to: {output_path}")


__all__ = [
    "extract_thinking_and_summary",
    "save_evaluation_result",
]
