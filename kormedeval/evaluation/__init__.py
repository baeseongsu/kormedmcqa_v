"""Evaluation module for KorMedMCQA-V."""

from .evaluator import evaluate_dataset, evaluate_sample

__all__ = [
    "evaluate_sample",
    "evaluate_dataset",
]
