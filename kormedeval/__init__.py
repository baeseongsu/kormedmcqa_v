"""
KorMedEval: Korean Medical Evaluation Framework

A lightweight evaluation framework for medical QA models
supporting API-based inference with OpenAI-compatible endpoints.
"""

from .datasets import BaseDataset, get_dataset
from .evaluation.evaluator import evaluate_dataset, evaluate_sample
from .utils import extract_thinking_and_summary, save_evaluation_result

__version__ = "1.0.0"

__all__ = [
    # Datasets
    "get_dataset",
    "BaseDataset",
    # Evaluation
    "evaluate_dataset",
    "evaluate_sample",
    # Utils
    "extract_thinking_and_summary",
    "save_evaluation_result",
]
