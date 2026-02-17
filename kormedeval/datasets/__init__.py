"""
Medical QA Datasets Module

Support for KorMedMCQA (text-only), KorMedMCQA-V (multimodal),
and KorMedMCQA-Mixed (mixed) datasets.
"""

from .base import BaseDataset
from .kormedmcqa import KorMedMCQADataset
from .kormedmcqa_mixed import KorMedMCQAMixed
from .kormedmcqa_v import KorMedMCQAVDataset

DATASET_REGISTRY = {
    "kormedmcqa": KorMedMCQADataset,
    "kormedmcqa_v": KorMedMCQAVDataset,
    "kormedmcqa_mixed": KorMedMCQAMixed,
}


def get_dataset(dataset_name: str, config: dict) -> BaseDataset:
    """Get dataset instance by name."""
    if dataset_name not in DATASET_REGISTRY:
        raise ValueError(
            f"Unknown dataset: {dataset_name}. "
            f"Available datasets: {list(DATASET_REGISTRY.keys())}"
        )
    return DATASET_REGISTRY[dataset_name](config)


__all__ = [
    "BaseDataset",
    "KorMedMCQADataset",
    "KorMedMCQAVDataset",
    "KorMedMCQAMixed",
    "get_dataset",
]
