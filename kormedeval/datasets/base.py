"""
Base class for medical QA datasets.

Provides a consistent interface for different medical QA datasets.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional


class BaseDataset(ABC):
    """Base class for medical QA datasets"""

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize dataset with configuration.

        Args:
            config: Dataset configuration dictionary
                - name: Dataset name
                - subset: Subset name (e.g., 'doctor', 'nurse')
                - split: Data split (e.g., 'train', 'test', 'dev')
                - with_image: Whether to include images (for multimodal datasets)
        """
        self.config = config
        self.name = config.get("name", "unknown")
        self.subset = config.get("subset", "all")
        self.split = config.get("split", "test")
        self.with_image = config.get("with_image", True)

    @abstractmethod
    def load_data(self) -> List[Dict[str, Any]]:
        """
        Load dataset samples.

        Returns:
            List of sample dictionaries
        """
        pass

    @abstractmethod
    def format_user_prompt(self, sample: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Build a structured user prompt with question, choices, and instruction.
        Better for API models as each component is separate.
        Must be implemented by subclasses as each dataset may have different formats.

        Args:
            sample: Dataset sample

        Returns:
            List of content dictionaries with 'type' and 'text'/'image_url' keys
            e.g., [
                {"type": "text", "text": "질문: ..."},
                {"type": "text", "text": "A. Choice 1"},
                {"type": "text", "text": "B. Choice 2"},
                ...
            ]
        """
        pass

    @abstractmethod
    def extract_answer(self, response: str) -> Optional[str]:
        """
        Parse answer choice (A-E or A-D) from model response.

        Args:
            response: Model's generated response

        Returns:
            Parsed answer letter or None if not found
        """
        pass

    @abstractmethod
    def evaluate_answer(self, predicted: str, correct: str) -> bool:
        """
        Check if predicted answer is correct.

        Args:
            predicted: Predicted answer (letter)
            correct: Correct answer (can be letter or number)

        Returns:
            True if correct, False otherwise
        """
        pass

    @abstractmethod
    def get_choices(self) -> List[str]:
        """
        Get available answer choices for this dataset.

        Returns:
            List of choice letters (e.g., ['A', 'B', 'C', 'D', 'E'])
        """
        pass

    def get_dataset_info(self) -> Dict[str, Any]:
        """
        Get dataset information for logging.

        Returns:
            Dictionary with dataset metadata
        """
        return {
            "name": self.name,
            "subset": self.subset,
            "split": self.split,
            "with_image": self.with_image,
            "choices": self.get_choices(),
        }
