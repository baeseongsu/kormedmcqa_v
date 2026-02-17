"""
KorMedMCQA-V Dataset

Korean Medical Multiple Choice Question Answering dataset with multimodal support.
Supports doctor, nurse, pharm, and dentist subsets with images.
"""

import ast
import json
import logging
import re
from typing import Any, Dict, List, Optional

from datasets import load_dataset as hf_load_dataset

from .base import BaseDataset

logger = logging.getLogger(__name__)


class KorMedMCQAVDataset(BaseDataset):
    """
    Korean Medical Multiple Choice Question Answering Dataset (Multimodal)

    Features:
        - Text and image questions
        - Four subsets: doctor, nurse, pharm, dentist
        - 5 choices (A-E)
    """

    CHOICE_KEYS = ["A", "B", "C", "D", "E"]
    HF_PATH = "seongsubae/KorMedMCQA-V"
    SUPPORTED_SUBSETS = ["doctor"]
    SAMPLE_KEYS = ["uid", "question", "images", *CHOICE_KEYS, "answer"]
    INSTRUCTION = (
        "각 문제에서 가장 적절한 답을 하나만 고르시오.\n"
        "정답은 A, B, C, D, E 중 하나입니다.\n"
        "출력은 반드시 JSON 객체 1개만 반환하고, JSON 밖의 다른 텍스트(설명/마크다운 등)는 쓰지 마세요.\n"
        '출력 형식: {"정답": "A"}'
    )

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize KorMedMCQA-V dataset.

        Args:
            config: Dataset configuration dictionary
                - name: Dataset name (e.g., 'kormedmcqa_v')
                - subset: Subset name ('doctor', 'nurse', 'pharm', 'dentist', or 'all')
                - split: Data split ('test', 'test_full')
                - with_image: Whether to include images
        """
        super().__init__(config)
        self.hf_path = self.HF_PATH
        self.supported_subsets = self.SUPPORTED_SUBSETS

    @staticmethod
    def _extract_answer(response: str, choice_keys: List[str]) -> Optional[str]:
        """
        Extract answer letter from model responses.

        Supports (in order):
        - JSON format: {"정답": "A"} or {"answer": "A"}
        - LaTeX box format (fallback): $\boxed{E}$ or $\boxed{E}

        Args:
            response: Model's generated response
            choice_keys: Valid choice letters (e.g., ['A', 'B', 'C', 'D', 'E'])

        Returns:
            Parsed answer letter (uppercase) or None if not found
        """
        if not response:
            return None

        # 1. Try JSON format first
        fenced = re.search(
            r"```json\s*(\{.*?\})\s*```", response, re.DOTALL | re.IGNORECASE
        )
        blocks = (
            [fenced.group(1)] if fenced else re.findall(r"\{.*?\}", response, re.DOTALL)
        )

        # Iterate in reverse so the last JSON block wins. This handles models
        # that echo the prompt (which contains an example like {"정답": "A"})
        # before emitting the actual answer.
        for block in reversed(blocks):
            parsed: Any = None
            try:
                parsed = json.loads(block)
            except Exception:
                try:
                    parsed = ast.literal_eval(block)
                except Exception:
                    parsed = None

            if not isinstance(parsed, dict):
                continue

            choice_pattern = "|".join(choice_keys)
            for key in ("정답", "answer"):
                if key in parsed and parsed[key] is not None:
                    match = re.search(
                        rf"([{choice_pattern}])", str(parsed[key]), re.IGNORECASE
                    )
                    if match:
                        ans = match.group(1).upper()
                        if ans in choice_keys:
                            return ans

        # 2. Fallback: Try LaTeX box format (e.g., $\boxed{E}$)
        choice_pattern = "|".join(choice_keys)
        latex_match = re.search(
            rf"\\boxed\{{([{choice_pattern}])\}}", response, re.IGNORECASE
        )
        if latex_match:
            ans = latex_match.group(1).upper()
            if ans in choice_keys:
                return ans

        return None

    @staticmethod
    def _normalize_answer_to_letter(answer: Any, max_choices: int = 5) -> Optional[str]:
        """
        Normalize numeric or letter answers to a choice letter.

        Args:
            answer: Answer value (int, str, etc.)
            max_choices: Maximum number of choices (default 5)

        Returns:
            Normalized choice letter (A-E) or None if invalid
        """
        if answer is None:
            return None

        if isinstance(answer, int):
            value = answer
        else:
            text = str(answer).strip()
            if text.isdigit():
                value = int(text)
            else:
                match = re.search(
                    rf"([A-{chr(64 + max_choices)}])", text, re.IGNORECASE
                )
                return match.group(1).upper() if match else None

        if 1 <= value <= max_choices:
            return chr(64 + value)
        return None

    @staticmethod
    def _normalize_image_base64(image_base64: str) -> str:
        image_base64 = str(image_base64).strip()
        if image_base64.startswith(("http://", "https://")):
            return image_base64
        if image_base64.startswith("data:"):
            return image_base64

        # Some datasets may provide raw base64 without data URL prefix.
        image_base64 = re.sub(r"\s+", "", image_base64)

        mime_type = "image/jpeg"
        if image_base64.startswith("iVBORw0KGgo"):
            mime_type = "image/png"
        elif image_base64.startswith("/9j/"):
            mime_type = "image/jpeg"
        elif image_base64.startswith("R0lGOD"):
            mime_type = "image/gif"
        elif image_base64.startswith("UklGR"):
            mime_type = "image/webp"

        return f"data:{mime_type};base64,{image_base64}"

    @classmethod
    def _extract_image_from_sample(cls, sample: Dict[str, Any]) -> Optional[List[str]]:
        """Extract image data URLs from a sample.

        Expected schema:
            sample["images"] = [
                {"pic_num": "1", "modality": "xray", "image_base64": "..."},
                {"pic_num": "2-1", "modality": "xray", "image_base64": "..."},
                ...
            ]
        """
        if "images" not in sample:
            return None

        images = sample["images"]
        if not images:
            return None

        if isinstance(images, str):
            try:
                images = json.loads(images)
            except json.JSONDecodeError:
                return None

        if not isinstance(images, list):
            return None

        urls = []
        for img in images:
            if isinstance(img, dict) and "image_base64" in img:
                base64 = img["image_base64"]
                urls.append(cls._normalize_image_base64(base64))

        return urls if urls else None

    def get_image_count(self, sample: Dict[str, Any]) -> int:
        """
        Get number of images for a sample.

        Args:
            sample: Dataset sample dictionary

        Returns:
            Number of images (0 if none)
        """
        images = self._extract_image_from_sample(sample)
        return len(images) if images else 0

    def load_data(self) -> List[Dict[str, Any]]:
        """
        Load KorMedMCQA-V dataset from Hugging Face.

        Loads samples for specified subset(s) and converts numeric answers
        (1-5) to letter choices (A-E).

        Returns:
            List of sample dictionaries with keys:
                - uid: Unique identifier (subject-year-period-q_number)
                - question: Question text
                - images: List of image objects (optional)
                - A, B, C, D, E: Choice options
                - answer: Correct answer letter (A-E)
        """
        logger.info(
            f"Loading dataset: {self.hf_path}, subset={self.subset}, split={self.split}"
        )

        try:
            subsets = self.supported_subsets if self.subset == "all" else [self.subset]
            all_samples = []

            for subset_name in subsets:
                logger.info(f"Loading subset: {subset_name}")
                dataset = hf_load_dataset(
                    self.hf_path, name=subset_name, split=self.split
                )
                samples = list(dataset)
                for sample in samples:
                    answer = sample["answer"]
                    normalized = self._normalize_answer_to_letter(answer, max_choices=5)
                    if normalized is None:
                        raise ValueError(f"Invalid answer format: {answer}")
                    sample["answer"] = normalized
                    sample["uid"] = (
                        f"{sample['subject']}-{sample['year']}-"
                        f"{sample['period']}-{sample['q_number']}"
                    )
                    filtered = {key: sample[key] for key in self.SAMPLE_KEYS}
                    all_samples.append(filtered)

            logger.info(
                f"Loaded {len(all_samples)} samples from {self.subset} ({self.split} split)"
            )
            return all_samples
        except Exception as e:
            logger.error(f"Error loading dataset: {e}")
            raise

    def format_user_prompt(self, sample: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Build a structured user prompt with question, image(s), choices, and instruction.

        Constructs API-compatible prompt format with separate content items
        for question, image(s), each choice, and instruction.

        Args:
            sample: Dataset sample containing question, images, and choice keys (A-E)
                - images: list[dict] with `image_base64` (data URL or raw base64)

        Returns:
            List of content dictionaries:
                [
                    {"type": "text", "text": "질문: ..."},
                    {"type": "image_url", "image_url": {"url": "..."}},
                    ...
                    {"type": "text", "text": "A. Choice 1\n"},
                    {"type": "text", "text": "B. Choice 2\n"},
                    ...
                    {"type": "text", "text": "INSTRUCTION"}
                ]
        """
        content: List[Dict[str, Any]] = []

        content.append({"type": "text", "text": f"질문: {sample['question']}\n\n"})

        image_urls = self._extract_image_from_sample(sample)

        if image_urls:
            for image_url in image_urls:
                content.append({"type": "image_url", "image_url": {"url": image_url}})

        for opt in self.CHOICE_KEYS:
            if opt in sample and sample[opt]:
                content.append({"type": "text", "text": f"{opt}. {sample[opt]}\n"})

        content.append({"type": "text", "text": "\n" + self.INSTRUCTION})

        return content

    def extract_answer(self, response: str) -> Optional[str]:
        """
        Parse answer choice (A-E) from model response.

        Delegates to static method _extract_answer with dataset's choice keys.

        Args:
            response: Model's generated response

        Returns:
            Parsed answer letter (A-E) or None if not found
        """
        return self._extract_answer(response, self.CHOICE_KEYS)

    def evaluate_answer(self, predicted: str, correct: str) -> bool:
        """
        Check if predicted answer is correct.

        Case-insensitive comparison of predicted and correct answers.

        Args:
            predicted: Predicted answer letter
            correct: Correct answer letter

        Returns:
            True if answers match, False otherwise
        """
        if predicted is None or correct is None:
            return False

        return predicted.strip().upper() == str(correct).strip().upper()

    def get_choices(self) -> List[str]:
        """
        Get available answer choices for this dataset.

        Returns:
            List of choice letters: ['A', 'B', 'C', 'D', 'E']
        """
        return self.CHOICE_KEYS.copy()


def main() -> None:
    """
    Test KorMedMCQA-V dataset loader.

    Loads doctor subset and prints sample count and prompt structure.
    """
    config = {
        "name": "kormedmcqa_v",
        "subset": "doctor",
        "split": "test",
        "with_image": True,
    }
    dataset = KorMedMCQAVDataset(config)
    samples = dataset.load_data()

    if not samples:
        print("No samples loaded.")
        return

    print(f"Loaded {len(samples)} samples.")

    image_count_dist = {}
    for sample in samples:
        images = dataset._extract_image_from_sample(sample)
        count = len(images) if images else 0
        image_count_dist[count] = image_count_dist.get(count, 0) + 1

    print("\n이미지 개수 분포:")
    for count in sorted(image_count_dist.keys()):
        print(f"- {count}개 이미지: {image_count_dist[count]}개 문제")
    print(f"\n총 문제 수: {len(samples)}")

    sample = samples[0]
    prompt = dataset.format_user_prompt(sample)
    print(f"\nPrompt parts (first sample): {len(prompt)}")


if __name__ == "__main__":
    main()
