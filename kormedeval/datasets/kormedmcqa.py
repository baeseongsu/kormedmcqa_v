"""
KorMedMCQA Dataset

Korean Medical Multiple Choice Question Answering dataset (text-only).
Supports doctor, nurse, pharm, and dentist subsets.
"""

import ast
import json
import logging
import re
from typing import Any, Dict, List, Optional

from datasets import load_dataset as hf_load_dataset

from .base import BaseDataset

logger = logging.getLogger(__name__)


class KorMedMCQADataset(BaseDataset):
    """
    Korean Medical Multiple Choice Question Answering Dataset (Text-only)

    Features:
        - Text-only questions (no images)
        - Four subsets: doctor, nurse, pharm, dentist
        - 5 choices (A-E)
    """

    CHOICE_KEYS = ["A", "B", "C", "D", "E"]
    HF_PATH = "sean0042/KorMedMCQA"
    SUPPORTED_SUBSETS = ["doctor"]
    SAMPLE_KEYS = ["uid", "question", *CHOICE_KEYS, "answer"]
    INSTRUCTION = (
        "각 문제에서 가장 적절한 답을 하나만 고르시오.\n"
        "정답은 A, B, C, D, E 중 하나입니다.\n"
        "출력은 반드시 JSON 객체 1개만 반환하고, JSON 밖의 다른 텍스트(설명/마크다운 등)는 쓰지 마세요.\n"
        '출력 형식: {"정답": "A"}'
    )

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.hf_path = self.HF_PATH
        self.supported_subsets = self.SUPPORTED_SUBSETS

    @staticmethod
    def _extract_answer(response: str, choice_keys: List[str]) -> Optional[str]:
        """
        Extract answer letter from model responses.

        Supports (in order):
        - JSON format: {"정답": "A"} or {"answer": "A"}
        - LaTeX box format (fallback): $\\boxed{E}$
        """
        if not response:
            return None

        fenced = re.search(
            r"```json\s*(\{.*?\})\s*```", response, re.DOTALL | re.IGNORECASE
        )
        blocks = (
            [fenced.group(1)] if fenced else re.findall(r"\{.*?\}", response, re.DOTALL)
        )

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
        """Normalize numeric or letter answers to a choice letter."""
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

    def load_data(self) -> List[Dict[str, Any]]:
        """
        Load KorMedMCQA dataset from Hugging Face.

        Returns:
            List of sample dictionaries with keys:
                - uid: Unique identifier (subject-year-period-q_number)
                - question: Question text
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
        Build a structured user prompt with question, choices, and instruction.

        Text-only format (no images).

        Args:
            sample: Dataset sample containing question and choice keys (A-E)

        Returns:
            List of content dictionaries
        """
        content: List[Dict[str, Any]] = []

        content.append({"type": "text", "text": f"질문: {sample['question']}\n\n"})

        for opt in self.CHOICE_KEYS:
            if opt in sample and sample[opt]:
                content.append({"type": "text", "text": f"{opt}. {sample[opt]}\n"})

        content.append({"type": "text", "text": "\n" + self.INSTRUCTION})

        return content

    def extract_answer(self, response: str) -> Optional[str]:
        return self._extract_answer(response, self.CHOICE_KEYS)

    def evaluate_answer(self, predicted: str, correct: str) -> bool:
        if predicted is None or correct is None:
            return False
        return predicted.strip().upper() == str(correct).strip().upper()

    def get_choices(self) -> List[str]:
        return self.CHOICE_KEYS.copy()
