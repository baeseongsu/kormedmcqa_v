"""
Synchronous evaluation module for API-based models.

Provides core evaluation functions for running API-based models (OpenAI, Gemini, etc.)
on medical QA datasets.
"""

import logging
from typing import Any, Dict, Optional

from openai import OpenAI
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)
from tqdm import tqdm

from kormedeval.utils import extract_thinking_and_summary

logger = logging.getLogger(__name__)


@retry(
    stop=stop_after_attempt(10),
    wait=wait_exponential(multiplier=1, min=1, max=10),
    retry=retry_if_exception_type((TimeoutError, ConnectionError)),
    reraise=True,
)
def evaluate_sample(
    client: OpenAI,
    model_name: str,
    sample: Dict[str, Any],
    dataset: Any,
    temperature: float,
    max_tokens: int,
    is_vlm: bool = False,
    reasoning_effort: Optional[str] = None,
) -> Dict[str, Any]:
    """Evaluate a single sample with retry logic.

    Args:
        client: OpenAI client
        model_name: Model name
        sample: Test sample
        dataset: Dataset instance
        temperature: Generation temperature
        max_tokens: Maximum tokens to generate
        is_vlm: Whether model supports vision-language
        reasoning_effort: Reasoning effort parameter (e.g., 'medium' for GPT-5.2)

    Returns:
        Result dictionary
    """

    formatted_content = dataset.format_user_prompt(sample)

    if is_vlm:
        messages = [{"role": "user", "content": formatted_content}]
    else:
        prompt = "".join([item["text"] for item in formatted_content])
        messages = [{"role": "user", "content": prompt}]

    try:
        completion_params = {
            "model": model_name,
            "messages": messages,
        }

        # GPT-5 models only support temperature=1.0 (default)
        if not model_name.startswith("gpt-5"):
            completion_params["temperature"] = temperature

        # Use max_completion_tokens for GPT-5 models, max_tokens for others
        if model_name.startswith("gpt-5"):
            completion_params["max_completion_tokens"] = max_tokens
            if reasoning_effort is not None:
                completion_params["reasoning_effort"] = reasoning_effort
        else:
            completion_params["max_tokens"] = max_tokens

        response = client.chat.completions.create(**completion_params)

        generated_text = response.choices[0].message.content

        # Extract thinking blocks if present (e.g., Qwen3-VL-Thinking models)
        _, clean_text = extract_thinking_and_summary(generated_text)

        predicted_answer = dataset.extract_answer(clean_text)

        is_correct = dataset.evaluate_answer(predicted_answer, sample["answer"])

        return {
            "uid": sample["uid"],
            "question": sample["question"],
            "correct_answer": sample["answer"],
            "predicted_answer": predicted_answer,
            "generated_text": generated_text,
            "is_correct": is_correct,
            "success": True,
        }

    except Exception as e:
        logger.error(f"Error evaluating sample: {e}")

        return {
            "uid": sample["uid"],
            "question": sample["question"],
            "correct_answer": sample["answer"],
            "predicted_answer": None,
            "generated_text": str(e),
            "is_correct": False,
            "success": False,
        }


def evaluate_dataset(
    client: OpenAI,
    model_name: str,
    dataset: Any,
    max_samples: Optional[int],
    temperature: float,
    max_tokens: int,
    is_vlm: bool = False,
    reasoning_effort: Optional[str] = None,
) -> Dict[str, Any]:
    """Evaluate on a dataset subset with sequential processing.

    Args:
        client: OpenAI client
        model_name: Model name
        dataset: Dataset instance
        max_samples: Maximum number of samples to evaluate
        temperature: Generation temperature
        max_tokens: Maximum tokens to generate
        is_vlm: Whether model supports vision-language
        reasoning_effort: Reasoning effort parameter (e.g., 'medium' for GPT-5.2)

    Returns:
        Results dictionary
    """
    subset = dataset.subset
    split = dataset.split

    samples = dataset.load_data()
    if max_samples:
        samples = samples[:max_samples]

    logger.info(
        f"Loaded {len(samples)} samples from {dataset.name}/{subset} ({split} split)"
    )

    results = []
    correct_count = 0
    total_count = 0

    for sample in tqdm(samples, desc=f"Evaluating {dataset.name}/{subset}"):
        result = evaluate_sample(
            client=client,
            model_name=model_name,
            sample=sample,
            dataset=dataset,
            temperature=temperature,
            max_tokens=max_tokens,
            is_vlm=is_vlm,
            reasoning_effort=reasoning_effort,
        )

        results.append(result)

        if result["success"] and result["is_correct"]:
            correct_count += 1

        if result["success"]:
            total_count += 1

    accuracy = correct_count / total_count if total_count > 0 else 0.0

    logger.info(
        f"Dataset: {dataset.name}/{subset}, Accuracy: {accuracy:.4f} "
        f"({correct_count}/{total_count})"
    )

    return {
        "dataset_name": dataset.name,
        "subset": subset,
        "split": split,
        "total_samples": len(samples),
        "evaluated_samples": total_count,
        "correct": correct_count,
        "accuracy": accuracy,
        "predictions": results,
    }


__all__ = ["evaluate_sample", "evaluate_dataset"]
