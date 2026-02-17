# KorMedMCQA-V: A Multimodal Benchmark for Evaluating Vision-Language Models on the Korean Medical Licensing Examination

[![Paper](https://img.shields.io/badge/Paper-arXiv:2602.13650-B31B1B.svg)](https://arxiv.org/abs/2602.13650)
[![Dataset](https://img.shields.io/badge/Dataset-HuggingFace-FFD21E.svg)](https://huggingface.co/datasets/seongsubae/KorMedMCQA-V)
[![Code](https://img.shields.io/badge/Code-GitHub-181717.svg)](https://github.com/baeseongsu/kormedmcqa_v)
[![Leaderboard](https://img.shields.io/badge/Leaderboard-Website-blue.svg)](https://kormedmcqa-v.github.io/)

We introduce KorMedMCQA-V, a Korean medical licensing-exam-style multimodal multiple-choice question answering benchmark for evaluating vision-language models (VLMs). The dataset consists of 1,534 questions with 2,043 associated images from Korean Medical Licensing Examinations (2012-2023), with about 30% containing multiple images requiring cross-image evidence integration. Images cover clinical modalities including X-ray, computed tomography (CT), electrocardiography (ECG), ultrasound, endoscopy, and other medical visuals. We benchmark over 50 VLMs across proprietary and open-source categories-spanning general-purpose, medical-specialized, and Korean-specialized families-under a unified zero-shot evaluation protocol. The best proprietary model (Gemini-3.0-Pro) achieves 96.9% accuracy, the best open-source model (Qwen3-VL-32B-Thinking) 83.7%, and the best Korean-specialized model (VARCO-VISION-2.0-14B) only 43.2%. We further find that reasoning-oriented model variants gain up to +20 percentage points over instruction-tuned counterparts, medical domain specialization yields inconsistent gains over strong general-purpose baselines, all models degrade on multi-image questions, and performance varies notably across imaging modalities. By complementing the text-only KorMedMCQA benchmark, KorMedMCQA-V forms a unified evaluation suite for Korean medical reasoning across text-only and multimodal conditions.

## Updates

- **[2026/02/17]** [Paper](https://arxiv.org/abs/2602.13650), [evaluation code](https://github.com/baeseongsu/kormedmcqa_v), [dataset](https://huggingface.co/datasets/seongsubae/KorMedMCQA-V), and [leaderboard](https://kormedmcqa-v.github.io/) are publicly released.

> See [VERSIONS.md](VERSIONS.md) for version information of each artifact.

## Leaderboard

Visit our [leaderboard page](https://kormedmcqa-v.github.io/) for detailed evaluation results of over 50 VLMs on KorMedMCQA-V.

## Getting Started

### Installation

```bash
uv venv --python 3.12
source .venv/bin/activate
uv pip install -e .
cp .env.example .env  # then fill in your API keys
```

### Evaluation

```bash
# OpenAI GPT-5 Mini
python scripts/eval.py \
    --model-name gpt-5-mini-2025-08-07 \
    --dataset kormedmcqa_v \
    --subset doctor \
    --split test_full \
    --reasoning-effort medium

# Google Gemini 3 Flash (auto-detects API key and endpoint from model name)
python scripts/eval.py \
    --model-name gemini-3-flash-preview \
    --dataset kormedmcqa_v \
    --subset doctor \
    --split test_full

# Open-source model via vLLM (OpenAI-compatible server)
# 1. Start vLLM server: vllm serve Qwen/Qwen2.5-VL-72B-Instruct --tensor-parallel-size 4
python scripts/eval.py \
    --model-name Qwen/Qwen2.5-VL-72B-Instruct \
    --dataset kormedmcqa_v \
    --subset doctor \
    --split test_full \
    --base-url http://localhost:8000/v1 \
    --api-key dummy

# Quick test with limited samples
python scripts/eval.py \
    --model-name gpt-5-mini-2025-08-07 \
    --dataset kormedmcqa_v \
    --subset doctor \
    --split test_full \
    --max-samples 10
```

<details>
<summary>All CLI arguments</summary>

| Argument | Default | Description |
|----------|---------|-------------|
| `--model-name` | (required) | Model name for API |
| `--dataset` | `kormedmcqa_v` | Dataset: `kormedmcqa_v` or `kormedmcqa_mixed` |
| `--subset` | `doctor` | Dataset subset |
| `--split` | `test` | Data split: `test`, `test_full`, `dev`, `train` |
| `--api-key` | Auto-detected | API key (auto: `GEMINI_API_KEY` for Gemini models, `OPENAI_API_KEY` otherwise) |
| `--base-url` | Auto-detected | API base URL (auto-detected from model name) |
| `--temperature` | `0.7` | Generation temperature |
| `--max-tokens` | `8192` | Maximum tokens to generate |
| `--max-samples` | `None` | Limit samples (for debugging) |
| `--output-dir` | `./results` | Output directory |
| `--seed` | `42` | Random seed |
| `--reasoning-effort` | `None` | Reasoning effort for supported models |

</details>

### Python API

```python
from openai import OpenAI
from kormedeval import get_dataset, evaluate_dataset

# Load dataset
dataset = get_dataset("kormedmcqa_v", {
    "name": "kormedmcqa_v",
    "subset": "doctor",
    "split": "test_full",
})

# Create client
client = OpenAI(api_key="your-key")

# Run evaluation
result = evaluate_dataset(
    client=client,
    model_name="gpt-5-mini-2025-08-07",
    dataset=dataset,
    max_samples=None,
    temperature=0.7,
    max_tokens=8192,
    is_vlm=True,
)

print(f"Accuracy: {result['accuracy']:.4f}")
```

## Citation

If you use this benchmark in your research, please cite our paper:

```bibtex
@article{choi2026kormedmcqav,
  title={KorMedMCQA-V: A Multimodal Benchmark for Evaluating Vision-Language Models on the Korean Medical Licensing Examination},
  author={Choi, Byungjin and Bae, Seongsu and Kweon, Sunjun and Choi, Edward},
  journal={arXiv preprint arXiv:2602.13650},
  year={2026}
}
```

## License

The code in this repository is licensed under the [MIT License](LICENSE).
The KorMedMCQA-V dataset is licensed under [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/).

## Contact

For questions or concerns regarding the dataset or code, please contact Byungjin Choi ([choi328328@ajou.ac.kr](mailto:choi328328@ajou.ac.kr)) or Seongsu Bae ([seongsu@kaist.ac.kr](mailto:seongsu@kaist.ac.kr)).
