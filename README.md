# KorMedMCQA-V: A Multimodal Benchmark for Evaluating Vision-Language Models on the Korean Medical Licensing Examination

[![License: MIT](https://img.shields.io/badge/Code-MIT-yellow.svg)](LICENSE)
[![License: CC BY-NC-SA 4.0](https://img.shields.io/badge/Dataset-CC%20BY--NC--SA%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by-nc-sa/4.0/)

We introduce KorMedMCQA-V, a Korean medical licensing-exam-style multimodal multiple-choice question answering benchmark for evaluating vision-language models (VLMs). The dataset consists of 1,534 questions with 2,043 associated images from Korean Medical Licensing Examinations (2012-2023), with about 30% containing multiple images requiring cross-image evidence integration. Images cover clinical modalities including X-ray, computed tomography (CT), electrocardiography (ECG), ultrasound, endoscopy, and other medical visuals. We benchmark over 50 VLMs across proprietary and open-source categories-spanning general-purpose, medical-specialized, and Korean-specialized families-under a unified zero-shot evaluation protocol. The best proprietary model (Gemini-3.0-Pro) achieves 96.9% accuracy, the best open-source model (Qwen3-VL-32B-Thinking) 83.7%, and the best Korean-specialized model (VARCO-VISION-2.0-14B) only 43.2%. We further find that reasoning-oriented model variants gain up to +20 percentage points over instruction-tuned counterparts, medical domain specialization yields inconsistent gains over strong general-purpose baselines, all models degrade on multi-image questions, and performance varies notably across imaging modalities. By complementing the text-only KorMedMCQA benchmark, KorMedMCQA-V forms a unified evaluation suite for Korean medical reasoning across text-only and multimodal conditions.

## Updates

- [Coming Soon] Paper, evaluation code, and dataset will be publicly released.

## Citation

If you use this benchmark in your research, please cite our paper (coming soon):

```
KorMedMCQA-V: A Multimodal Benchmark for Evaluating Vision-Language Models
on the Korean Medical Licensing Examination
Byungjin Choi, Seongsu Bae, Sunjun Kweon and Edward Choi
```

## License

The code in this repository is licensed under the [MIT License](LICENSE).
The KorMedMCQA-V dataset is licensed under [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/).

## Contact

For questions or concerns regarding the dataset or code, please contact Seongsu Bae ([seongsu@kaist.ac.kr](mailto:seongsu@kaist.ac.kr)).
