# Version Tracker

Tracks the current version and links for each artifact.

| Artifact | Version / Revision | URL | Updated |
|---|---|---|---|
| Paper | arXiv:2602.13650 | https://arxiv.org/abs/2602.13650 | 2026-02-17 |
| Dataset | v1.0.0 | https://huggingface.co/datasets/seongsubae/KorMedMCQA-V | 2026-02-17 |
| Code | v1.0.0 | https://github.com/baeseongsu/kormedmcqa_v | 2026-02-17 |
| Leaderboard | - | https://kormedmcqa-v.github.io/ | 2026-02-17 |

## Version Update Guide

Follow these steps when updating an artifact:

1. Update the Version / Updated columns in the table above
2. Add a dated changelog entry to the Updates section in `README.md`
3. For code changes, also bump the version in `pyproject.toml` and `kormedeval/__init__.py`
4. Create a git tag (e.g., `git tag v1.1.0`)
