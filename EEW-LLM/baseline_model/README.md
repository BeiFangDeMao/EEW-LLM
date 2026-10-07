# Baseline model code

This folder contains code copied from `model_compare` for the comparison models, without checkpoints, logs, prediction tables, images, or datasets.

| Model | Tasks / training entry points |
| --- | --- |
| EQTransformer | P-wave picking and event detection: `EQTransformer/traineqtnoise.py` |
| PhaseNet | P-wave picking and event detection: `PhaseNet/trainpnnoise.py` |
| MagNet | Magnitude: `MagNet/trainmagnet.py` |
| SeisT | Magnitude, epicentral distance, back-azimuth: `SeisT/trainemg.py`, `traindis.py`, `trainazi.py` |
| SeisMoLLM | P-wave picking and event detection, magnitude, epicentral distance, back-azimuth: `SeisMoLLM/train_*.py` |

Each model's architecture code is in the same directory as its training script. The SeisMoLLM regression model depends on `SeisMoLLM/original_models/SeisMoLLM.py` and `_factory.py`, copied from `SeisMoLLMAnnotatedEdition/models`; the copied wrapper loads those local files. Training datasets and pretrained language-model weights are not included; the existing training scripts retain their configured paths.

EEW-LLM/BART is the proposed model, not a baseline, and is not included here. BazNet appears in the manuscript comparison but a matching BazNet training script was not found in `model_compare` or the project Python files; it has not been substituted with another implementation.
