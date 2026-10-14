# Model E: Tempo-Conditioned Piano Audio Diffusion

Experimental latent diffusion model for tempo-customizable piano audio
using preprocessed MAESTRO recordings and ASAP beat annotations.

## Current implementation

- Stereo audio at 8,000 Hz.
- Training examples: 65,536 samples, or 8.192 seconds.
- Pretrained frozen AudioVAE1D.
- Fixed latent scaling estimated from the training split.
- Scalar BPM conditioning through FiLM.
- Multiscale MIDI-onset and beat-grid conditioning.
- Target tempos: 60, 75, 90, 105, 120, 135, 150, 165, 180 BPM.

Tempo control remains under evaluation. This implementation does not
establish independent scalar-BPM-only control.

## Entry points

- preprocess_maestro_asap.py: dataset preprocessing.
- train_model_e_conditioned.py: training and checkpoint resume.
- check_model_e_conditioned.py: implementation checks.
- generate_model_e_conditioned_30s.py: 30-second generation.
- generate_model_e_tempo_sweep.py: controlled multi-tempo generation.

Use --help with an entry point to see its arguments.

## Environment

The current training environment uses Python 3.8, PyTorch 1.13.1,
and PyTorch Lightning 1.9.5.

See requirements-environment.txt for the captured package versions.
Dataset preprocessing also requires the Rubber Band command-line tool.

## External files

Dataset audio, MIDI, conditioning arrays, pretrained VAE weights,
diffusion checkpoints, and generated outputs are stored separately.

Training and generation require the appropriate external paths.

## Acknowledgments

Based on work from:
- https://github.com/Harmonai-org/sample-generator
- https://github.com/Remi908/VAE-Audio-Diffusion

Data sources:
- https://magenta.tensorflow.org/datasets/maestro
- https://github.com/fosfrancesco/asap-dataset

Preserve applicable upstream copyright and license notices.
