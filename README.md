# FocusAnchor

Official code of *FocusAnchor: Zero-Shot Metric Depth by Anchoring a Depth Foundation Prior to Focus Cues*, by Ashkan Ganj, Hang Su and Tian Guo.

[Project page](https://cake-lab.github.io/FocusAnchor-website/) | [Weights](https://huggingface.co/AshkanGanj/FocusAnchor)

FocusAnchor predicts metric depth from a focal stack and its all-in-focus image. A CNN encodes each slice. A focus volume scores every slice at every pixel. A GRU then refines inverse depth by reading this volume, starting from the relative depth of a frozen Depth Anything V2 (DAv2) ViT-L. In every refinement step the DAv2 depth is refitted locally to the current estimate, weighted by a predicted confidence, and fed back to the GRU. The state is inverse depth divided by the nearest focus plane, so the same weights run on any camera and any number of slices.

## Layout

```text
src/
  cli.py                            entry point (LightningCLI): fit and test
  finetune.py                       fit from the weights of another checkpoint
  model/focusAnchor.py              Lightning module: units, loss, logging
  model/modules/defom/              the network (MIT, LICENSE.DEFOM)
  model/modules/depth_anything_v2/  frozen DAv2 (Apache-2.0, LICENSE.DAV2)
  dataloader/                       NYUv2, DDFF12, ZEDD, Infinigen, iBims-1, HAMMER, ARKitScenes
  utils/                            metrics, figures, focal-stack renderer
  configs/                          fit_*.yaml and test_*.yaml
checkpoints/                        weights, ignored by Git
reports/, checkpoints_best/         generated outputs, ignored by Git
```

## Setup

The project uses `uv` with Python 3.12. Run from the repository root:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh   # only if uv is missing
uv python install 3.12
uv sync --frozen
uv run python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

## Weights

Put all weights in `checkpoints/`.

- DAv2 ViT-L in the original Depth-Anything-V2 format, not the `-hf` one: `depth_anything_v2_vitl.pth` ([download](https://huggingface.co/depth-anything/Depth-Anything-V2-Large/resolve/main/depth_anything_v2_vitl.pth)). Building the model needs this file, for testing too. Its license is CC-BY-NC-4.0.
- FocusAnchor weights from [Hugging Face](https://huggingface.co/AshkanGanj/FocusAnchor). Each file holds the trained weights only. The frozen DAv2 ViT-L is left out, because the model loads it from `depth_anything_v2_vitl.pth`.

| File | Model |
|---|---|
| `FocusAnchor.ckpt` | FocusAnchor, trained on NYU Depth v2, five slices |
| `FocusAnchor-RS.ckpt` | FocusAnchor-flex, trained with 2 to 5 slices |
| `FocusAnchor-NYU.ckpt` | NYU in-domain variant |
| `FocusAnchor-DDFF12.ckpt` | FocusAnchor fine-tuned on DDFF12 |
| `FocusAnchor-RS-DDFF12.ckpt` | FocusAnchor-flex fine-tuned on DDFF12 |

```bash
for f in FocusAnchor FocusAnchor-RS FocusAnchor-NYU FocusAnchor-DDFF12 FocusAnchor-RS-DDFF12; do
  curl -L -o checkpoints/$f.ckpt https://huggingface.co/AshkanGanj/FocusAnchor/resolve/main/$f.ckpt
done
```

## Test

Set the dataset root in `src/configs/test_*.yaml`. Then run from the repository root:

```bash
uv run python src/cli.py test --config src/configs/test_ZEDD.yaml
```

Each run prints a "Final Evaluation Results" table and logs the same numbers to TensorBoard as `test/*`. The YAML builds the model and `--ckpt_path` only swaps the weights.

- FocusAnchor-flex: add `--ckpt_path checkpoints/FocusAnchor-RS.ckpt`. On DDFF12 add `--ckpt_path checkpoints/FocusAnchor-RS-DDFF12.ckpt`.
- NYU in-domain variant: `test_NYU.yaml` with `--ckpt_path checkpoints/FocusAnchor-NYU.ckpt` gives AbsRel 0.0077.
- ZEDD test: the test ground truth is hidden, so the benchmark server scores it. `PYTHONPATH=src uv run python src/dataloader/ZEDD/submit.py checkpoints/FocusAnchor.ckpt OUT_DIR --root ZEDD_ROOT` writes `OUT_DIR/zedd_outputs.zip`. Check it with `uv run python src/dataloader/ZEDD/check_format.py --zip OUT_DIR/zedd_outputs.zip`.
- Protocols: iBims-1, HAMMER and ARKitScenes use FOSSA's rendered stacks (`src/dataloader/RGBD/loader.py`: F/1.4, 15x15 Gaussian PSF, 36 mm sensor, five planes even in disparity between the 5th and 95th percentile of the ground truth). ARKitScenes is resized from 1920x1440 to 480x640. NYU uses a fixed render (`renderSyntheticFocalStack` in `src/dataloader/batch.py`): the NYU camera, planes over each image's depth range, and a 384x512 centre crop. Infinigen is scored at 480x640.
- The 4th decimal can differ between GPUs, because the model runs in bf16.

ZEDD is [FOSSA](https://github.com/princeton-vl/FOSSA)'s real-camera benchmark. Download `ZEDD.zip` from [Hugging Face](https://huggingface.co/datasets/venkatsubra/ZEDD). Unzip it and move `metadata.txt`, `val/` and `test/` out of `tmp_zedd_release_upload/` into the root set in `test_ZEDD.yaml`. `--data.init_args.aif f16` uses the F/16 shot as the all-in-focus image.

## Train

FocusAnchor and FocusAnchor-flex train only on NYU Depth v2. `fit_NYU.yaml` is the FocusAnchor recipe: DAv2 ViT-L, `prior_fusion: input`, FOSSA's random cameras and focus planes (`fossa_synthesis: True`), per-slice noise (`slice_noise: 0.01`), five slices, seed 0. It runs 60k steps on two GPUs with DDP, four images per GPU. `devices: auto` takes every visible GPU, so set `CUDA_VISIBLE_DEVICES` to two of them. Every epoch is saved (1.6 GB each). The paper uses `last.ckpt`.

```bash
# FocusAnchor
uv run python src/cli.py fit --config src/configs/fit_NYU.yaml
```

FocusAnchor-flex trains in two stages. Stage 1 is the same run with 2 to 5 random slices per batch. Stage 2 loads only the stage-1 weights and fine-tunes them for 20k steps with a fresh optimizer at lr 5e-5. `src/finetune.py` reads the start weights from `INIT_CKPT`, an environment variable, because DDP restarts the script with the same arguments. It has no `fit` subcommand. The paper's flex model is epoch 5 of stage 2 (`epoch05-*.ckpt`).

```bash
# FocusAnchor-flex, stage 1: 60k steps
uv run python src/cli.py fit --config src/configs/fit_NYU.yaml \
  --model.train_n_stack "[2, 5]" \
  --trainer.logger.init_args.name nyu_flex_60k \
  --trainer.callbacks.init_args.dirpath checkpoints_best/nyu_flex_60k

# FocusAnchor-flex, stage 2: 20k-step fine-tune from the stage-1 weights
INIT_CKPT=checkpoints_best/nyu_flex_60k/last.ckpt uv run python src/finetune.py \
  --config src/configs/fit_NYU.yaml \
  --model.train_n_stack "[2, 5]" --model.lr 5e-5 --trainer.max_steps 20000 --seed_everything 1 \
  --trainer.logger.init_args.name nyu_flex_ft20k \
  --trainer.callbacks.init_args.dirpath checkpoints_best/nyu_flex_ft20k
```

The NYU in-domain variant turns both randomizations off. It then trains on the same render as the NYU test stacks:

```bash
uv run python src/cli.py fit --config src/configs/fit_NYU.yaml \
  --model.fossa_synthesis False --model.slice_noise 0.0 \
  --trainer.logger.init_args.name nyu_indomain \
  --trainer.callbacks.init_args.dirpath checkpoints_best/nyu_indomain
```

The DDFF12 models are fine-tuned from the NYU weights for 20k steps on one GPU with five slices. The paper uses `last.ckpt`.

```bash
# FocusAnchor on DDFF12
INIT_CKPT=checkpoints/FocusAnchor.ckpt uv run python src/finetune.py --config src/configs/fit_DDFF12.yaml

# FocusAnchor-flex on DDFF12
INIT_CKPT=checkpoints/FocusAnchor-RS.ckpt uv run python src/finetune.py --config src/configs/fit_DDFF12.yaml \
  --trainer.logger.init_args.name ddff12_ft_flex \
  --trainer.callbacks.init_args.dirpath checkpoints_best/ddff12_ft_flex
```

Give each run its own logger `name` and checkpoint `dirpath`, as above. Otherwise a new run overwrites the `last.ckpt` of an old one. `--trainer.callbacks.init_args.*` sets the last callback in the YAML, so `ModelCheckpoint` stays last there. Any other Lightning flag can be overridden the same way, for example `--trainer.max_steps 100`.

NYU has no real focal stack. A plain PyTorch Gaussian-PSF renderer (`src/utils/synthetic/camera_lib.py`) makes one on the training device, after the batch is moved there. With `model.fossa_synthesis: True`, training uses FOSSA's random cameras and focus planes instead. Validation keeps the fixed render.

Outputs, all ignored by Git:

- TensorBoard logs: `reports/logs/<logger name>/`
- Fit checkpoints: `checkpoints_best/<run>/`
- Test metrics: `reports/results/csv/metrics.csv`
- Fallback figures: `reports/results/imgs/`


## Credits

DEFOM-Stereo: Jiang et al., CVPR 2025. Depth Anything V2: Yang et al., NeurIPS 2024. FOSSA and ZEDD: Zuo et al., "Zero-Shot Depth from Defocus", ECCV 2026. HybridDepth: Ganj et al., WACV 2025.

## Citation

```bibtex
@article{ganj2026focusanchor,
  title   = {FocusAnchor: Zero-Shot Metric Depth by Anchoring a Depth Foundation Prior to Focus Cues},
  author  = {Ganj, Ashkan and Su, Hang and Guo, Tian},
  journal = {arXiv preprint},
  year    = {2026}
}
```

## License

The code is under GPL-3.0 (`LICENSE`). Third-party parts keep their own licenses: DEFOM-Stereo (MIT, `LICENSE.DEFOM`), the Depth Anything V2 code (Apache-2.0, `LICENSE.DAV2`) and FOSSA (BSD-3, `LICENSE.FOSSA`). The Depth Anything V2 ViT-L weights are CC-BY-NC-4.0 and FocusAnchor needs them at run time, so check that license before any commercial use. The datasets have their own licenses.
