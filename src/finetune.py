"""Fine-tune: build the model and data from the YAMLs, load only the weights of $INIT_CKPT, then fit
with a fresh optimizer and schedule. The path comes from the environment, not argv, because DDP
relaunches this script for the second GPU with the same argv. Used for FocusAnchor-flex and the DDFF12 fine-tune (README).
Usage: INIT_CKPT=x.ckpt uv run python src/finetune.py --config a.yaml [--model.lr 5e-5 ...]"""
import os

import torch
from pytorch_lightning.cli import LightningCLI

from model.focusAnchor import FocusAnchorModule

init = os.environ["INIT_CKPT"]
torch.set_float32_matmul_precision("high")
cli = LightningCLI(FocusAnchorModule, datamodule_class=None, run=False)
sd = torch.load(init, map_location="cpu", weights_only=False)["state_dict"]
cli.model.load_state_dict(sd, strict=True)
print("loaded weights from", init, "lr", cli.model.hparams.lr, flush=True)
cli.trainer.fit(cli.model, cli.datamodule)
