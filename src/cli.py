import torch
from pytorch_lightning.cli import LightningCLI

from model.focusAnchor import FocusAnchorModule


class CLI(LightningCLI):
    def _parse_ckpt_path(self) -> None:
        # Lightning writes the checkpoint's hparams over the YAML and the flags. Skip that:
        # the YAML builds the model and the checkpoint gives the weights. So a DDFF12
        # checkpoint (invert_depth False) can be tested on metric data. A wrong encoder
        # still fails in the strict weight load.
        pass


def cliMain():
    """FocusAnchor."""
    torch.set_float32_matmul_precision("high")
    cli = CLI(FocusAnchorModule, datamodule_class=None)

if __name__ == "__main__":
    cliMain()
