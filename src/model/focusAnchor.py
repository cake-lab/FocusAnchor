from time import perf_counter
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from pytorch_lightning.utilities import grad_norm

from dataloader.batch import prepareBatch
from model.modules.defom.defom import DefomFocus
from utils.io import IO, toNumpy
from utils.metric_cal import calcMetrics


RESULTS_CSV_PATH = "reports/results/csv/metrics.csv"
RESULTS_IMAGE_DIR = "reports/results/imgs"
# A stride rather than hand-picked batch indices: it scales to any test-set size,
# and keeps meaning the same when batch_size changes.
FIGURE_EVERY = 10
# Confidence target: the state counts as right when |log pred - log gt| is below this.
CONF_TOL = 0.05


class FocusAnchorModule(pl.LightningModule):
    """Lightning wrapper around DefomFocus.

    Units live here and nowhere else. `invert_depth` says the batch is in metres (NYU,
    ARKitScenes), so the model gets 1 / focus_dist. Otherwise the batch is already
    inverse depth (DDFF12 disparity). The model sees inverse depth divided by the
    nearest focus plane, and its output goes back to the batch units before the loss
    and metrics.
    """

    def __init__(
        self,
        invert_depth: bool = True,
        lr: float = 2e-4,
        wd: float = 1e-5,
        dinov2_encoder: str = "vits",  # DAv2: vits, vitb, vitl.
        n_downsample: int = 2,
        train_iters: int = 12,
        valid_iters: int = 16,
        scale_iters: int = 4,
        corr_levels: int = 2,
        corr_radius: int = 4,
        corr_step: float = 0.5,
        scale_list: list[float] = [0.5, 0.67, 0.8, 0.9, 1.0, 1.1, 1.25, 1.5, 2.0],
        scale_corr_radius: int = 2,
        mixed_precision: bool = True,
        loss_gamma: float = 0.9,
        volume_loss_weight: float = 1.0,
        # Train on FOSSA's random cameras and focus planes (batch.py:renderFossaBatch).
        # Validation and test keep the fixed render (batch.py:renderSyntheticFocalStack).
        fossa_synthesis: bool = False,
        # Largest sigma of Gaussian sensor noise added to each rendered slice in training,
        # in [0, 1] units (batch.py:normalizeStack). Kills the render's noise-as-focus cue
        # on blank surfaces. 0 is off. ZEDD measures about 0.002.
        slice_noise: float = 0.0,
        # DAv2 fusion in the delta iterations (defom.py). 'input': a confidence head, and
        # DAv2 fitted locally to the state by that confidence, both go into the GRU.
        # 'blend': the same, then the state becomes conf * GRU + (1 - conf) * fitted DAv2.
        # 'none' is off.
        prior_fusion: str = "none",
        conf_loss_weight: float = 0.1,
        # Focus volume scales: 1 is the 1/4 volume only, 3 adds 1/8 and 1/16.
        volume_scales: int = 1,
        # (min, max): the FOSSA render draws a random stack size per training batch.
        # Needs fossa_synthesis. Validation and test keep the loader's size.
        train_n_stack: tuple[int, int] | None = None,
    ) -> None:
        super().__init__()
        if train_n_stack and not fossa_synthesis:
            raise ValueError("train_n_stack needs fossa_synthesis.")
        self.save_hyperparameters()
        self.model = DefomFocus(SimpleNamespace(
            **self.hparams, n_gru_layers=3, hidden_dims=[128] * 3, context_norm="batch"))
        self.io = IO(RESULTS_CSV_PATH, RESULTS_IMAGE_DIR)
        self.test_metrics: list[dict[str, float]] = []
        self.confidences: list[torch.Tensor] = []

    def forward(self, rgb_aif, focal_stack, focus_dist, test_mode=True):
        """Returns (list of predictions, DAv2 initialization, volume prediction) in the
        batch units."""
        hp = self.hparams
        inverse = 1 / focus_dist if hp.invert_depth else focus_dist
        nearest = inverse[:, :1]
        # Confidence logits are kept on the module, so the return stays 3 values.
        predictions, init, volumePred, self.confidences = self.model(
            rgb_aif, focal_stack, inverse / nearest,
            iters=hp.valid_iters if test_mode else hp.train_iters,
            scaleIters=hp.scale_iters, testMode=test_mode)
        nearest = nearest.view(-1, 1, 1, 1)
        if hp.invert_depth:
            return [1 / (p * nearest) for p in predictions], 1 / (init * nearest), 1 / (volumePred * nearest)
        return [p * nearest for p in predictions], init * nearest, volumePred * nearest

    def sequenceLoss(self, predictions, gt, mask):
        """DEFOM's gamma-weighted loss over all iterations, with L1 on log depth instead
        of L1 on disparity. The log makes it the same in metres and in disparity."""
        valid = mask.bool() & (gt > 0)
        logGt = gt[valid].log()
        n = len(predictions)
        gamma = self.hparams.loss_gamma ** (15 / max(n - 1, 1))
        return sum(gamma ** (n - i - 1) * (p[valid].clamp(min=1e-6).log() - logGt).abs().mean()
                   for i, p in enumerate(predictions))

    def training_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        return self.sharedStep(batch, batch_idx, stage="train")

    def validation_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        return self.sharedStep(batch, batch_idx, stage="val")

    def sharedStep(self, batch: Any, batch_idx: int, stage: str) -> torch.Tensor:
        sizes = self.hparams.train_n_stack if stage == "train" else None
        depth_batch = prepareBatch(batch, fossa=stage == "train" and self.hparams.fossa_synthesis,
                                   noise=self.hparams.slice_noise if stage == "train" else 0.0,
                                   nStack=torch.randint(sizes[0], sizes[1] + 1, ()).item() if sizes else None)
        predictions, init, volumePred = self.forward(depth_batch.rgb_aif, depth_batch.focal_stack,
                                                     depth_batch.focus_dist, test_mode=stage != "train")
        # A single prediction makes sequenceLoss a plain L1 on log depth.
        volumeLoss = self.sequenceLoss([volumePred], depth_batch.depth_gt, depth_batch.mask)
        loss = (self.sequenceLoss(predictions, depth_batch.depth_gt, depth_batch.mask)
                + self.hparams.volume_loss_weight * volumeLoss)
        if stage == "train" and self.confidences:
            # Each confidence judges the state before its own update, so it pairs with
            # the prediction one iteration earlier.
            valid = depth_batch.mask.bool() & (depth_batch.depth_gt > 0)
            logGt = depth_batch.depth_gt[valid].log()
            judged = predictions[len(predictions) - len(self.confidences) - 1:]
            confLoss = sum(F.binary_cross_entropy_with_logits(
                logit[valid], ((p.detach()[valid].clamp(min=1e-6).log() - logGt).abs() < CONF_TOL).float())
                for logit, p in zip(self.confidences, judged)) / len(self.confidences)
            loss = loss + self.hparams.conf_loss_weight * confLoss
            self.log("train/conf_loss", confLoss, batch_size=len(depth_batch.rgb_aif))

        # Batches are dicts, so Lightning cannot infer the batch size on its own.
        batch_size = len(depth_batch.rgb_aif)
        self.log(f"{stage}/total_loss", loss, prog_bar=True, sync_dist=True, batch_size=batch_size)
        self.log(f"{stage}/volume_loss", volumeLoss, sync_dist=True, batch_size=batch_size)

        # Depth metrics on val only: they force a host sync, and on train the loss is
        # the signal anyway.
        metrics = None
        if stage == "val":
            metrics = calcMetrics(toNumpy(predictions[-1]), toNumpy(depth_batch.depth_gt),
                                  toNumpy(depth_batch.mask))
            self.log_dict({f"val/{name}": value for name, value in metrics.items()},
                          sync_dist=True, batch_size=batch_size)

        # Once per epoch: global_step does not advance during validation, so logging
        # more often would just overwrite the same tag.
        if batch_idx == 0 and self.global_rank == 0:
            self.io.plotDiagnostics(depth_batch.rgb_aif, init, predictions[-1], depth_batch.depth_gt,
                                  depth_batch.mask, metrics=metrics, tag=stage, step=self.global_step,
                                  experiment=self.experiment())
        return loss

    def test_step(self, batch: Any, batch_idx: int) -> None:
        depth_batch = prepareBatch(batch)
        # CUDA launches are async, so both ends need a sync or this times kernel
        # dispatch instead of the forward pass.
        if depth_batch.rgb_aif.is_cuda:
            torch.cuda.synchronize()
        start_time = perf_counter()
        predictions, init, _ = self.forward(depth_batch.rgb_aif, depth_batch.focal_stack, depth_batch.focus_dist)
        if depth_batch.rgb_aif.is_cuda:
            torch.cuda.synchronize()
        elapsed_time = perf_counter() - start_time

        metrics = calcMetrics(toNumpy(predictions[-1]), toNumpy(depth_batch.depth_gt), toNumpy(depth_batch.mask))
        metrics["seconds"] = elapsed_time

        # Deliberately not self.log: anything logged during test makes Lightning print
        # its own mean-only table next to the fuller one in on_test_epoch_end.
        self.test_metrics.append(metrics)
        self.io.saveMetricsToCsv(metrics, batch_idx)
        if batch_idx % FIGURE_EVERY == 0:
            self.io.plotDiagnostics(depth_batch.rgb_aif, init, predictions[-1], depth_batch.depth_gt,
                                  depth_batch.mask, metrics=metrics, tag="test", step=batch_idx,
                                  experiment=self.experiment())

    def on_load_checkpoint(self, checkpoint: dict) -> None:
        # The released checkpoints leave out the frozen DAv2, which __init__ already loads
        # from checkpoints/depth_anything_v2_<encoder>.pth. Fill those weights in here, so the
        # strict load still checks every other weight.
        state = checkpoint["state_dict"]
        for name, value in self.state_dict().items():
            if ".depth_anything.pretrained." in name or ".depth_anything.depth_head." in name:
                state.setdefault(name, value)

    def on_fit_start(self) -> None:
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        frozen = sum(p.numel() for p in self.parameters() if not p.requires_grad)
        self.print(f"Training {trainable:,} parameters; {frozen:,} DAv2 parameters are frozen.")
        if self.hparams.fossa_synthesis:
            # seed_everything gives every DDP rank the same seed, so all GPUs would draw the
            # same FOSSA cameras. Offset it by rank. The weights are already initialized.
            torch.manual_seed(torch.initial_seed() + self.global_rank)

    def on_test_start(self) -> None:
        self.test_metrics = []

    def on_test_epoch_end(self) -> None:
        if not self.test_metrics:
            return
        table = self.io.formatMetricsTable(self.test_metrics)
        self.print(f"\n{table}\n")
        experiment = self.experiment()
        if hasattr(experiment, "add_scalar"):
            for name in self.test_metrics[0]:
                experiment.add_scalar(f"test/{name}", np.mean([row[name] for row in self.test_metrics]))
            experiment.add_text("test/summary", f"<pre>{table}</pre>")

    def experiment(self):
        """The TensorBoard writer, or None when no logger can take figures/scalars."""
        return getattr(self.logger, "experiment", None)

    def on_before_optimizer_step(self, optimizer: torch.optim.Optimizer) -> None:
        if self.trainer.global_step % self.trainer.log_every_n_steps == 0:
            self.log("train/grad_norm", grad_norm(self, norm_type=2)["grad_2.0_norm_total"])

    def configure_optimizers(self):
        # DEFOM's schedule: AdamW, then a one-cycle LR with 1% warmup and a linear decay.
        optimizer = torch.optim.AdamW([p for p in self.parameters() if p.requires_grad],
                                      lr=self.hparams.lr, weight_decay=self.hparams.wd, eps=1e-8)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer, self.hparams.lr, total_steps=int(self.trainer.estimated_stepping_batches) + 100,
            pct_start=0.01, cycle_momentum=False, anneal_strategy="linear")
        return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "interval": "step"}}
