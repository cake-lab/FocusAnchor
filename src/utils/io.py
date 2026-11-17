import csv
import os

import matplotlib.pyplot as plt
import numpy as np

# Inches per diagnostic panel; the figure height follows from the image aspect.
PANEL_WIDTH = 5.0


def toNumpy(tensor):
    return tensor.detach().cpu().numpy()


class IO:
    """Writes the per-batch metric CSV, the summary table and the test figures."""

    def __init__(self, csv_path, img_dir):
        self.csv_path = csv_path
        self.img_dir = img_dir
        self.header = None

    def saveMetricsToCsv(self, metrics, batch_idx=None):
        """Append one row of metrics. `metrics` is a name -> float dict."""
        row = {"batch": batch_idx, **metrics}

        # The stale file is cleared on the first write, not in the constructor:
        # building an IO you never write to must not delete anyone's results.
        if self.header is None:
            os.makedirs(os.path.dirname(self.csv_path), exist_ok=True)
            if os.path.isfile(self.csv_path):
                os.remove(self.csv_path)

        with open(self.csv_path, mode="a", newline="") as file:
            writer = csv.writer(file)
            if self.header is None:
                self.header = list(row)
                writer.writerow(self.header)
            writer.writerow([row[name] for name in self.header])

    @staticmethod
    def formatMetricsTable(metrics, title="Final Evaluation Results"):
        """Render mean/std/min/max per metric from a list of name -> float dicts."""
        names = list(metrics[0])
        values = np.array([[row[name] for name in names] for row in metrics], dtype=float)

        rows = [["Metric", "Mean", "Std", "Min", "Max"]]
        rows += [
            [name, f"{column.mean():.5f}", f"{column.std():.5f}",
             f"{column.min():.5f}", f"{column.max():.5f}"]
            for name, column in zip(names, values.T)
        ]

        widths = [max(len(row[col]) for row in rows) for col in range(len(rows[0]))]
        separator = "+-" + "-+-".join("-" * width for width in widths) + "-+"
        lines = [f"{title} (batches: {len(values)})", separator]
        for index, row in enumerate(rows):
            lines.append("| " + " | ".join(cell.ljust(widths[col]) for col, cell in enumerate(row)) + " |")
            if index == 0:
                lines.append(separator)
        lines.append(separator)

        return "\n".join(lines)

    def plotDiagnostics(self, rgb_aif, init, pred, gt, mask=None, metrics=None,
                        tag="test", step=0, experiment=None):
        """One row: RGB, the DAv2 start, the final prediction, GT, and the error of the
        start and of the final prediction on one shared range. The gap between the two
        error panels is what the focus volume and the GRU added.

        Goes to TensorBoard when `experiment` can take a figure, and falls back to a
        PNG on disk otherwise, so this still works with no logger configured."""
        def sample(tensor):
            # One sample per figure; a larger batch just shows its first image.
            return toNumpy(tensor[0]).squeeze()

        rgb = np.transpose(sample(rgb_aif), (1, 2, 0)).astype(np.float32).clip(0, 1)
        init, pred, gt = sample(init), sample(pred), sample(gt)
        # Invalid GT is stored as 0. Blanking it keeps the shared color range off zero.
        if mask is not None:
            gt = np.where(sample(mask).astype(bool), gt, np.nan)

        vmin = min(np.nanmin(gt), np.nanpercentile(pred, 1))
        vmax = max(np.nanmax(gt), np.nanpercentile(pred, 99))
        initError, error = init - gt, pred - gt
        # Clipped at the 99th percentile, so a few outlier pixels cannot flatten the rest.
        spread = float(np.nanpercentile(np.abs(np.stack([initError, error])), 99)) or 1e-6

        panels = [
            (rgb, "RGB (AiF)", None, None, None),
            (init, "DAv2 init", "plasma", vmin, vmax),
            (pred, "Predicted", "plasma", vmin, vmax),
            (gt, "Ground truth", "plasma", vmin, vmax),
            (initError, "Error of init", "RdBu_r", -spread, spread),
            (error, "Error of prediction", "RdBu_r", -spread, spread),
        ]
        title = f"{tag} step {step}"
        if metrics is not None:
            title += (f"    RMSE {metrics['rmse']:.3f}    AbsRel {metrics['absRel']:.3f}"
                      f"    logRMSE {metrics['logRmse']:.3f}    d1 {metrics['d1']:.4f}")

        # Size the figure from the image aspect so TensorBoard shows no grey padding.
        aspect = pred.shape[0] / pred.shape[1]
        fig, axes = plt.subplots(1, len(panels), layout="constrained",
                                 figsize=(PANEL_WIDTH * len(panels), PANEL_WIDTH * 0.91 * aspect + 0.5))
        fig.suptitle(title, fontsize=15)
        for ax, (image, name, cmap, low, high) in zip(axes, panels):
            ax.axis("off")
            ax.set_title(name)
            if cmap is None:
                ax.imshow(image)
                continue
            handle = ax.imshow(image, cmap=cmap, vmin=low, vmax=high)
            fig.colorbar(handle, ax=ax, fraction=0.046, pad=0.02)

        if hasattr(experiment, "add_figure"):
            experiment.add_figure(f"{tag}/diagnostics", fig, global_step=step)
        else:
            os.makedirs(self.img_dir, exist_ok=True)
            plt.savefig(f"{self.img_dir}/{tag}_{step}.png", bbox_inches="tight")
        plt.close(fig)
