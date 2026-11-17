"""ZEDD test submission for FOSSA's server (zedd.cs.princeton.edu/submissions/new).

Runs a FocusAnchor checkpoint on the full frame at 480x720, the same scale as the 480x640 val crop,
and upsamples the depth in metres to the server's 1216x1824. Running at native 1216x1824
makes the blur ~2.5x larger in pixels than in training: on ZEDD val the volume goes from
AbsRel 0.143 to 0.195 and the output from 0.073 to 0.094. File
zedd_output_{i+1:04d}.npy is the i-th sorted scene, as in FOSSA. The zip is written the way FOSSA's zedd_test/zedd_test.py writes it. With
--split val nothing is saved. It scores against the full-res GT instead, the way the
server scores test.

PYTHONPATH=src uv run python src/dataloader/ZEDD/submit.py CKPT OUT_DIR [--root ZEDD_ROOT] [--split val]
uv run python src/dataloader/ZEDD/check_format.py --zip OUT_DIR/zedd_outputs.zip
"""
import argparse
import zipfile
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from pytorch_lightning.utilities import move_data_to_device

from dataloader.batch import prepareBatch
from dataloader.ZEDD.loader import MAX_DEPTH, MIN_DEPTH, ZEDDLoader
from model.focusAnchor import FocusAnchorModule
from utils.metric_cal import calcMetrics

parser = argparse.ArgumentParser()
parser.add_argument("ckpt")
parser.add_argument("out", help="folder for zedd_outputs.zip")
parser.add_argument("--root", default="/project/zedd")
parser.add_argument("--split", default="test", choices=["test", "val"])
parser.add_argument("--aif", default="fused", choices=["fused", "f16"])
args = parser.parse_args()

# The checkpoint's own hparams build the model: backbone, prior fusion, 16 iterations.
module = FocusAnchorModule.load_from_checkpoint(args.ckpt, map_location="cuda").eval()
data = ZEDDLoader(args.root, aif=args.aif, split=args.split, crop=False)
out = Path(args.out)
out.mkdir(parents=True, exist_ok=True)
files, rows = [], []
with torch.no_grad():
    for i, scene in enumerate(data.scenes):
        b = prepareBatch(move_data_to_device({k: v[None] for k, v in data[i].items()}, "cuda"))
        pred = module(b.rgb_aif, b.focal_stack, b.focus_dist)[0][-1].float()
        pred = F.interpolate(pred, (1216, 1824), mode="bilinear", align_corners=True)[0, 0].cpu().numpy()
        if args.split == "test":
            files.append(out / f"zedd_output_{i + 1:04d}.npy")
            np.save(files[-1], pred)
        else:
            gt = np.load(scene / "gt" / "depth.npy").astype(np.float32)
            mask = np.isfinite(gt) & (gt >= MIN_DEPTH) & (gt <= MAX_DEPTH)
            rows.append(calcMetrics(pred[None], np.where(mask, gt, 0)[None], mask[None]))

if args.split == "test":
    # As in FOSSA: deflate, flat names, then delete the .npy files.
    with zipfile.ZipFile(out / "zedd_outputs.zip", "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for f in files:
            zf.write(f, arcname=f.name)
    for f in files:
        f.unlink()
    print(f"Zipped {len(files)} predictions to {out / 'zedd_outputs.zip'}")
else:
    m = {k: float(np.mean([r[k] for r in rows])) for k in rows[0]}
    print(f"{args.ckpt} val, model at 480x720, scored at 1216x1824: " + " ".join(f"{k} {m[k]:.4f}" for k in ["d105", "d115", "d1", "absRel", "rmse"]))
