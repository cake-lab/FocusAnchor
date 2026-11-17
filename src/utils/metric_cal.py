import numpy as np


# d105 and d115 (ratio < 1.05, < 1.15) are FOSSA's tighter thresholds, used on ZEDD.
METRIC_NAMES = ("mse", "rmse", "logRmse", "absRel", "sqRel", "d105", "d115", "d1", "d2", "d3")

# Predictions are clamped to this before entering a log, so a single bad pixel
# cannot turn a whole run's LogRMSE into nan.
MIN_DEPTH = 1e-6


def calcMetrics(pred, target, mask):
    """Depth metrics computed per image, then averaged over the batch.

    `pred`, `target` and `mask` are numpy arrays whose first axis is the batch.
    A pixel counts only where the mask is set and the ground truth is positive;
    images with no valid pixel at all are skipped. Returns a name -> float dict,
    all-nan if the whole batch was invalid.
    """
    pred = pred.reshape(len(pred), -1)
    target = target.reshape(len(target), -1)
    mask = mask.reshape(len(mask), -1).astype(bool)

    perImage = []
    for predImg, targetImg, maskImg in zip(pred, target, mask):
        valid = maskImg & (targetImg > 0)
        if not valid.any():
            continue

        # Clamp rather than drop: a non-positive prediction is a real error and
        # should still be counted everywhere except inside the logs.
        predImg = np.clip(predImg[valid], MIN_DEPTH, None)
        targetImg = targetImg[valid]

        error = targetImg - predImg
        ratio = np.maximum(targetImg / predImg, predImg / targetImg)
        mse = np.mean(error ** 2)

        perImage.append((
            mse,
            np.sqrt(mse),
            np.sqrt(np.mean((np.log(targetImg) - np.log(predImg)) ** 2)),
            np.mean(np.abs(error) / targetImg),
            np.mean(error ** 2 / targetImg),
            (ratio < 1.05).mean(),
            (ratio < 1.15).mean(),
            (ratio < 1.25).mean(),
            (ratio < 1.25 ** 2).mean(),
            (ratio < 1.25 ** 3).mean(),
        ))

    if not perImage:
        return {name: float("nan") for name in METRIC_NAMES}

    return dict(zip(METRIC_NAMES, np.mean(perImage, axis=0).tolist()))
