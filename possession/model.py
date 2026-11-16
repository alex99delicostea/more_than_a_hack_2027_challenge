"""Time-block split, LightGBM training and scoring for 2_lightgbm.ipynb."""
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

from .data import ATHLETIC, BARCA, DEAD, FPS, HALVES, N_FRAMES, SECOND_HALF_KICKOFF, rolling_mean

BLOCK = 120 * FPS        # validation is cut in 2-minute blocks ...
VALIDATION_EVERY = 5     # ... every 5th block (20%)
PURGE = 5 * FPS          # training skips frames this close to a validation block (5 s averages)


def validation_blocks():
    """True on validation frames: every VALIDATION_EVERY-th 2-minute block of each half from kickoff."""
    is_val = np.zeros(N_FRAMES, bool)
    for half, first, last in HALVES:
        start = SECOND_HALF_KICKOFF if half == 2 else first
        block = (np.arange(start, last + 1) - start) // BLOCK
        is_val[start:last + 1] = block % VALIDATION_EVERY == 2
    return is_val


def near(mask, frames):
    """True within `frames` of any True value."""
    return rolling_mean(mask.astype(float), 2 * frames + 1) > 0


def train(X, labels, train_mask):
    model = lgb.LGBMClassifier(n_estimators=300, learning_rate=0.05, num_leaves=63, min_child_samples=200,
                               subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
                               random_state=0, verbose=-1,
                               # same model on any machine: fixed summation order, fixed strategy, one thread
                               deterministic=True, force_row_wise=True, n_jobs=1)
    rows = np.flatnonzero(train_mask)[::2]  # neighbouring frames are near-copies: every 2nd is enough
    model.fit(X.iloc[rows], labels[rows])
    return model


def smooth(probabilities, frames):
    """Average the class probabilities over `frames` frames, then take the most likely class."""
    return pd.DataFrame(probabilities).rolling(frames, center=True, min_periods=1).mean() \
        .to_numpy().argmax(axis=1).astype(np.int8)


def scores(truth, prediction):
    """Frame scores. 'H vs A' = share right on frames the truth calls H or A."""
    on_ball = truth != DEAD
    f1 = f1_score(truth, prediction, labels=[BARCA, ATHLETIC, DEAD], average=None)
    return {'accuracy': np.mean(truth == prediction), 'F1 H': f1[0], 'F1 A': f1[1], 'F1 D': f1[2],
            'macro F1': f1.mean(), 'H vs A': np.mean(truth[on_ball] == prediction[on_ball])}


def carry_forward(labels, labelled, blocks):
    """Baseline: every block gets the last label seen before it (default D)."""
    guess = np.full(len(labels), DEAD, np.int8)
    for a, b in blocks:
        before = np.flatnonzero(labelled[:a])
        if len(before):
            guess[a:b] = labels[before[-1]]
    return guess


def fill_gaps(labels, blocks):
    """Inside each block, unlabelled frames take the previous label (the first ones the next label)."""
    filled = labels.copy()
    for a, b in blocks:
        seg = pd.Series(np.where(labels[a:b] >= 0, labels[a:b], np.nan)).ffill().bfill()
        filled[a:b] = seg.fillna(DEAD).to_numpy().astype(np.int8)
    return filled
