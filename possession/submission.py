"""Write the submission CSV: one row per video frame, H / A / D on graded frames, X elsewhere."""
import numpy as np
import pandas as pd

from .data import LABEL_NAME, UNLABELLED


def write_csv(labels, graded, path):
    assert not np.any(labels[graded] == UNLABELLED), 'every graded frame needs H, A or D'
    names = np.where(graded, pd.Series(labels).map(LABEL_NAME).to_numpy(), 'X')
    pd.DataFrame({'frame': np.arange(len(labels)), 'label': names}).to_csv(path, index=False)
