from __future__ import annotations

import numpy as np


def cnapap_lpa2v(mu_in: float, mu_mem: float, fl: float) -> float:
    mu_in = float(np.clip(mu_in, 0.0, 1.0))
    mu_mem = float(np.clip(mu_mem, 0.0, 1.0))
    fl = float(np.clip(fl, 0.0, 1.0))
    lambda_ = 1.0 - mu_mem
    mu_next = ((mu_in - lambda_) * fl + 1.0) / 2.0
    return float(np.clip(mu_next, 0.0, 1.0))
