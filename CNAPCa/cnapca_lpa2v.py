from __future__ import annotations

import numpy as np


def cnapca_lpa2v(mu: float, lambda_: float, ftce: float, ftct: float) -> dict:
    mu = float(np.clip(mu, 0.0, 1.0))
    lambda_ = float(np.clip(lambda_, 0.0, 1.0))
    ftce = float(np.clip(ftce, 0.0, 1.0))
    ftct = float(np.clip(ftct, 0.0, 1.0))

    lambdac = 1.0 - lambda_
    gce = mu - lambdac
    gun = mu + lambdac - 1.0
    mur = (gce + 1.0) / 2.0

    if abs(gce) > ftce:
        s1 = mur
        s2 = 0.0
    else:
        if (abs(gun) > ftct) and (abs(gun) > abs(gce)):
            s1 = mur
            s2 = abs(gun)
        else:
            s1 = 0.5
            s2 = 0.0

    return {
        'mu': mu,
        'lambda': lambda_,
        'lambdac': lambdac,
        'Gce': gce,
        'Gun': gun,
        'mur': mur,
        'S1': s1,
        'S2': s2,
    }
