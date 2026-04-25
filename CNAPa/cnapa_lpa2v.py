from __future__ import annotations


def clamp01(x: float) -> float:
    return max(0.0, min(1.0, float(x)))


def cnapa_lpa2v(mu: float, lambda_: float, ftc: float, ftct: float) -> dict:
    mu = clamp01(mu)
    lambda_ = clamp01(lambda_)
    ftc = clamp01(ftc)
    ftct = clamp01(ftct)

    gc = mu - lambda_
    gct = mu + lambda_ - 1.0

    mu_e = (gc + 1.0) / 2.0
    mu_ctr = (gct + 1.0) / 2.0
    phi_e = 1.0 - abs(gct)

    vcve = (1.0 + ftc) / 2.0
    vcfa = (1.0 - ftc) / 2.0
    vcic = (1.0 + ftct) / 2.0
    vcpa = (1.0 - ftct) / 2.0

    cond1 = (vcic > mu_ctr) and (mu_ctr > vcpa)
    cond2 = (vcve <= mu_e) or (mu_e <= vcfa)

    if cond1 and cond2:
        s1 = mu_e
    else:
        s1 = 0.5

    s2 = phi_e
    return {
        'mu': mu,
        'lambda': lambda_,
        'Gc': gc,
        'Gct': gct,
        'muE': mu_e,
        'muCtr': mu_ctr,
        'phiE': phi_e,
        'Vcve': vcve,
        'Vcfa': vcfa,
        'Vcic': vcic,
        'Vcpa': vcpa,
        'S1': s1,
        'S2': s2,
    }
