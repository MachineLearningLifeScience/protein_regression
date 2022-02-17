import numpy as np

def prep_for_logdict(y, mu, unc, err2, baseline):
    if type(mu)==np.ndarray:
        trues = list(np.hstack(y))
        mus = list(np.hstack(mu))
        uncs = list(np.hstack(unc))
        errs = list(np.hstack(err2/baseline))
    else:
        trues = list(np.hstack(y))
        mus = list(np.hstack(mu.cpu().numpy()))
        uncs = list(np.hstack(unc.cpu().numpy()))
        errs = list(np.hstack(err2/baseline))
    return trues, mus, uncs, errs