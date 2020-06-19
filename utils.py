import torch
import numpy as np
from train import cuda

class Meter(object):
    """Computes and stores the min, max, avg, and current values"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        self.max = -float("inf")
        self.min = float("inf")

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        self.max = max(self.max, val)
        self.min = min(self.min, val)


class SplittingMethodStats(object):
    def __init__(self):
        self.fwd_iters = Meter()
        self.bkwd_iters = Meter()
        self.fwd_time = Meter()
        self.bkwd_time = Meter()

    def reset(self):
        self.fwd_iters.reset()
        self.fwd_time.reset()
        self.bkwd_iters.reset()
        self.bkwd_time.reset()

    def report(self):
        print('Fwd iters: {:.2f}\tFwd Time: {:.4f}\tBkwd Iters: {:.2f}\tBkwd Time: {:.4f}\n'.format(
                self.fwd_iters.avg, self.fwd_time.avg,
                self.bkwd_iters.avg, self.bkwd_time.avg))

def compute_eigval(lin_module, method="power", compute_smallest=False, largest=None):
    with torch.no_grad():
        if method == "direct":
            W = lin_module.W.weight
            eigvals = torch.symeig(W + W.T)[0]
            return eigvals.detach().cpu().numpy()[-1] / 2

        elif method == "power":
            z0 = tuple(torch.randn(*shp).to(lin_module.U.weight.device) for shp in lin_module.z_shape(1))
            lam = power_iteration(lin_module, z0, 100,
                                  compute_smallest=compute_smallest,
                                  largest=largest)
            return lam

def power_iteration(linear_module, z, T,  compute_smallest=False, largest=None):
    n = len(z)
    for i in range(T):
        za = linear_module.multiply(*z)
        zb = linear_module.multiply_transpose(*z)
        if compute_smallest:
            zn = tuple(-2*largest*a + 0.5*b + 0.5*c for a,b,c in zip(z, za, zb))
        else:
            zn = tuple(0.5*a + 0.5*b for a,b in zip(za, zb))
        x = sum((zn[i]*z[i]).sum().item() for i in range(n))
        y = sum((z[i]*z[i]).sum().item() for i in range(n))
        lam = x/y
        z = tuple(zn[i]/np.sqrt(y) for i in range(n))
    return lam +2*largest if compute_smallest else lam

def get_splitting_stats(dataLoader, model):
    model = cuda(model)
    model.train()
    model.mon.save_abs_err = True
    for batch in dataLoader:
        data, target = cuda(batch[0]), cuda(batch[1])
        model(data)
        return model.mon.errs