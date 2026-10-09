import torch
from scipy.spatial.distance import squareform
from sklearn.manifold import _t_sne

EPS = _t_sne.MACHINE_EPSILON

class GPUExactObjective:
    def __init__(self):
        self.p_ref = None

    def __call__(self, params, P, degrees_of_freedom, n_samples, n_components,
                 skip_num_points=0, compute_error=True):
        assert degrees_of_freedom == 1 and n_components == 2
        if self.p_ref is not P:
            self.p_gpu = torch.from_numpy(squareform(P)).cuda()
            self.p_ref = P
            self.p_first = float(P[0])
            assert self.p_first > 0
        scale = float(P[0]) / self.p_first
        y = torch.from_numpy(params.reshape(n_samples, 2)).cuda()
        yd = y.double()
        kernel = (yd[:, None, 0] - yd[None, :, 0]).square()
        kernel.add_((yd[:, None, 1] - yd[None, :, 1]).square())
        kernel.add_(1).reciprocal_()
        kernel.fill_diagonal_(0)
        q = (kernel / kernel.sum()).clamp_min_(EPS)
        p = self.p_gpu * scale
        error = float((p * (p.clamp_min(EPS) / q).log()).sum()) if compute_error else float('nan')
        weights = (p - q) * kernel
        grad = torch.empty_like(y)
        for d in range(2):
            delta = y[:, None, d] - y[None, :, d]
            grad[:, d] = (weights * delta).sum(dim=1)
        grad[:skip_num_points] = 0
        grad *= 4
        return error, grad.flatten().cpu().numpy()


def fit_exact(model, distance):
    if not torch.cuda.is_available():
        return model.fit_transform(distance)
    assert model.method == 'exact' and model.n_components == 2
    reference = _t_sne._kl_divergence
    _t_sne._kl_divergence = GPUExactObjective()
    try:
        return model.fit_transform(distance)
    finally:
        _t_sne._kl_divergence = reference
