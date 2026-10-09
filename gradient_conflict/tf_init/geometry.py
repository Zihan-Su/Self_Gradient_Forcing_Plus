import math

def geometry(a, b):
    norm2, ref_norm2, dot = map(float, ((a*a).sum(), (b*b).sum(), (a*b).sum()))
    return combined_geometry([dict(norm2=norm2, ref_norm2=ref_norm2, dot=dot)])


def combined_geometry(rows):
    result = {k: sum(r[k] for r in rows) for k in ['norm2', 'ref_norm2', 'dot']}
    a, b, dot = (result[k] for k in ['norm2', 'ref_norm2', 'dot'])
    result['cosine'] = max(-1., min(1., dot/math.sqrt(a*b))) if a > 0 and b > 0 else None
    return result


def split_weight_gradient(x, dy):
    assert x.shape[:-1] == dy.shape[:-1] and x.shape[1] % 2 == 0
    mid = x.shape[1] // 2
    return tuple(dy[:, start:stop].reshape(-1, dy.shape[-1]).float().T @
                 x[:, start:stop].reshape(-1, x.shape[-1]).float()
                 for start, stop in [(0, mid), (mid, x.shape[1])])
