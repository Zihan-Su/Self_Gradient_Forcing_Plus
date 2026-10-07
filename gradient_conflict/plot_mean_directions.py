"""Estimate mean gradient directions from coordinate sketches."""
import numpy as np
from plot_utils import FAMILIES, exact_cosines, inputs, load, save, vectors
import paper_style


def main():
    a = inputs(__doc__).parse_args()
    rows, groups, metrics = load(a.data)
    metrics.update(method='Cosine of raw mean gradients in fixed coordinate sketches; no per-pair normalization', families={})
    for family in FAMILIES:
        cs, gs = [], []
        gram = np.zeros((len(rows), 3))
        for name in groups[family]:
            c, g = vectors(rows, name)
            cs.append(c.mean(axis=0)); gs.append(g.mean(axis=0))
            gram[:, 0] += (c*g).sum(axis=1)
            gram[:, 1] += (c*c).sum(axis=1)
            gram[:, 2] += (g*g).sum(axis=1)
        c, g = np.concatenate(cs), np.concatenate(gs)
        nc, ng = np.linalg.norm(c), np.linalg.norm(g)
        if nc <= 0 or ng <= 0:
            raise ValueError('Zero mean-gradient norm')
        cosine = float(np.clip(c@g/(nc*ng), -1, 1))
        angle = float(np.degrees(np.arccos(cosine)))
        ratio = float(ng/nc)
        end = np.array([ratio*cosine, ratio*np.sqrt(1-cosine*cosine)])
        sampled = gram[:, 0]/np.sqrt(gram[:, 1]*gram[:, 2])
        exact = exact_cosines(rows, groups[family])
        metrics['families'][family] = dict(cosine=cosine, angle_degrees=angle,
            norm_context=float(nc), norm_generation=float(ng), norm_ratio=ratio,
            context=[1., 0.], denoising=end.tolist(),
            paired_sketch_cosine_max_abs_error=float(np.abs(sampled-exact).max()),
            note='Per-pair sketch error does not bound error of mean-gradient geometry.')
    fig = paper_style.means(metrics)
    save(fig, a.out, 'b_mean_gradient_directions', metrics)


if __name__ == '__main__':
    main()
