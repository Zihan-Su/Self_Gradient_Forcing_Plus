"""Exact paired cosine distributions by nominal denoising timestep."""
from plot_utils import FAMILIES, NOMINAL, exact_cosines, inputs, load, save
import paper_style


def main():
    a = inputs(__doc__).parse_args()
    rows, groups, metrics = load(a.data)
    metrics.update(method='Exact cosine of concatenated family gradients for each prompt and exit', families={})
    data = {}
    for family in FAMILIES:
        values = exact_cosines(rows, groups[family])
        data[family] = values.reshape(-1, 4)
        metrics['families'][family] = {}
        for t in range(4):
            v = data[family][:, t]
            metrics['families'][family][str(NOMINAL[t])] = dict(mean_cosine=float(v.mean()),
                negative_fraction=float((v<0).mean()), cosines=v.tolist())
    fig = paper_style.paired(data)
    save(fig, a.out, 'c_paired_gradient_cosine', metrics)


if __name__ == '__main__':
    main()
