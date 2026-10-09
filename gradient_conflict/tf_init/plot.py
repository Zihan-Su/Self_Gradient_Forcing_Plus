import argparse
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import paper_style

FAMILIES = ('attention', 'ffn')


def load(data):
    cache = data / '.analysis_cache'
    summary = json.loads((cache / 'summary.json').read_text())
    with np.load(cache / 'analysis.npz') as saved:
        arrays = {k: saved[k] for k in saved.files}
    assert summary['tsne']['method'] == 'exact'
    assert arrays['paired_cosine'].shape == (2,128,50)
    assert all(np.isfinite(v).all() for v in arrays.values())
    assert all(arrays[f+'_embedding'].shape == (12800,2) for f in FAMILIES)
    plt.rcParams.update({'font.family':'DejaVu Serif', 'mathtext.fontset':'dejavuserif',
                         'text.color':'#222222', 'font.size':7, 'pdf.fonttype':42,
                         'svg.fonttype':'path'})
    metrics = dict(prompt_indices=list(range(128)), prompts=128, pairs=6400,
                   sketch_coordinates_per_matrix=[4096])
    return arrays, summary, metrics


def save(fig, out, stem, metrics):
    out.mkdir(parents=True, exist_ok=True)
    for ext in ('png', 'pdf', 'svg'):
        fig.savefig(out/f'{stem}.{ext}', dpi=200, facecolor='white')
    (out/f'{stem}.json').write_text(json.dumps(metrics, indent=2)+'\n')
    plt.close(fig)
    print('SAVED', stem, flush=True)


def distributions(a, arrays, summary, metrics):
    settings = summary['tsne']
    metrics.update(method='t-SNE on sketch-estimated normalized angular distance',
                   perplexity=settings['perplexity'], seed=settings['seed'], families={})
    embeddings = {f: arrays[f+'_embedding'] for f in FAMILIES}
    for family in FAMILIES:
        metrics['families'][family] = dict(kl_divergence=summary['families'][family]['tsne_kl'],
                                            points_per_role=6400)
    fig = paper_style.distributions(embeddings, arrays['timesteps'])
    save(fig, a.out, 'a_gradient_distributions', metrics)
    np.savez_compressed(a.out/'a_gradient_distributions_embeddings.npz', **embeddings)


def means(a, arrays, summary, metrics):
    metrics.update(method='Cosine of raw mean gradients in fixed coordinate sketches; no per-pair normalization', families={})
    for family in FAMILIES:
        info = summary['families'][family]
        angle, ratio = info['angle_degrees'], info['norm_ratio']
        theta = np.deg2rad(angle)
        metrics['families'][family] = dict(cosine=info['cosine'], angle_degrees=angle,
            norm_context=info['norm_context'], norm_generation=info['norm_denoising'],
            norm_ratio=ratio, context=[1.,0.], denoising=[ratio*np.cos(theta),ratio*np.sin(theta)],
            paired_sketch_cosine_max_abs_error=info['paired_sketch_cosine_max_abs_error'],
            note='Per-pair sketch error does not bound error of mean-gradient geometry.')
    fig = paper_style.means(metrics)
    save(fig, a.out, 'b_mean_gradient_directions', metrics)


def paired(a, arrays, summary, metrics):
    metrics.update(method='Exact cosine of concatenated family gradients for each prompt and exit', families={})
    data = {}
    for i, family in enumerate(FAMILIES):
        values = arrays['paired_cosine'][i]
        data[family] = values
        metrics['families'][family] = {}
        for j, timestep in enumerate(arrays['timesteps']):
            v = values[:,j]
            metrics['families'][family][str(int(timestep))] = dict(mean_cosine=float(v.mean()),
                negative_fraction=float((v<0).mean()), cosines=v.tolist())
    fig = paper_style.paired(data, arrays['timesteps'])
    save(fig, a.out, 'c_paired_gradient_cosine', metrics)


def main():
    p = argparse.ArgumentParser(description='Plot gradient conflict figures')
    p.add_argument('--data', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    arrays, summary, metrics = load(a.data)
    for plot in (distributions, means, paired):
        with plt.rc_context():
            plot(a, arrays, summary, metrics.copy())


if __name__ == '__main__':
    main()
