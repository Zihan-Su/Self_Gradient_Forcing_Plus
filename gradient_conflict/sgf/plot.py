import argparse
import json
from pathlib import Path
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import paper_style
from sklearn.manifold import TSNE
from exact_tsne import fit_exact

FAMILIES = ('attention', 'ffn')
NOMINAL = (1000, 750, 500, 250)


def load(data):
    torch.set_num_threads(4)
    paths = sorted(data.glob('p*_t?.pt'))
    rows = [torch.load(p, map_location='cpu', weights_only=True) for p in paths]
    if not rows:
        raise ValueError('No gradient records found')
    rows.sort(key=lambda r: (r['prompt_index'], r['exit_index']))
    ids = sorted({r['prompt_index'] for r in rows})
    observed = {(r['prompt_index'], r['exit_index']) for r in rows}
    if len(rows) != len(observed) or observed != {(i, t) for i in ids for t in range(4)}:
        raise ValueError('Expected unique, complete prompt x four-exit grid')
    groups = {f: sorted(n for n in rows[0]['module_stats']
                        if ('.self_attn.' if f == 'attention' else '.ffn.') in n) for f in FAMILIES}
    assert len(groups['attention']) == 120 and len(groups['ffn']) == 60
    names = set(groups['attention'] + groups['ffn'])
    for r in rows:
        if set(r['module_stats']) != names:
            raise ValueError('Inconsistent module set')
    protocols = [json.loads((data/f'prompt_{i:03d}_protocol.json').read_text()) for i in ids]
    first = protocols[0]
    for protocol in protocols:
        for key in ('modules', 'checkpoint', 'checkpoint_step', 'checkpoint_key', 'config'):
            if protocol[key] != first[key]:
                raise ValueError(f'Inconsistent acquisition protocol: {key}')
    if first['checkpoint_key'] != 'generator':
        raise ValueError('Expected raw generator diagnostic protocol')
    sketch_sizes = sorted({int(v['sketch_context'].numel()) for r in rows for v in r['module_stats'].values()})
    plt.rcParams.update({'font.family': 'DejaVu Serif', 'mathtext.fontset': 'dejavuserif', 'text.color': '#222222', 'font.size': 10,
                         'axes.titlesize': 13, 'axes.spines.top': False,
                         'axes.spines.right': False, 'svg.fonttype': 'none', 'pdf.fonttype': 42})
    return rows, groups, dict(prompt_indices=ids, prompts=len(ids), pairs=len(rows),
                              sketch_coordinates_per_matrix=sketch_sizes)


def vectors(rows, name):
    stats = [r['module_stats'][name] for r in rows]
    return (torch.stack([s['sketch_context'] for s in stats]).double().numpy(),
            torch.stack([s['sketch_generation'] for s in stats]).double().numpy())


def exact_cosines(rows, names):
    result = []
    for r in rows:
        v = [r['module_stats'][n] for n in names]
        dot = sum(s['dot'] for s in v)
        norm = np.sqrt(sum(s['norm2_context'] for s in v) * sum(s['norm2_generation'] for s in v))
        if norm <= 0:
            raise ValueError('Grouped gradient has zero norm')
        result.append(dot/norm)
    values = np.clip(result, -1, 1)
    if not np.isfinite(values).all():
        raise ValueError('Non-finite cosine')
    return values


def save(fig, out, stem, metrics):
    out.mkdir(parents=True, exist_ok=True)
    for ext in ('png', 'pdf', 'svg'):
        fig.savefig(out/f'{stem}.{ext}', dpi=200, facecolor='white')
    (out/f'{stem}.json').write_text(json.dumps(metrics, indent=2)+'\n')
    plt.close(fig)
    print('SAVED', stem, flush=True)


def distributions(a, rows, groups, metrics):
    count = len(rows)
    if not 0 < a.perplexity < 2*count:
        raise ValueError('Perplexity must be positive and smaller than point count')
    metrics.update(method='t-SNE on sketch-estimated normalized angular distance',
                   perplexity=a.perplexity, seed=a.seed, families={})
    metadata = json.loads((a.data/f"prompt_{rows[0]['prompt_index']:03d}_protocol.json").read_text())['modules']
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    embeddings = {}
    for family in FAMILIES:
        gram = np.zeros((2*count, 2*count), dtype=np.float64)
        names = sorted(groups[family], key=lambda n: (metadata[n]['layer'], metadata[n]['family'], metadata[n]['projection']))
        for name in names:
            stats = [r['module_stats'][name] for r in rows]
            v = torch.stack([s['sketch_context'] for s in stats] +
                            [s['sketch_generation'] for s in stats]).to(device)
            gram += (v @ v.T).double().cpu().numpy()
        norm = np.sqrt(np.diag(gram))
        if (norm <= 0).any():
            raise ValueError('Zero grouped sketch norm')
        distance = np.arccos(np.clip(gram/np.outer(norm, norm), -1, 1))/np.pi
        distance = (distance + distance.T)/2
        np.fill_diagonal(distance, 0)
        model = TSNE(n_components=2, perplexity=a.perplexity, metric='precomputed',
                     init='random', learning_rate='auto', method='exact',
                     max_iter=2000, random_state=a.seed)
        xy = fit_exact(model, distance)
        if xy[:count, 0].mean() > xy[count:, 0].mean():
            xy[:, 0] *= -1
        embeddings[family] = xy
        metrics['families'][family] = {'kl_divergence': float(model.kl_divergence_),
                                     'points_per_role': count}
    fig = paper_style.distributions(embeddings, count)
    save(fig, a.out, 'a_gradient_distributions', metrics)
    np.savez_compressed(a.out/'a_gradient_distributions_embeddings.npz', **embeddings)


def means(a, rows, groups, metrics):
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


def paired(a, rows, groups, metrics):
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


def main():
    p = argparse.ArgumentParser(description='Plot gradient conflict figures')
    p.add_argument('--data', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--perplexity', type=float, default=20)
    p.add_argument('--seed', type=int, default=23092026)
    a = p.parse_args()
    rows, groups, metrics = load(a.data)
    for plot in (distributions, means, paired):
        with plt.rc_context():
            plot(a, rows, groups, metrics.copy())


if __name__ == '__main__':
    main()
