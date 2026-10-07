"""Load and validate gradient data for plotting."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

FAMILIES = ('attention', 'ffn')
NOMINAL = (1000, 750, 500, 250)


def inputs(description):
    p = argparse.ArgumentParser(description=description)
    p.add_argument('--data', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    return p


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
