import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch


def summarize(rows):
    result = {}
    for family in ('attention', 'ffn'):
        result[family] = {}
        for exit_index in range(4):
            values = np.array([r['cosine'] for r in rows
                               if r['family'] == family and r['exit_index'] == exit_index])
            if not len(values):
                continue
            result[family][str(exit_index)] = {
                'n': len(values), 'mean_cosine': float(values.mean()),
                'negative_fraction': float((values < 0).mean()),
                'mean_angular_distance': float((np.arccos(np.clip(values, -1, 1)) / np.pi).mean())}
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    rows, max_error, pairs = [], 0., set()
    for path in sorted(a.data.glob('p*_t?.pt')):
        row = torch.load(path, map_location='cpu', weights_only=True)
        pair = (row['prompt_index'], row['exit_index'])
        if pair in pairs:
            raise ValueError(f'Duplicate pair {pair}')
        pairs.add(pair)
        stats = row['module_stats']
        if len(stats) != 180:
            raise ValueError(f'Unexpected protocol in {path}')
        max_error = max(max_error, max(v['reconstruction_relative_error'] for v in stats.values()))
        for family, marker in (('attention', '.self_attn.'), ('ffn', '.ffn.')):
            group = [v for k, v in stats.items() if marker in k]
            dot = sum(v['dot'] for v in group)
            norm = np.sqrt(sum(v['norm2_context'] for v in group) * sum(v['norm2_generation'] for v in group))
            rows.append(dict(prompt=pair[0], exit_index=pair[1], family=family,
                             cosine=float(np.clip(dot / max(norm, 1e-30), -1, 1))))
    if not rows or not all(np.isfinite(r['cosine']) for r in rows):
        raise ValueError('No finite gradient results')
    prompt_ids = sorted({p for p, _ in pairs})
    if pairs != {(p, t) for p in prompt_ids for t in range(4)}:
        raise ValueError('Each sampled prompt must contain all four exits')
    result = dict(prompts=prompt_ids, pairs=len(pairs), max_reconstruction_relative_error=max_error,
                  summary=summarize(rows))
    a.out.mkdir(parents=True, exist_ok=True)
    with (a.out/'values.csv').open('w') as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys());w.writeheader();w.writerows(rows)
    (a.out/'summary.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
