import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

import lmdb
import numpy as np


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--dataset', type=Path, required=True, help='Directory containing ODE6KCausal_chunkwise_0..14')
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    dataset, out = a.dataset.resolve(), a.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    shards, index = [], []
    for k in range(15):
        path = dataset / f'ODE6KCausal_chunkwise_{k}'
        env = lmdb.open(str(path), readonly=True, lock=False, readahead=False, meminit=False)
        with env.begin() as txn:
            shape = tuple(map(int, txn.get(b'latents_shape').decode().split()))
        shards.append((path, env, shape))
        index.extend((k, j) for j in range(shape[0]))
    assert len(index) == 6505, 'This protocol targets the original 6505-record release'
    ids = sorted(np.random.default_rng(20261008).choice(len(index), 128, replace=False).tolist())
    rows = []
    for i, global_id in enumerate(ids):
        k, j = index[global_id]
        path, env, shape = shards[k]
        with env.begin() as txn:
            prompt = txn.get(f'prompts_{j}_data'.encode()).decode('utf-8')
        rows.append(dict(index=i, global_id=global_id, shard=path.name, local_id=j,
                         source_shape=list(shape[1:]), clean_endpoint=-1, prompt=prompt))
    manifest = out / 'manifest.json'
    if manifest.exists():
        previous = json.loads(manifest.read_text())
        assert [{k:r[k] for k in rows[0]} for r in previous] == rows, 'Existing selection differs'
    import torch
    root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(root))
    os.chdir(root)
    from utils.wan_wrapper import WanTextEncoder
    torch.set_num_threads(3)
    encoder = None
    (out / 'cache').mkdir(exist_ok=True)
    for row in rows:
        k, _ = index[row['global_id']]
        _, env, shape = shards[k]
        with env.begin() as txn:
            raw = txn.get(f"latents_{row['local_id']}_data".encode())
        clean = np.ascontiguousarray(np.frombuffer(raw, dtype=np.float16).reshape(shape[1:])[-1:])
        assert clean.shape == (1,21,16,60,104) and np.isfinite(clean).all()
        digest = hashlib.sha256(clean.tobytes()).hexdigest()
        dest = out / 'cache' / f"{row['index']:02d}.pt"
        if dest.exists():
            cache = torch.load(dest, map_location='cpu', weights_only=False)
            assert cache['metadata']['prompt'] == row['prompt']
            assert hashlib.sha256(cache['latent'].half().numpy().tobytes()).hexdigest() == digest
            assert torch.isfinite(cache['prompt_embeds']).all()
            del cache
        else:
            if encoder is None:
                encoder = WanTextEncoder().to(device='cuda', dtype=torch.bfloat16)
            with torch.no_grad():
                condition = encoder([row['prompt']])['prompt_embeds'].cpu()
            assert torch.isfinite(condition).all()
            temp = dest.with_suffix('.tmp')
            torch.save(dict(latent=torch.from_numpy(clean.copy()).float(),
                            prompt_embeds=condition, metadata=row), temp)
            temp.replace(dest)
        row['clean_fp16_sha256'] = digest
        print('CACHED', row['index'], row['global_id'], flush=True)
    manifest.write_text(json.dumps(rows, ensure_ascii=False, indent=2))
    for _, env, _ in shards:
        env.close()
    print('Selected 128 unique records from 6505; seed 20261008', flush=True)


if __name__ == '__main__':
    main()
