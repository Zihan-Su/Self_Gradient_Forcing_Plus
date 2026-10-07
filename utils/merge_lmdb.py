import os, shutil, lmdb, numpy as np
from tqdm import tqdm

BASE = "dataset"
BATCH = 512
MAP_MULT = 2.2

def read_shape(env, name):
    with env.begin() as txn:
        v = txn.get(f"{name}_shape".encode())
        if v is None:
            raise KeyError(f"missing key: {name}_shape")
        return tuple(map(int, v.decode().split()))

def list_array_names(env):
    out = []
    with env.begin() as txn:
        for k, _ in txn.cursor():
            if k.endswith(b"_shape"):
                out.append(k[:-6].decode())
    out = sorted(set(out))
    if not out:
        raise RuntimeError("no *_shape keys found")
    return out

def ensure_empty_dir(path):
    os.makedirs(path, exist_ok=True)
    if os.listdir(path):
        raise RuntimeError(f"dst_dir not empty: {path}")

def safe_mapsize(env):
    ms = env.info().get("map_size", 0)
    return ms if ms and ms > (1 << 30) else (1 << 30)

def get_bytes(txn, key):
    v = txn.get(key)
    if v is None:
        raise KeyError(f"missing key: {key!r}")
    return v

def latents_bytes_to_out(row_bytes, in_row_shape, out_row_shape):
    a = np.frombuffer(row_bytes, dtype=np.float16).reshape(in_row_shape)  # (S,F,C,H,W)
    if tuple(a.shape) != out_row_shape:
        raise RuntimeError(f"latents row shape mismatch: got {a.shape} vs expect {out_row_shape}")
    return np.ascontiguousarray(a).tobytes()

def merge_many(src_dirs_all, dst_dir):
    src_dirs = [d for d in tqdm(src_dirs_all, desc=f"scan -> {dst_dir}", unit="dir") if os.path.isdir(d)]
    if not src_dirs:
        print(f"nothing to merge for {dst_dir}")
        return []

    ensure_empty_dir(dst_dir)

    envs = [lmdb.open(s, readonly=True, lock=False, readahead=False, meminit=False) for s in src_dirs]
    try:
        names0 = list_array_names(envs[0])
        if "latents" not in names0:
            raise RuntimeError("missing 'latents' in *_shape keys")

        for e in envs[1:]:
            n = list_array_names(e)
            if n != names0:
                raise RuntimeError(f"array names mismatch:\n{src_dirs[0]}={names0}\n{e.path()}={n}")
        names = names0

        shapes, Ns, mapsum = [], [], 0

        # infer (S,F,C,H,W) from the first env's latents row shape
        sh0 = {n: read_shape(envs[0], n) for n in names}
        if len(sh0["latents"]) != 6:
            raise RuntimeError(f"expected latents shape (N,S,F,C,H,W), got {sh0['latents']}")
        out_lat_row = sh0["latents"][1:]  # (S,F,C,H,W)

        for e in tqdm(envs, desc=f"read meta -> {dst_dir}", unit="lmdb"):
            mapsum += safe_mapsize(e)
            sh = {n: read_shape(e, n) for n in names}
            N = sh[names[0]][0]
            for n in names:
                if sh[n][0] != N:
                    raise RuntimeError(f"inconsistent N in {e.path()} for '{n}': {sh[n][0]} vs {N}")

            # require exact row-shape match for all arrays (including latents)
            for n in names:
                if sh[n][1:] != sh0[n][1:]:
                    raise RuntimeError(f"shape mismatch for '{n}': {sh0[n]} vs {sh[n]}")

            if sh["latents"][1:] != out_lat_row:
                raise RuntimeError(f"latents row shape mismatch: expect {out_lat_row} got {sh['latents'][1:]}")

            shapes.append(sh)
            Ns.append(N)

        out = lmdb.open(dst_dir, map_size=int(mapsum * MAP_MULT), subdir=True, lock=True, readahead=False, meminit=False)

        def write_batch(src_txn, src_i0, src_i1, out_offset):
            while True:
                try:
                    with out.begin(write=True) as wtxn:
                        for i in range(src_i0, src_i1):
                            out_i = out_offset + i
                            for arr in names:
                                k = f"{arr}_{i}_data".encode()
                                b = get_bytes(src_txn, k)
                                if arr == "latents":
                                    b = latents_bytes_to_out(b, out_lat_row, out_lat_row)
                                wtxn.put(f"{arr}_{out_i}_data".encode(), b)
                    return
                except lmdb.MapFullError:
                    cur = out.info()["map_size"]
                    out.set_mapsize(int(cur * 1.5) + (1 << 30))

        totalN = sum(Ns)
        pbar = tqdm(total=totalN, desc=f"merge -> {dst_dir}", unit="row")
        offset = 0
        for idx, (e, N, src_path) in enumerate(zip(envs, Ns, src_dirs)):
            pbar.set_postfix_str(f"{idx+1}/{len(envs)} {os.path.basename(src_path)}", refresh=False)
            rtxn = e.begin()
            for s in range(0, N, BATCH):
                t = min(s + BATCH, N)
                write_batch(rtxn, s, t, offset)
                pbar.update(t - s)
            offset += N
        pbar.close()

        with out.begin(write=True) as wtxn:
            for arr in names:
                new_shape = (totalN, *sh0[arr][1:])
                wtxn.put(f"{arr}_shape".encode(), (" ".join(map(str, new_shape))).encode())

        out.sync()
        out.close()
    finally:
        for e in envs:
            try:
                e.close()
            except:
                pass

    return src_dirs

def rm_dirs(dirs, desc="remove dirs"):
    for d in tqdm(dirs, desc=desc, unit="dir"):
        if os.path.exists(d):
            shutil.rmtree(d)
            
print('Begin merging ...')
# -------- framewise --------
print('Begin merging framewise data...')
fw_src_all = [os.path.join(BASE, f"ODE6KCausal_framewise_{i}") for i in range(15)]
fw_dst = os.path.join(BASE, "ODE6KCausal_framewise")
fw_merged = merge_many(fw_src_all, fw_dst)
rm_dirs(fw_merged, desc="remove framewise shards")

# -------- chunkwise --------
print('Begin merging framewise data...')
cw_src_all = [os.path.join(BASE, f"ODE6KCausal_chunkwise_{i}") for i in range(15)]
cw_dst = os.path.join(BASE, "ODE6KCausal_chunkwise")
cw_merged = merge_many(cw_src_all, cw_dst)
rm_dirs(cw_merged, desc="remove chunkwise shards")

print("done")
