import os, shutil, lmdb, numpy as np
from tqdm import tqdm

BASE = "dataset"
BATCH = 512
MAP_MULT = 2.2

def read_shape(env, name):
    with env.begin() as txn:
        v = txn.get(f"{name}_shape".encode())
        if v is None: raise KeyError(f"missing key: {name}_shape")
        return tuple(map(int, v.decode().split()))

def list_array_names(env):
    out = []
    with env.begin() as txn:
        for k, _ in txn.cursor():
            if k.endswith(b"_shape"):
                out.append(k[:-6].decode())
    out = sorted(set(out))
    if not out: raise RuntimeError("no *_shape keys found")
    return out

def ensure_empty_dir(path):
    os.makedirs(path, exist_ok=True)
    if os.listdir(path): raise RuntimeError(f"dst_dir not empty: {path}")

def safe_mapsize(env):
    ms = env.info().get("map_size", 0)
    return ms if ms and ms > (1 << 30) else (1 << 30)

def get_bytes(txn, key):
    v = txn.get(key)
    if v is None: raise KeyError(f"missing key: {key!r}")
    return v

def latents_bytes_to_out(row_bytes, in_row_shape, out_row_shape):
    a = np.frombuffer(row_bytes, dtype=np.float16)
    if len(in_row_shape) == 4:
        a = a.reshape(in_row_shape)[None, ...]
    elif len(in_row_shape) == 5:
        a = a.reshape(in_row_shape)
        if a.shape[0] != 1: a = a[-1:]
    else:
        raise RuntimeError(f"unsupported latents row shape: {in_row_shape}")
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
        if "latents" not in names0: raise RuntimeError("missing 'latents' in *_shape keys")
        for e in envs[1:]:
            n = list_array_names(e)
            if n != names0:
                raise RuntimeError(f"array names mismatch:\n{src_dirs[0]}={names0}\n{e.path()}={n}")
        names = names0

        shapes, Ns, lat_rows, mapsum = [], [], [], 0
        for e in tqdm(envs, desc=f"read meta -> {dst_dir}", unit="lmdb"):
            mapsum += safe_mapsize(e)
            sh = {n: read_shape(e, n) for n in names}
            N = sh[names[0]][0]
            for n in names:
                if sh[n][0] != N: raise RuntimeError(f"inconsistent N in {e.path()} for '{n}': {sh[n][0]} vs {N}")
            shapes.append(sh); Ns.append(N)
            r = sh["latents"][1:]
            if len(r) not in (4, 5): raise RuntimeError(f"unsupported latents shape: {sh['latents']}")
            lat_rows.append(r)

        for n in names:
            if n == "latents": continue
            ref = shapes[0][n][1:]
            for sh in shapes[1:]:
                if sh[n][1:] != ref:
                    raise RuntimeError(f"shape mismatch for '{n}': {shapes[0][n]} vs {sh[n]}")

        def rest(row): return row[1:] if len(row) == 5 else row
        ref_rest = rest(lat_rows[0])
        for r in lat_rows[1:]:
            if rest(r) != ref_rest:
                raise RuntimeError(f"latents spatial dims mismatch: {lat_rows[0]} vs {r}")
        out_lat_row = (1, *ref_rest)

        out = lmdb.open(dst_dir, map_size=int(mapsum * MAP_MULT), subdir=True, lock=True, readahead=False, meminit=False)

        def write_batch(src_txn, src_i0, src_i1, out_offset, lat_in_row):
            while True:
                try:
                    with out.begin(write=True) as wtxn:
                        for i in range(src_i0, src_i1):
                            out_i = out_offset + i
                            for arr in names:
                                k = f"{arr}_{i}_data".encode()
                                b = get_bytes(src_txn, k)
                                if arr == "latents":
                                    b = latents_bytes_to_out(b, lat_in_row, out_lat_row)
                                wtxn.put(f"{arr}_{out_i}_data".encode(), b)
                    return
                except lmdb.MapFullError:
                    cur = out.info()["map_size"]
                    out.set_mapsize(int(cur * 1.5) + (1 << 30))

        totalN = sum(Ns)
        pbar = tqdm(total=totalN, desc=f"merge -> {dst_dir}", unit="row")
        offset = 0
        for idx, (e, N, lat_in_row, src_path) in enumerate(zip(envs, Ns, lat_rows, src_dirs)):
            pbar.set_postfix_str(f"{idx+1}/{len(envs)} {os.path.basename(src_path)}", refresh=False)
            rtxn = e.begin()
            for s in range(0, N, BATCH):
                t = min(s + BATCH, N)
                write_batch(rtxn, s, t, offset, lat_in_row)
                pbar.update(t - s)
            offset += N
        pbar.close()

        with out.begin(write=True) as wtxn:
            for arr in names:
                new_shape = (totalN, *out_lat_row) if arr == "latents" else (totalN, *shapes[0][arr][1:])
                wtxn.put(f"{arr}_shape".encode(), (" ".join(map(str, new_shape))).encode())

        out.sync()
        out.close()
    finally:
        for e in envs:
            try: e.close()
            except: pass

    return src_dirs

def rm_dirs(dirs, desc="remove dirs"):
    for d in tqdm(dirs, desc=desc, unit="dir"):
        if os.path.exists(d):
            shutil.rmtree(d)

# -------- chunkwise only -> dataset/clean_data --------
print('Begin merge ...')
cw_src_all = [os.path.join(BASE, f"ODE6KCausal_chunkwise_{i}") for i in range(15)]
cw_dst = os.path.join(BASE, "clean_data")
cw_merged = merge_many(cw_src_all, cw_dst)
rm_dirs(cw_merged, desc="remove shards")

cw_src_all = [os.path.join(BASE, f"ODE6KCausal_framewise_{i}") for i in range(15)]
src_dirs = [d for d in tqdm(cw_src_all, unit="dir") if os.path.isdir(d)]
rm_dirs(src_dirs, desc="remove shards")
print("done")
