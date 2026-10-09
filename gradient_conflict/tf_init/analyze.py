import argparse, csv, hashlib, json, time
from pathlib import Path
import numpy as np
import torch
from sklearn.manifold import TSNE

from exact_tsne import fit_exact

def summarize(rows):
    result = {}
    for family in ('attention', 'ffn'):
        result[family] = {}
        for exit_index in sorted({r['exit_index'] for r in rows}):
            values = np.array([r['cosine'] for r in rows
                               if r['family'] == family and r['exit_index'] == exit_index])
            if not len(values):
                continue
            result[family][str(exit_index)] = {
                'n': len(values), 'mean_cosine': float(values.mean()),
                'negative_fraction': float((values < 0).mean()),
                'mean_angular_distance': float((np.arccos(np.clip(values, -1, 1)) / np.pi).mean())}
    return result


def save_statistics(out, cosines, max_error):
    rows = [dict(prompt=i, exit_index=j, family=family, cosine=float(cosines[f,i,j]))
            for i in range(128) for j in range(50)
            for f,family in enumerate(('attention','ffn'))]
    result = dict(prompts=list(range(128)), pairs=128*50,
                  max_reconstruction_relative_error=max_error, summary=summarize(rows))
    with (out/'values.csv').open('w') as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    (out/'summary.json').write_text(json.dumps(result, indent=2)+'\n')


parser=argparse.ArgumentParser()
parser.add_argument('--data',type=Path,required=True)
parser.add_argument('--out',type=Path,required=True)
args=parser.parse_args()
D=args.data.resolve();O=args.out.resolve();O.mkdir(parents=True,exist_ok=True)
CACHE=D/'.analysis_cache';CACHE.mkdir(exist_ok=True)
E=CACHE/'exact_tsne';E.mkdir(exist_ok=True)
plan=json.loads(Path(__file__).with_name('plan.json').read_text());ts=plan['timesteps'];n=128*50
fingerprint=hashlib.sha256(json.dumps(plan,sort_keys=True).encode())
assert len(ts)==50 and len(set(ts))==50 and all(1<=t<=1000 for t in ts)
protocols=[json.loads(p.read_text()) for p in (D/'results').glob('protocol_*.json')]
assert protocols
meta=protocols[0]['modules'];names=protocols[0]['module_order']
assert len(names)==180
assert all(p['modules']==meta and p['module_order']==names and p['plan']==plan
           and p['checkpoint']==protocols[0]['checkpoint'] and p['dtype']==protocols[0]['dtype']
           and p['loss']==protocols[0]['loss'] for p in protocols)
vectors=np.empty((180,2*n,4096),dtype=np.float32)
cos=np.empty((2,128,50));sketch_cos=np.empty_like(cos);max_recon=0.
for i in range(128):
    for j,t in enumerate(ts):
        stem=D/'results'/f'v{i:03d}_t{j:02d}'
        r=json.loads(stem.with_suffix('.json').read_text())
        assert (r['index'],r['point'],r['t'],r['context_t'],r['noise_seed'])==(i,j,t,0,20260928+i*1009)
        assert set(r['modules'])==set(names) and np.isfinite(r['loss'])
        with np.load(stem.with_suffix('.npz')) as z:
            v=z['sketches'];assert v.shape==(180,2,4096) and np.isfinite(v).all()
            assert int(z['index'])==i and int(z['t'])==t
            fingerprint.update(v.tobytes())
            vectors[:,i*50+j]=v[:,0];vectors[:,n+i*50+j]=v[:,1]
        assert np.isfinite(r['reconstruction_max'])
        max_recon=max(max_recon,r['reconstruction_max']);assert max_recon<.01
        for f,fam in enumerate(['attention','ffn']):
            cos[f,i,j]=r['aggregate'][fam]['role']['cosine']
            sketch_cos[f,i,j]=r['aggregate'][fam]['sketch_geometry']['cosine']
    if i%8==7: print('LOADED',i+1,'videos',flush=True)
assert np.isfinite(cos).all() and (np.abs(cos)<=1).all()
input_sha256=fingerprint.hexdigest()
torch.set_num_threads(8);torch.set_float32_matmul_precision('highest');torch.backends.cuda.matmul.allow_tf32=False
result=dict(status='PASS',video_count=128,timestep_count=50,pair_count=n,point_count_per_tsne=2*n,
    plan=plan,max_reconstruction_error=max_recon,source_protocol=protocols[0],families={},
    tsne=dict(metric='arccos(sketch cosine)/pi',perplexity=20,seed=23092026,max_iter=2000,
              method='exact',note='All 12800 points; exact t-SNE.'))
arrays=dict(timesteps=np.array(ts),paired_cosine=cos)
for f,fam in enumerate(['attention','ffn']):
    idx=[k for k,name in enumerate(names) if meta[name]['family']==fam]
    cached=E/f'{fam}_exact.npz'
    if cached.exists():
        with np.load(cached) as check:
            assert str(check['input_sha256'])==input_sha256, 'Stale embedding cache; remove data/.analysis_cache or use a new --data directory'
    gram=torch.zeros((2*n,2*n),device='cuda',dtype=torch.float64)
    means=[];start=time.monotonic()
    for k in idx:
        v=torch.from_numpy(vectors[k]).cuda()
        gram.add_((v@v.T).double())
        means.append(np.stack([vectors[k,:n].mean(0,dtype=np.float64),vectors[k,n:].mean(0,dtype=np.float64)]))
        del v
    gram=gram.cpu().numpy()
    lengths=np.sqrt(np.diag(gram));assert np.all(lengths>0)
    observed=gram[np.arange(n),n+np.arange(n)]/(lengths[:n]*lengths[n:])
    assert np.max(np.abs(observed-sketch_cos[f].reshape(-1)))<1e-4
    means=np.concatenate(means,axis=1);c,g=means
    def geometry(c,g):
        nc=np.linalg.norm(c);ng=np.linalg.norm(g);co=float(np.clip(c@g/(nc*ng),-1,1))
        return dict(cosine=co,angle_degrees=float(np.degrees(np.arccos(co))),norm_context=float(nc),norm_denoising=float(ng),norm_ratio=float(ng/nc))
    info=geometry(c,g)
    info['paired_sketch_cosine_max_abs_error']=float(np.max(np.abs(observed-cos[f].reshape(-1))))
    info['mean_method']='Raw arithmetic mean over 128 videos and 50 timesteps in fixed coordinate sketches; no per-sample normalization'
    print('GRAM_DONE',fam,'seconds',time.monotonic()-start,flush=True)
    if cached.exists():
        with np.load(cached) as saved:
            xy=saved['embedding'];kl=float(saved['kl'])
        del gram
    else:
        gram/=lengths[:,None];gram/=lengths[None,:]
        np.clip(gram,-1,1,out=gram);np.arccos(gram,out=gram);gram/=np.pi
        distance=(gram+gram.T)*.5;del gram
        np.fill_diagonal(distance,0);assert np.isfinite(distance).all()
        model=TSNE(n_components=2,perplexity=20,metric='precomputed',init='random',learning_rate='auto',
                   method='exact',max_iter=2000,random_state=23092026,n_jobs=8,verbose=2)
        xy=fit_exact(model, distance);kl=float(model.kl_divergence_)
        np.savez_compressed(cached,embedding=xy,kl=kl,input_sha256=input_sha256)
        del distance
    assert xy.shape==(2*n,2) and np.isfinite(xy).all()
    if xy[:n,0].mean()>xy[n:,0].mean():xy[:,0]*=-1
    arrays[fam+'_embedding']=xy
    info['tsne_kl']=kl;result['families'][fam]=info
    print('TSNE_DONE',fam,'seconds',time.monotonic()-start,flush=True)
np.savez_compressed(CACHE/'analysis.npz',**arrays)
(CACHE/'summary.json').write_text(json.dumps(result,indent=2))
save_statistics(O,cos,max_recon)
print('ANALYSIS_DONE',flush=True)
