import argparse, gc, hashlib, json, math, os, sys, time
from pathlib import Path
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from geometry import geometry, combined_geometry, split_weight_gradient

def file_hash(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--indices',nargs='+',type=int,required=True)
    p.add_argument('--data',type=Path,required=True,help='Prepared directory containing manifest.json and cache/')
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--checkpoint',type=Path,required=True)
    args=p.parse_args()
    from utils.wan_wrapper import WanDiffusionWrapper
    R=args.data.resolve(); D=args.out.resolve(); D.mkdir(parents=True,exist_ok=True)
    CKPT=str(args.checkpoint.resolve())
    manifest=json.loads((R/'manifest.json').read_text())
    assert len(manifest)==128 and len(set(args.indices))==len(args.indices)
    assert all(0<=i<len(manifest) for i in args.indices)
    plan=json.loads(Path(__file__).with_name('plan.json').read_text())
    os.chdir(ROOT)
    provenance=dict(checkpoint_sha256=file_hash(Path(CKPT)),manifest_sha256=file_hash(R/'manifest.json'),plan=plan)
    protocol_path=D/'run_protocol.json'
    if protocol_path.exists(): assert json.loads(protocol_path.read_text())==provenance, 'Use a new output directory for changed inputs'
    else:
        try:
            with protocol_path.open('x') as f: json.dump(provenance,f,indent=2)
        except FileExistsError:
            assert json.loads(protocol_path.read_text())==provenance
    timesteps=plan['timesteps']
    out=D/'results';out.mkdir(exist_ok=True)
    torch.set_num_threads(3)
    torch.set_float32_matmul_precision('highest');torch.backends.cuda.matmul.allow_tf32=False
    m=WanDiffusionWrapper(model_name='Wan2.1-T2V-1.3B',timestep_shift=5.,is_causal=True,local_attn_size=-1,sink_size=0)
    state=torch.load(CKPT,map_location='cpu',weights_only=True,mmap=True)
    m.load_state_dict(state['generator'],strict=True);del state
    m.to(device='cuda',dtype=torch.bfloat16).eval().requires_grad_(True)
    m.model.num_frame_per_block=1;m.enable_gradient_checkpointing()
    names=[f'model.blocks.{d}.{s}' for d in range(30) for s in ['self_attn.q','self_attn.k','self_attn.v','self_attn.o','ffn.0','ffn.2']]
    mods=dict(m.named_modules());metadata={};coords={}
    for name in names:
        numel=mods[name].weight.numel();size=min(4096,numel)
        seed=int.from_bytes(hashlib.sha256((plan['sketch_seed_prefix']+name).encode()).digest()[:8],'little')%(2**32)
        idx=np.random.default_rng(seed).choice(numel,size=size,replace=False)
        coords[name]=torch.from_numpy(idx).cuda()
        metadata[name]=dict(layer=int(name.split('.')[2]),family='attention' if '.self_attn.' in name else 'ffn',
                            shape=list(mods[name].weight.shape),numel=numel,sketch_size=size,sketch_scale=math.sqrt(numel/size),sketch_seed=seed)
    protocol=dict(plan=plan,checkpoint=CKPT,
        modules=metadata,module_order=names,dtype='BF16 forward/backward; full FP32 gradient geometry; TF32 disabled',
        loss='Generation-supervised unweighted flow MSE, target=noise-clean; no DMD or optimizer updates',
        role_definition='Two token-row contributions to one shared loss',
        sketch='4096 fixed coordinates per matrix, scaled by sqrt(numel/4096)')
    (out/f'protocol_{os.getpid()}.json').write_text(json.dumps(protocol,indent=2))
    records={};pending={};sketches={};active=[False]
    def fh(name):
        def forward(mod,inputs,output):
            if not active[0] or not torch.is_grad_enabled() or not output.requires_grad:return
            x=inputs[0].detach();assert x.shape[1]==65520
            def backward(dy):
                assert name not in records
                c,g=split_weight_gradient(x,dy)
                idx=coords[name];scale=metadata[name]['sketch_scale']
                sc=torch.stack([torch.take(c,idx),torch.take(g,idx)])*scale
                sketches[name]=sc.cpu().numpy()
                records[name]=dict(role=geometry(c,g),sketch_geometry=geometry(sc[0],sc[1]))
                pending[name]=c+g
                return dy
            output.register_hook(backward)
        return forward
    def wh(name):
        def backward(grad):
            if active[0]:
                full=pending.pop(name);ref=grad.detach().float();den=torch.linalg.vector_norm(ref)
                records[name]['reconstruction_relative_error']=float(torch.linalg.vector_norm(full-ref)/den) if den>0 else 0.
            return grad
        return backward
    for name in names:mods[name].register_forward_hook(fh(name));mods[name].weight.register_hook(wh(name))
    print('MODEL_LOADED',len(names),'matrices',flush=True)
    for index in args.indices:
        cache=torch.load(R/'cache'/f'{index:02d}.pt',map_location='cpu',weights_only=False)
        clean=cache['latent'].cuda().float();condition=cache['prompt_embeds'].cuda().to(torch.bfloat16)
        torch.manual_seed(20260928+index*1009);noise=torch.randn_like(clean);target=noise-clean
        for j,t in enumerate(timesteps):
            dest=out/f'v{index:03d}_t{j:02d}.json';data=dest.with_suffix('.npz')
            if dest.exists() and data.exists():
                row=json.loads(dest.read_text());assert row['index']==index and row['t']==t
                with np.load(data) as saved: assert saved['sketches'].shape==(180,2,4096) and np.isfinite(saved['sketches']).all()
                assert row['reconstruction_max']<.01
                continue
            started=time.monotonic();m.zero_grad(set_to_none=True);records.clear();pending.clear();sketches.clear()
            xc=clean.to(torch.bfloat16);xg=((1-t/1000)*clean+(t/1000)*noise).to(torch.bfloat16)
            ct=torch.zeros((1,21),device='cuda');gt=torch.full((1,21),float(t),device='cuda')
            torch.cuda.reset_peak_memory_stats();active[0]=True
            pred=m.model(xg.permute(0,2,1,3,4),t=gt,context=condition,seq_len=32760,
                         clean_x=xc.permute(0,2,1,3,4),aug_t=ct).permute(0,2,1,3,4)
            loss=(pred.float()-target).square().mean();loss.backward();active[0]=False
            assert len(records)==180 and not pending
            recon=max(v['reconstruction_relative_error'] for v in records.values());assert recon<.01
            aggregate={}
            for fam in ['attention','ffn']:
                nn=[n for n in names if metadata[n]['family']==fam]
                aggregate[fam]={k:combined_geometry([records[n][k] for n in nn]) for k in ['role','sketch_geometry']}
                assert aggregate[fam]['role']['cosine'] is not None
            array=np.stack([sketches[n] for n in names]);assert array.shape==(180,2,4096) and np.isfinite(array).all()
            tmp=data.with_suffix('.tmp')
            with tmp.open('wb') as f:np.savez_compressed(f,sketches=array,index=index,t=t)
            tmp.replace(data)
            row=dict(index=index,point=j,t=t,context_t=0,noise_seed=20260928+index*1009,loss=float(loss),
                     reconstruction_max=recon,aggregate=aggregate,modules=records,seconds=time.monotonic()-started,
                     peak_gpu_gb=torch.cuda.max_memory_allocated()/1e9)
            tmp=dest.with_suffix('.tmp');tmp.write_text(json.dumps(row,allow_nan=False));tmp.replace(dest)
            print('SAVED',index,j,t,'seconds',round(row['seconds'],2),flush=True)
            del pred,loss,xc,xg,array;gc.collect()
        del clean,noise,target,condition,cache;gc.collect();torch.cuda.empty_cache()
    print('WORKER_DONE',args.indices,flush=True)

if __name__=='__main__':main()
