"""Collect context-writing and denoising weight gradients."""
import argparse
from contextlib import nullcontext
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time

import numpy as np
import torch
from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)


def stable_seed(name: str) -> int:
    digest = hashlib.sha256(('sgf-all-layers:' + name).encode()).digest()
    return int.from_bytes(digest[:8], 'little') % (2**32)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--config', type=Path, default=Path('configs/self_gradient_forcing_framewise.yaml'))
    parser.add_argument('--prompt-indices', type=int, nargs='+', required=True)
    parser.add_argument('--manifest', type=Path, default=Path('gradient_conflict/prompts.jsonl'))
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--exits', type=int, nargs='+', default=[0, 1, 2, 3])
    parser.add_argument('--sketch-size', type=int, default=4096)
    parser.add_argument('--cpu-offload', action='store_true',
                        help='Offload score models and saved activations to CPU.')
    args = parser.parse_args()
    from model.dmd import DMD
    if any(t not in range(4) for t in args.exits):
        raise ValueError(f'exits must be in 0..3, got {args.exits}')
    args.out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    torch.set_float32_matmul_precision('high')
    device = torch.device('cuda:0')
    torch.cuda.set_device(device)

    config = OmegaConf.merge(
        OmegaConf.load('configs/default_config.yaml'),
        OmegaConf.load(args.config))
    manifest = [json.loads(line) for line in args.manifest.read_text().splitlines() if line.strip()]
    invalid = [i for i in args.prompt_indices if not 0 <= i < len(manifest)]
    if invalid:
        raise ValueError(f'prompt indices outside manifest range 0..{len(manifest)-1}: {invalid}')
    if len(set(args.prompt_indices)) != len(args.prompt_indices):
        raise ValueError('prompt indices must be unique')
    samples = [(i, manifest[i]) for i in args.prompt_indices]
    print('INITIALIZING', args.prompt_indices, time.ctime(), flush=True)
    torch.manual_seed(0)
    model = DMD(config, device)
    model.generator.to(device=device, dtype=torch.bfloat16)
    score_device = torch.device('cpu') if args.cpu_offload else device
    model.real_score.to(device=score_device, dtype=torch.bfloat16).requires_grad_(False)
    model.fake_score.to(device=score_device, dtype=torch.bfloat16).requires_grad_(False)
    model.text_encoder.to(device=device, dtype=torch.bfloat16)
    with torch.no_grad():
        conditionals = {i: model.text_encoder([sample['model_prompt']])
                        for i, sample in samples}
        unconditional = model.text_encoder([config.negative_prompt])
    model.text_encoder.cpu()
    gc.collect()
    torch.cuda.empty_cache()
    model._initialize_inference_pipeline()
    pipe = model.inference_pipeline

    names = []
    for depth in range(30):
        names.extend(f'model.blocks.{depth}.self_attn.{projection}'
                     for projection in ('q', 'k', 'v', 'o'))
        names.extend(f'model.blocks.{depth}.ffn.{index}' for index in (0, 2))
    modules = dict(model.generator.named_modules())
    missing = [name for name in names if name not in modules]
    if missing:
        raise RuntimeError(f'Missing target modules: {missing}')
    non_linear = [name for name in names if not isinstance(modules[name], torch.nn.Linear)]
    if non_linear:
        raise TypeError(f'Targets must be Linear: {non_linear}')

    expected_tokens = 2 * 21 * 1560
    active = [False]
    records = {}
    pending_combined = {}
    coord_indices = {}
    module_meta = {}
    for name in names:
        module = modules[name]
        out_features, in_features = module.weight.shape
        numel = module.weight.numel()
        k = min(args.sketch_size, numel)
        rng = np.random.default_rng(stable_seed(name))
        index = rng.choice(numel, size=k, replace=False)
        coord_indices[name] = torch.from_numpy(index.astype(np.int64)).to(device)
        parts = name.split('.')
        family = 'attention' if parts[3] == 'self_attn' else 'ffn'
        projection = parts[-1]
        if family == 'ffn':
            projection = 'up' if projection == '0' else 'down'
        module_meta[name] = dict(
            layer=int(parts[2]), family=family, projection=projection,
            shape=[int(out_features), int(in_features)], numel=int(numel),
            sketch_size=int(k), sketch_scale=math.sqrt(numel / k),
            sketch_seed=stable_seed(name))

    def make_forward_hook(name):
        def forward_hook(module, inputs, output):
            if not active[0] or not torch.is_grad_enabled() or not output.requires_grad:
                return
            x = inputs[0].detach()
            if args.cpu_offload:
                x = x.cpu()
            if x.shape[1] != expected_tokens:
                raise RuntimeError(f'Unexpected TF token count: {name} {tuple(x.shape)}')

            def backward_hook(dy):
                if name in records or name in pending_combined:
                    raise RuntimeError(f'Duplicate gradient hook: {name}')
                midpoint = x.shape[1] // 2
                gradients = []
                for start, stop in ((0, midpoint), (midpoint, x.shape[1])):
                    xx = x[:, start:stop].reshape(-1, x.shape[-1]).to(dy.device).float()
                    dd = dy[:, start:stop].reshape(-1, dy.shape[-1]).float()
                    gradients.append(dd.T @ xx)
                context, generation = gradients
                norm2_context = torch.sum(context * context)
                norm2_generation = torch.sum(generation * generation)
                dot = torch.sum(context * generation)
                index = coord_indices[name]
                scale = module_meta[name]['sketch_scale']
                sketch_context = torch.take(context, index) * scale
                sketch_generation = torch.take(generation, index) * scale
                records[name] = dict(
                    norm2_context=float(norm2_context),
                    norm2_generation=float(norm2_generation),
                    dot=float(dot),
                    sketch_context=sketch_context.cpu(),
                    sketch_generation=sketch_generation.cpu())
                pending_combined[name] = context + generation
                return dy
            output.register_hook(backward_hook)
        return forward_hook

    def make_weight_hook(name):
        def weight_hook(grad):
            if not active[0]:
                return grad
            if name not in pending_combined:
                raise RuntimeError(f'Weight hook ran without partitioned gradient: {name}')
            combined = pending_combined.pop(name)
            reference = grad.detach().float()
            relative = torch.linalg.vector_norm(combined - reference) / \
                torch.linalg.vector_norm(reference).clamp_min(1e-30)
            records[name]['reconstruction_relative_error'] = float(relative)
            return grad
        return weight_hook

    handles = []
    if args.cpu_offload:
        def load_score(module, inputs):
            module.to(device)

        def unload_score(module, inputs, output):
            module.cpu()
            torch.cuda.empty_cache()

        for score in (model.real_score, model.fake_score):
            handles.append(score.register_forward_pre_hook(load_score))
            handles.append(score.register_forward_hook(unload_score))

    for name in names:
        handles.append(modules[name].register_forward_hook(make_forward_hook(name)))
        handles.append(modules[name].weight.register_hook(make_weight_hook(name)))

    checkpoint = args.checkpoint
    state = torch.load(checkpoint, map_location='cpu', weights_only=True, mmap=True)
    checkpoint_step = state.get('step')
    for key, module in (('generator', model.generator), ('critic', model.fake_score)):
        raw = state[key]
        normalized = {k.replace('model._fsdp_wrapped_module.', 'model.', 1)
                      if k.startswith('model._fsdp_wrapped_module.') else k: v
                      for k, v in raw.items()}
        module.load_state_dict(normalized, strict=True)
        print('LOADED', key, len(normalized), flush=True)
    del state, raw, normalized

    for prompt_index, sample in samples:
        conditional = conditionals[prompt_index]
        protocol = dict(
            protocol='SGF generator with paired critic; no optimizer updates',
            checkpoint=str(checkpoint), checkpoint_step=checkpoint_step, checkpoint_key='generator',
            critic='Paired checkpoint critic', teacher='Wan2.1-T2V-14B',
            prompt_index=prompt_index, prompt=sample, manifest=str(args.manifest), manifest_size=len(manifest),
            config=OmegaConf.to_container(config, resolve=True), modules=module_meta,
            measurement='FP32 exact token-partitioned shared weight-gradient geometry for all 30 blocks; biases excluded',
            role_definition='First/second half token-row contribution to the same joint teacher-forcing DMD loss gradient',
            token_count=expected_tokens, tokens_per_role=expected_tokens // 2,
            sketch='Deterministic uniform coordinate sample without replacement, scaled by sqrt(weight_numel/sketch_size)')
        (args.out / f'prompt_{prompt_index:03d}_protocol.json').write_text(
            json.dumps(protocol, indent=2, ensure_ascii=False))
        print('READY', prompt_index, 'GPU_GB', torch.cuda.memory_allocated()/1e9, flush=True)

        for exit_index in args.exits:
            out_file = args.out / f'p{prompt_index:03d}_t{exit_index}.pt'
            if out_file.exists():
                print('EXISTS', out_file, flush=True)
                continue
            started = time.monotonic()
            model.generator.zero_grad(set_to_none=True)
            records.clear()
            pending_combined.clear()
            torch.cuda.reset_peak_memory_stats()
            torch.manual_seed(int(sample['seed']))
            noise = torch.randn([1, 21, 16, 60, 104], device=device, dtype=torch.bfloat16)
            rng_seed = 710000 + prompt_index * 100 + exit_index * 10000
            while True:
                torch.cuda.manual_seed(rng_seed)
                rng_state = torch.cuda.get_rng_state()
                drawn = int(torch.randint(0, 4, (1,), device=device).item())
                if drawn == exit_index:
                    torch.cuda.set_rng_state(rng_state)
                    break
                rng_seed += 1
            active[0] = True
            print('ROLLOUT', prompt_index, exit_index, flush=True)
            saved_tensors = (torch.autograd.graph.save_on_cpu() if args.cpu_offload
                             else nullcontext())
            with saved_tensors:
                pred, from_t, to_t, actual_exit = pipe.inference_with_trajectory(
                    noise=noise, return_sim_step=True, **conditional)
                if actual_exit != exit_index + 1:
                    raise RuntimeError((actual_exit, exit_index))
                print('DMD', prompt_index, exit_index, flush=True)
                torch.manual_seed(900000 + prompt_index)
                loss, logs = model.compute_distribution_matching_loss(
                    pred, conditional, unconditional, denoised_timestep_from=from_t,
                    denoised_timestep_to=to_t)
                print('BACKWARD', prompt_index, exit_index, float(loss), flush=True)
                loss.backward()
            active[0] = False
            if pending_combined:
                raise RuntimeError(f'Unconsumed weight-gradient checks: {sorted(pending_combined)}')
            if set(records) != set(names):
                raise RuntimeError(f'Missing hooks: {sorted(set(names) - set(records))}')
            errors = {name: records[name]['reconstruction_relative_error'] for name in names}
            if max(errors.values()) > .025:
                raise RuntimeError(f'Gradient partition verification failed: {max(errors.values())}')
            result = dict(
                checkpoint_step=checkpoint_step, prompt_index=prompt_index, exit_index=exit_index,
                generator_timestep=float(pipe.denoising_step_list[exit_index]),
                score_timestep=float(logs['timestep'].flatten()[0]), seed=int(sample['seed']),
                loss=float(loss), module_stats=records.copy(),
                cpu_offload=args.cpu_offload,
                elapsed_seconds=time.monotonic() - started,
                peak_gpu_gb=torch.cuda.max_memory_allocated()/1e9)
            temporary = out_file.with_suffix('.tmp')
            torch.save(result, temporary)
            temporary.replace(out_file)
            print('SAVED', out_file, 'seconds', result['elapsed_seconds'],
                  'peak_gpu_gb', result['peak_gpu_gb'],
                  'max_partition_error', max(errors.values()), flush=True)
            del pred, loss, logs, result, noise
            model.generator.zero_grad(set_to_none=True)
            records.clear()
            gc.collect()
            torch.cuda.empty_cache()
        print('PROMPT_COMPLETE', prompt_index, time.ctime(), flush=True)

    for handle in handles:
        handle.remove()
    print('COMPLETE', args.prompt_indices, time.ctime(), flush=True)


if __name__ == '__main__':
    main()
