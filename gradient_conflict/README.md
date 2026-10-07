# Gradient Conflict Experiments

Reproduce the gradient-conflict analysis between context writing and denoising in SGF.

## Setup

Follow the [installation and weight download instructions](../README.md), then install the additional plotting dependency and download the diagnostic checkpoint. Run all commands from the repository root.

```bash
pip install scikit-learn==1.7.2
hf download ZihanSu/Self_Gradient_Forcing_Plus diagnostics/sgf.pt --local-dir hf_weights
```

The experiment uses the SGF raw generator and paired critic in `hf_weights/diagnostics/sgf.pt`, together with the Wan2.1-T2V-14B teacher downloaded by the main setup script.

Gradient collection requires approximately 110 GB of GPU memory per worker. You can use `--cpu-offload` to reduce GPU memory usage.

## Collect Gradients

Collect gradients from self-attention and FFN layers over **128 prompts × 4 denoising steps**, without updating the model:

```bash
PYTHONPATH=. python gradient_conflict/collect.py \
  --checkpoint hf_weights/diagnostics/sgf.pt \
  --prompt-indices $(seq 0 127) \
  --sketch-size 16384 \
  --out outputs/gradient_conflict_128
```

<details>
<summary>Run on eight GPUs</summary>

Use this command instead of the single-GPU command above. Each GPU processes a different subset of prompts.

```bash
pids=()
for gpu in $(seq 0 7); do
  CUDA_VISIBLE_DEVICES="$gpu" PYTHONPATH=. \
    python gradient_conflict/collect.py \
    --checkpoint hf_weights/diagnostics/sgf.pt \
    --prompt-indices $(seq "$gpu" 8 127) \
    --sketch-size 16384 \
    --out outputs/gradient_conflict_128 &
  pids+=("$!")
done
for pid in "${pids[@]}"; do
  wait "$pid" || exit 1
done
```

</details>

## Analyze and Plot

```bash
python gradient_conflict/analyze.py \
  --data outputs/gradient_conflict_128 \
  --out outputs/gradient_conflict_128/analysis

for plot in plot_gradient_distributions plot_mean_directions plot_paired_cosine; do
  python gradient_conflict/${plot}.py \
    --data outputs/gradient_conflict_128 \
    --out outputs/gradient_conflict_128/figures
done
```

Statistics are saved in `analysis/`; figures are saved in `figures/` as PNG, PDF and SVG, with accompanying JSON data.

| Figure | Description |
| --- | --- |
| A: Gradient distributions | t-SNE visualization of context-writing and denoising gradients |
| B: Mean gradient directions | Mean directions and their angle in Attention and FFN |
| C: Conflict across timesteps | Paired gradient cosine similarities at each denoising step |
