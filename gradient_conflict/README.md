# Gradient Conflict Experiments

Reproduce the SGF and TF-initialization gradient-conflict experiments. Follow the [installation and weight download instructions](../README.md), then run the commands below from the repository root.

## SGF

### Setup

Install the additional plotting dependency and download the diagnostic checkpoint:

```bash
pip install scikit-learn==1.7.2
hf download ZihanSu/Self_Gradient_Forcing_Plus diagnostics/sgf.pt --local-dir hf_weights
```

The experiment uses the SGF raw generator and paired critic in `hf_weights/diagnostics/sgf.pt`, together with the Wan2.1-T2V-14B teacher downloaded by the main setup script.

Gradient collection requires approximately 110 GB of GPU memory per worker. You can use `--cpu-offload` to reduce GPU memory usage.

### Collect Gradients

Collect gradients from self-attention and FFN layers over **128 prompts × 4 denoising steps**, without updating the model:

```bash
PYTHONPATH=. python gradient_conflict/sgf/collect.py \
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
    python gradient_conflict/sgf/collect.py \
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

### Analyze and Plot

The commands below generate **Figure 3(a–c) in our paper**: gradient distributions, mean gradient directions, and paired cosine similarities.

```bash
python gradient_conflict/sgf/analyze.py \
  --data outputs/gradient_conflict_128 \
  --out outputs/gradient_conflict_128/analysis

python gradient_conflict/sgf/plot.py \
  --data outputs/gradient_conflict_128 \
  --out outputs/gradient_conflict_128/figures
```

Statistics are saved in `analysis/`; figures are saved in `figures/` as PNG, PDF and SVG, with accompanying JSON data.

| Figure | Description |
| --- | --- |
| A: Gradient distributions | t-SNE visualization of context-writing and denoising gradients |
| B: Mean gradient directions | Mean directions and their angle in Attention and FFN |
| C: Conflict across timesteps | Paired gradient cosine similarities at each denoising step |

## TF Initialization

### Setup

This experiment reuses `checkpoints/init/framewise/ar_diffusion.pt` from the main setup.

Install the additional plotting dependencies and prepare the official Causal Forcing Stage 1 data:

```bash
pip install scikit-learn==1.7.2
hf download zhuhz22/Causal-Forcing-data --local-dir dataset

PYTHONPATH=. python gradient_conflict/tf_init/prepare.py \
  --dataset dataset \
  --out outputs/tf_init/data
```

Analysis requires a GPU and at least 64 GB host RAM.

### Collect Gradients

Collect gradients from self-attention and FFN layers over **128 videos × 50 denoising timesteps**, using the original prompts and without updating the model.

```bash
PYTHONPATH=. python gradient_conflict/tf_init/collect.py \
  --checkpoint checkpoints/init/framewise/ar_diffusion.pt \
  --data outputs/tf_init/data \
  --out outputs/tf_init/collection \
  --indices $(seq 0 127)
```

<details>
<summary>Run on eight GPUs</summary>

Use this command instead of the single-GPU command above. Each GPU processes a different subset of videos.

```bash
pids=()
for gpu in $(seq 0 7); do
  CUDA_VISIBLE_DEVICES="$gpu" PYTHONPATH=. \
    python gradient_conflict/tf_init/collect.py \
    --checkpoint checkpoints/init/framewise/ar_diffusion.pt \
    --data outputs/tf_init/data \
    --out outputs/tf_init/collection \
    --indices $(seq "$gpu" 8 127) &
  pids+=("$!")
done
for pid in "${pids[@]}"; do
  wait "$pid" || exit 1
done
```

</details>

### Analyze and Plot

The commands below generate **Figure 7(a–c) in our paper**: gradient distributions, mean gradient directions, and paired cosine similarities.

```bash
python gradient_conflict/tf_init/analyze.py \
  --data outputs/tf_init/collection \
  --out outputs/tf_init/analysis

python gradient_conflict/tf_init/plot.py \
  --data outputs/tf_init/collection \
  --out outputs/tf_init/figures
```

Statistics are saved in `analysis/`; figures are saved in `figures/` as PNG, PDF and SVG, with accompanying JSON data.

| Figure | Description |
| --- | --- |
| A: Gradient distributions | t-SNE visualization of context-writing and denoising gradients |
| B: Mean gradient directions | Mean directions and their angle in Attention and FFN |
| C: Conflict across timesteps | Paired gradient cosine similarities at each denoising step |
