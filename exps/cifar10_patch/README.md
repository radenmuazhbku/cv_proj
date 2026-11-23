# CIFAR-10 discrete patch exemplar model

One script contains model, training, inference, evaluation, checkpointing, and visualization. It uses non-overlapping patch sizes 8 (one level) or 8 then 4 (two levels). The finer level's categorical prior is conditioned on its size-8 parent code. Codebooks are periodically projected onto actual training patches.

Training supports minibatch gradient descent (`--trainer gradient`, default) and optional layerwise hard EM/K-means (`--trainer em`). The same checkpoint supports sample and evaluation inference in either case. This is a discrete code/exemplar model, not a Gaussian density model.

The two patch-4 configs reproduce the referenced legacy HMP settings:

```bash
.venv/bin/python exps/cifar10_patch/train.py --config exps/cifar10_patch/config/overfit_s4_k2048.py
.venv/bin/python exps/cifar10_patch/train.py --config exps/cifar10_patch/config/overlap_s4_k2048_o2.py
```

Both use 100 train/test images, one level, patch size 4, K=2048, hard assignments, continuous centroids, and layerwise EM. The no-overlap image came from `test_overfit.py` with 3 iterations; the overlap image came from `run_overlap_baseline_ablation.py` with 5 iterations. Overlap 2 changes the extracted metric patch context. The configs use those iteration counts and the original image basenames as run names.

```bash
.venv/bin/python exps/cifar10_patch/train.py --config exps/cifar10_patch/config/sanity_bijective.py
.venv/bin/python exps/cifar10_patch/train.py --config exps/cifar10_patch/config/sanity_compressed.py
.venv/bin/python exps/cifar10_patch/train.py --config exps/cifar10_patch/config/full_cifar10.py
```

Configs define a `CONFIG` dictionary. CLI values override config values; the resolved config and loaded config path are printed and saved. Choose EM training with `--trainer em`; all configs can be overridden, for example:

```bash
.venv/bin/python exps/cifar10_patch/train.py --config exps/cifar10_patch/config/full_cifar10.py --trainer em
.venv/bin/python exps/cifar10_patch/train.py --mode eval --checkpoint logs/cifar_patch/full_cifar10/model.pt --config exps/cifar10_patch/config/full_cifar10.py
.venv/bin/python exps/cifar10_patch/train.py --mode sample --checkpoint logs/cifar_patch/full_cifar10/model.pt --config exps/cifar10_patch/config/full_cifar10.py --class-id 3 --samples 64
```

Artifacts go under `logs/cifar_patch/<run_name>/`: config snapshot, training log, checkpoint, metrics JSON, reconstruction and sample grids. Evaluation reports reconstruction MSE/PSNR, class accuracy/confusion matrix, class-conditional class NLL in nats/image, and unconditional and class-conditional discrete-code NLL in nats/patch. These NLLs score code sequences under the categorical prior; they are not pixel-space likelihoods.

Set `umap_enabled=True` in a config (or pass `--umap-enabled`) to save the patch UMAP. It displays actual sampled train/test patches, annotated with the assigned code index; red/blue outlines identify train/test. `--umap-patch-scale` scales the base 36-pixel thumbnail (default `1.5`, yielding 54-pixel tiles); resizing uses nearest-neighbor so pixels stay crisp. `codebook_exemplars.png` shows the corresponding projected exemplars. Install `umap-learn` to enable this optional, expensive visualization.

The bijective config gives 100 training images 1,600 size-8 exemplars, one per training patch, as an end-to-end reconstruction sanity run. The compressed config uses 128 exemplars and tracks observed train/test reconstruction metrics; approximately 99% reconstruction similarity is a target, not a guarantee.
