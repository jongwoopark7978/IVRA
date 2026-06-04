# IVRA

Project page for **IVRA: Improving Visual-Token Relations for Robot Action Policy with Training-Free Hint-Based Guidance**.

- Project page: https://jongwoopark7978.github.io/IVRA/
- arXiv PDF: https://arxiv.org/pdf/2601.16207

## Code

This repository provides a compact public release of the IVRA inference code.
The FLOWER VLA integration is provided as an overlay so the repository stays
small and does not vendor pretrained weights, datasets, or the full FLOWER code.

## IVRA + FLOWER VLA Inference

The integration lives under `integrations/flower_vla`. It adds:

- `flower/evaluation/forward_IVRA.py`: IVRA forward/encoding functions.
- `flower/evaluation/flower_eval_libero_IVRA.py`: LIBERO evaluation entrypoint.

`forward_IVRA.py` defines the custom visual-token refinement logic, and
`patch_encode_observations` swaps it into FLOWER before inference.

### Setup

Install FLOWER VLA and LIBERO following the upstream FLOWER instructions:

https://github.com/intuitive-robots/flower_vla_calvin

Then apply the IVRA overlay:

```bash
git clone --recurse-submodules https://github.com/intuitive-robots/flower_vla_calvin.git
git clone https://github.com/jongwoopark7978/IVRA.git

export FLOWER_ROOT=/path/to/flower_vla_calvin
export IVRA_ROOT=/path/to/IVRA

rsync -a $IVRA_ROOT/integrations/flower_vla/flower/ $FLOWER_ROOT/flower/
```

Download the FLOWER pretrained weights and LIBERO data separately. They are not
included in this repository.

### Run LIBERO Evaluation

```bash
cd $FLOWER_ROOT
PYTHONPATH=$FLOWER_ROOT python flower/evaluation/flower_eval_libero_IVRA.py \
  train_folder=/path/to/flower_libero_10 \
  checkpoint=/path/to/flower_libero_10/model.safetensors \
  dataset_path=/path/to/libero_dataset \
  benchmark_name=libero_10 \
  device=0 \
  n_eval=50 \
  num_videos=0 \
  log_wandb=false \
  log_dir=/path/to/eval_logs
```

The same entrypoint can be used for other LIBERO benchmarks by changing
`benchmark_name` and the corresponding FLOWER checkpoint paths.
