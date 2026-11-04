# NEXUS

Official implementation of **NEXUS: Neural Energy Fields for Physically Consistent Contact-Rich 3D Object Dynamics**.

NEXUS models deformable 3D object dynamics using structural graphs, dynamic contact
graphs, and learned energy and dissipation terms. This repository provides the
multi-object implementation for scenes containing two, three, or five objects.

## Installation

Create and activate the `nexus` Conda environment using the provided configuration:

```bash
conda env create -f nexus.yml
conda activate nexus
```

## Data Generation

Generate datasets for two, three, and five objects from PhysCtrl `.h5` geometry files:

```bash
SRC_H5_DIR=/path/to/h5 DATA_ROOT=/path/to/data bash scripts/generate_all_multiobj_data.sh
```

## Training

Train a NEXUS model using the corresponding dataset:

```bash
DATA_DIR=/path/to/data/dataset_multiobj2 N_OBJECTS=2 GPU=0 EXP=nexus_2obj bash scripts/train_multiobj_base.sh
```

For two, three, or five objects, set `N_OBJECTS` to `2`, `3`, or `5`, and change
`DATA_DIR` to `dataset_multiobj2`, `dataset_multiobj3`, or `dataset_multiobj5`,
respectively; use a matching experiment name such as `nexus_3obj` for `EXP`.

## Evaluation

Evaluate a trained NEXUS checkpoint on the validation split:

```bash
python -m nexus.eval_multi_object_rollout --data /path/to/data/dataset_multiobj2 --ckpt /path/to/best.pth --out results/nexus_2obj --split valid --max_per_k_tuple 0 --horizons 6 12 24 48 --require_full_horizon
```

## Trajectory Export and Visualization

Export predicted and ground-truth trajectories, and visualize a generated ground-truth episode:

```bash
python scripts/export_multiobj_trajectories.py --data /path/to/data/dataset_multiobj2 --ckpt /path/to/best.pth --out exported/nexus_2obj --horizon 48 --max_per_k_tuple 1
python plot_multi_object_gif_3d.py --data /path/to/data/dataset_multiobj2 --episode 0 --out visualizations/nexus_2obj_gt.gif --show_floor --trail
```

## License

This project is licensed under the [MIT License](LICENSE).
