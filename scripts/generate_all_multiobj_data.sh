#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR=${PROJECT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
SRC_H5_DIR=${SRC_H5_DIR:?set SRC_H5_DIR to the directory containing PhysCtrl .h5 geometry files}
DATA_ROOT=${DATA_ROOT:-$PROJECT_DIR/generated_data_true_h48}
OBJECT_COUNTS=${OBJECT_COUNTS:-"2 3 5"}
EXPECTED_CONDA_ENV=${CONDA_ENV:-nexus}

if [ ! -d "$SRC_H5_DIR" ]; then
  echo "SRC_H5_DIR does not exist: $SRC_H5_DIR" >&2
  exit 1
fi

if ! compgen -G "$SRC_H5_DIR/*.h5" >/dev/null; then
  echo "No .h5 files found directly under SRC_H5_DIR=$SRC_H5_DIR" >&2
  exit 1
fi

if [ "${CONDA_DEFAULT_ENV:-}" != "$EXPECTED_CONDA_ENV" ]; then
  echo "[warn] active Conda environment is '${CONDA_DEFAULT_ENV:-none}', expected '$EXPECTED_CONDA_ENV'" >&2
  echo "[warn] run: conda activate $EXPECTED_CONDA_ENV" >&2
fi

python -c "import h5py, numpy, scipy, tqdm" >/dev/null
mkdir -p "$DATA_ROOT"

for nobj in $OBJECT_COUNTS; do
  case "$nobj" in
    2)
      build_script="$PROJECT_DIR/scripts/build_2object_floor_gravity_force.sh"
      max_groups=${MAX_GROUPS_2:-600}
      repeats=${REPEAT_PER_GROUP_2:-2}
      seed=${SEED_2:-2026}
      ;;
    3)
      build_script="$PROJECT_DIR/scripts/build_3object_floor_gravity_force.sh"
      max_groups=${MAX_GROUPS_3:-600}
      repeats=${REPEAT_PER_GROUP_3:-1}
      seed=${SEED_3:-2026}
      ;;
    5)
      build_script="$PROJECT_DIR/scripts/build_5object_floor_gravity_force.sh"
      max_groups=${MAX_GROUPS_5:-600}
      repeats=${REPEAT_PER_GROUP_5:-1}
      seed=${SEED_5:-2026}
      ;;
    *)
      echo "Unsupported object count '$nobj'; use OBJECT_COUNTS='2 3 5' or a subset" >&2
      exit 1
      ;;
  esac

  dst_dir="$DATA_ROOT/dataset_multiobj${nobj}"
  if compgen -G "$dst_dir/*.npz" >/dev/null; then
    echo "Refusing to overwrite existing NPZ files in $dst_dir" >&2
    echo "Choose another DATA_ROOT or move the existing dataset first." >&2
    exit 1
  fi

  echo "[generate] n_objects=$nobj max_groups=$max_groups repeats=$repeats seed=$seed"
  SRC_H5_DIR="$SRC_H5_DIR" \
  DST_DIR="$dst_dir" \
  MAX_GROUPS="$max_groups" \
  REPEAT_PER_GROUP="$repeats" \
  SEED="$seed" \
  bash "$build_script"
done

echo "[done] generated datasets under $DATA_ROOT"
