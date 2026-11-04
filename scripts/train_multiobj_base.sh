#!/usr/bin/env bash
set -euo pipefail
PROJECT_DIR=${PROJECT_DIR:-$(pwd)}
DATA_DIR=${DATA_DIR:?set DATA_DIR}
CONDA_SH=${CONDA_SH:-$HOME/miniconda3/etc/profile.d/conda.sh}
CONDA_ENV=${CONDA_ENV:-nexus}
GPU=${GPU:-0}
EXP=${EXP:-multiobj_run}
N_OBJECTS=${N_OBJECTS:-3}
if [ -f "$CONDA_SH" ]; then
  source "$CONDA_SH"
  conda activate "$CONDA_ENV"
fi
cd "$PROJECT_DIR"
export PYTHONPATH="$PROJECT_DIR:${PYTHONPATH:-}"
export OMP_NUM_THREADS=${CPU_THREADS:-2}
export MKL_NUM_THREADS=${CPU_THREADS:-2}
export NUMEXPR_NUM_THREADS=${CPU_THREADS:-2}
CUDA_VISIBLE_DEVICES=$GPU python -m nexus.train_multi_object \
  --data "$DATA_DIR" \
  --n_objects "$N_OBJECTS" \
  --device cuda \
  --epochs ${EPOCHS:-50} \
  --lr ${LR:-1e-4} \
  --hidden ${HIDDEN:-128} \
  --mlp_depth ${MLP_DEPTH:-2} \
  --encoder_type ${ENCODER_TYPE:-simple} \
  --split ${SPLIT:-0.9} \
  --split_by group_index \
  --min_stored_states ${MIN_STORED_STATES:-49} \
  --file_load_workers ${FILE_LOAD_WORKERS:-2} \
  --num_workers_train ${NUM_WORKERS_TRAIN:-3} \
  --num_workers_valid ${NUM_WORKERS_VALID:-1} \
  --epoch_train_samples ${TRAIN_SAMPLES:-3072} \
  --epoch_valid_samples ${VALID_SAMPLES:-256} \
  --scale_samples ${SCALE_SAMPLES:-64} \
  --rollout_steps ${ROLLOUT_STEPS:-6} \
  --rollout_schedule ${ROLLOUT_SCHEDULE:-0:6,8:12,30:24} \
  --val_rollout_steps ${VAL_ROLLOUT_STEPS:-24} \
  --rollout_decay ${ROLLOUT_DECAY:-0.97} \
  --rollout_x_weight 1.0 \
  --rollout_v_weight 0.25 \
  --rollout_strain_weight ${STRAIN_WEIGHT:-0.10} \
  --detach_rollout_state \
  --oversample_contact ${OVERSAMPLE_CONTACT:-6} \
  --contact_oversample_mode metadata \
  --contact_window_radius 6 \
  --struct_graph_preset solid \
  --struct_add_mst \
  --struct_long_k ${STRUCT_LONG_K:-8} \
  --struct_long_start_rank 12 \
  --struct_long_end_rank 96 \
  --struct_long_strategy ${STRUCT_LONG_STRATEGY:-random} \
  --edge_coef_mode ${EDGE_COEF_MODE:-inv_rest} \
  --edge_coef_clip_max 100 \
  --k_obj ${K_OBJ:-18} \
  --k_knn ${K_KNN:-12} \
  --k_pair_contact ${K_PAIR_CONTACT:-4} \
  --k_floor_contact ${K_FLOOR_CONTACT:-8} \
  --model_family nexus \
  --struct_mode ${STRUCT_MODE:-learned_times_k} \
  --struct_scale ${STRUCT_SCALE:-100} \
  --pair_contact_scale ${PAIR_CONTACT_SCALE:-60000} \
  --floor_contact_scale ${FLOOR_CONTACT_SCALE:-80000} \
  --pair_contact_damping_scale ${PAIR_DAMPING:-150} \
  --floor_contact_damping_scale ${FLOOR_DAMPING:-150} \
  --n_substeps ${N_SUBSTEPS:-4} \
  --acc_clip ${ACC_CLIP:-100} \
  --grad_clip ${GRAD_CLIP:-0.5} \
  --tf32 \
  --exp_name "$EXP" \
  "$@"
