#!/usr/bin/env bash
set -euo pipefail
SRC_H5_DIR=${SRC_H5_DIR:?set SRC_H5_DIR}
DST_DIR=${DST_DIR:?set DST_DIR}
python build_multi_object_springmass_from_physctrl_h5.py \
  --src_h5_dir "$SRC_H5_DIR" \
  --dst_dir "$DST_DIR" \
  --n_objects 3 \
  --max_files ${MAX_FILES:-400} \
  --max_groups ${MAX_GROUPS:-400} \
  --repeat_per_group ${REPEAT_PER_GROUP:-1} \
  --target_n_obj_each ${TARGET_N_OBJ_EACH:-128} \
  --target_n_env ${TARGET_N_ENV:-1024} \
  --T ${T:-49} \
  --dt ${DT:-0.0416667} \
  --substeps ${SIM_SUBSTEPS:-64} \
  --k_values "${K_VALUES:-10,10,10;10,100,100;100,10,100;100,100,10;10,100,500;500,100,10;50,50,50;100,100,100;500,100,100;100,500,100;100,100,500;500,500,500}" \
  --layout ${LAYOUT:-ring} \
  --graph_preset solid \
  --long_strategy random \
  --normalize_diameter 1.0 \
  --particle_radius ${PARTICLE_RADIUS:-0.04} \
  --relative_speed_min ${REL_SPEED_MIN:-1.0} \
  --relative_speed_max ${REL_SPEED_MAX:-1.8} \
  --contact_time_min ${CONTACT_TIME_MIN:-0.25} \
  --contact_time_max ${CONTACT_TIME_MAX:-0.45} \
  --floor_mode grid \
  --floor_height 0.0 \
  --floor_lift ${FLOOR_LIFT:-0.35} \
  --gravity_mode always \
  --gravity_y ${GRAVITY_Y:--4.0} \
  --force_mode ${FORCE_MODE:-one_of_objects} \
  --force_prob ${FORCE_PROB:-0.5} \
  --force_direction ${FORCE_DIRECTION:-push_towards_center} \
  --force_mask_mode ${FORCE_MASK_MODE:-local_patch} \
  --force_mask_fraction ${FORCE_MASK_FRACTION:-0.10} \
  --force_mag_min ${FORCE_MAG_MIN:-2.0} \
  --force_mag_max ${FORCE_MAG_MAX:-8.0} \
  --k_pair_contact_stiffness ${PAIR_CONTACT_STIFFNESS:-60000} \
  --pair_contact_damping_ratio ${PAIR_CONTACT_DAMPING_RATIO:-0.20} \
  --pair_contact_k ${PAIR_CONTACT_K:-4} \
  --k_floor ${FLOOR_CONTACT_STIFFNESS:-80000} \
  --floor_contact_damping_ratio ${FLOOR_CONTACT_DAMPING_RATIO:-0.20} \
  --episodes_per_npz ${EPISODES_PER_NPZ:-64} \
  --seed ${SEED:-2026}
