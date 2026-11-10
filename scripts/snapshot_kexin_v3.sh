#!/bin/bash
# Code snapshot of Kexin's checkout for Route B (self-play in her V3 game): her working tree as is (uncommitted
# simulator edits included) without data, results or git history, plus the low-level controller and her 50k Q and
# Phi models. Idempotent: does nothing once SNAPSHOT_DONE exists. Our v3_selfplay package is rsynced in separately.
set -u
K=~/kexin/ping-pong-game-290
SNAP=~/angelina/kexin-v3-snapshot
mkdir -p $SNAP
cd $SNAP
[ -e SNAPSHOT_DONE ] && exit 0
rsync -a --max-size=100m --exclude='.git/' --exclude='data/' --exclude='skill_eval/' --exclude='models/' \
    --exclude='logs/' --exclude='runs*/' --exclude='analysis/' --exclude='venv/' --exclude='.venv/' \
    --exclude='wandb/' --exclude='__pycache__/' --exclude='*.pkl' --exclude='*.npz' --exclude='*.mp4' \
    --exclude='v3_selfplay/' $K/ $SNAP/
mkdir -p models/v3 logs/best_model_tracker1
cp $K/models/v3/potential_50k_aggressive.pth $K/models/v3/q_preposition_onehot_50k_aggressive.pth models/v3/
cp -r $K/logs/best_model_tracker1/. logs/best_model_tracker1/
{ date -Is; (cd $K && git rev-parse HEAD && git status --short | grep -v '^??'); \
  md5sum models/v3/*.pth logs/best_model_tracker1/best_model.zip; } > SNAPSHOT_SOURCE.txt
touch SNAPSHOT_DONE
