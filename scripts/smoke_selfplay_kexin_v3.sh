#!/bin/bash
# Smoke test for Route B in the snapshot of Kexin's checkout: 2 training iterations, then one rally per side of
# head-to-head and one rally per opponent of her Table 1 grid. Prints SMOKE_OK when all three finish.
set -u
bash ~/angelina/selfplay_kexin_v3/snapshot_kexin_v3.sh
cd ~/angelina/kexin-v3-snapshot
PY=~/angelina/envs/designb/bin/python
export MUJOCO_GL=egl OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=.
Q=models/v3/q_preposition_onehot_50k_aggressive.pth
PHI=models/v3/potential_50k_aggressive.pth
S=runs_selfplay_v3_smoke
rm -rf $S && mkdir -p $S
timeout 1800 $PY -u -m v3_selfplay.train --iterations 2 --rallies-per-iter 4 --round-iters 1 --snapshot-every 1 \
    --workers 2 --seed 0 --output $S/sp.pth --log $S/sp.csv > $S/train.log 2>&1 && echo "train ok" || { tail -30 $S/train.log; exit 1; }
cat $S/sp.csv
timeout 1800 $PY -u -m v3_selfplay.her_eval h2h --register sp=$S/sp.pth --tested sp --opponents potential_minimax \
    --model $Q --potential-model $PHI --seeds 0 --episodes 1 --both-sides --output-json $S/h2h.json \
    > $S/h2h.log 2>&1 && tail -3 $S/h2h.log || { tail -30 $S/h2h.log; exit 1; }
timeout 2400 $PY -u -m v3_selfplay.her_eval grid --register sp=$S/sp.pth -- --policies sp --model $Q \
    --potential-model $PHI --seeds 0 --episodes 1 --steps 600 --output-dir $S/grid > $S/grid.log 2>&1 \
    && tail -5 $S/grid.log || { tail -30 $S/grid.log; exit 1; }
echo SMOKE_OK
