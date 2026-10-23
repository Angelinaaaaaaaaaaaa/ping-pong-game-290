#!/bin/bash
# Paper experiments with the fixed joint-state encoder (attack_defend_joint_v2), Design B + preparation.
#   A  self-play PPO (3 seeds) || alpha-potential data (20 pure + 20 mixed-theta shards)
#   B  Phi/Q: main (gamma 0.7, 3 seeds), gamma 0.99 (3 seeds), MSE, pure-only, 25% / 50% data
#   C  Table 1 alpha-potential: PGA + argmax (main), PGA (gamma 0.99)
#   D  Table 1 self-play: sampled + argmax, head-to-heads vs PGA and argmax (Phi main seed 0)
set -u
cd ~/angelina/ping-pong-game-290-designb
PY=~/angelina/envs/designb/bin/python
export MUJOCO_GL=egl OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=.
R=runs_joint
mkdir -p $R/data_shards $R/selfplay $R/phi $R/table1 $R/table1_gamma099
{ echo "launched: $(date -Is)"; echo "encoder: attack_defend_joint_v2"; md5sum logs/best_model_tracker1/best_model.zip
  $PY -c "import mujoco, stable_baselines3, torch; print('mujoco', mujoco.__version__, 'sb3', stable_baselines3.__version__, 'torch', torch.__version__)"; } > $R/MANIFEST.txt 2>&1

for s in 0 1 2; do
  $PY -u nash_skills/v2/selfplay_attack_defend.py --iterations 500 --rallies-per-iter 32 --seed $s \
      --defend-mechanism preparation --reset-targets --save-history \
      --output $R/selfplay/prep_seed$s.pth --log $R/selfplay/prep_seed$s.csv > $R/selfplay/prep_seed$s.log 2>&1 &
done

pids=()
for i in $(seq 0 19); do
  for mode in grid mixed; do
    base=$([ $mode = grid ] && echo 100 || echo 200)
    $PY -u nash_skills/v2/collect_data_attack_defend.py --sampling $mode --decisions 450 --max-rallies 1000 \
        --max-steps 2500 --defend-mechanism preparation --reset-targets --seed $((base + i)) --progress-every 0 \
        --output $R/data_shards/${mode}_$i.pkl > $R/data_shards/${mode}_$i.log 2>&1 &
    pids+=($!)
  done
done
wait "${pids[@]}"

M=scripts/merge_rally_shards.py
$PY $M --output $R/rallies_prep_grid.pkl $R/data_shards/grid_*.pkl > $R/merge.log 2>&1
$PY $M --output $R/rallies_prep_mixed.pkl $R/data_shards/mixed_*.pkl >> $R/merge.log 2>&1
$PY $M --output $R/rallies_prep_combined.pkl $R/data_shards/grid_*.pkl $R/data_shards/mixed_*.pkl >> $R/merge.log 2>&1
NTRAIN=$($PY -c "import pickle; print(int(round(0.8 * len(pickle.load(open('$R/rallies_prep_combined.pkl', 'rb'))))))")
C="--rallies $R/rallies_prep_combined.pkl"
pids=()
phi() { name=$1; shift
  OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 $PY -u nash_skills/v2/train_q_model_5skill_v3_attack_defend.py "$@" \
      --out-dir $R/phi/$name > $R/phi/$name.log 2>&1 &
  pids+=($!); }
for s in 0 1 2; do
  phi main_s$s $C --phi-loss lse --theta-mode mixed --seed $s
  phi gamma099_s$s $C --phi-loss lse --theta-mode mixed --seed $s --gamma 0.99
done
phi mse_mixed $C --phi-loss mse --theta-mode mixed --seed 0
phi grid_lse_pure --rallies $R/rallies_prep_grid.pkl --phi-loss lse --theta-mode pure --seed 0
phi vol25 $C --phi-loss lse --theta-mode mixed --seed 0 --train-rally-limit $((NTRAIN / 4))
phi vol50 $C --phi-loss lse --theta-mode mixed --seed 0 --train-rally-limit $((NTRAIN / 2))
wait "${pids[@]}"
touch $R/phi/DONE

E="$PY -u scripts/eval_table1_attack_defend.py --episodes 100 --workers 34 --defend-mechanism preparation"
( for s in 0 1 2; do
    $E --method nash-p-pga --model-dir $R/phi/main_s$s --seed $s --output $R/table1/nash_p_pga_s$s.json > $R/table1/nash_p_pga_s$s.log 2>&1
  done
  for s in 0 1 2; do
    $E --method nash-p-pga --model-dir $R/phi/gamma099_s$s --seed $s --output $R/table1_gamma099/nash_p_pga_s$s.json > $R/table1_gamma099/nash_p_pga_s$s.log 2>&1
  done ) &
c1=$!
( for s in 0 1 2; do
    $E --method nash-p-hard --model-dir $R/phi/main_s$s --seed $s --output $R/table1/nash_p_hard_s$s.json > $R/table1/nash_p_hard_s$s.log 2>&1
  done ) &
c2=$!
wait $c1 $c2
touch $R/table1/ALPHA_DONE

until grep -q "^Done" $R/selfplay/prep_seed0.log && grep -q "^Done" $R/selfplay/prep_seed1.log \
      && grep -q "^Done" $R/selfplay/prep_seed2.log; do sleep 120; done
S="$E --model-dir $R/phi/main_s0 --opponents static random ibr nash-p-pga nash-p-hard"
( for s in 0 1 2; do
    $S --method selfplay:$R/selfplay/prep_seed$s.pth --seed $((20 + s)) --output $R/table1/selfplay_s$s.json > $R/table1/selfplay_s$s.log 2>&1
  done ) &
d1=$!
( for s in 0 1 2; do
    $S --method selfplay-argmax:$R/selfplay/prep_seed$s.pth --seed $((30 + s)) --output $R/table1/selfplay_argmax_s$s.json > $R/table1/selfplay_argmax_s$s.log 2>&1
  done ) &
d2=$!
wait $d1 $d2
touch $R/ALL_DONE
