#!/bin/bash
# (A) Check Q early stopping (--q-select val) on the main setting, both profiles:
#     held-out Q error, Phi vs constant potential, sign agreement, Table 1 static + random.
# (B) 10x data: 108 grid + 108 mixed extra shards x 1500 decisions per profile.
set -u
cd ~/angelina/ping-pong-game-290-designb
PY=~/angelina/envs/designb/bin/python
export MUJOCO_GL=egl OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=.
R=runs_paper
SET="--defend-mechanism preparation_cal --decision-timing contact"
( for P in aggressive moderate_aggressive; do
    ( OMP_NUM_THREADS=2 $PY -u nash_skills/v2/train_q_model_5skill_v3_attack_defend.py --rallies $R/$P/rallies.pkl \
          --theta-mode mixed --phi-loss lse --phi-select val_residual --q-select val --gamma 0.99 --seed 0 \
          --out-dir $R/$P/phi/g099_qval_s0 > $R/$P/phi/g099_qval_s0.log 2>&1
      $PY scripts/phi_sign_agreement.py --rallies $R/$P/rallies.pkl --model-dir $R/$P/phi/g099_qval_s0 \
          --output $R/sign_agreement/${P}_g099_qval_s0.json > $R/sign_agreement/${P}_g099_qval_s0.log 2>&1
      $PY -u scripts/eval_table1_attack_defend.py $SET --skill-profile $P --workers 16 --method nash-p-pga \
          --model-dir $R/$P/phi/g099_qval_s0 --seed 0 --episodes 100 --opponents static random \
          --output $R/$P/table1/pga_g099_qval_s0.json > $R/$P/table1/pga_g099_qval_s0.log 2>&1 ) &
  done; wait; touch $R/QFIX_DONE ) &
( for P in aggressive moderate_aggressive; do
    mkdir -p $R/$P/data10x
    for i in $(seq 0 107); do for mode in grid mixed; do echo "$P $mode $i"; done; done
  done | xargs -P 40 -L 1 bash -c 'P=$0; mode=$1; i=$2; base=$([ $mode = grid ] && echo 2000 || echo 3000)
      '"$PY"' -u nash_skills/v2/collect_data_attack_defend.py --sampling $mode --decisions 1500 --max-rallies 3000 \
          --max-steps 2500 '"$SET"' --skill-profile $P --reset-targets --seed $((base + i)) --progress-every 0 \
          --output '"$R"'/$P/data10x/${mode}_$i.pkl > '"$R"'/$P/data10x/${mode}_$i.log 2>&1'
  for P in aggressive moderate_aggressive; do
    $PY scripts/merge_rally_shards.py --output $R/$P/rallies_10x.pkl $R/$P/data_shards/*.pkl $R/$P/data10x/*.pkl \
        > $R/$P/merge_10x.log 2>&1
  done
  touch $R/DATA10X_DONE ) &
wait
