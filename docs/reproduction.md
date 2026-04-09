# 재현 가이드

## 1. 환경 준비

프로젝트 루트에서 실행:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

참고:

- `controllers/mpc.py`는 `solver="auto"`일 때 `osqp`가 있으면 OSQP를, 없으면 SciPy 최적화를 사용한다.
- 기본 requirements에는 SciPy 경로가 포함되어 있으므로 추가 설치 없이 동작한다.
- 헤드리스 서버에서 렌더링이 필요하면 `export MUJOCO_GL=egl`를 권장한다. [smoke_mujoco.py](../smoke_mujoco.py)는 기본값으로 `egl`을 설정한다.

## 2. 기본 동작 확인

MuJoCo 환경이 제대로 열리는지 먼저 확인:

```bash
python smoke_mujoco.py
```

무제약 MPC와 LQR이 수치 오차 범위에서 일치하는지 확인:

```bash
python -m experiments.fd_compare.sanity_unconstrained_mpc \
  --seed 0 \
  --horizon 20 \
  --u-max 100 \
  --tol-u 1e-3 \
  --tol-x 1e-3
```

## 3. 빠른 비교 실행

짧은 sanity용 sweep:

```bash
python -m experiments.fd_compare.eval_sweep_fd_compare \
  --mode metrics \
  --controllers lqr_fd,mpc_fd \
  --amps 250,275 \
  --seeds 3 \
  --disturbance-kind window \
  --t0 100 \
  --duration 5 \
  --theta0 0 \
  --termination-theta 1.5708 \
  --termination-x-limit 1.0 \
  --x-fail-limit 1.0 \
  --x-fail-eps 0.0 \
  --x-fail-hold 1 \
  --actuator-u-max 3.0 \
  --actuator-u-min -3.0 \
  --actuator-du-max 2.6 \
  --metric-du-threshold 2.6 \
  --sat-tol 0.02 \
  --steps 250 \
  --step-log-dir logs/fd_compare/steps_quick \
  --out logs/fd_compare/summary_quick.csv
```

플롯 생성:

```bash
python -m experiments.fd_compare.plot_fd_compare \
  --csv logs/fd_compare/summary_quick.csv \
  --outdir plots/fd_compare_quick \
  --step-log-dir logs/fd_compare/steps_quick \
  --u-seed 0
```

## 4. 대표 figure 재현

README와 가장 가까운 명령은 아래 조합이다.

```bash
python -m experiments.fd_compare.eval_sweep_fd_compare \
  --mode metrics \
  --controllers lqr_fd,mpc_fd \
  --amps 50,75,100,125,150,175,200,225,250,275 \
  --seeds 50 \
  --disturbance-kind window \
  --t0 100 \
  --duration 5 \
  --theta0 0 \
  --termination-theta 1.5708 \
  --termination-x-limit 1.0 \
  --x-fail-limit 1.0 \
  --x-fail-eps 0.0 \
  --x-fail-hold 1 \
  --actuator-u-max 3.0 \
  --actuator-u-min -3.0 \
  --actuator-du-max 2.6 \
  --du-tol 0.01 \
  --mpc-state-constraint x \
  --mpc-x-margin 0.02 \
  --metric-du-threshold 2.6 \
  --sat-tol 0.02 \
  --steps 250 \
  --actuation-delay-steps 1 \
  --step-log-dir logs/fd_compare/steps_metrics \
  --out logs/fd_compare/summary_force.csv
```

```bash
python -m experiments.fd_compare.plot_fd_compare \
  --csv logs/fd_compare/summary_force.csv \
  --outdir plots/fd_compare \
  --step-log-dir logs/fd_compare/steps_metrics \
  --u-seed 2 \
  --u-amp-idx 5,10 \
  --u-energy-amp 250 \
  --u-energy-seed 2
```

대표 출력:

- `plots/fd_compare/success_rate.png`
- `plots/fd_compare/sat_rate.png`
- `plots/fd_compare/recovery_time.png`
- `plots/fd_compare/u_timeseries_all_amps_seed2.png`

## 5. du_max 선택 실험

전이 구간에서 `delta u` 제한을 어떻게 고를지 비교하려면 아래 스크립트를 사용한다.

```bash
python -m experiments.fd_compare.sweep_du_high_amp \
  --du-values 0.8,1.0,1.2,1.6,2.0,2.2,2.4,2.6,2.8,3.0 \
  --amps 270,275,280,285,290,295,300 \
  --seeds 6 \
  --steps 250 \
  --disturbance-kind window \
  --t0 100 \
  --duration 5 \
  --theta0 0 \
  --termination-theta 0.5 \
  --termination-x-limit 1.0 \
  --x-fail-limit 1.0 \
  --x-fail-eps 0.0 \
  --x-fail-hold 1 \
  --actuator-u-max 3.0 \
  --actuator-u-min -3.0 \
  --mpc-state-constraint x \
  --mpc-x-margin 0.02 \
  --sat-tol 0.02 \
  --outdir logs/fd_compare/du_tuning_270_300 \
  --plotdir plots/fd_compare/du_tuning_270_300
```

이 스크립트는 각 `du` 후보마다 CSV를 만들고, 성공률 gap과 추천 `du`를 자동으로 정리한다.

## 6. 주요 출력 경로

- `logs/fd_compare/*.csv`: run-level aggregate metrics
- `logs/fd_compare/steps_*/*.csv`: step-level traces
- `plots/fd_compare/*.png`: aggregate plots
- `videos/`: `RecordVideo`가 생성하는 렌더링 결과