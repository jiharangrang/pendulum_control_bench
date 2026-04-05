# 방법론 메모

## 상태와 모델

환경 상태는 다음 벡터로 둔다.

```text
x = [x, theta, xdot, thetadot]
```

- `x`: 카트 위치
- `theta`: 직립 기준 막대 각도
- `xdot`, `thetadot`: 각 상태의 시간 미분

메인 비교는 MuJoCo 시뮬레이터에서 직접 구한 finite-difference 선형화에 기반한다.

- 선형화 함수: [experiments/fd_compare/run_fd_compare.py](../experiments/fd_compare/run_fd_compare.py)
- 이론 모델과 ZOH 이산화: [controllers/lqr.py](../controllers/lqr.py)

## 제어기 구성

### LQR

`Ad`, `Bd`, `Q`, `R`에 대해 discrete algebraic Riccati equation을 풀고, `u = -Kx` 형태의 feedback gain을 사용한다.

구현:

- [controllers/lqr.py](../controllers/lqr.py)

### MPC

MPC는 condensed prediction matrix를 구성한 뒤, finite horizon quadratic program을 매 step 푼다.

지원 제약:

- input box constraint
- rate constraint `delta u`
- state constraint on selected state index, 기본적으로 `x`

구현:

- [controllers/mpc.py](../controllers/mpc.py)

solver 선택:

- `solver="auto"`이면 `osqp`가 설치되어 있을 때 OSQP 사용
- 그렇지 않으면 SciPy optimization 사용

## 실험 현실화 요소

환경 wrapper는 비교를 공정하게 만들기 위한 현실화 요소를 제공한다.

- 외란 주입: `ActionDisturbance`, `ForceDisturbance`
- 종료 조건: `TerminationOverride`
- 입력 지연: `ActuationDelay`
- 센서 노이즈: `ObservationNoise`

구현:

- [envs/wrappers.py](../envs/wrappers.py)

## 공정 비교 원칙

LQR와 MPC는 가능한 한 같은 조건을 공유한다.

- 같은 `Ad`, `Bd`
- 같은 `Q`, `R`
- 같은 disturbance schedule
- 같은 actuator limits
- 같은 seed 기반 노이즈 시퀀스

차이는 오직 예측 기반 제약 처리 여부에 두는 것이 목표다.

## 성공 / 실패 판정

메인 sweep는 다음 종류의 실패를 분리해서 기록한다.

- `theta_fail`: 각도 한계 초과
- `x_fail`: rail position 한계 초과
- `nan_fail`: 수치 불안정
- `other_fail`: 환경 termination만 감지된 경우

대표 판정 파라미터:

- `termination_theta`
- `termination_x_limit`
- `x_fail_limit`
- `x_fail_eps`
- `x_fail_hold`

집계 구현:

- [experiments/fd_compare/eval_sweep_fd_compare.py](../experiments/fd_compare/eval_sweep_fd_compare.py)

## 주요 지표

### Constraint activity

- `act_rate_u`: 입력 포화 활성 비율
- `act_rate_du`: 입력 변화율 제약 활성 비율
- `max_run_u`, `max_run_du`: 연속 활성 최대 길이
- `min_margin_x`: 위치 제약까지 남은 최소 margin

### Post-disturbance response

- `theta_max_post`
- `theta_rms_post`
- `recovery_time`
- `u_energy`
- `J_emp`

이 지표들은 "성공률만 높다"가 아니라, 어떤 방식으로 성공하고 실패하는지를 읽기 위한 용도다.

## Sanity checks

### 무제약 MPC = LQR

`P = DARE(Q, R)`를 terminal cost로 두고 제약을 충분히 크게 열면, linear MPC의 첫 입력은 LQR과 일치해야 한다. 이 레포에서는 다음 스크립트로 이를 수치적으로 확인한다.

- [experiments/fd_compare/sanity_unconstrained_mpc.py](../experiments/fd_compare/sanity_unconstrained_mpc.py)

### du_max 전이 구간 선택

MPC와 LQR의 차이가 가장 잘 드러나는 `delta u` 영역을 찾기 위해 high-amp 영역에서 `du_max`를 sweep한다.

- [experiments/fd_compare/sweep_du_high_amp.py](../experiments/fd_compare/sweep_du_high_amp.py)

이 실험은 너무 빡빡한 제약과 너무 느슨한 제약을 피하고, 차이가 가장 정보량 있게 드러나는 구간을 찾기 위한 보조 분석이다.
