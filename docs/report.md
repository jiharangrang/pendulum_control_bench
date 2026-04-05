# 실험 보고서

## 왜 이 프로젝트를 했는가

LQR와 MPC는 제어 이론에서 자주 함께 등장하지만, 실제로 둘의 차이가 언제 드러나는지 코드를 직접 짜서 확인해본 적은 많지 않다. 이 프로젝트의 목표는 MuJoCo 도립진자에서 두 제어기를 같은 조건으로 비교하고, 차이가 발생하는 정확한 지점을 실패 모드 중심으로 정리하는 것이다.

핵심 질문은 세 가지였다.

- 무제약 조건에서 MPC는 LQR과 같은 해를 재현하는가
- 입력, 입력 변화율, 위치 제약이 생기면 어느 순간부터 성능이 갈리는가
- 지연과 센서 노이즈가 추가되면 어떤 제어기가 먼저 취약해지는가

## 실험 셋업

환경은 Gymnasium의 `InvertedPendulum-v5`를 바탕으로 하되, MuJoCo 모델 파라미터와 시뮬레이터 step을 그대로 이용했다. 상태는 `x = [x, theta, xdot, thetadot]`를 사용하고, 제어 입력은 카트에 작용하는 수평 힘에 대응하는 단일 action이다.

제어기는 두 가지다.

- `LQR`: MuJoCo 환경에서 finite difference로 선형화한 `Ad`, `Bd`에 대해 discrete LQR을 계산
- `MPC`: 같은 `Ad`, `Bd`, 같은 `Q`, `R`을 사용하되 `u`, `delta u`, `x` 제약을 예측 지평 내부에서 직접 반영

대표 가중치는 다음과 같다.

- `Q = diag([1, 80, 1, 10])`
- `R = 0.1`

대표 제약은 다음과 같다.

- `|u| <= 3.0`
- `|x| <= 1.0`
- `|delta u| <= 2.6`

## 해석 기준

이 레포는 단순 평균 보상보다 실패 모드와 post-disturbance 응답을 더 중요하게 본다. 그래서 결과 CSV와 플롯에는 다음 항목들을 함께 기록한다.

- `success_rate`
- `fail_theta`, `fail_x`, `fail_nan`
- `act_rate_u`, `act_rate_du`
- `max_run_u`, `max_run_du`
- `min_margin_x`
- `theta_max_post`, `theta_rms_post`
- `recovery_time`
- `u_energy`, `J_emp`

구체 정의는 [docs/methodology.md](methodology.md)에 정리했다.

## 핵심 결과

### 1. 무제약 구간에서는 MPC와 LQR이 사실상 같은 입력을 낸다

이 프로젝트에서 가장 먼저 확인한 것은 구현 검증이다. `terminal cost = P`인 무제약 선형 MPC는 이론적으로 LQR과 첫 입력이 같아야 한다. [experiments/fd_compare/sanity_unconstrained_mpc.py](../experiments/fd_compare/sanity_unconstrained_mpc.py)는 바로 이 점을 수치적으로 확인한다.

이 검증 덕분에 이후 실험에서 차이가 나타나면, 그 원인을 제약이나 지연, 노이즈에서 찾을 수 있다.

### 2. 제약이 활성화되기 전까지는 둘의 거동이 거의 같다

외란이 약하거나 rail limit에 충분한 여유가 있을 때는 제약이 거의 켜지지 않는다. 이 구간에서는 LQR의 해가 이미 좋은 해이고, MPC도 같은 모델과 같은 비용을 사용하므로 결과가 비슷하게 나온다.

### 3. 차이는 `x` 제약이 미래 예측에 들어가는 순간부터 벌어진다

대표 실험에서는 `window disturbance`를 넣고 `|u|`, `|delta u|`, `|x|` 제약을 동시에 켰다. 이때 LQR은 현재 상태만 보고 즉각적인 안정화 입력을 만든다. 반면 MPC는 예측 지평 안에서 미래 `x` 경계를 같이 본다.

그래서 외란이 커질수록 두 제어기의 전략이 달라진다.

- LQR은 막대 각도 복원에 집중하다가 카트가 레일 끝에 먼저 닿는 경우가 늘어난다.
- MPC는 더 이른 시점에 braking 성격의 입력을 분배해 rail limit를 피하고, 동일 외란에서도 더 오래 성공한다.

대표 figure는 README에만 남기고, 이 문서에서는 해석 요약만 유지한다. 핵심 관찰은 다음과 같다.

- 제약이 거의 비활성인 구간에서는 LQR과 MPC의 결과가 거의 겹친다.
- `x` 제약과 `delta u` 제약이 동시에 병목이 되는 구간부터 LQR의 `x_fail` 비율이 먼저 증가한다.
- 노이즈나 지연이 추가되면 같은 제약 조건에서도 실패 양상이 더 빨리 드러난다.

## 정리

이 레포에서 얻은 가장 중요한 결론은 "MPC가 항상 더 좋다"가 아니다. 차이는 제약이 실제로 문제를 만들기 시작할 때만 커진다. 무제약 혹은 제약이 비활성인 구간에서는 LQR과 MPC의 차이가 거의 없지만, rail limit와 input-rate limit가 동시에 성능 병목이 되는 구간에서는 미래 제약을 직접 다루는 MPC의 장점이 분명해진다.

다음 단계는 같은 프레임을 유지한 채 다음 비교를 추가하는 것이다.

- 지연 step 증가에 따른 성공률 붕괴 지점
- 센서 노이즈 크기 증가에 따른 민감도
- horizon, terminal cost, state margin에 대한 sensitivity
- observer가 포함된 출력 feedback 버전
