# Piper 학습 robustness 설정

이 문서는 기본 MuJoCo-Warp PPO 경로의 시간 기준과 불확실성 모델을 설명한다.
목표는 실제 Piper에서 관측될 수 있는 작은 편차를 학습 중에 경험하게 하는 것이며,
아래 수치는 Piper 실측·식별 결과가 아니다. 실기 궤적을 수집하면 범위를 다시
추정해야 한다.

## 기본 시간과 관측

Warp 물리 timestep은 `0.002 s`이고 policy가 10개의 물리 step마다 action을
결정하므로 기본 주기는 50Hz다. 한 episode는 600 policy tick, 즉 12초이며,
유효한 배치가 16 tick (0.32초) 지속되어야 성공으로 기록된다. 25Hz 초기 CPU
환경의 `0.002 s × 20`과 기존 300 tick checkpoint는 이 시간 기준과 다르다.
policy action은 7차원 normalized incremental target command다. zero action은
현재 actuator target을 유지한다. 처음 6개 값은 각 관절 target에 더하는
`0.035 rad/tick` increment이고 마지막 값은 gripper target에 더하는
`0.004 m/tick` increment다. 결과 target은 기존 joint/gripper limits 안에서
clamp한다. SDK/저수준 제어기에 absolute target을 보내는 변환은 학습 정책
contract 바깥의 별도 adapter다. 50Hz에서 평균 관절 속도가 초기 25Hz 경로와
크게 달라지지 않도록 tick별 이동 제한은 초기 정책 tick 제한의 절반 수준으로 둔다.
PPO는 이 control tick 기준으로 `gamma=sqrt(0.99)`, `lambda=sqrt(0.95)`를
사용한다. 따라서 discount와 GAE horizon도 25Hz에서 유지하던 physical-time
비율을 기준으로 조정된다.

관측은 63차원 벡터다. 앞의 56차원은 raw-SI 기반 state이고 뒤의 7차원은
직전 policy raw action이다. actor에는 episode에 고정된 관절 영점 offset,
균등 센서 오차, 0–2 policy-tick의 sensor-history delay를 state에 적용한다.
critic에는 같은 63차원 layout의 clean observation을 제공한다. actor와 critic은
각자 empirical observation normalization을 사용한다. 이 분리는 critic에게만
추가 privileged field를 붙이는 방식이 아니다. actor의 state 56차원에는 현재 task의
접촉 및 상태 flag처럼 시뮬레이터 계산값이 여전히 있으므로 완전한 실기 배포
관측 분리로 약속하지 않는다.

센서 delay는 50Hz에서 0–40ms다. 관절 명령, 직전 raw action, simulation current
clock은 지연시키지 않으며, observation을 같은 tick에 여러 번 읽어도 같은 delayed
sample을 돌려준다. reset 때 history와 episode-local offset을 함께 초기화해
이전 episode sample이 새 episode로 새지 않게 한다. 현재 Piper task에는 IMU가
없으므로 IMU 전용 noise를 추가하지 않는다.

## Domain randomization

randomization은 각 Warp world에 대해 episode reset 때 샘플한다. 기본 policy
학습에서는 다음 보수적인 범위를 최종 난이도의 기준으로 사용한다.

| 대상 | 최종 범위/방식 |
| --- | --- |
| 이동 링크 질량 | nominal 대비 `±10%` |
| cube 질량 | nominal 40g 대비 `±25%` |
| 링크·cube 관성 | sampled mass scale에 추가 `±10%` (positive definite 유지) |
| link·cube center of mass | 각 축 `±2 mm` |
| table·cube·finger friction | nominal 대비 `±25%`, 양수 clamp |

각 world는 독립적으로 샘플하므로 한 reset batch 안에서도 값이 다를 수 있다.
partial reset에서는 완료된 world의 물리값과 상태만 다시 샘플한다. reference의
초기 trunk-only mass `±5%`, armature `±10%`, 그리고 BAM friction `±10%` 값은
Microduck 쪽 설계 참고값이다. Piper 측정값인 것처럼 복사하지 않고, 현재 draft의
per-episode moving-link/cube uncertainty를 실제 데이터에 맞춰 조정해야 한다.

전압 randomization과 Piper용 BAM actuator는 아직 추가하지 않았다. 기존
`gravcomp=1`과 low-level position actuator는 유지하지만, high-level 정책 입력은
직접 joint target으로 바뀐다. 이 문서의 friction 범위를 BAM calibration의
대체물로 해석하지 않는다.

## Sensor uncertainty

noise toggle이 켜져 있을 때 사용하는 uniform half-ranges는 다음과 같다.

| 신호 | half-range |
| --- | --- |
| joint position | `0.001 rad` |
| joint velocity | `0.25 rad/s` |
| episode joint zero offset | `0.005 rad`, episode 동안 고정 |
| finger position | `0.0002 m` |
| finger velocity | `0.002 m/s` |
| Cartesian position | `0.002 m` |
| orientation | `0.01 rad` axis-angle perturbation |
| linear velocity | `0.02 m/s` |
| angular velocity | `0.05 rad/s` |
| observation delay | integer `0..2` policy ticks |

이 값들은 raw-SI 신호에 적용하고 그 뒤 empirical normalization을 수행한다.
`--no-sensor-noise`는 오차, zero offset, delay를 모두 끈다. `--no-domain-randomization`
은 물리 parameter sampling을 끄지만 sensor noise와 curriculum toggle에는 영향을
주지 않는다.

## Curriculum

기본 curriculum은 성능이나 성공률을 보고 자동 승급하지 않는다. 전체 vector
environment가 수집한 `training_steps`를 경계와 비교해 stage를 선택한다. 따라서
stage step은 world 수나 PPO `steps_per_env` batch 크기와 독립적이다. reward
weight와 home-start 비율도 이 표의 step 경계에서 함께 바뀐다.

| `training_steps` 시작 | randomization scale | layout scale | home probability | action-rate weight |
| ---: | ---: | ---: | ---: | ---: |
| 0 | 0.20 | 0.55 | 0.00 | 0.001 |
| 32,000 | 0.50 | 0.75 | 0.25 | 0.002 |
| 64,000 | 0.75 | 0.90 | 0.50 | 0.005 |
| 96,000 | 1.00 | 1.00 | 0.80 | 0.010 |

`--no-curriculum`은 stage를 즉시 최종 난이도(마지막 행)로 고정한다. `above_cube`
와 `home`은 `--start-mode`로 직접 선택할 수 있다. 명시한 start mode는 curriculum의
home probability보다 우선하며, mode를 지정하지 않는 기본 경로에서만 stage의
home probability를 사용한다. action-rate penalty는 policy가 낸 현재 raw action과
관측에 포함된 직전 raw action의 차이에 적용한다. reset 시 직전 incremental action은
zero로 초기화하므로, zero command는 reset target을 유지하고 첫 tick의 변화 penalty가
없다.

## 실행과 checkpoint

기본 실행은 curriculum, domain randomization, sensor noise를 모두 켠다.

```bash
# 기본 curriculum + physical/sensor uncertainty
bash scripts/run_piper_pick_place.sh --headless

# 고정된 최종 난이도에서 clean sensor로 확인
bash scripts/run_piper_pick_place.sh --headless \
    --no-curriculum --no-domain-randomization --no-sensor-noise \
    --start-mode above_cube

# home reset을 명시적으로 선택
bash scripts/run_piper_pick_place.sh --headless --start-mode home
```

checkpoint에는 `training_steps`가 저장되며 resume 시 curriculum stage가 그
값에서 이어진다. 새 checkpoint schema는 `3`이며 `tanh_squashed_gaussian_v1`
action distribution, incremental action contract, reward-v3 placement gate를 포함한다.
25Hz/300-tick 초기 경로의
checkpoint와 SB3 `.zip`, 이전 observation layout 또는 다른 actor/critic 입력을
가진 `.pt`는 호환되지 않으므로 새 run을 시작해야 한다. checkpoint 호환 여부는
단순히 파일 확장자가 아니라 schema와 시간/관측 설정을 함께 확인해 판단한다.

reward-v3에서는 성공 stability가 `has_lifted`, goal 내부, table 위, release,
robot-cube 비접촉을 같은 tick에 만족한 뒤 시작한다. 이후에도 robot-cube contact가
한 tick이라도 발생하면 stability가 0으로 돌아가며, 16개의 연속된 접촉 없는 정지
tick을 다시 채워야 한다. TensorBoard의 `Episode/success_count`는 기존 full-task
성공을, `Episode/success_contact_anomaly_count`는 placement/recontact 조건을
위반한 성공 신호를, `Episode/clean_success_count`는 보수적인 성공 수를 기록한다.
`Episode/rolling_*_denominator`와 함께 읽어 빈 terminal batch를 성공률 0으로
오해하지 않도록 한다. reward-v1/v2 checkpoint는 새 reward contract와 호환되지
않으므로 normal resume를 거부한다.

## 해석 범위

이 설정은 contact-based pick-and-place physics와 fixed-gravity-compensation
controller를 계속 사용한다. 짧은 smoke run, shaped reward, 또는 한 seed의
성공률만으로 수렴·일반화·sim2real 준비를 주장할 수 없다. 실제 Piper의 관절
응답, friction, sensor timestamp와 비교한 뒤 범위·delay·controller를 보정하고,
actor에 남아 있는 simulator-only contact/state 값을 실기에서 측정할 수 있는지
검토해야 한다.
