# Piper real2sim 비교와 IMU·BAM 모델 검토

작성일: 2026-09-21

Piper의 실측 데이터를 시뮬레이션에 반영하는 real2sim과, 시뮬레이션에서 학습한
정책을 실기에 적용하는 sim2real을 준비하기 위한 비교 문서다. 레퍼런스는
[Microduck RL](https://github.com/pollen-robotics/microduck_rl/tree/cb70b792312d559a4da09064d92009079671815f)의 대표 보행 설정을
기준으로 하며, 개별 과제와 옵션에 따라 설정은 달라질 수 있다.
로컬 `reference/`는 이 저장소에 포함하지 않으며, 레퍼런스 링크는 비교에 사용한
원본 커밋을 가리킨다.

## 작성 시점의 코드 상태

아래 비교표는 대화 당시의 **초기 Piper 구현(SB3 + CPU MuJoCo)**을 기준으로
보존한 것이다. 작성 시점에는 기본 Piper 실행 경로가 이미 변경되어 있다.

| 항목 | 작성 시점의 기본 Piper 경로 |
| --- | --- |
| 실행 진입점 | `scripts/run_piper_pick_place.sh` → `scripts/train_piper_rsl.py` |
| 물리 계산·학습 | MuJoCo Warp GPU 물리 + `rsl_rl` PPO, 기본 병렬 환경 128개 |
| 관측 정규화 | actor와 critic 모두 `obs_normalization=True` |
| 신경망 | actor·critic 각각 `256 → 128 → 64`, Tanh |
| 관측·행동 | 58차원 수치 관측, 손끝 이동량과 그리퍼 변화량의 4차원 행동 |
| actor·critic 입력 | 둘 다 같은 `policy` 관측 사용. 실기용 관측과 학습 전용 정보의 분리는 아직 없음 |
| 정책 실행 주기 | 25Hz 유지: 물리 2ms × 20스텝 |
| 저장·배포 | RSL-RL `model.pt` 및 중간 체크포인트 저장. 프로젝트의 ONNX 내보내기·실기 비교 경로는 아직 없음 |

즉, 초기 비교의 CPU 물리 계산과 관측 정규화 관련 차이는 현재 기본 경로에
그대로 적용되지 않는다. 반면 정확한 시뮬레이션 상태를 정책에 제공하는 점,
고정된 모터·물리 설정, 센서 오차 모델과 실기 연결의 부재는 별도로 보완해야 한다.
GPU 전환만으로 실제 Piper와의 동작 차이가 보정되는 것은 아니다.

근거: [현재 학습 설정](../scripts/train_piper_rsl.py),
[현재 Warp 환경](../scripts/piper_warp_env.py), [실행 안내](../README.md).

## 레퍼런스와의 비교 — 초기 SB3 구현 기준

| 항목 | 초기 Piper | 레퍼런스 |
| --- | --- | --- |
| 정책이 보는 정보 | 정확한 물체 위치·자세·속도·접촉 정보를 시뮬레이터에서 직접 제공 | 실기에서 얻을 관측과 학습용 critic에만 제공할 정보를 구분 |
| 관측 정규화 | 속도에 `0.1`을 곱하는 등 수동 스케일 조정 | 관측 통계로 정규화하고 배포 모델에도 포함 |
| 모터 모델 | 고정된 위치 제어 게인·마찰, 로봇 링크에 `gravcomp=1` | XL330용 BAM 모델로 전압·부하에 따른 마찰·명령 지연 등을 반영 |
| 물리 조건의 변화 | 물체·목표 위치 랜덤화 중심. 물체 질량은 40g으로 고정 | 질량·관성·무게중심·마찰·전압 등을 변화시키며 학습 |
| 센서 오차 | 명시적인 관측 잡음·센서 지연 모델 없음 | 관절·IMU 잡음, 관절 영점 오차, 관측 지연 반영 |
| 정책 실행 주기 | 25Hz: 물리 2ms × 20스텝마다 행동 결정 | 50Hz |
| 부드러운 움직임 | 행동 크기 벌점과 관절 목표 변화량 제한 | 행동 변화량 벌점, 학습 단계에 따라 벌점 강도 조절 |
| 학습 난이도 조절 | 시작 모드를 직접 선택, 자동 단계 조절 없음 | 보상 가중치·명령 비율·일부 랜덤화 범위를 단계적으로 조절 |
| 배포·실기 비교 | SB3 `model.zip` 저장 중심 | 정규화를 포함한 ONNX 내보내기, 추론 리허설, 실기 궤적 비교 코드 |

관련 코드:

- [초기 Piper 환경](../scripts/piper_pick_place_env.py)
- [초기 SB3 학습기](../scripts/train_reacher_parallel.py)
- [Piper 물리 모델](../third_party/mujoco_menagerie/agilex_piper/piper.xml)
- [Piper 작업 장면](../scenes/piper_pick_place.xml)
- [레퍼런스 보행 환경 설정](https://github.com/pollen-robotics/microduck_rl/blob/cb70b792312d559a4da09064d92009079671815f/src/mjlab_microduck/tasks/microduck_velocity_env_cfg.py)
- [레퍼런스 모터 설정](https://github.com/pollen-robotics/microduck_rl/blob/cb70b792312d559a4da09064d92009079671815f/src/mjlab_microduck/robot/microduck_constants.py)
- [레퍼런스 ONNX 내보내기](https://github.com/pollen-robotics/microduck_rl/blob/cb70b792312d559a4da09064d92009079671815f/src/mjlab_microduck/export.py)
- [레퍼런스 실기·시뮬레이션 비교](https://github.com/pollen-robotics/microduck_rl/blob/cb70b792312d559a4da09064d92009079671815f/scripts/testbench_sim2real.py)

Piper의 25Hz 제어와 손끝 목표를 IK로 관절 목표로 변환하는 구조는 작업에 따른
설계 선택이다. 레퍼런스의 50Hz·관절 목표 출력과 똑같이 바꿀 필요는 없으며,
학습과 실제 제어의 주기·단위·좌표계·명령 처리 방식을 맞추는 것이 중요하다.

현재 Piper 정책이 사용하는 물체 위치·자세·속도·접촉 정보는 실기에서도 측정하거나
추정할 수 있어야 한다. 확보하기 어려운 정보는 정책 입력을 재설계하거나,
학습용 critic에만 제공하는 구성을 검토할 수 있다.

## Piper 학습에서 IMU의 역할

현재처럼 베이스가 고정된 Piper로 집기·옮기기를 수행한다면 IMU의 우선순위는
낮다고 판단한다. 이는 성능 실험 결과가 아니라 작업 구조에 따른 판단이다.
Microduck은 몸체의 기울기와 회전 상태가 균형 유지에 중요하지만, 고정형 Piper는
관절 각도와 로봇 모델로 베이스 기준 손끝 위치·자세를 계산할 수 있다.

| 상황 | 예상되는 IMU의 역할 |
| --- | --- |
| 고정된 베이스에 부착 | 움직임이 적어 정책에 추가되는 정보가 제한적 |
| 손목에 부착 | 빠른 움직임에서 진동·충격·회전 응답을 측정하는 보조 정보 |
| 이동로봇 위에 Piper 설치 | 베이스의 흔들림·기울기를 추정하는 데 유용 |
| real2sim 모델 보정 | 실제 진동·동적 응답과 시뮬레이션을 비교하는 보조 측정 |

IMU의 가속도·각속도 측정은 물체 위치나 잡힘 상태를 직접 제공하지 않는다.
현재 작업에서는 다음 정보의 확보를 우선 검토한다.

1. 관절 위치·속도와 명령·피드백 타임스탬프: 제어 응답과 지연 보정.
2. 카메라로 추정한 물체 위치·자세: 실기에서의 작업 관측 확보.
3. 그리퍼 벌어짐과 전류·힘 관련 피드백: 잡기·놓기 상태 판단.

IMU를 추가한다면 장착 위치와 센서 오차를 시뮬레이션에도 반영하고, IMU를 사용하지
않는 정책과 성공률·진동 등을 비교해 실제 이득을 확인한다. 정책 입력으로 사용하기
전에 모델 보정용 측정 장비로 활용하는 것도 가능하다.

참고: [Piper SDK의 관절·그리퍼 피드백 및 순기구학 API](https://github.com/agilexrobotics/piper_sdk/blob/master/asserts/V2/INTERFACE_V2.MD),
[MuJoCo 가속도 센서 정의](https://mujoco.readthedocs.io/en/3.3.7/XMLreference.html#sensor-accelerometer).

## BAM 모델과 모터 피드백의 차이

BAM은 **Better Actuator Models**의 약자로, Rhoban이 개발한 액추에이터 모델링·
파라미터 추정 도구다. 실제 모터를 측정해 시뮬레이션용 모델을 만들고, 일부 모터에
대해서는 이미 추정된 파라미터를 제공한다. 모터가 실시간으로 보내는 피드백이나
모터 제조사가 기본 제공하는 기능과는 구분해야 한다.
[BAM 공식 설명](https://github.com/Rhoban/bam)

| 구분 | 의미 |
| --- | --- |
| 모터의 피드백 | 지금 관절의 위치·속도·전류 등 측정값 |
| BAM 모델 | 명령과 부하가 주어지면 모터가 어떻게 움직일지 예측하는 모델 |
| 모터별 파라미터 | 실제 움직임을 재현하도록 실험으로 맞춘 마찰·관성 등의 수치 |

레퍼런스는 XL330에 대해 미리 측정·추정된 BAM 파라미터를 사용한다.
설정의 `motor_name="xl330"`은 대상 모터를, `model="m6"`는 마찰 모델의 종류를
선택한다. M6는 모터 제품명이 아니다.

BAM의 파라미터 추정은 실제 궤적과 시뮬레이션 궤적의 오차를 줄이는 방식이다.
공식 절차는 여러 부하·제어 조건에서 데이터를 기록하고, 파라미터를 최적화한 뒤
별도로 남긴 데이터로 검증하도록 안내한다.
[데이터 수집](https://bam.readthedocs.io/en/latest/identification/acquisition.html),
[파라미터 추정 과정](https://bam.readthedocs.io/en/latest/identification/fitting.html)

## Piper에 적용하는 방향

2026-09-21 확인한 [BAM 공식 공개 목록](https://bam.readthedocs.io/en/latest/usage/actuators.html)에서는
Piper용으로 바로 사용할 수 있는 모델을 찾지 못했다. 이는 공개된 준비 모델을
확인하지 못했다는 의미이며, Piper에 대한 모델을 만들 수 없다는 의미는 아니다.

Piper에는 위치·속도·전류 피드백과 제어 인터페이스가 있으므로, 명령 대비 실제
관절 응답을 기록할 기반은 있다. SDK의 `effort`는 전류에서 계수로 환산한 값이므로
독립적인 토크 센서의 실측값과 구분해야 한다.
[Piper SDK 피드백 정의](https://github.com/agilexrobotics/piper_sdk/blob/master/asserts/V2/INTERFACE_V2.MD#getarmhighspdinfomsgs)

우리 프로젝트에서는 다음 순서로 진행할 수 있다.

1. 사용할 Piper 제어 모드와 펌웨어 버전, 제어 주기, 부하 조건을 기록한다.
2. 목표 명령과 실제 관절 위치·속도, 가능한 전류 피드백을 타임스탬프와 함께 수집한다.
3. 동일한 명령을 시뮬레이션에서 재생하고 관절별 궤적 오차를 비교한다.
4. 지연·제어 게인·마찰 등 선택한 모델의 파라미터를 맞춘다. 링크 질량·관성·중력
   보상 설정도 함께 점검해 오차의 원인을 구분한다.
5. 보정에 사용하지 않은 궤적과 부하 조건으로 검증하고, 남은 불확실성을 학습의
   물리 조건 랜덤화 범위에 반영한다.

Piper에서는 모터뿐 아니라 감속기와 내부 제어기의 영향도 포함한 관절 응답을
맞춰야 한다. XL330의 BAM 수치를 그대로 복사해서 사용하면 안 된다.
BAM 자체에 연결하려면 Piper 제어 방식에 맞는 액추에이터 구현과 데이터 수집
어댑터가 필요하다. 먼저 현재 시뮬레이터의 관절 응답을 보정하고, 검증 결과에 따라
더 상세한 마찰 모델이나 BAM 통합을 검토할 수 있다.
[BAM의 새 액추에이터 지원 방법](https://bam.readthedocs.io/en/latest/identification/acquisition.html#recording)

이 문서는 설계 차이와 진행 방향을 정리한 것으로, Piper 실기와의 일치도나
학습 정책의 실물 성공을 검증한 결과는 아니다.
