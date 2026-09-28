# piper_sim2real (Isaac Sim)

AgileX Piper 팔의 pick-and-place state 정책을 Isaac Sim 5.1 + Isaac Lab 2.3.2에서
Docker로 학습하는 저장소입니다. 이전 MuJoCo-Warp 구현은 `main` 브랜치에 남아 있고,
이 브랜치는 Isaac Sim만 사용합니다. 실기 연동은 아직 범위에 넣지 않았습니다.

> **현재 상태 (작업 중)**: USD 변환, torch FK/IK, 환경 코드 작성까지 끝났습니다.
> PhysX에서 Piper articulation이 첫 step에 발산(NaN)하는 문제를 조사하고 있으므로
> 아직 학습은 실행되지 않습니다.

## 환경 스펙

| 항목 | 값 |
| --- | --- |
| Base image | `nvcr.io/nvidia/isaac-lab:2.3.2` (Isaac Sim 5.1.0, Python 3.11) |
| 학습 라이브러리 | rsl-rl-lib 3.1.2, torch 2.7.0+cu128 |
| 검증한 호스트 | RTX 5090 32GB, 드라이버 580, Ubuntu 24.04, NVIDIA Container Toolkit |
| 로봇 모델 | `third_party/mujoco_menagerie/agilex_piper` (고정 커밋) → USD 변환 |

## 준비

```bash
git submodule update --init third_party/mujoco_menagerie   # Piper MJCF와 mesh
docker build -t piper-isaac:5.1 -f Dockerfile.isaac .
bash scripts/run_isaac.sh convert      # .cache/isaac/assets/piper/piper.usd 생성
bash scripts/run_isaac.sh test         # FK/IK, MJCF 변환 테스트
```

`scripts/run_isaac.sh`는 호스트 UID로 컨테이너를 실행하므로 `runs/`와 `.cache/`의
파일은 현재 사용자 소유입니다. Kit shader/cache는 `.cache/isaac/`에 보존됩니다.

## 실행

```bash
# 물리 검증: 정책 action API만으로 scripted pick-and-place 수행
bash scripts/run_isaac.sh python scripts/isaac/check_env.py --headless --num_envs 64

# headless PPO 학습 (로그: runs/isaac/piper_pick_place/<timestamp>)
bash scripts/run_isaac.sh train --headless --num_envs 4096
bash scripts/run_isaac.sh train --headless --no-curriculum --no-domain-randomization --no-sensor-noise
bash scripts/run_isaac.sh train --headless --resume runs/isaac/piper_pick_place/<run>/model_1000.pt

bash scripts/run_isaac.sh tensorboard --port 6006
```

## 정책 contract (MuJoCo 구현에서 유지)

- 제어 주기는 50Hz(20ms), episode는 600 tick(12초)입니다. PhysX는 5ms × 4 substep으로 계산합니다.
- Action은 7차원 normalized incremental target입니다. 관절 6개는 `0.035 rad/tick`,
  gripper는 `0.004 m/tick`의 배수이며, 0은 현재 target을 유지합니다.
- Observation은 63차원입니다. actor는 noise, 관절 영점 offset, 0–2 tick delay가 적용된
  값을 받고, critic은 clean 값을 받습니다. 필드 순서는 `isaac_piper/pick_place_env.py`의
  `OBS_*` slice가 정의합니다.
- 보상은 reward-v3(potential shaping, recontact penalty)이고, 성공은 grasp → lift →
  goal 배치 → release → 16 tick 비접촉 정지를 모두 만족해야 합니다.
- Domain randomization(질량, 관성, CoM, 마찰)과 curriculum 4단계는 이전 값과 같습니다.

## MuJoCo 구현과 다른 점

| 항목 | MuJoCo-Warp | Isaac Sim |
| --- | --- | --- |
| 충돌 형상 | mesh에 맞춘 capsule (compile 시 생성) | 같은 capsule을 MJCF에 명시해 변환 |
| gravcomp | `gravcomp="1"` | `disable_gravity=True` |
| finger 결합 | equality `joint8 = -joint7` | 두 finger drive에 `+g`/`-g` 명령 |
| 접촉 판정 | 접촉 거리 1mm 이하 | PhysX 법선력 0.01N 초과 |
| cube 각속도 | body frame | world frame |
| reset IK | CPU MuJoCo DLS | GPU torch DLS를 2mm 격자로 미리 풀고 보간 |
| action 분포 | tanh-squashed Gaussian | Gaussian + env에서 clamp |
| 테이블 마찰 randomization | 적용 | 미적용 (정적 collider, max 결합이라 cube 마찰이 지배) |

gain, 마찰, 질량 범위는 Menagerie 값과 이전 draft 값이며 Piper 실측값이 아닙니다.
실기 데이터로 real2sim 식별을 한 뒤에 교체해야 합니다.

## 구성

```
Dockerfile.isaac, requirements-isaac.txt   Isaac 이미지
isaac_piper/                               task 패키지 (Gym ID: Piper-PickPlace-Direct-v0)
  mjcf.py            Menagerie MJCF → importer용 MJCF
  kinematics.py      torch FK/IK
  robot_cfg.py       ArticulationCfg (drive gain, 한계)
  pick_place_env*.py DirectRLEnv와 설정
  agents/            RSL-RL PPO 설정
scripts/run_isaac.sh                       Docker 실행기
scripts/isaac/                             convert / train / check_env
tests/isaac/                               컨테이너 안에서 실행하는 테스트
```
