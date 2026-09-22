# piper_sim2real

## 처음 내려받기

Piper 모델은 `third_party/mujoco_menagerie` 서브모듈의 고정된 커밋을 사용합니다.
다른 로봇의 대용량 모델을 모두 받지 않도록 처음에는 Piper 경로만 선택합니다.

```bash
git clone https://github.com/jh79783/piper_sim2real.git
cd piper_sim2real
git clone --filter=blob:none --no-checkout https://github.com/google-deepmind/mujoco_menagerie.git third_party/mujoco_menagerie
git -C third_party/mujoco_menagerie sparse-checkout set agilex_piper
git submodule update --init third_party/mujoco_menagerie
```

이후 모델 버전은 `git submodule update --init third_party/mujoco_menagerie`로
현재 프로젝트 커밋에 맞출 수 있습니다. `reference/`, `runs/`, `.cache/`는 로컬에만
유지하며 Git에 포함하지 않습니다.

[Piper real2sim 비교와 IMU·BAM 검토](docs/piper_real2sim_reference_comparison.md)에서
초기 구현과 현재 구성, 실기 적용 시 필요한 작업을 확인할 수 있습니다.
[관절 command/feedback trace와 offline replay](docs/piper_real2sim_reference_comparison.md#관절-명령피드백-trace와-오프라인-replay)는
실기 명령을 보내지 않고 MOVEJ 기록을 검증하는 절차를 설명합니다.
[학습 robustness 설정](docs/piper_training_robustness.md)에는 50Hz 시간 기준,
물리·센서 randomization, curriculum stage와 checkpoint 조건을 정리했습니다.

## Docker 준비 및 이미지 빌드

학습은 Docker 컨테이너에서 실행합니다. Linux Docker Engine에서는 NVIDIA
드라이버와 [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)이
필요하며, Docker Desktop for Windows에서는
[WSL 2 GPU 지원](https://docs.docker.com/desktop/features/gpu/)을 설정하세요.
현재 사용자로 `docker info`가 성공해야 하며, 다음 명령으로 컨테이너의 GPU 접근을 확인합니다.

```bash
docker run --rm --gpus all ubuntu:24.04 nvidia-smi
```

프로젝트 루트에서 아래 순서대로 빌드합니다. `Dockerfile`은 legacy Reacher용
`piper-rl:gpu`를 만들고, `Dockerfile.rsl`은 이를 기반으로 Piper용
`piper-rsl:gpu`를 만듭니다. 기본 Docker builder를 사용해 같은 Docker Engine에
이미지를 빌드하세요.

```bash
docker build -t piper-rl:gpu -f Dockerfile .
docker build -t piper-rsl:gpu -f Dockerfile.rsl .

# RGB 정책을 사용할 때만 추가 빌드
docker build -t piper-rsl:vision -f Dockerfile.vision .
```

GPU 학습 스크립트는 `--gpus all`로 GPU를 연결하고 호스트 UID/GID로 실행하므로
`runs/`와 `.cache/`에 생성한 파일도 현재 사용자 소유입니다. 컨테이너의
`HOME`은 쓰기 가능한 `/tmp`이고, 학습 캐시는 프로젝트의 `.cache/`에 보존합니다.
학습 코드와 모델은 `/workspace`에 마운트하므로 코드 변경 시 재빌드는 필요하지 않습니다.
`PIPER_RSL_IMAGE`와 `PIPER_RL_IMAGE`로 사용할 이미지 태그를 바꿀 수 있습니다.

미리보기 창은 호스트의 `DISPLAY`와 해당 X11 socket을 사용합니다. Docker
컨테이너에는 선택한 display cookie만 담은 임시 read-only Xauthority 파일을
마운트하므로 `xhost+`가 필요하지 않습니다. WSLg `:0`과 native/Xrdp display
`:10` 같은 local display를 지원하며, 화면이 없으면 `--headless`를 사용하세요.

## Piper pick-and-place (기본 실행)

기본 학습 과제는 MuJoCo Warp GPU 물리와 `rsl_rl` 5.0.1 PPO를 사용하는
Piper arm pick-and-place입니다. Docker 이미지 `piper-rsl:gpu`에 프로젝트를
마운트해 실행합니다.
Actor와 critic MLP의 기본 activation은 `tanh`이며, 이전 ELU 설정으로 만든
checkpoint는 호환되지 않으므로 새 학습을 시작해야 합니다. 현재 기본 경로는
물리 `0.002 s × 10`의 50Hz, 600 policy tick (12초)이며, 초기 25Hz/300-tick
Piper checkpoint와 schema 3 이전 checkpoint는 호환되지 않습니다. 정책은 7차원
normalized incremental joint/gripper target command를 내고, 63차원 state/action-history를
사용합니다. actor에는 noisy/delayed sensor state, critic에는 clean observation을
제공하며 둘 다 empirical normalization을 켭니다. 스크립트 변경만으로 적용되므로
이미지 재빌드는 필요하지 않습니다.

Incremental action에서 zero는 현재 actuator target을 유지하는 hold command입니다.
처음 6개 값은 기존 joint target rate (`0.035 rad/tick`)의 배수이고 마지막 값은
기존 gripper rate (`0.004 m/tick`)의 배수입니다. 적용 target은 기존 joint limit와
gripper limit 안에서 clamp됩니다. SDK/저수준 제어기에 absolute target을 보내는
변환은 학습 정책 contract 바깥의 별도 adapter가 담당해야 합니다.

```bash
cd /home/mjung11/workspace/piper_sim2real
bash scripts/run_piper_pick_place.sh
```

한 명령으로 CUDA GPU Warp 환경 128개, 하나의 PPO 정책, 그리고 실제 학습
환경 ID 0–3을 합친 하나의 2×2 preview 창이 시작됩니다. TensorBoard도 같은
컨테이너의 소유 child로 시작하며 주소는 기본적으로
[http://localhost:6006](http://localhost:6006)입니다. 호스트
`127.0.0.1`에만 바인딩되고, 포트 충돌 시 기존 프로세스를 건드리지 않고
명확한 오류로 종료합니다.

```bash
# 화면 없이 GPU 학습
bash scripts/run_piper_pick_place.sh --headless

# TensorBoard 끄기 또는 다른 loopback 포트 사용
bash scripts/run_piper_pick_place.sh --no-tensorboard
bash scripts/run_piper_pick_place.sh --tensorboard-port 6007

# bounded smoke: 64 environments × 10 steps/env = 640 transitions/env
# 600-tick episode timeout 경계까지 확인
bash scripts/run_piper_pick_place.sh --headless --num-envs 64 --steps-per-env 64 --iterations 10 \
    --tensorboard-port 16006

# 호환 가능한 schema-3 RSL-RL .pt 재개 / fixed home pose 변형
bash scripts/run_piper_pick_place.sh --resume runs/piper_pick_place/<run>/model.pt
bash scripts/run_piper_pick_place.sh --start-mode home

# uncertainty/curriculum controls
bash scripts/run_piper_pick_place.sh --no-domain-randomization
bash scripts/run_piper_pick_place.sh --no-sensor-noise
bash scripts/run_piper_pick_place.sh --no-curriculum
```

기본 실행은 curriculum, domain randomization, sensor noise를 켭니다. `--no-curriculum`은
최종 난이도를 바로 사용하며, `--start-mode above_cube`와 `--start-mode home`으로
시작 pose를 선택할 수 있습니다. 기본 `above_cube` 시작은 탐색을 돕기 위해 TCP를 cube 근처에 둡니다. 이는
이미 풀렸다는 뜻이 아니며 `home`은 접근과 grasp부터 학습해야 합니다. Cube
XY와 직사각형 goal XY는 매 reset마다 독립적으로 randomize됩니다. 성공은
reward가 높거나 cube에 닿은 것이 아니라, 실제 finger 접촉 grasp, lift,
goal 내부 배치, gripper release, 안정 정지를 모두 만족해야 합니다. reset은
초기 배치를 설정하지만 action step 중 cube를 weld하거나 강제로 teleport하지
않습니다. 짧은 smoke 또는 shaped reward만으로 수렴,
일반화, 실물 성공을 주장할 수 없으므로 여러 seed의 `task/success_rate`를
확인하세요.

Observation은 63차원이며, 앞의 56개 state field 뒤에 직전 7차원 raw action이
붙습니다. 마지막 state `has_placed` bit는 처음에는 0입니다.
실제 grasp/lift 뒤 cube가 goal 안에 있고 table 위에 있으며 release되고,
그 control step에 robot-cube contact가 없을 때 latch가 1이 되어 episode 끝까지
유지됩니다. 그 뒤 cube가 goal 밖으로 움직이더라도 latch는 유지되며, 이후
contact가 감지된 control step에만 0.2의 `recontact_penalty`를 적용합니다.
따라서 접촉이 없는 남은 step 전체에 매번 penalty를 주는 것은 아닙니다.
이는 full-task `task/success_rate` 정의를 바꾸지 않는 reward-v3 동작입니다. 성공 안정
카운트는 lift, goal 내부, table 위, gripper release, robot-cube 비접촉을 모두 만족한
control tick에서만 시작하고, 이후 16개의 연속된 비접촉·정지 tick이 필요합니다.
`task/recontact_rate`와 episode의 양의 누적 비용인 `task/recontact_penalty`를
함께 확인하세요. 현재 계수는 검증용 설정이며 최적 benchmark라고 주장하지 않습니다.
기존 reward-v1/v2 또는 58차원 run의 reward/return은 새 run과 직접 비교하지 말고,
`--resume` 없이 새 학습을 시작하세요.

첫 실행은 Warp CUDA kernel compile으로 잠시 지연될 수 있으며 cache는
`.cache/warp`에 보존됩니다. 학습 물리와 PPO는 GPU에서 실행되며, episode
초기화용 IK helper는 CPU를 사용할 수 있습니다. RGB 정책의 GUI 실행은
GPU EGL 정책 renderer를 유지하고, 작은 child process가 GLFW preview 창만
담당합니다. `PIPER_RENDERER=gpu`는 native/WSLg accelerated preview를
사용하고, `PIPER_RENDERER=software`는 preview child만 software GLFW로
실행합니다. State-only GUI는 하나의 GLFW backend를 사용합니다. 필요한
GPU backend가 없으면 wrapper가 조용히 CPU로 전환하지 않고 오류를 냅니다.
`--headless`는 preview와 X11 checks를 건너뛰고 RGB 정책을 EGL로 실행합니다.
rollout physics를 CPU에서 별도로 step하지 않습니다. Esc, 창 닫기, Ctrl+C는
checkpoint, summary, 환경, TensorBoard child를 정리합니다. `--device cpu`는 지원하지 않으며
CUDA가 없으면 hard error입니다. 이 코드는 simulation 검증용이고 실제
Piper/CAN/motor 명령을 보내지 않습니다.

RGB actor observation contract v2는 55차원 numeric state와 frozen 4096차원
vision feature를 사용합니다. Base state의 cube XYZ와 이를 직접 재구성하는
cube-minus-TCP/goal-minus-cube 세 그룹만 actor에서 제외하고, orientation·velocity·
contact·lift·placement·time·command·previous action fields는 유지합니다. retained
fields는 설계상 간접적인 object cue를 제공할 수 있습니다. 따라서
RGB actor 입력은 4151차원이고 critic은 기존 clean 63차원을 유지합니다. v1 RGB
PPO checkpoint는 새 actor observation과 호환되지 않아 새 학습이 필요하며,
standalone frozen encoder는 재사용할 수 있습니다.

기본 설정은 `10,000 × 128 × 64 ≈ 81.9M` transitions이므로 긴 학습입니다.
먼저 위의 bounded smoke 명령으로 설치와 reset/metric 경로를 확인하세요.

실행 결과는 `runs/piper_pick_place/<timestamp>/`에 저장됩니다: `model.pt`,
`model_<iteration>.pt`, `tensorboard/`, `config.json`, `summary.json`, 그리고
GUI 사용 시 `preview.png`. TensorBoard에서 `task/success_rate` (full-task),
`task/lift_rate`, `task/final_goal_distance`, `task/recontact_rate`,
`task/recontact_penalty`, `Train/mean_reward`를 보세요.
Reward만으로 과제 성공을 판단하지 마세요.

종료 후 로그를 보려면 standalone helper를 사용합니다. 역시 loopback only이며
임의의 서버에 붙거나 다른 Docker 컨테이너를 정리하지 않습니다.

```bash
bash scripts/run_tensorboard.sh
bash scripts/run_tensorboard.sh 6007
```

## Legacy Reacher 확인용

아래 기존 경로는 **SB3/Gymnasium Reacher-v5** 검증용으로 남아 있습니다.
`run_piper_pick_place.sh`의 기본 Piper 경로와 달리 CPU MuJoCo worker와 SB3 PPO를
사용하며 Warp GPU task가 아닙니다.
MuJoCo 환경 4개는 CPU의 별도 프로세스에서 실행합니다. 필요하면 legacy
스크립트의 `--device cpu` 또는 `--device cuda`를 선택할 수 있습니다.

WSL Ubuntu에서 실행하세요. 스크립트는 프로젝트를 마운트하므로 이미지 재빌드는 필요 없습니다.

```bash
cd /home/mjung11/workspace/piper_sim2real
bash scripts/run_reacher_parallel.sh
```

- 창 하나에 실제 학습 환경 4개를 위쪽 왼쪽부터 순서대로 표시합니다.
- 표시용으로 별도 에이전트를 실행하는 것이 아니라, 학습 환경의 현재 RGB 화면을 가져옵니다.
- 창 제목에 전체 환경의 누적 스텝과 학습 장치를 표시합니다.
- **Esc, 창 닫기, 터미널 Ctrl+C**로 중단하면 현재 모델을 저장하고 종료합니다.
- 각 환경에서 256스텝씩 모은 뒤 PPO를 업데이트합니다. 기본 4개 환경에서는 첫 업데이트까지 1,024스텝입니다.
- `--steps`는 전체 환경 합계이며, PPO는 완전한 rollout을 수집하므로 지정값보다 조금 더 실행될 수 있습니다.
- 40,000스텝은 수렴 보장이 아닌 테스트 설정입니다.
- 화면은 기본 최대 15 FPS로 갱신하며, 모든 물리 스텝을 실시간 속도로 재생하는 것은 아닙니다.
- 시각화에도 비용이 들므로 환경 수만큼 학습 속도가 빨라지는 것은 아닙니다.
- `LIBGL_ALWAYS_SOFTWARE=1`은 WSLg 렌더링 설정입니다. PyTorch 학습은 CUDA를 사용합니다.
- 작은 MLP 기반 PPO는 GPU 사용률이 낮거나 CPU보다 느릴 수 있습니다. SB3의 관련 경고는 오류가 아닙니다.

```bash
# 짧은 확인용 실행. 2,000스텝은 환경 4개를 합친 값입니다.
bash scripts/run_reacher_parallel.sh --steps 2000

# 더 오래 학습하면서 화면 갱신 부담 줄이기
bash scripts/run_reacher_parallel.sh --steps 200000 --fps 5

# 화면 없이 학습하기
bash scripts/run_reacher_parallel.sh --headless --steps 200000

# CPU만 사용하는 진단 실행
bash scripts/run_reacher_parallel.sh --headless --device cpu --steps 2000

# 전체 옵션
bash scripts/run_reacher_parallel.sh --help
```

`--n-envs`의 기본값은 4입니다. 창을 사용할 때는 최대 4개이며 빈 칸은 검정으로 표시합니다.
다중 프로세스 실행을 위해 Python 파일과 `spawn`을 사용하므로 코드를 `python -`로 실행하지 마세요.

각 실행은 `runs/reacher_parallel/<실행시각>/`에 별도로 저장되며 이전 실행을 덮어쓰지 않습니다.

- `model.zip`: 마지막 PPO 정책 및 최적화기 상태. 기존 SAC 모델과는 호환되지 않습니다.
- `best/best_model.zip`: 별도 평가 환경에서 평균 보상이 가장 높았던 PPO 모델
- `tensorboard/`: 학습 지표
- `evaluation/evaluations.npz`: 평가 결과. 학습 전 기준값과 기본 5,000스텝마다 5개 에피소드의 보상
- `config.json`, `summary.json`: 실행 설정, 종료 상태, 누적 스텝 및 최적화 epoch 수
- `preview.png`: 마지막으로 표시한 2×2 원본 화면. 화면을 켠 실행에서만 저장됩니다.

현재 창은 관찰용 합성 화면이므로 MuJoCo viewer의 마우스 물체 조작 기능은 제공하지 않습니다.
WSLg 창이 다시 작업표시줄에만 나타나면 작업을 저장한 뒤 Windows PowerShell에서
`wsl --shutdown`으로 WSL을 종료하고 Ubuntu를 다시 열어보세요. 이 명령은 다른 WSL 작업도 종료합니다.

Reacher의 SB3 `.zip`과 Piper의 RSL-RL `.pt` checkpoint는 서로 호환되지
않습니다. Piper를 시작할 때 SB3 모델을 `--resume`하지 말고 새 RSL-RL
학습을 시작하세요.

## Legacy Reacher 학습 그래프 보기

Piper 기본 실행은 TensorBoard를 자동으로 시작하므로 별도 터미널이 필요하지 않습니다.
아래 standalone 명령은 **legacy Reacher 실행을 종료한 뒤** 그래프를 볼 때만
새 WSL 터미널에서 실행하세요.

```bash
cd /home/mjung11/workspace/piper_sim2real
bash scripts/run_tensorboard.sh
```

Windows 브라우저에서 <http://localhost:6006>을 열고 Scalars 탭과 현재 실행을 선택합니다.
이 서버는 호스트의 localhost에만 노출되며, GPU는 사용하지 않습니다.
포트가 사용 중이면 `bash scripts/run_tensorboard.sh 6007`을 실행하고 6007로 접속하세요.

| 그래프 | 해석 |
| --- | --- |
| `eval/mean_reward` | 가장 먼저 볼 지표. 학습과 분리한 환경에서 탐색 잡음 없이 평가한 평균 보상 |
| `rollout/ep_rew_mean` | 학습 중 에피소드 보상의 이동 평균 |
| `train/value_loss` | 가치 예측 오차. 이것만 감소한다고 과제를 잘하는 것은 아님 |
| `train/approx_kl`, `train/clip_fraction` | PPO 업데이트 크기와 클리핑 정도를 보는 보조 지표 |

현재 Reacher의 보상은 음수이므로 **덜 음수가 되는 방향**이 좋습니다.
각 평가에서는 같은 별도 seed로 초기화하므로 체크포인트 간 비교가 가능합니다.
다만 5개 평가 에피소드는 제한적이며 다른 seed에서도 확인해야 일반화 성능을 판단할 수 있습니다.
학습 보상은 기본 약 1,024스텝 이후부터, 평가 보상은 시작 시점과 기본 5,000스텝마다 기록됩니다.
그래프가 비어 있으면 몇 초 기다려 새로고침하고 현재 실행의 체크박스를 확인하세요.

구현 참고: [SB3 SubprocVecEnv](https://stable-baselines3.readthedocs.io/en/v2.9.0/guide/vec_envs.html#subprocvecenv),
[Gymnasium Reacher](https://gymnasium.farama.org/environments/mujoco/reacher/).
PPO 및 그래프 참고: [SB3 PPO](https://stable-baselines3.readthedocs.io/en/v2.9.0/modules/ppo.html),
[TensorBoard](https://stable-baselines3.readthedocs.io/en/v2.9.0/guide/tensorboard.html).
