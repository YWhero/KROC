# Clone and reproduce on a new GPU server

See [Reproduction guide](docs/REPRODUCING.md). From a private clone at `/workspace/KROC`, run:

```bash
bash scripts/bootstrap.sh
```

The source repository excludes datasets, virtual environments, caches, credentials and checkpoints. It includes pinned downloads and conversion steps, the approved study annotations/splits, and the personal-PC checkpoint relay. This command prepares the environment/data and does not start training.

---

# KROC: ACT / ACT+EMG / ACT+EMG encoder

2026-11-15에 `180.189.55.41`의 `/workspace/KROC`에 설치하고,
원시 데이터 전체 변환과 세 모델의 실제 데이터 검증 학습을 완료했습니다.
각 모델 20 step(batch 2), 10 step 체크포인트에서 재개, 저장 전후 예측 일치를 확인했습니다.
전체 100000 step 학습은 아직 실행하지 않았습니다. 체크포인트 공개 업로드는 개인 PC 중계로 지원합니다.
자세한 결과는 [검증 보고서](reports/VALIDATION.md)에 있습니다.
기존 서버와 같은 ROBOTIS-GIT/lerobot-cyclo 커밋
`240b4a0314ae0879cdd928c7f4bdc1eee9a01b3b`를 `vendor/lerobot`에 체크아웃했습니다.
원본 ACT 소스는 수정하지 않습니다. 이 프로젝트의 구현은 `src/kroc_act`에 있습니다.

## 세 모델

| 설정 | 입력 | 구현 |
|---|---|---|
| `configs/01_baseline.json` | 카메라 + 로봇 state | upstream ACT |
| `configs/02_concat.json` | 카메라 + `[정규화한 로봇 state, 정규화한 EMG RMS]` | ACT state projection 입력 확장 |
| `configs/03_window.json` | 카메라 + 로봇 state + EMG 토큰 | 과거 EMG → Conv1d → pool → Linear → LayerNorm → ACT attention |

현재 로봇 프로필을 적용하면 ① state 6차원, ② state 10차원, action은 모두 6차원입니다.
차원과 카메라 키/크기는 실제 변환된 데이터 메타데이터에서 읽습니다.
실험은 영상 입력을 유지하며, ②의 EMG 특징은 우선 **과거 50ms RMS 4개**로 정했습니다.
원신호를 한 점씩 다운샘플링하는 실험과는 다릅니다. 비교 실험의 정의를 검토해 주세요.

③은 2,000Hz EMG의 과거 200ms `[400,4]`를 `[512]`로 인코딩합니다.
이를 ACT가 이미 지원하는 `observation.environment_state`에 넣습니다.
ACT 내부에서 별도의 projection/위치 임베딩을 가진 **독립된 EMG 토큰**이 되며,
로봇 state 및 카메라 토큰과 같은 Transformer encoder self-attention을 통과합니다.
인코더까지 행동 손실로 함께 학습됩니다. state 벡터에 붙이는 ②와 구조적으로 다릅니다.
시간 방향 Conv1d는 과거 윈도우 내부에서만 처리하므로 현재 이후 데이터는 보지 않습니다.

VAE posterior는 upstream 구조를 유지합니다. 따라서 ②의 확장된 state는 VAE에도 들어가고,
③의 EMG 토큰은 주 Transformer에만 들어갑니다. 논문 비교 시 이 차이를 명시해야 합니다.

## 공통 데이터와 공정한 비교

세 모델은 `data/lerobot_v3`와 `data/emg`를 함께 사용합니다. 영상/로봇 데이터 3벌을 만들지 않습니다.
EMG sidecar는 획득 시각, 사용 가능 시각, 유효 마스크, 채널 값을 보관합니다.
공통 로더가 각 학습 시점의 과거 윈도우와 RMS를 계산합니다.
기본 ACT에도 동일한 EMG 유효성 조건을 적용해 **세 실험의 사용 프레임을 맞춥니다**.
즉 baseline 모델 자체는 EMG를 받지 않지만, 데이터 선별은 EMG 모델과 같습니다.

- 모든 모델에 같은 `splits.json` 적용: 에피소드 단위 train/val, 프레임 무작위 분할 금지.
- EMG/로봇 시계 정렬 확인 전에는 학습을 거부합니다.
- 미래 EMG, 아직 도착하지 않은 EMG, 에피소드 경계 이전 EMG는 사용하지 않습니다.
- warmup/누락/지연 때문에 200ms 윈도우를 만들 수 없으면 그 프레임은 세 모델 모두 제외합니다.
- 정규화 통계는 선택된 train 에피소드에서만 계산합니다. validation 전체 통계를 사용하지 않습니다.
- 카메라는 RGB/ImageNet mean·std 정규화, action과 state는 train mean·std 정규화합니다.
- 윈도우는 현재까지 도착한 최신 유효 EMG의 획득 시각을 끝점으로 구성합니다.
  `max_gap_ms=2`는 윈도우 내부 누락 허용 한계이고, `max_emg_age_ms=100`은 최신 신호의 최대 지연입니다.
  아직 도착하지 않은 시점까지 마지막 값을 늘여서 200ms를 채우지 않습니다.

실제 원시 데이터 → 공통 데이터 변환의 책임 경계와 스키마는 [DATA_CONTRACT](docs/DATA_CONTRACT.md)를 참고하세요.
실제 MCAP/HDF5/영상 스키마를 감사한 변환기는 `src/kroc_act/convert_raw.py`입니다.
이번 원본의 단일 EMG segment 구조를 지원하며, 관절명 순서를 맞추고 30Hz causal 수신 기준으로 정렬합니다.
영상은 원본 비율을 유지한 320×180으로 변환했고, 원본 파일은 `raw/`에 보존했습니다.
`prepare.py`는 별도 검증된 중간 데이터가 있을 때 사용하는 대안 경로입니다.

## 로컬 검토

주요 코드는 [model.py](src/kroc_act/model.py), [data.py](src/kroc_act/data.py),
[emg.py](src/kroc_act/emg.py), [train.py](src/kroc_act/train.py)입니다.
설정 경로는 설정 파일 위치를 기준으로 해석하므로 실행 디렉터리와 무관합니다.

PowerShell에서 설정만 출력하기(학습/데이터 접근 없음):

```powershell
$env:PYTHONPATH = (Resolve-Path ./src).Path
python -m kroc_act.train --config configs/01_baseline.json --dry-run
python -m kroc_act.train --config configs/02_concat.json --dry-run
python -m kroc_act.train --config configs/03_window.json --dry-run
python -m unittest discover -s tests -v
```

로컬 Python 3.11 + CPU torch에서는 EMG 정렬과 인코더 단위 검증만 가능합니다.
세 ACT 모델의 통합 검증은 Python 3.12 + pinned LeRobot이 설치된 환경이 필요합니다.
서버에서는 전체 9개 테스트가 skip 없이 통과했습니다. 통합 테스트는 다운로드 없는 작은 ResNet18 ACT로
세 모델의 역전파, EMG 영향, checkpoint state 복원을 검사합니다.

## 서버에서 실행할 명령

`KROC` 전체를 서버의 새 폴더로 복사합니다. `vendor/lerobot`도 포함해야 합니다.
기존 전역 Python/가상환경 대신 이 폴더에 독립 `.venv`를 만듭니다.

```bash
bash scripts/setup_server.sh
# 현재 서버는 설치와 변환까지 완료되어 아래 학습 명령부터 사용할 수 있습니다.
bash scripts/train.sh --config configs/01_baseline.json
bash scripts/train.sh --config configs/02_concat.json
bash scripts/train.sh --config configs/03_window.json
```

각 명령은 **별도의 한 실험**을 실행합니다. 한 GPU에서는 순서대로 실행하세요.
기본값: batch 8, chunk 30, n_action_steps 15, 100000 steps, lr 1e-5, seed 1000, fp32, AdamW,
ResNet18 ImageNet 초기화, gradient clip 10. upstream ACT 모델/손실을 사용하지만
학습 루프는 세 모델을 지원하기 위한 이 프로젝트 코드이며 Cyclo 서비스 학습 루프 자체는 아닙니다.
validation은 전체 유효 프레임의 zero-latent L1과 동작별·에피소드별 지표를 측정합니다.
best checkpoint 선택을 지원합니다. early stopping과 실시간 로봇 평가는 아직 구현하지 않았습니다.

설정된 `output_dir`가 이미 있으면 새 학습은 덮어쓰지 않고 종료합니다.
checkpoint는 10000 step마다, 최고 validation 갱신 시, 종료 시 저장합니다.
개인 PC 중계가 Hugging Face 업로드의 크기·해시를 확인한 체크포인트만 서버에서 정리합니다.
한 개의 `checkpoint.pt`에 모델·EMG 인코더·정규화 통계·설정·optimizer·RNG가 함께 저장됩니다.
업스트림 `pretrained_model/` 형식과는 다르며 아래 어댑터로 로딩해야 합니다.

```bash
bash scripts/train.sh --config configs/03_window.json \
  --resume outputs/03_window/checkpoints/00010000/checkpoint.pt
```

resume는 모델/optimizer/RNG를 복원하지만 DataLoader는 새 epoch 경계에서 다시 시작합니다.
따라서 중단 시점과 샘플 순서를 비트 단위로 재현하는 resume는 아닙니다.
100 step 로그, checkpoint와 실행 설정/통계/입력 manifest 및 split 해시는 실험 출력 아래 남습니다.
학습 로그의 KL 포함 training loss와 zero-latent validation L1은 서로 다른 지표입니다.
검증용 `--stop-after 10`은 experiment의 목표 steps를 바꾸지 않고 중간에 저장/종료하므로,
같은 config로 `--resume`해 남은 구간을 실제로 실행할 수 있습니다.

## Cyclo 추론 연결

[inference.py](src/kroc_act/inference.py)의 `CycloACTAdapter`는 공통 checkpoint를 로딩하고,
원래 로봇 단위의 action을 반환합니다. 카메라 RGB CHW, state, 현재 시각과 EMGStream을 받습니다.
학습과 같은 sampler/RMS/정규화/인코더를 재사용합니다.

```python
from kroc_act.inference import CycloACTAdapter
policy = CycloACTAdapter('outputs/03_window/checkpoints/00100000/checkpoint.pt')
policy.reset()  # 태스크/에피소드 시작 시; 센서 버퍼도 함께 비울 것
action = policy.action(state, images, now_ns, stream=aligned_emg_stream)
```

**Cyclo의 기존 UI/엔진에 자동 등록된 상태는 아닙니다.** 모델 로더에 어댑터를 연결하고,
실시간 EMG 수집/같은 causal 필터/시계 변환/고정 길이 버퍼/센서 장애 시 hold·stop을 구현해야 합니다.
이 프로젝트는 로봇에 명령을 보내지 않습니다.
세 모델 모두 `chunk_size=30`, `n_action_steps=15`입니다.
30스텝의 행동을 예측하고 앞의 15스텝을 실행한 뒤, 새 관측으로 다시 예측합니다.
새 EMG는 재예측 시점에 반영되며, 정책 호출이 30Hz이면 재예측 간격은 약 0.5초입니다.


## Updated study and checkpoint workflow

See [2026-11-15 training updates](docs/TRAINING_UPDATES_20261115.md) for the
24/6 stratified split, linked collection annotations, full validation, best model
selection, token-free PC relay, verified cleanup and public checkpoint restore.
Keep the PC relay running before starting production training.
