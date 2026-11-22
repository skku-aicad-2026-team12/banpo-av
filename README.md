# 잠수교 보행화 전·후, 반포대교 남단을 지나는 자율주행차 (banpo-av)

성균관대 2026-2 **자율주행인공지능및제어** 12조 테슬라팀 프로젝트.
팀원: 김재령 · 배준서 · 윤동건 · 정상현 (가나다순)
서울 반포대교 남단 ~ 신반포사거리를 SUMO로 실제 규모대로 구현하고, 잠수교 차량 통제(2026-10-26) **전·후** 두 교통 상황에서 자율주행차(AV)의 의사결정 모델(BC → 선생님 보정 PPO)을 학습 · 평가하고, 주행 데이터를 수집했다.

- **프로젝트 페이지** · <https://skku-aicad-2026-team12.github.io/banpo-av/> (준비 중)
- **데이터셋 (Hugging Face)** · <https://huggingface.co/datasets/skku-aicad-2026-team12/banpo-av-dataset>
- **주행 영상 5편** (YouTube) · 통제 전 [고가 → 잠수교](https://youtu.be/5h2dk-oDcSA) · [올림픽대로 → 잠수교](https://youtu.be/hS5VRAD5RiQ) / 통제 후 [측도 → 반포대교](https://youtu.be/WhpvJpyW3jg) · [고가 → 반포대교](https://youtu.be/KPofBx5-zJc) · [신반포로 → 반포대교](https://youtu.be/WEecXe-vcdk)

![전체 도로망](environment/road/images/01_전체.png)

## 결과

최종 모델 `results/team_rl_1120_U5/model_261632.pt` — 무작위 출발 차로, 학습에 쓰지 않은 새 시드 3,000판 (5개 경로 × 600판, 그중 판 600~799는 모델 비교에도 쓰지 않은 확정 평가).

| 지표 | 값 |
|---|---|
| 도착률 | **98.8%** (2,957 / 2,994, SUMO 차량 삽입 오류 6판 제외) |
| 충돌률 | **0.6%** (충돌 18판) |
| 그 밖의 실패 | 신호 위반 10 · 비정상 정지 4 · 시간초과 5 |

경로별 (확정 평가 1,000판, 판 번호 600~799 — [`evaluation/eval/final_U5_261632/report.md`](evaluation/eval/final_U5_261632/report.md)):

| 시나리오 | 경로 | 도착 |
|---|---|---|
| 통제 전 | `goga-jamsu` 고가 → 잠수교 | 100% |
| 통제 전 | `oly-jamsu` 올림픽대로 → 잠수교 | 94% (충돌 8 · 정지 2 · 시간초과 1) |
| 통제 후 | `side-banpo` 측도 → 반포대교 | 100% (시간초과 1) |
| 통제 후 | `goga-banpo` 고가 → 반포대교 | 100% |
| 통제 후 | `sbr-banpo` 신반포로 → 반포대교 | 100% |

## 무엇을 만들었나

| 항목 | 내용 |
|---|---|
| 도로 | 반포대로 북행 축, 강남터미널고가 · 신반포사거리 · 올림픽대로 · 잠수교 지하차도 · 회전교차로 · 반포대교 상층 (고가 / 지상 / 둔치 3층 입체 접속부). 합류 · 분기 · 차로 감소 · 신호 사거리 · 회전교차로 · 엮임 · 버스전용차로 |
| 교통 | 실측 기반 — 수요 TOPIS(통제 전) · t-data(통제 후), 신호 5곳 C-ITS 신호 기록, 공식 버스 34개 노선, 보행자 (원자료와 출처: [`environment/road/measured/`](environment/road/measured/)) |
| 시나리오 | 도로망은 하나, 통제 전 · 후는 수요 · 버스 경로 · 잠수교 통행 여부만 다름 |
| AV 관측 | 49차원 = 수업 기본 31 + 경로 4(분기 2개의 방향 · 거리) + 내 차로 진행 방향 4 + 신호 3(색 · 정지선 거리 · 남은 시간) + 경합 차량 4 + 차로 제한속도 3 |
| AV 학습 | 규칙 정책 선생님 → 시범 수집 · BC · DAgger → 선생님 보정 PPO (PPO 업데이트마다 rollout 관측에 선생님 정답으로 BC를 한 번 더) |
| 데이터 | 수업 기본 칼럼 + 추가 칼럼 (`sim_time` · `ego_route` · `signal_tls` · `signal_state` · `signal_remaining_s` · `end_reason`), 10,558,595행 · 26,000판 |

만든 과정 · 결정 · 문제 해결 기록은 [`docs/프로젝트_기록.md`](docs/프로젝트_기록.md), 교통 실측 원자료는 [`environment/road/measured/`](environment/road/measured/), 수집 데이터 설명은 [Hugging Face 데이터셋 카드](https://huggingface.co/datasets/skku-aicad-2026-team12/banpo-av-dataset), 도로망 명령과 코드 안의 규칙은 [`environment/road/README.md`](environment/road/README.md).

## 폴더 구조

```
.
├── README.md
├── requirements.txt           Python 패키지 버전
├── repo_paths.py              폴더 경로를 한곳에서 정함 (모든 실행 스크립트가 맨 위에서 불러 씀)
├── environment/               [Environment] SUMO 환경
│   ├── road/                  반포 도로망 정의 · 생성 · 검증 (road_spec.py가 기준) + 스크린샷 images/ · 교통 실측 원자료 measured/
│   ├── env/                   Gym 환경 — 수업 코드 + 반포 연결 (sumo_env · mdp_config · road_config · road_builder)
│   └── view_road.py           배경 교통이 흐르는 도로를 sumo-gui로 보기
├── training/                  [Training] train_team.py (BC → 병렬 PPO · 선생님 보정), 수업 원본 PPO · BC 알고리즘
├── evaluation/                [Evaluation] 주행 지표 · 기준 지표 · 모델 평가 (baseline.py) — 보고서는 eval/<이름>/
├── data_collection/           [Data Collection] 수업 기본 칼럼 + 추가 칼럼 수집 (collect_pomdp_data.py)
├── results/team_rl_1120_U5/   최종 모델 (model_261632.pt)과 학습 당시 설정
└── docs/                      프로젝트 페이지 (index.html · assets/) · 프로젝트 기록 (프로젝트_기록.md)
```

`data/`(수집 데이터), `environment/road/videos/`(주행 영상), 실행할 때마다 생기는 도로망 `environment/env/sumo/banpo/`는 git에 없다 — 아래 명령으로 다시 만든다.

## 설치

1. **Python 3.14** (3.11에서도 동작 확인)와 **SUMO 1.27** (Windows는 설치할 때 `Set SUMO_HOME` 체크)
2. `python -m pip install -r requirements.txt` (`traci` · `sumolib` · `libsumo`는 설치한 SUMO 버전과 맞춘다. `libsumo`가 있으면 화면 없는 평가 · 수집이 훨씬 빠르다)
3. 주행 영상을 만들 때만 **ffmpeg** (Windows: `winget install Gyan.FFmpeg`)
4. 명령은 **저장소 루트**에서 실행한다. 한글 출력 때문에 `python -X utf8`로 실행한다.

## 실행

```bash
# 1. 도로망 생성 + 검증
python -X utf8 environment/road/build_net.py
python -X utf8 environment/road/check_net.py              # 마지막 줄 "결과: 통과"
python -X utf8 environment/road/check_env.py              # 수업 환경 연결 (AV 주행 + 배경 교통)

# 도로 보기 — 배경 교통이 흐르는 도로를 sumo-gui로 (▶ 누르면 시작)
python -X utf8 environment/view_road.py

# 2. 최종 모델 평가 — 5개 경로, 무작위 출발 차로 → evaluation/eval/<이름>/report.md · metrics.csv
#    --episodes는 경로당 횟수 (200이면 1,000판). 발표 수치는 --episode-start 100 / 400 / 600 세 묶음
python -X utf8 evaluation/baseline.py --model results/team_rl_1120_U5/model_261632.pt --episode-start 600 --episodes 200 --workers 7 --depart-lane random --depart-speed limit --safety off --name final_U5_261632

# 3. 데이터 수집 — 최종 모델, 안전장치 끔 → data/<out-dir>/
python -X utf8 data_collection/collect_pomdp_data.py --model results/team_rl_1120_U5/model_261632.pt --safety off --depart-lane random --depart-speed limit --episodes 1000 --workers 7 --seed 100000 --out-dir data/final --name final_100000_safetyoff

# 4. 주행 영상 (ffmpeg 필요) → environment/road/videos/
python -X utf8 environment/road/record_video.py --model results/team_rl_1120_U5/model_261632.pt --route goga-jamsu --episode 22 --depart-lane 0 --depart-speed limit --name goga-jamsu_ep22

# 기준 지표 — SUMO 기본 운전자, 경로 3개 · 경로 차로 출발 (evaluation/baseline_metrics.csv · baseline_report.md를 덮어쓴다)
python -X utf8 evaluation/baseline.py --episodes 30 --workers 7
```

저장소의 `evaluation/baseline_report.md` · `baseline_metrics.csv`는 11/16에 잰 옛 기준(경로 3개 · 경로 차로 출발)이다. 최종 모델 평가 보고서의 기준 줄은 5개 경로 · 무작위 출발로 잰 `evaluation/baseline_metrics_random.csv`를 쓴다 (출발 옵션이 무작위면 `baseline.py`가 자동으로 고름).

학습을 처음부터 다시 하려면 `training/train_team.py` (`bc` → `rl --init <BC 모델> --teacher-lr 2e-4`). 최종 모델의 학습 설정은 [`results/team_rl_1120_U5/args.json`](results/team_rl_1120_U5/args.json).

## 수업 코드와의 관계

수업 저장소 [bmil-ssu/Artificial-Intelligence-and-Control-for-Autonomous-Driving](https://github.com/bmil-ssu/Artificial-Intelligence-and-Control-for-Autonomous-Driving)의 코드를 Environment / Training / Evaluation / Data Collection으로 나눠 옮겼다.

| 수업 문서의 위치 | 이 저장소 |
|---|---|
| `env/` (road_config · road_builder · mdp_config · sumo_env) | `environment/env/` |
| `view_road.py` | `environment/view_road.py` |
| `train.py` · `algorithms/` · `utils/` | `training/` 아래 같은 이름 |
| `collect_pomdp_data.py` | `data_collection/collect_pomdp_data.py` |
| `test.py` · `train_bc.py` | 지움 — 평가는 `evaluation/baseline.py`, BC는 `training/train_team.py bc` |
| `env/sumo/highway.*` · 예제 데이터 · 예제 모델 | 지움 — 수업 예제 직선 도로 전용 |

`environment/env/` · `environment/view_road.py` · `data_collection/collect_pomdp_data.py` · `training/train.py`는 반포 도로망 연결 · 폴더 경로 · 추가 옵션 때문에 고쳐져 있다 (내역: [`environment/road/README.md`](environment/road/README.md) "수업 코드 연결"). 수업 데이터 기본 칼럼은 지우지 않고 추가만 했다.
