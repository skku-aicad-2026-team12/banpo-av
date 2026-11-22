# environment/road/ — 반포대교 남단 ~ 신반포사거리 SUMO 도로망

수업 코드(`environment/env/`)가 쓰는 도로망을 만들고 검증하는 폴더다 (`environment/env/road_config.py`의 `ROAD["network"] = "banpo"`). 도로의 기준은 **`road_spec.py` 하나**다.
이 문서는 **사용법과 코드 안의 규칙**만 다룬다. 교통 실측 원자료와 출처는 [`measured/`](measured/), 설계 이유 · 검증 수치 · 문제 해결은 `docs/프로젝트_기록.md` (§3 도로 · §5 교통 · §7 AV 학습 · 부록 문제와 해결).

도로 · 교통 설정은 최종 모델 학습 · 데이터 수집에 쓴 그대로다. 고쳐야 하면 `road_spec.py` 수정 → 아래 1 → 2를 다시 통과시키고, 교통 설정을 바꿨으면 `check_long.py`도 돌린다.

## 명령

```bash
# 1. 도로망 생성 + 자동 검증 (길이 · 경로 · 차로 규칙 · 경로별 1대 주행)
python -X utf8 environment/road/build_net.py
python -X utf8 environment/road/check_net.py            # 주행 테스트 생략: --no-sim

# 2. 수업 환경(environment/env/) 검증 — AV 경로 주행 + 배경 교통 30분 (두 시나리오)
python -X utf8 environment/road/check_env.py
python -X utf8 environment/road/check_env.py --gui --episodes 1 --no-traffic-test   # AV 주행을 화면으로
python -X utf8 environment/road/check_env.py --scenario after                       # 통제 후만
python -X utf8 environment/road/check_env.py --policy rule [--safety]               # 관측만 보는 규칙 정책 = BC · DAgger · 보정 PPO의 정답을 만드는 선생님 (관측 · 신호 위반 판정 확인)
python -X utf8 environment/road/check_env.py --policy hold                          # AV를 정책처럼 직접 제어 (파이프라인 오류 확인용)
python -X utf8 environment/road/check_long.py                                       # 장시간 안정성: 배경 교통 4시간 × 시드 3 × 두 시나리오, 병렬 약 6~8분
                                                                        # (교착 · 서서히 자라는 대기열 · 투입 부족 — 교통 설정을 바꾸면 꼭 다시)

# 3. 교통 수요 · 신호 근거 검산 (실측 대비 측정 링크 교통량, 실측 신호 기준 포화도 · 신반포로 진입량)
python -X utf8 environment/road/calibrate.py

# 4. 배경 교통이 흐르는 도로를 sumo-gui로 (▶ 누르면 시작)
python -X utf8 environment/view_road.py

# 5. 주행 영상 (sumo-gui 화면을 매 스텝 저장 → ffmpeg mp4, environment/road/videos/)
python -X utf8 environment/road/record_video.py --policy sumo --route side-banpo --scenario after
#    --model <model.pt> · --episode N(평가와 같은 시드) · --speed · --zoom(기본 1500 ≈ 시야 100 m)

# 기타: 도로망 그림 / 도로망만 sumo-gui로
python -X utf8 environment/road/preview_net.py --scenario before
sumo-gui -n environment/road/build/before/banpo.net.xml
```

| 결과 | 위치 |
|---|---|
| 검증용 도로망 · 리포트 (`build_net.py` · `check_net.py`) | `environment/road/build/<before\|after>/` — `banpo.net.xml`, `check_report.txt`, `netconvert.log` 등. 언제든 지우고 다시 만들 수 있다 (git 제외) |
| 환경 검증 리포트 (`check_env.py` · `check_long.py` · `calibrate.py`) | `environment/road/build/check_env_report.txt` · `check_long_report.txt` · `calibrate_report.txt` |
| 수업 환경이 쓰는 도로망 · 수요 · 화면 (`environment/env/road_builder.py`가 실행 때마다 생성, git 제외) | `environment/env/sumo/banpo/` — `banpo.net.xml`(신호 포함), `banpo.tll.xml`, `banpo.before.rou.xml` · `banpo.after.rou.xml`(배경차 · 버스 · 보행자), `banpo.stops.xml`, `banpo.background.xml`, `banpo.layers.xml`, `banpo.gui.xml`, `banpo.sumocfg` |

## 파일

| 파일 | 역할 |
|---|---|
| `road_spec.py` | **도로 정의.** 좌표 상수, 노드, 엣지(차로 수 · 제한속도 · 형상 · 버스전용 · 차로 변경 금지), 차로 단위 연결, 층 높이 z, AV 경로(`AV_ROUTES` — 무작위 출발 차로 범위 `depart_lanes`: `oly-jamsu` 0~2번 · `sbr-banpo` 0 · 1번(2번 제외 — `docs/프로젝트_기록.md` 부록 #81), 나머지 경로는 전 차로), 시나리오, 진입점 · 출구, 확정안 길이, 실측 신호 5곳(`SIGNALS` — 옛 추정 `SIGNAL_PLAN`은 기록용), 보도(`SIDEWALKS`) · 보행자(`PEDESTRIANS`) |
| `build_net.py` | `road_spec.py` → XML → netconvert(`--default.disallow pedestrian` — 보행자는 보도 · 횡단보도로만) → 신호 프로그램 적용. `--scenario`로 하나만 |
| `check_net.py` | 도로망 자동 검증 |
| `check_env.py` | 수업 환경 연결 검증 (AV 경로 주행 · 관측 차원 · 값 범위 · 잠수교 통제 · `signal` 칼럼 · 배경 교통 30분) |
| `check_long.py` | 장시간 안정성 (배경 교통 4시간 × 시드 3 × 두 시나리오, 병렬) |
| `calibrate.py` | 교통 수요 · 신호 근거 계산 (실측 → 측정 링크 비교, 4절 실측 신호 기준 포화도 → 신반포로 진입량, 3절 Webster는 옛 추정 기록). 입력 `measured/` |
| `routing.py` | 차로 단위 경로 탐색 (check_net · calibrate · environment/env/road_builder 공용) |
| `record_video.py` | 주행 영상 녹화 |
| `preview_net.py` | 도로망 그림 |
| `images/` · `videos/` | 최종 도로망 스크린샷 · 녹화 영상 출력 폴더 |

## 수업 코드 연결

| 파일 | 바꾼 점 |
|---|---|
| `environment/env/road_config.py` | `ROAD`: `network="banpo"`, `scenario`(before/after/**mixed**), `av_routes`, vmax 22.22 m/s. `EGO`: `depart_lane="best"`, `sumo_driver`, `sumo_vtype`(SUMO 운전자 모드 AV 차종: accel 2.6 · decel 4.5 · sigma 0). `TRAFFIC`: 배경차 차선변경 켬, 프리필 끔, `warmup_s`, `demand_scale`(1.0), `od`(진입점 → 출구 교통량, "출구\|경유"로 경로 지정), `buses` · `bus_stops`, ACC · EIDM `apparent_decel`, 배경차 `impatience` 0.5 |
| `environment/env/road_builder.py` | `build()`가 항상 `_build_network()`(수업 원본 직선 도로 생성 코드는 11/16 지움): 도로망 하나 + 시나리오별 수요 파일(배경차 flow는 900 대/시 이하로 나눔, 투입 차로 `best`, 버스 flow · 정차, 보행자) + 버스 정류소 + 화면 표시 → `environment/env/sumo/banpo/banpo.sumocfg`. 수요 파일에 AV 차종 두 개(`ego_type` RL · `ego_sumo` SUMO 운전자). **덮어쓰기 방지**: 임시 폴더에 다 만든 뒤 내용이 다른 파일만 `os.replace`, 도로망 입력 지문(`banpo.net.stamp`)이 같으면 netconvert 생략, 학습 중에는 `.in_use_<PID>` 표시로 다른 설정 build를 막음 |
| `environment/env/sumo_env.py` — 반포 연결 | 에피소드마다 시나리오 · AV 경로 선택(`reset(options={"scenario": "after", "route": "goga-banpo"})`) · 통제 후 잠수교 통제(`apply_after_closure()`, 환경 · `check_long` 공용) · 배경차를 `warmup_s`초 먼저 흘림 · 교차로 내부 충돌 검사 · 종료 판정(충돌 · 도착 · 경로 이탈 · 비정상 정지 · 순간이동 · 신호 위반) → `info["end_reason"]` · AV 앞 신호 상태(`get_signal_state()`, 데이터 칼럼 `signal`) · SUMO 기본 운전자 모드 · **수집 최적화**(state 조회를 TraCI 구독으로, 화면 없을 때 libsumo — `SUMO_LIBSUMO=0`이면 기존 TraCI) |
| `environment/env/sumo_env.py` · `environment/env/mdp_config.py` — AV 학습 | 관측 49차원(11/19 `lane_dir_obs` — 끄면 기존 모델용 43차원, 칸 번호표 `sumo_env.obs_index`), 잘못된 차로 조기 종료(`wrong_lane_end_m` — 11/19 낮부터 0 = 끔. 켜면 차로 끝 30 m 안에서 `off_route`), 차로 제한속도 기준 속도 상한 · 보상, 신호 위반 종료 · 감점, 비정상 정지 벌점(`stuck_penalty` — 학습 값 20은 `training/train.py`), 안전장치 스위치 `ACTION["safety_brake"]`(기본 끔) — 상세 `docs/프로젝트_기록.md` §7 |
| `evaluation/` (12팀) | `metrics.py` 주행 지표(기존 칼럼만, 빨간불 대기는 `signal` 칼럼 + 정지선 거리 — 정지선 위치는 `environment/env/sumo/banpo/banpo.net.xml`에서 찾으므로 환경 파일이 먼저 있어야 함), `lanes.py` 차량 기준 차로 번호, `baseline.py` 기준 지표 · 모델 평가 (11/19~ `--route` · `--depart-lane` best · random · 차로 번호 · `--episode-start` · `--base-metrics`, `metrics.csv`에 `depart_lane` 칼럼 — 무작위 출발 기준 지표는 `baseline_metrics_random.csv`) |
| `data_collection/collect_pomdp_data.py` | 기존 칼럼 그대로 + 추가 칼럼 `sim_time` · `ego_route` · `signal_tls` · `signal_state` · `signal_remaining_s` · `end_reason` (JSONL · NPZ 모두 코드값, 11/19 `scenario` 뺌 — `ego_route`로 정해짐), `--policy sumo` · `--scenario` · `--route` · `--paired-seeds` · `--depart-lane random` · `--depart-speed limit`. reset 실패(AV 삽입 실패) 판은 건너뛰어 `episode` 번호가 빌 수 있다. 같은 폴더 `relabel_with_rule.py`(11/19, 12팀)는 기존 시범 NPZ의 정답을 지금 선생님으로 다시 매긴다 (SUMO 안 띄움) |
| `environment/view_road.py` | 도로망 파일 경로를 sumocfg 폴더 기준으로 |
| `training/train.py` | 폴더 경로만 (결과 `results/` · 설정 복사 · 도로망 생성 위치를 `repo_paths`로) — 학습 내용은 수업 원본 그대로 |
| `repo_paths.py` (12팀, 저장소 루트) | 폴더 경로를 한곳에서 정함. 실행 스크립트가 맨 위에서 불러, 어느 폴더에 있든 `env` · `road_spec` · `algorithms` · `train` · `metrics` · `collect_pomdp_data`를 같은 이름으로 import한다 (11/16 폴더 구조 변경) |
| `training/train_team.py` (12팀) | BC → 병렬 PPO 학습 — `docs/프로젝트_기록.md` §7.4 |

- **차량 id = 진입 · 출구**: 배경차 `f_<진입>__<출구>[_via_<경유>]_<번호>`, 버스 `bus_<노선>__<진입>__<출구>` (+ SUMO가 붙이는 `.<n>`). 주변 차량 목적지는 id로 계산한다.
- **도착 판정**: SUMO는 충돌로 제거된 차도 "도착 목록"에 넣는다 → 도착은 "충돌 아님 + 경로 마지막 도로에서 사라짐"일 때만 인정.

## 코드 안의 도로 규칙 (`road_spec.py` · `road_builder.py`를 고칠 때 주의)

- 좌표: m, x = 동쪽, y = 북쪽. 차로 번호는 SUMO 규칙대로 0 = 가장 오른쪽.
- **겹치는 도로**(강남터미널고가 · 반포대교 · 올림픽대로 · 지하차도)는 노드를 공유하지 않는다 → 교차로가 생기지 않는다.
- **차로 단위 연결을 직접 지정**한다 (netconvert 자동 추정에 맡기지 않음). ① 지하차도: 고가 쪽 왼쪽 차로만 `m1`의 4번 차로(① 전용)로 이어지고, `m1`의 3번 차로는 왼쪽 · 4번 차로는 오른쪽 차로 변경 금지(긴급차만) → 측도에서 온 차는 ①로 못 간다.
- **교차로 안 차로 변경 금지**: `build_net.py`가 모든 지정 연결에 `changeLeft/changeRight="emergency authority"`. **사거리 출구 짧은 도로**(`side_a8a` · `sbr_e1a` · `sbr_e2a`)도 `NO_LC` — AV도 여기서는 차로를 못 바꾼다.
- **한 차로가 여러 차로로 부채꼴로 갈라지는 연결을 만들지 않는다** — 버스가 옆 차로 차와 교차로 안에서 부딪힌다. 측도 남행 좌회전 포켓은 `sb_2`(5차로) → `sb_a5` 5 → 5 일직선, 우회전 도로 `slip_a5`는 직진 차로에서 6 m 바깥.
- 측도 북행은 분기에서 4차로로 오다가 사거리 50 m 앞(`a4s`)에서 우회전 도로 `slip_a4`가 갈라진다 (`side_a4u` → `side_a4` · `slip_a4`). 우회전 도로는 사거리 갈래 보도 바깥 0.3 m.
- 회전교차로: 반시계 1차로. 우회전 우회로 3개(`byp_se` · `byp_ws` · `byp_nw`), 서쪽 · 북쪽 우회로는 회전차로 바깥 0.8 m(`RB_BYP`). 지하차도는 남동쪽(`RB_SE` = 320°)에서 중심을 향해 정면 진입. 지하차도 북행 갈래 `und_j`와 남행 합류 `und_m`은 다른 노드 — 한 교차로에 두면 교차로가 올림픽대로 밑까지 늘어난다.
- 지하차도 남행 → 강남터미널고가 남행 합류점은 지하차도 남행 우선(`priority=10`) — 양보하게 두면 실측 수요에서 회전교차로까지 막힌다.
- **보도**는 `SIDEWALKS` 엣지에만(폭 2.5 m). netconvert가 보도를 0번 차로로 넣어 **차량 차로 번호가 1씩 밀린다** — `road_spec`의 차로 번호는 차량 기준이고 `net_lane()`으로 바꾼다. 수집 데이터의 `ego_lane` · `lane` · `event.lane_*`는 SUMO 번호 그대로라 분석에서는 `evaluation/lanes.py` `veh_lane(lane_id)`.
- 지도 북쪽 끝 `Y_END`: 반포대교 · 잠수교 북행 끝 = 잠수교 남행 진입. 북행이 먼저 끝나면 다리 한가운데에 막다른 차로 끝 막대가 보인다.
- 보행자 flow는 6시간에서 끝난다(`road_builder.PED_END_S`) — SUMO는 보행자 flow를 시작할 때 한꺼번에 펼쳐 만들어, 끝을 크게 두면 매 reset마다 수십만 명을 만드느라 34 s가 걸렸다. 6시간보다 오래 돌릴 일이 있으면 늘린다.
- 차로 변경 시간 `road_builder.LC_DURATION` = 0 (SUMO 기본, 한 스텝에 옆 차로로) — 연속 차로 변경 · sublane 모델은 합류부 측면 충돌이 생겨 쓰지 않는다 (`docs/프로젝트_기록.md` §8).

## 화면 표시 — 그리는 방법

높이별 색(고가 갈색 · 지상 검정 · 경사로 연두 · 한강 둔치 진초록)으로 층을 구분한다. `road_spec.z_at`이 화면용 높이를, `road_builder.BANDS` · `_write_layers`가 색을 정한다 — **높이는 층 순서만 실제와 맞춘 화면용 값**이고 주행에는 영향 없다. 색 상수: 버스전용 차선 `BUS_LINE_COLOR`, 정류소 `BUS_STOP_COLOR`, 보도 `SIDEWALK_COLOR`, 보행자 `PERSON_COLOR`(3배 크기), 중앙선 `CENTER_LINE_COLOR`, 버스 노선 유형 색 `road_config.BUS_COLORS`(노선 번호 라벨은 `busLine` → `vehicleTextParam`). 배경은 `_write_background` → `banpo.background.xml`(도로 밑 layer −5, 우리 좌표에 맞춰 직접 그림 · 위치는 대략). 범례는 sumo-gui 글꼴에 한글이 없어 영어.

sumo-gui 동작에 맞춘 것 — 바꿀 때 주의:
- 지하 · 둔치 구간의 원래 차로 · 교차로는 화면 설정에서 투명(alpha **0**)으로 숨기고, 색 다각형을 **차로 위**(layer 5.2~5.6)에 다시 그린다. 차로 밑에 그리면 sumo-gui가 투명 차로에도 그리는 차선 경계가 비친다. alpha를 0이 아닌 값으로 주면 불투명하게 그려진다.
  - 예외: 둔치 층(≤ −8 m) **차로**는 불투명(둔치 색)으로 둔다 — sumo-gui는 투명 차로의 신호 막대(정지선 · 횡단보도 끝 빨강/초록)를 그리지 않는다. 그래서 신호 막대가 다른 도로 밑에 오지 않게 교차로 위치를 잡는다.
- sumo-gui는 **시작점이 투명한 차로를 통째로 건너뛰고**, 꼭짓점 사이 선분을 시작점 높이 색 하나로 칠한다 → 지하에서 시작하는 차로의 지상 구간과, 지하 → 지상 경계 선분의 지상 쪽 나머지는 직접 검정으로 그린다.
- 아래층 도로 · **교차로** 위를 지나는 위층 구간(올림픽대로 × 지하차도)은 겹치는 곳만 layer 6.x에 덧칠한다 (교차로 다각형도 겹침 판정에 넣음). 고가 상판은 layer 10. 차량은 항상 맨 위.
- 색은 높이 경계(2 m · −3 m · −8 m)를 지나는 정확한 지점에서 바꾼다 → 북행 · 남행이 같은 지점에서 바뀐다.
- 다각형 위 차선 경계는 sumo-gui와 같은 위상의 점선(6 m 주기, 차로 시작부터 2.5~5.5 m). 보도와 맞닿은 경계는 점선 없이 색으로 구분.
- 가장자리(테두리)는 차로 · 교차로 면 바로 밑(L − 0.01)에 양옆 0.35 m 넓은 테두리색 띠(교차로는 굵은 닫힌 선)로 깐다 → 같은 층 면이 안쪽을 덮어 도로 바깥 둘레에만 테두리가 남는다. 차로마다 선을 그리면 교차로 안 이음매 · 층이 바뀌는 지점에 선이 생기고 꺾이는 곳에 틈이 난다.
- 보행 구역(walkingarea) · 횡단보도(crossing)는 차로 그리기 · 위층 덧칠에서 빼고 따로 그린다 (차로로 읽으면 폭 4 m짜리 덩어리가 생김).
- 남은 netconvert 경고 1종: 강남터미널고가 남행 진입로가 42°로 붙는 곳에서 연결 속도를 줄임 (그대로 둠).
