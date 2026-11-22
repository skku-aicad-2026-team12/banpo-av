"""SUMO 고속도로 RL 환경 — Gymnasium 인터페이스 + TraCI.

┌─────────────────────────────────────────────────────────────┐
│  강화학습 루프에서 이 클래스의 역할                              │
│                                                             │
│    obs, _ = env.reset()          # 에피소드 시작, 초기 관측     │
│    obs, r, term, trunc, _ = env.step(action)  # 1스텝 진행    │
│                                                             │
│  내부적으로는 TraCI(Traffic Control Interface)라는 소켓 API로  │
│  SUMO 시뮬레이터를 원격 조종한다:                               │
│    traci.simulationStep()  → 시뮬레이션을 1스텝 진행            │
│    traci.vehicle.setSpeed() → 차량 속도 명령                   │
│    traci.vehicle.getSpeed() 등 → 상태 조회                     │
└─────────────────────────────────────────────────────────────┘

관측/행동/보상의 구체적인 정의는 configs/mdp_config.py 에 있고,
이 파일은 그 설정을 받아서 실행만 한다. (MDP를 바꿀 때 이 파일 수정 불필요)
"""
import os
import sys

import numpy as np
import gymnasium as gym
from gymnasium import spaces

# SUMO 설치 폴더의 tools/ 안에 traci, sumolib 파이썬 모듈이 들어있다.
# pip로도 설치했지만, 버전 불일치를 피하려면 SUMO_HOME 쪽을 우선 경로에 추가.
#
# SUMO_HOME이 설정 안 돼 있으면 OS별 기본 설치 경로를 순서대로 탐색해서
# 자동으로 잡아준다 (환경변수 설정을 깜빡해도 동작하도록).
if "SUMO_HOME" not in os.environ:
    _candidates = [
        # macOS (Homebrew) — Apple Silicon / Intel
        "/opt/homebrew/opt/sumo/share/sumo",
        "/usr/local/opt/sumo/share/sumo",
        # Linux (apt)
        "/usr/share/sumo",
        # Windows (기본 인스톨러 경로)
        r"C:\Program Files (x86)\Eclipse\Sumo",
        r"C:\Program Files\Eclipse\Sumo",
    ]
    for _p in _candidates:
        if os.path.isdir(_p):
            os.environ["SUMO_HOME"] = _p
            print(f"[sumo_env] SUMO_HOME 자동 감지: {_p}")
            break
    else:
        raise RuntimeError(
            "SUMO_HOME 환경변수가 없고 기본 경로에서도 SUMO를 찾지 못했습니다.\n"
            "  Windows: 인스톨러 재설치(SUMO_HOME 체크) 또는 환경변수 수동 설정\n"
            "  macOS  : ./setup_mac.sh 실행 또는 README_MAC.md 참고\n"
            "  Linux  : sudo apt install sumo sumo-tools 후 "
            "export SUMO_HOME=/usr/share/sumo")

sys.path.append(os.path.join(os.environ["SUMO_HOME"], "tools"))

import traci
import traci.constants as _tc
from sumolib import checkBinary

# [12팀 수집 최적화] 화면 없이 돌 때는 libsumo(SUMO를 파이썬 프로세스 안에서 직접 호출)를 쓴다.
# TraCI 소켓 왕복·결과 해석이 사라져 같은 결과를 약 2배 빨리 낸다 (같은 SUMO 1.27.1 → 출력 동일, 비교 검증함).
# 설치: pip install libsumo==<SUMO 버전>. 없거나 SUMO_LIBSUMO=0이면 기존 TraCI 그대로. GUI는 항상 TraCI.
try:
    import libsumo as _libsumo
except ImportError:
    _libsumo = None
_traci_socket = traci

# get_privileged_state가 차량마다 구독하는 변수 (getPosition · getSpeed · getLaneIndex · getLaneID · getRoadID · getLanePosition)
_STATE_VARS = (_tc.VAR_POSITION, _tc.VAR_SPEED, _tc.VAR_LANE_INDEX,
               _tc.VAR_LANE_ID, _tc.VAR_ROAD_ID, _tc.VAR_LANEPOSITION)


def _road_spec():
    """environment/road/road_spec.py (도로 정의의 단일 원천)."""
    road_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "road")
    if road_dir not in sys.path:
        sys.path.insert(0, road_dir)
    import road_spec
    return road_spec


# [12팀] 잠수교 보행화 후 처리 — 환경(reset)과 environment/road/check_long.py가 이 두 함수를 같이 쓴다 (11/15, 코드 한 벌)
def after_closure_targets():
    """보행화 후 막을 엣지와 끌 신호 (environment/road/road_spec.py 기준).

    엣지: 보행화 전에만 있는 엣지(CLOSED_AFTER = 잠수교 남·북행). 도로는 그대로 두고 차량만 통제한다.
    신호: 보행화 전에만 작동하는 신호 (잠수교 횡단보도 C-ITS 4616 · 24616 — 통제 후 점멸 운영이라 끈다)
    """
    road_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "road")
    if road_dir not in sys.path:
        sys.path.insert(0, road_dir)
    import road_spec
    tls = [tl for tl, sp in road_spec.SIGNALS.items() if tuple(sp.get("scenarios") or ()) == ("before",)]
    return list(road_spec.CLOSED_AFTER), tls


def apply_after_closure(tc, closed_edges, tls_off, mark=False):
    """보행화 후: 막을 엣지의 모든 차로를 보행자 전용으로, 잠수교 신호를 끈다 (SUMO 시작 직후 1번).

    tc: 지금 연결된 traci 또는 libsumo 모듈. mark=True면 sumo-gui에 막힌 차로를 빨간 선으로 표시한다.
    """
    for e in closed_edges:
        for i in range(tc.edge.getLaneNumber(e)):
            tc.lane.setAllowed(f"{e}_{i}", ["pedestrian"])
            if mark:
                tc.polygon.add(f"closed_{e}_{i}", tc.lane.getShape(f"{e}_{i}"),
                               (220, 40, 40, 200), fill=False, layer=12, lineWidth=3.0)
    have = set(tc.trafficlight.getIDList())
    for tl in tls_off:
        if tl in have:
            tc.trafficlight.setProgram(tl, "off")


# 신호색 묶음: 같은 묶음 안에서 바뀌는 것(G ↔ g)은 "색이 바뀐 것"으로 보지 않는다
_SIG_GROUP = {"G": "green", "g": "green", "y": "yellow", "Y": "yellow", "u": "yellow",
              "r": "red", "R": "red", "s": "red", "o": "off", "O": "off"}


class SumoHighwayEnv(gym.Env):
    metadata = {"render_modes": ["human"]}

    def __init__(self, cfg_path, road, ego, mdp_sim, mdp_obs, action, reward,
                 traffic=None, gui=False, gui_autostart=True):
        """
        Args:
            cfg_path: road_builder.build()가 만든 .sumocfg 경로
            road:     configs.road_config.ROAD
            ego:      configs.road_config.EGO (가속/감속 한계가 여기서 옴)
            mdp_sim:  configs.mdp_config.SIMULATION
            mdp_obs:  configs.mdp_config.OBSERVATION
            action:   configs.mdp_config.ACTION (행동 차원 수)
            reward:   configs.mdp_config.REWARD
            traffic:  configs.road_config.TRAFFIC (프리필/컨트롤러 정보).
                None이면 프리필 없이 flow만으로 동작 (하위호환)
            gui:      True면 sumo-gui(화면 있는 버전)로 실행
            gui_autostart: (gui=True일 때만 의미)
                True  → 창이 뜨면 시뮬레이션이 자동으로 시작
                False → 사용자가 ▶(플레이) 버튼을 누를 때까지 대기.
                        sumo-gui는 --start 옵션이 없으면 TraCI의
                        simulationStep() 호출도 플레이 전까지 블록되므로,
                        에피소드 전체가 "▶를 눌러야 시작"되는 효과가 난다.
        """
        super().__init__()
        self.cfg_path = cfg_path
        self.gui = gui
        global traci
        traci = (_libsumo if (_libsumo is not None and not gui and os.environ.get("SUMO_LIBSUMO", "1") != "0")
                 else _traci_socket)   # 한 프로세스에 환경 하나 기준 (수집 워커·학습·평가 모두 그렇다). 한 프로세스에 env를 둘 이상(특히 화면 유무가 섞이게) 만들지 말 것
        self.gui_autostart = gui_autostart

        # ----- 도로/차량 설정에서 필요한 값들 -----
        self.vmax = float(road["speed_limit"])   # 속도 정규화 기준
        self.num_lanes = int(road["num_lanes"])
        # 시작 차선: "center"면 중앙 차선(차선수//2)으로 자동 결정.
        #   3차선 → 1번(가운데), 5차선 → 2번. 차선 수를 바꿔도 항상 가운데라
        #   좌/우 양쪽으로 변경 여지가 대칭으로 생긴다 (탐험에 유리).
        _dl = ego["depart_lane"]
        if str(_dl).lower() == "center":
            self.ego_depart_lane = str(int(road["num_lanes"]) // 2)
        else:
            self.ego_depart_lane = str(_dl)
        # ---- 프리필(도로 전체 사전 배치) 설정 ----
        self.traffic = traffic or {}
        pf = (self.traffic.get("prefill") or {})
        self.prefill_enabled = bool(pf.get("enabled", False))
        self.prefill_start = float(pf.get("start_offset", 80.0))
        self.prefill_end_margin = float(pf.get("end_margin", 50.0))
        self.prefill_jitter = float(pf.get("spacing_jitter", 0.3))
        # ---- 차선변경 안전 게이트 설정 (mdp_config.ACTION["lane_change_safety"]) ----
        lcs = (action.get("lane_change_safety") or {})
        self.lc_safety_enabled = bool(lcs.get("enabled", True))
        self.lc_front_min = float(lcs.get("min_front_gap", 5.0))   # m
        self.lc_rear_min = float(lcs.get("min_rear_gap", 5.0))     # m
        self.lc_front_tau = float(lcs.get("front_tau", 0.3))       # s
        self.lc_rear_tau = float(lcs.get("rear_tau", 0.5))         # s
        self.lc_spawn_protect = float(lcs.get("spawn_protect", 60.0))  # m
        self.lc_horizon = float(lcs.get("approach_horizon", 1.0))       # s
        self.road_length = float(road["length"])
        self.traffic_max_speed = float(self.traffic.get("max_speed", self.vmax))
        # 차선당 배치 간격: 지정값이 없으면 교통류 이론대로 자동 계산
        #   차선당 유량 q_lane = (vehs/h) / 3600 / 차선수  [대/초]
        #   간격 = 속도 / q_lane   (그 속도로 흐를 때 차 사이 평균 거리)
        if pf.get("spacing"):
            self.prefill_spacing = float(pf["spacing"])
        else:
            vph = float(self.traffic.get("vehs_per_hour", 1800))
            q_lane = max(vph / 3600.0 / max(self.num_lanes, 1), 1e-6)
            self.prefill_spacing = self.traffic_max_speed / q_lane
        # 컨트롤러 타입 샘플링 준비 (rou.xml의 car_<이름> vType과 대응)
        ctrls = self.traffic.get("controllers") or {}
        if ctrls:
            names = list(ctrls.keys())
            probs = np.array([float(ctrls[n].get("probability", 1.0)) for n in names])
            self._prefill_types = [f"car_{n}" for n in names]
            self._prefill_probs = probs / probs.sum()
        else:
            self._prefill_types, self._prefill_probs = ["car"], np.array([1.0])
        self.ego_depart_speed = str(ego["depart_speed"])
        # ---- 실제 도로망 모드 (road["network"]) ----
        #   AV 경로: 시나리오 수요 파일(banpo.<scenario>.rou.xml)의 <route id="av_이름"> 중 road["av_routes"]에서 에피소드마다 하나를 고른다.
        #   초기 교통: 프리필 대신 traffic["warmup_s"]초 동안 배경차를 먼저 흘려 둔다
        #             (분기·합류가 많은 도로망에서 경로·차선이 맞는 자연스러운 교통 상태가 된다).
        self.network = road.get("network")
        self.av_routes = list(road.get("av_routes") or [])
        self.route_name = None          # 이번 에피소드의 AV 경로 이름
        self._route_ids = {}            # 시나리오별로 수요 파일에 실제로 있는 AV 경로
        # 보행화 전·후: 도로망은 하나, 시나리오마다 수요 파일(banpo.<시나리오>.rou.xml)만 다르고
        # 보행화 후에는 잠수교 차로를 보행자 전용으로 바꾼다. "mixed"면 에피소드마다 무작위.
        _sc = road.get("scenario", "before")
        self.scenarios = ["before", "after"] if _sc == "mixed" else [_sc]
        self.scenario = None            # 이번 에피소드의 시나리오
        self.closed_after = []
        self.tls_before_only = []
        self._tls_phases = {}           # 신호별 [(지속 s, state 문자열)] — reset마다 SUMO에서 읽음 (get_signal_state)
        if self.network:
            self.closed_after, self.tls_before_only = after_closure_targets()
        self.warmup_s = float(self.traffic.get("warmup_s", 0.0)) if self.network else 0.0
        # SUMO 기본 운전자 모드: 정책 행동을 무시하고 SUMO가 AV를 운전한다 (검증·BC 전문가용).
        # info의 accel_raw/lane_change는 실제 움직임에서 거꾸로 계산한 값.
        self.sumo_driver = bool(ego.get("sumo_driver", False))
        # [12팀 11/17] SUMO 운전자 시범의 앞차 정지 간격 (m). None이면 차종 기본(2.5 m = 충돌 판정 경계에 정확히 선다).
        # 수집 --demo-min-gap에서만 준다 (기준 지표는 차종 기본 그대로).
        self.sumo_min_gap = ego.get("sumo_min_gap")
        # 행동 a ∈ [-1,1] 을 실제 가속도로 변환할 때 쓰는 스케일:
        #   a = +1 → +max_accel (풀 액셀),  a = -1 → -max_decel (풀 브레이크)
        self.max_accel = float(ego["accel"])   # m/s^2
        self.max_decel = float(ego["decel"])   # m/s^2 (양수로 저장, 부호는 계산 시)

        # ----- MDP 설정 -----
        self.step_length = float(mdp_sim["step_length"])
        self.max_steps = int(mdp_sim["max_steps"])
        self.warmup_steps = int(mdp_sim["warmup_steps"])
        # 비정상 정지: ego가 이 시간(초) 이상 계속 멈춰 있으면 에피소드 종료 (0이면 끔)
        self.stuck_steps = int(float(mdp_sim.get("stuck_seconds", 0)) / self.step_length)
        self.visibility = float(mdp_obs["visibility"])   # 관측 거리 W
        # 밀도 정규화용 "차량 1대가 점유하는 길이" (차체 + 최소 간격)
        self.density_veh_len = float(mdp_obs.get("density_veh_length", 7.5))
        # [12팀 11/15] 경로 · 신호 관측 정규화 거리 (mdp_config.OBSERVATION 주석)
        self.route_horizon = float(mdp_obs.get("route_horizon", 300.0))
        self.signal_horizon = float(mdp_obs.get("signal_horizon", 150.0))
        # [12팀 11/16] 선택 관측 (스위치 — 바꾸면 시범 · BC부터 다시): 경합 차량 4 · 차로 제한속도 3 (mdp_config 주석 [39..])
        self.conflict_obs = bool(mdp_obs.get("conflict_obs", False))
        self.conflict_horizon = float(mdp_obs.get("conflict_horizon", 100.0))
        self.speed_limit_obs = bool(mdp_obs.get("speed_limit_obs", False))
        # [12팀 11/17] 신호 남은 시간 1 (컨설팅 질문 2 답: C-ITS 기준으로 넣어도 됨 — mdp_config 주석 [42])
        self.signal_time_obs = bool(mdp_obs.get("signal_time_obs", False))
        self.signal_time_horizon = float(mdp_obs.get("signal_time_horizon", 170.0))
        # [12팀 11/19] 분기 2개 · 차로 진행 방향 (mdp_config 주석 [31..38]). 끄면 43차원 그대로.
        # 환경변수 SUMO_LANE_DIR_OBS=0/1이 있으면 설정을 덮어쓴다 (43차원 모델 평가 — 평가 워커는 spawn이라 환경변수를 물려받는다)
        _ldo = os.environ.get("SUMO_LANE_DIR_OBS")
        self.lane_dir_obs = bool(mdp_obs.get("lane_dir_obs", False)) if _ldo in (None, "") else _ldo.strip() not in ("0", "false", "False")
        self.branch_horizon = float(mdp_obs.get("branch_horizon", 600.0))
        # [12팀 11/19] 잘못된 차로 조기 종료 거리 (m, 0이면 끔) — mdp_config SIMULATION 주석
        self.wrong_lane_end_m = float(mdp_sim.get("wrong_lane_end_m", 0.0))
        self._rank_cache = {}            # (시나리오, 분기 엣지) → {나가는 도로: +1 맨 왼쪽 .. -1 맨 오른쪽}
        self._lane_out_cache = {}        # (시나리오, 차로) → 승용차가 갈 수 있는 다음 차로 목록
        self._continue_cache = {}        # (시나리오, 차로, 경로 인덱스, 경로) → 잘못된 차로 판정 (_lane_continues)
        self._lane_reach_cache = {}      # (시나리오, 차로) → ([(차로 끝부터 분기 끝까지 m, 방향 집합)] × 첫째 · 둘째 분기)
        self._edge_speed = {}            # 엣지 → (제한속도, 길이) (도로망은 고정, 엣지 안 차로는 모두 같은 제한속도)
        self._turn_cache = {}            # (분기 엣지, 다음 경로 엣지) → 방향 (도로망 고정)
        self._branch_cache = {}          # (시나리오, 엣지) → 끝에서 다른 도로로도 나갈 수 있는가 (시나리오마다 한 번만 계산)
        self.act_dim = int(action["dim"])
        self.accel_index = int(action.get("accel_index", 0))
        self.lane_change_index = int(action.get("lane_change_index", 1))
        self.lane_change_threshold = float(action.get("lane_change_threshold", 0.5))
        self.lane_change_duration = float(action.get("lane_change_duration", 1.0))
        # [12팀 11/15] 안전장치 스위치: 최소 충돌 방지 제동 (RL 모드에서만, mdp_config.ACTION["safety_brake"])
        sb = action.get("safety_brake") or {}
        self.safety_brake = bool(sb.get("enabled", False))
        self.safety_min_gap = float(sb.get("min_gap", 2.0))
        self.safety_lookahead = float(sb.get("lookahead", 100.0))
        self.rw = reward

        # GUI는 ego를 따라가며 가까이 보이도록 기본값을 둔다.
        self.gui_zoom = float(mdp_sim.get("gui_zoom", 1200.0))
        self.gui_delay = float(mdp_sim.get("gui_delay", 100.0))
        self.gui_track_ego = bool(mdp_sim.get("gui_track_ego", True))

        self.ego_id = "ego"        # SUMO 안에서 ego 차량을 식별하는 ID
        self.step_count = 0
        self._started = False      # SUMO 프로세스가 떠 있는지 여부

        # Gymnasium 행동 공간:
        #   action[0] = 종방향 가감속 raw ∈ [-1,1]
        #   action[1] = 차선변경 raw ∈ [-1,1], 환경 내부에서 {-1,0,+1}로 양자화
        # PPO의 Gaussian 정책과 호환하기 위해 Box를 유지한다. 실제 적용된
        # 차선변경 명령은 info["lane_change"]에 정확한 -1/0/+1로 반환한다.
        self.action_space = spaces.Box(low=-1.0, high=1.0,
                                       shape=(self.act_dim,), dtype=np.float32)
        # 관측: 35차원 (+ 선택 관측: 경합 차량 4 · 차로 제한속도 3, 11/16 · 신호 남은 시간 1, 11/17 — 모두 켜면 43), 범위 [-1,1]. 레이아웃은 environment/env/mdp_config.py 주석 참고.
        #   [0] 내 속도, [1..20] 5개 차선 × (선행/후행 상대거리·상대속도),
        #   [21..25] 연결성(끊김 정도), [26..30] 전방 밀도 — 여기까지 수업 원본 31차원
        #   [31] 다음 분기 방향, [32] 다음 분기까지 거리 (12팀 11/17 — 차로별 경로 관측은 뺌), [33] 앞 신호색, [34] 정지선까지 거리
        #   (+ 11/19 lane_dir_obs 6: 둘째 분기 2 · 내 차로가 가는 쪽 4 — 켜면 49, [31..] 배치가 바뀐다 — 아래 self.obs_index)
        #   시야 차선: 좌2, 좌1, 현재, 우1, 우2 (5개) — 3차선 도로에서
        #   맨 왼쪽에 있어도 오프셋 -2로 맨 오른쪽 차선까지 보인다.
        self.obs_offsets = (+2, +1, 0, -1, -2)
        n_lanes_view = len(self.obs_offsets)
        self.obs_dim = 1 + n_lanes_view * 4 + n_lanes_view + n_lanes_view + 2 + 2   # 수업 31 + 경로 2 + 신호 2
        self.obs_dim += (4 if self.conflict_obs else 0) + (3 if self.speed_limit_obs else 0)   # [12팀 11/16] 선택 관측
        self.obs_dim += 1 if self.signal_time_obs else 0                                        # [12팀 11/17] 신호 남은 시간
        self.obs_dim += 6 if self.lane_dir_obs else 0                                           # [12팀 11/19] 둘째 분기 2 + 차로 진행 방향 4
        # [12팀 11/19] 칸 이름 → 번호 (mdp_config 관측 주석). 번호를 직접 쓰지 말고 이 표로 읽는다 (check_env.rule_policy 등).
        #   43차원(lane_dir_obs 끔, 기존 모델용): 경로 2 · 신호 2 · 경합 4 · 제한속도 3 · 신호 남은 시간 1
        #   49차원(lane_dir_obs 켬): 경로(분기 2개) 4 · 내 차로 진행 방향 4 · 신호 2 · 신호 남은 시간 1 · 경합 4 · 제한속도 3
        _blocks = {"route": ["route_dir1", "route_dist1"], "route2": ["route_dir2", "route_dist2"],
                   "reach": ["reach1_left", "reach1_right", "reach2_left", "reach2_right"],
                   "signal": ["sig_color", "sig_dist"], "sig_time": ["sig_time"],
                   "conflict": ["conf_av_dist", "conf_foe_dist", "conf_foe_speed", "conf_prio"],
                   "speed": ["lim_now", "lim_next", "lim_next_dist"]}
        if self.lane_dir_obs:
            self._obs_blocks = ["route", "route2", "reach", "signal"] + (["sig_time"] if self.signal_time_obs else []) \
                + (["conflict"] if self.conflict_obs else []) + (["speed"] if self.speed_limit_obs else [])
        else:
            self._obs_blocks = ["route", "signal"] + (["conflict"] if self.conflict_obs else []) \
                + (["speed"] if self.speed_limit_obs else []) + (["sig_time"] if self.signal_time_obs else [])
        self.obs_index, _i = {}, 1 + n_lanes_view * 6
        for _b in self._obs_blocks:
            for _n in _blocks[_b]:
                self.obs_index[_n] = _i
                _i += 1
        assert _i == self.obs_dim, (_i, self.obs_dim)
        self.observation_space = spaces.Box(-1.0, 1.0, shape=(self.obs_dim,),
                                            dtype=np.float32)

    # ==================================================================
    # SUMO 프로세스 시작
    # ==================================================================
    def _start_sumo(self, sumo_seed=None):
        # gui=True면 화면이 있는 sumo-gui, 아니면 화면 없는 sumo (학습용, 훨씬 빠름)
        binary = checkBinary("sumo-gui" if self.gui else "sumo")
        # 명시 시드가 있어도 기존 추첨을 한 번 소비해 출발 차로 등 이후 난수 순서를 유지한다.
        sampled_seed = np.random.randint(0, 100000)
        cmd = [
            binary, "-c", self.cfg_path,
            "--step-length", str(self.step_length),
            # 충돌한 차량을 시뮬레이션에서 제거. (기본값 teleport는 순간이동시켜버려서
            # RL 학습에는 부적합. remove로 해야 "충돌 → 에피소드 종료"가 깔끔함)
            "--collision.action", "remove",
            # 교차로(junction) 안의 충돌도 검사한다. 기본값은 교차로 내부 충돌을 무시해서
            # 교차로·회전교차로가 있는 도로에서는 충돌률이 실제보다 낮게 나온다.
            "--collision.check-junctions", "true",
            "--no-warnings", "true",   # 콘솔 경고 끄기
            "--no-step-log", "true",   # 스텝 로그 끄기
            # 매 에피소드 다른 시드 → 배경 교통 패턴이 달라져서 과적합 방지
            "--seed", str(sampled_seed if sumo_seed is None else sumo_seed),
        ]
        if self.network and self.scenario:
            # 이번 에피소드 시나리오의 수요 파일 (sumocfg의 기본값을 덮어씀)
            cmd += ["--route-files", os.path.join(os.path.dirname(self.cfg_path),
                                                  f"banpo.{self.scenario}.rou.xml")]
        if self.gui:
            cmd += ["--quit-on-end"]      # 에피소드 종료 시 창 정리
            cmd += ["--delay", str(self.gui_delay)]
            if self.gui_autostart:
                cmd += ["--start"]        # 자동 재생
            # --start가 없으면: 창이 뜬 뒤 ▶(플레이)를 눌러야 진행된다.
            #   매 에피소드가 reset()에서 SUMO를 새로 띄우므로,
            #   에피소드마다 ▶를 눌러 시작하게 된다.
        traci.start(cmd)   # SUMO를 자식 프로세스로 띄우고 소켓 연결
        self._started = True

    # ==================================================================
    # 에피소드 초기화
    # ==================================================================
    def _prefill_traffic(self):
        """도로 전 구간에 배경차를 일정 간격(±jitter)으로 미리 배치한다.

        flow는 도로 입구에서만 차를 만들기 때문에, 프리필이 없으면
        에피소드 초반의 긴 도로가 텅 비어 "빈 도로 주행"만 잔뜩 학습된다.
        프리필은 첫 스텝부터 ego가 실제 교통류 한가운데서 시작하게 만든다.

        - 위치: start_offset ~ (길이 - end_margin), 차선별 독립 배치
        - 간격: prefill_spacing × (1 ± jitter) 무작위
        - 타입: controllers 확률 비율로 샘플링 (rou.xml의 vType과 동일)
        - 속도: departSpeed="max" → 각 개체가 "자기 허용 최고속도"로 시작.
          컨트롤러별 max_speed/speedDev 차등이 프리필에도 그대로 반영되고,
          개체별 무작위 speedFactor 때문에 숫자 속도를 주면 SUMO가
          "허용속도 초과"로 거부할 수 있는 문제도 함께 피한다.
        배치된 차들은 SUMO의 각자 컨트롤러(IDM 등)가 정상적으로 운전한다.
        """
        rng = np.random.default_rng()
        count = 0
        # 차선별 간격 배율 (없으면 전 차선 동일). 인덱스 0 = 맨 오른쪽.
        factors = (self.traffic.get("prefill", {})
                   .get("lane_spacing_factor")
                   or [1.0] * self.num_lanes)
        for lane in range(self.num_lanes):
            f = float(factors[lane]) if lane < len(factors) else 1.0
            lane_spacing = self.prefill_spacing * f
            pos = self.prefill_start + rng.uniform(0, lane_spacing)
            while pos < self.road_length - self.prefill_end_margin:
                vid = f"pre_{lane}_{count}"
                vtype = str(rng.choice(self._prefill_types, p=self._prefill_probs))
                try:
                    traci.vehicle.add(
                        vid, "r0", typeID=vtype,
                        departLane=str(lane),
                        departPos=f"{pos:.1f}",
                        departSpeed="max")
                    count += 1
                except traci.TraCIException:
                    pass  # 겹침 등으로 실패한 자리는 건너뜀
                step = lane_spacing * rng.uniform(
                    1.0 - self.prefill_jitter, 1.0 + self.prefill_jitter)
                pos += max(step, 10.0)
        return count

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # 최종 수집에서만 판별 SUMO 시드를 지정한다. 미지정 경로(평가·학습)는 기존 추첨 그대로.
        sumo_seed = (options or {}).get("sumo_seed")
        if sumo_seed is not None:
            if (isinstance(sumo_seed, (bool, np.bool_))
                    or not isinstance(sumo_seed, (int, np.integer))
                    or not 0 <= sumo_seed < 2**31):
                raise ValueError("sumo_seed는 0 이상 2**31 미만의 정수여야 합니다")

        # 이번 에피소드의 시나리오 (보행화 전/후). options={"scenario": "after"}로 지정 가능
        want_sc = (options or {}).get("scenario")
        if want_sc is not None and want_sc not in ("before", "after"):
            raise ValueError(f"시나리오는 before/after: {want_sc}")
        # [12팀 11/19] 시나리오 · 경로를 둘 다 안 줬고 시나리오가 "mixed"면 쓸 수 있는 (시나리오, 경로) 쌍 전체에서 같은 확률로 뽑는다
        # (전에는 시나리오를 반반으로 먼저 뽑아 경로가 2개뿐인 통제 전 경로가 더 자주 나왔다). 하나라도 주면 예전 동작 · 난수 순서 그대로.
        pair_route = None
        if (want_sc is None and (options or {}).get("route") is None and len(self.scenarios) > 1 and self.network):
            pairs = self._route_pairs()
            want_sc, pair_route = pairs[np.random.randint(len(pairs))]
        self.scenario = want_sc or str(np.random.choice(self.scenarios))

        # SUMO 상태를 완전히 초기화하는 가장 확실한 방법은 껐다 켜는 것.
        if self._started:
            traci.close()
            self._started = False
        self._start_sumo(sumo_seed=sumo_seed)

        # 보행화 후: 잠수교 차로를 보행자 전용으로 (도로는 그대로, 차량만 통제). 화면에는 빨간 표시.
        # 잠수교 횡단보도 신호(C-ITS 4616 · 24616)는 통제 후 점멸 운영 (11/11 기록) → 신호를 끈다
        if self.network and self.scenario == "after":
            apply_after_closure(traci, self.closed_after, self.tls_before_only, mark=self.gui)
        # 지금 작동 중인 신호 프로그램의 현시 목록 (get_signal_state가 "AV 방향 신호색이 바뀔 때까지"를 계산할 때 씀)
        self._tls_phases = {}
        for tl in traci.trafficlight.getIDList():
            prog = traci.trafficlight.getProgram(tl)
            for lg in traci.trafficlight.getAllProgramLogics(tl):
                if lg.programID == prog:
                    self._tls_phases[tl] = [(float(p.duration), p.state) for p in lg.phases]

        # 도로 전 구간에 배경차를 미리 배치 (첫 스텝부터 실제 교통류 상태)
        if self.prefill_enabled:
            self._prefill_traffic()
        # 배치된 차들이 자리를 잡도록 몇 스텝만 정착시킨다.
        for _ in range(self.warmup_steps):
            traci.simulationStep()
        # 실제 도로망: 배경차를 warmup_s초 동안 먼저 흘려 둔다 (한 번의 호출로 그 시각까지 진행)
        if self.warmup_s > 0:
            traci.simulationStep(traci.simulation.getTime() + self.warmup_s)

        # 이번 에피소드의 AV 경로 (직선 도로 모드는 r0 하나)
        route_id = "r0"
        if self.network:
            if self.scenario not in self._route_ids:
                have = set(traci.route.getIDList())
                self._route_ids[self.scenario] = [r for r in self.av_routes if f"av_{r}" in have]
                if not self._route_ids[self.scenario]:
                    raise RuntimeError(f"{self.scenario} 수요 파일에 AV 경로가 없습니다: {self.av_routes}")
            ids = self._route_ids[self.scenario]
            want = (options or {}).get("route")
            if want is not None and want not in ids:
                raise ValueError(f"{self.scenario} 시나리오에 없는 AV 경로: {want} (가능: {ids})")
            if want is None and pair_route is not None and pair_route in ids:
                want = pair_route
            self.route_name = want or str(np.random.choice(ids))
            route_id = f"av_{self.route_name}"

        # 출발 차로 · 출발 속도 옵션 (주지 않으면 __init__에서 정한 EGO 값 그대로 — 기존 동작과 같음)
        #   options["depart_lane"]: 정수 = 그 차로, "random" = 출발 엣지의 승용차 허용 차로 중 하나 (전역 np.random)
        #   options["depart_speed"]: "limit" = 출발 차로 제한속도 × 0.8
        # 무작위 차로는 위의 SUMO --seed · 시나리오 · 경로 추첨 뒤에 뽑아 기존 난수 순서를 바꾸지 않는다.
        depart_lane, depart_speed = self.ego_depart_lane, self.ego_depart_speed
        opt_lane = (options or {}).get("depart_lane")
        opt_speed = (options or {}).get("depart_speed")
        if opt_lane is not None or opt_speed is not None:
            first_edge = traci.route.getEdges(route_id)[0]
            n_lanes = traci.edge.getLaneNumber(first_edge)
            # 승용차가 다닐 수 있는 차로 (allowed가 비어 있으면 모두 허용 · 보도 · 버스전용 제외)
            car_lanes = []
            for i in range(n_lanes):
                allowed = traci.lane.getAllowed(f"{first_edge}_{i}")
                if not allowed or "passenger" in allowed:
                    car_lanes.append(i)
            if opt_lane is not None:
                if str(opt_lane).lower() == "random":
                    # [12팀 11/19] 경로별 출발 차로 범위 road_spec.AV_ROUTES[...]["depart_lanes"] (없으면 승용 차로 전부)와의 교집합
                    dl = (_road_spec().AV_ROUTES.get(self.route_name) or {}).get("depart_lanes") if self.network else None
                    pick = [i for i in car_lanes if i in dl] if dl else car_lanes
                    pick = pick or car_lanes
                    depart_lane = str(int(pick[np.random.randint(len(pick))]))
                else:
                    if int(opt_lane) not in car_lanes:
                        raise ValueError(f"{first_edge}의 승용차 차로가 아닙니다: {opt_lane} (가능: {car_lanes})")
                    depart_lane = str(int(opt_lane))
            if opt_speed is not None:
                if str(opt_speed).lower() != "limit":
                    depart_speed = str(float(opt_speed))
                else:
                    # 차로가 정해졌으면 그 차로, "best" 등 미정이면 승용 차로 중 가장 낮은 제한속도
                    if depart_lane.isdigit():
                        vlim = traci.lane.getMaxSpeed(f"{first_edge}_{depart_lane}")
                    else:
                        vlim = min(traci.lane.getMaxSpeed(f"{first_edge}_{i}") for i in car_lanes)
                    depart_speed = f"{0.8 * vlim:.3f}"

        # ego 차량 투입 — 반드시 "경로 시작점(0m)"에서 출발
        # SUMO 운전자 모드(반포 도로망)는 사람 운전자 차종 ego_sumo (road_config.EGO["sumo_vtype"]) — RL 모드는 ego_type
        traci.vehicle.add(self.ego_id, route_id, typeID=("ego_sumo" if self.sumo_driver and self.network else "ego_type"),
                          departLane=depart_lane,
                          departPos="0",
                          departSpeed=depart_speed)

        if not self.sumo_driver:
            # ★ 중요: speedMode=0 → SUMO의 내장 안전장치(자동 안전제동 등)를 전부 끔.
            #   이걸 꺼야 RL이 속도를 온전히 제어하고, 잘못 배우면 실제로 충돌한다.
            #   (켜두면 SUMO가 알아서 브레이크를 밟아줘서 충돌이 거의 안 나고,
            #    그러면 "충돌 회피"라는 학습 신호 자체가 사라진다)
            traci.vehicle.setSpeedMode(self.ego_id, 0)
            # SUMO의 자율 차선변경은 끄고, RL이 내리는 TraCI 차선변경 명령만 사용한다.
            traci.vehicle.setLaneChangeMode(self.ego_id, 0)

        traci.simulationStep()  # ego가 실제로 도로에 나타나도록 1스텝 진행

        # ── ego 삽입 보증 ──
        # 도로 입구는 flow가 차를 스폰하는 자리라, 시작 차선 입구가 잠시
        # 막혀 있으면 SUMO가 삽입을 "지연"시킨다. 그 경우 ego 없이
        # 에피소드가 진행되어 즉시 종료되는 버그가 되므로, 삽입될 때까지
        # (최대 30스텝, 실제 도로망은 60스텝) 기다린다. 지연되어도 요청한 차선은 유지된다.
        _wait, _wait_max = 0, (60 if self.network else 30)
        while self.ego_id not in traci.vehicle.getIDList() and _wait < _wait_max:
            traci.simulationStep()
            _wait += 1
        # [12팀 11/20] 대체 경로 — 위 방식으로 _wait_max 안에 못 들어간 판에서만 작동한다 (정상 삽입 판은 한 스텝도 안 바뀜).
        #   정체가 아니라 출발 속도(제한속도×0.8)가 입구에 막 들어온 느린 배경차보다 빨라 안전한 틈이 안 나는 경우다
        #   (oly-jamsu wb_a 입구 10~35 m 차량 4~10 m/s). 대기 중인 ego를 빼고 같은 차로·같은 위치(0 m)에서
        #   출발 속도만 입구 근처(앞 50 m) 차량의 최저 속도로 낮춰 다시 넣고 (_wait_max 스텝), 그래도 안 되면 SUMO가
        #   매 스텝 앞차에 맞춰 고르는 안전 최대 속도("max")로 넣는다 (4×_wait_max 스텝 — 입구에 배경차 삽입 대기열이
        #   60~90대 쌓여 빈자리를 먼저 차지하므로 틈이 날 때까지 시간이 걸린다).
        #   배경 교통(차량 수 · 배치 · 순서)은 그대로 두고, 전역 np.random은 쓰지 않는다.
        if self.ego_id not in traci.vehicle.getIDList() and self.network:
            _edge0 = traci.route.getEdges(route_id)[0]
            _lanes = ([f"{_edge0}_{depart_lane}"] if str(depart_lane).isdigit()
                      else [f"{_edge0}_{i}" for i in range(traci.edge.getLaneNumber(_edge0))])
            _near = [traci.vehicle.getSpeed(v) for ln in _lanes for v in traci.lane.getLastStepVehicleIDs(ln)
                     if traci.vehicle.getLanePosition(v) < 50.0]
            _speeds = []
            if _near:
                _v = min(_near)
                try:
                    _v = min(_v, float(depart_speed))
                except (TypeError, ValueError):
                    pass
                _speeds.append(f"{_v:.3f}")
            _speeds.append("max")
            for _spd in _speeds:
                if self.ego_id in traci.vehicle.getIDList():
                    break
                traci.vehicle.remove(self.ego_id)      # 아직 출발 전(삽입 대기)인 ego만 지운다
                traci.simulationStep()
                traci.vehicle.add(self.ego_id, route_id,
                                  typeID=("ego_sumo" if self.sumo_driver and self.network else "ego_type"),
                                  departLane=depart_lane, departPos="0", departSpeed=_spd)
                if not self.sumo_driver:
                    traci.vehicle.setSpeedMode(self.ego_id, 0)
                    traci.vehicle.setLaneChangeMode(self.ego_id, 0)
                _wait = 0
                _lim = _wait_max if _spd != "max" else 4 * _wait_max    # "max"는 매 스텝 앞차에 맞춘 안전 속도라 더 기다린다
                while self.ego_id not in traci.vehicle.getIDList() and _wait < _lim:
                    traci.simulationStep()
                    _wait += 1
        if self.ego_id not in traci.vehicle.getIDList():
            raise RuntimeError(
                f"ego 삽입 실패: 도로 입구가 {_wait_max}스텝 이상 막혀 있습니다. "
                + ("TRAFFIC['demand_scale']을 낮추거나 AV 출발 도로(bp_nb) 정체를 확인하세요." if self.network
                   else "TRAFFIC['vehs_per_hour']를 낮추거나 depart_speed를 확인하세요."))
        if self.sumo_driver and self.sumo_min_gap:      # [12팀 11/17] 시범 운전자 정지 간격 (수집 --demo-min-gap)
            traci.vehicle.setMinGap(self.ego_id, float(self.sumo_min_gap))

        # GUI 기본 시점: ego를 화면 중앙에서 추적하고 충분히 가까이 확대한다.
        # 긴 직선도로 전체를 한 화면에 맞추는 기본 뷰는 차량이 점/삼각형처럼
        # 보이기 쉬우므로 학습/평가 관찰용으로는 이 설정이 더 직관적이다.
        if self.gui and self.ego_id in traci.vehicle.getIDList():
            try:
                view_ids = traci.gui.getIDList()
                if view_ids:
                    view_id = view_ids[0]
                    # tracking을 먼저 켜고 zoom을 마지막에 적용해야 SUMO가
                    # 차량 추적을 시작하면서 viewport를 다시 덮어쓰지 않는다.
                    if self.gui_track_ego:
                        traci.gui.trackVehicle(view_id, self.ego_id)
                    traci.gui.setZoom(view_id, self.gui_zoom)
            except traci.TraCIException:
                pass

        self.step_count = 0
        self._last_sumo_accel = 0.0                               # SUMO 운전자 모드: 마지막 스텝 행동 대체값
        self._ego_road = traci.vehicle.getRoadID(self.ego_id)   # 마지막으로 있던 도로 (도착 판정)
        self._route_last = traci.vehicle.getRoute(self.ego_id)[-1]
        self._stopped_steps = 0          # 연속 정지 스텝 수 (비정상 정지 판정)
        self.teleports = 0               # 이번 에피소드 동안 순간이동한 차량 수 (전체 차량)
        self.safety_interventions = 0    # 이번 에피소드 동안 안전장치가 속도 명령을 낮춘 스텝 수
        self._last_route_obs = [0.0, 1.0]    # 교차로 안에서 유지할 경로 관측 (방향 · 분기 거리)
        self._last_lane_dir_obs = [0.0, 1.0, 1.0, 1.0, 1.0, 1.0]   # [12팀 11/19] 같은 식으로 유지할 [33..38]
        # 실제 출발 차로 (삽입 직후 ego가 있는 차로 번호)
        self.depart_lane_actual = int(traci.vehicle.getLaneIndex(self.ego_id))
        return self._get_obs(), {"route": self.route_name, "scenario": self.scenario,
                                 "depart_lane": self.depart_lane_actual}

    # ==================================================================
    # 관측 만들기 (레이아웃은 environment/env/mdp_config.py 주석 참고)
    #
    # 핵심 설계: 특정 도로에 대한 하드코딩 없이, 매 스텝 SUMO 네트워크를
    # 실시간 조회해서 "지금 내 주변"을 계산한다. 그래서 도로 구조가
    # 어떻게 바뀌어도(차선 수, 합류, 차선 감소 등) 이 코드는 그대로 동작한다.
    #
    # 사용하는 TraCI API:
    #   getLeader / getFollower : 같은 차선의 앞/뒤 차량 (edge 경계도 넘어 탐색)
    #   getNeighbors(mode)      : 옆 차선의 앞/뒤 차량.
    #       mode 비트: bit0 = 0:왼쪽 / 1:오른쪽,  bit1 = 0:후행 / 1:선행
    #   lane.getLinks           : 차선이 다음 edge로 이어지는지 (연결성 판단)
    # ==================================================================

    # (차선 오프셋, 선행차 getNeighbors mode, 후행차 mode)
    #   오프셋 +1 = 왼쪽 차선 (SUMO는 차선 index가 왼쪽으로 갈수록 커짐)
    _NEIGHBOR_MODES = {+1: (0b10, 0b00),   # 왼쪽: 선행=2, 후행=0
                       -1: (0b11, 0b01)}   # 오른쪽: 선행=3, 후행=1

    def _neighbor(self, offset: int, leader: bool):
        """오프셋 차선(-1=오른쪽, 0=현재, +1=왼쪽)의 가장 가까운
        선행(leader=True)/후행 차량을 (거리 m, 그 차량 속도)로 반환.
        visibility 밖이거나 없으면 None."""
        if offset == 0:
            if leader:
                res = traci.vehicle.getLeader(self.ego_id, self.visibility)
                if res is None:
                    return None
                vid, gap = res
            else:
                vid, gap = traci.vehicle.getFollower(self.ego_id, self.visibility)
                if vid == "":
                    return None
        elif abs(offset) >= 2:
            # 두 칸 이상 옆 차선: SUMO getNeighbors는 "바로 옆"만 지원하므로
            # 차선 직접 스캔(_scan_lane_neighbors — 같은 edge의 차량 목록을
            # 훑어 가장 가까운 앞/뒤를 찾음)으로 감지한다.
            front, rear = self._scan_lane_neighbors(offset)
            res = front if leader else rear
            if res is None or res[0] < 0 or res[0] > self.visibility:
                return None
            return float(res[0]), float(res[1])
        else:
            mode_leader, mode_follower = self._NEIGHBOR_MODES[offset]
            pairs = traci.vehicle.getNeighbors(
                self.ego_id, mode_leader if leader else mode_follower)
            if not pairs:
                return None
            vid, gap = min(pairs, key=lambda p: p[1])   # 가장 가까운 차량

        if gap < 0 or gap > self.visibility:
            return None
        return float(gap), traci.vehicle.getSpeed(vid)

    def _route_pairs(self):
        """[12팀 11/19] 쓸 수 있는 (시나리오, 경로) 쌍 — road["av_routes"] × road_spec.AV_ROUTES[...]["scenarios"] (비어 있으면 모든 시나리오),
        이 환경의 시나리오 목록 안에서. 순서 고정 (경로 목록 순서 → 시나리오 순서)."""
        spec = _road_spec().AV_ROUTES
        out = []
        for r in self.av_routes:
            scs = tuple((spec.get(r) or {}).get("scenarios") or ()) or tuple(self.scenarios)
            out += [(sc, r) for sc in self.scenarios if sc in scs]
        if not out:
            raise RuntimeError(f"쓸 수 있는 (시나리오, 경로)가 없습니다: {self.scenarios} × {self.av_routes}")
        return out

    def _neighbor_seen(self, offset: int, leader: bool):
        """[12팀 11/19] lane_dir_obs 켬일 때의 관측용 _neighbor: 옆 차로(±1)는 getNeighbors와 차로 직접 스캔 중 더 가까운 쪽
        (안전 게이트 _lane_change_is_safe의 closer와 같은 방식), ±2는 스캔. 간격이 음수(AV와 겹쳐 나란히)인 차는 버리지 않고 0으로 —
        나란히 있는 차는 앞 거리 0 · 뒤 거리 0으로 양쪽에 다 나타난다 (수업 원본은 관측에서 빠져 차로 변경 판단을 못 했다).
        현재 차로(0)는 _neighbor 그대로. visibility 밖이면 None."""
        if offset == 0:
            return self._neighbor(0, leader)
        cands = []
        s = self._scan_lane_seen(offset, leader)
        if s is not None:
            cands.append(s)
        if abs(offset) == 1:
            mode_leader, mode_follower = self._NEIGHBOR_MODES[offset]
            try:
                pairs = traci.vehicle.getNeighbors(self.ego_id, mode_leader if leader else mode_follower)
            except traci.TraCIException:
                pairs = ()
            if pairs:
                vid, gap = min(pairs, key=lambda p: p[1])
                cands.append((max(float(gap), 0.0), float(traci.vehicle.getSpeed(vid))))
        if not cands:
            return None
        gap, sp = min(cands, key=lambda c: c[0])
        if gap > self.visibility:
            return None
        return gap, sp

    def _scan_lane_seen(self, offset: int, leader: bool):
        """_neighbor_seen용 차로 직접 스캔 (같은 엣지의 offset 차로): 가장 가까운 앞차(앞 범퍼가 AV 뒤끝보다 앞) 또는
        뒤차(뒤끝이 AV 앞 범퍼보다 뒤)의 (간격 m, 속도). 겹치는 차는 앞 · 뒤 모두에 간격 0으로 잡힌다. 없으면 None."""
        try:
            edge = traci.vehicle.getRoadID(self.ego_id)
            li = traci.vehicle.getLaneIndex(self.ego_id) + offset
            if edge.startswith(":") or li < 0 or li >= traci.edge.getLaneNumber(edge):
                return None
            my_pos = traci.vehicle.getLanePosition(self.ego_id)
            my_len = traci.vehicle.getLength(self.ego_id)
            best = None
            for vid in traci.lane.getLastStepVehicleIDs(f"{edge}_{li}"):
                p, ln = traci.vehicle.getLanePosition(vid), traci.vehicle.getLength(vid)
                if leader and p > my_pos - my_len:
                    g = max(p - ln - my_pos, 0.0)
                elif not leader and p - ln < my_pos:
                    g = max(my_pos - my_len - p, 0.0)
                else:
                    continue
                if best is None or g < best[0]:
                    best = (g, float(traci.vehicle.getSpeed(vid)))
            return best
        except traci.TraCIException:
            return None

    def _lane_connectivity(self, offset: int) -> float:
        """오프셋 차선의 '끊김 정도'를 반환.

        -1 : 그 차선이 존재하지 않음
         1 : visibility 이내에서 계속 이어짐
        0~1: (차선이 끝나는 지점까지의 거리 / visibility)
             → 값이 작을수록 곧 차선이 사라진다는 경고

        일반화 원리: 차선의 남은 길이와, 차선 끝에서 다음 edge로의
        연결(link) 존재 여부를 SUMO 네트워크에서 직접 조회한다.
        차선 감소/합류 지점은 "링크 없는 차선의 끝"으로 자연히 감지된다.
        """
        try:
            edge = traci.vehicle.getRoadID(self.ego_id)
            if edge.startswith(":"):          # 교차로(junction) 내부 통과 중
                return 1.0
            lane_index = traci.vehicle.getLaneIndex(self.ego_id) + offset
            if lane_index < 0 or lane_index >= traci.edge.getLaneNumber(edge):
                return -1.0                   # 그런 차선이 없음
            lane_id = f"{edge}_{lane_index}"
            # 보도 · 버스전용 차로처럼 승용차가 못 쓰는 차로는 "없는 차로"로 본다 (11/12 보도 추가)
            _allowed = traci.lane.getAllowed(lane_id)
            if _allowed and "passenger" not in _allowed:
                return -1.0
            remaining = (traci.lane.getLength(lane_id)
                         - traci.vehicle.getLanePosition(self.ego_id))
            if remaining > self.visibility:
                return 1.0                    # 시야 내에선 계속 이어짐
            if traci.lane.getLinks(lane_id):
                return 1.0                    # edge가 끝나도 다음으로 연결됨
            return float(np.clip(remaining / self.visibility, 0.0, 1.0))
        except traci.TraCIException:
            return 1.0                        # 조회 실패 시 중립값

    def _lane_density(self, offset: int) -> float:
        """오프셋 차선의 '전방 차량 밀도'를 0~1로 반환.

        밀도 = (시야 W 이내 전방 차량 수) / (그 구간의 최대 수용 대수)
               최대 수용 대수 = 사용가능거리 / density_veh_length

        인코딩 관례 (막힘 정도로 읽히도록):
            1.0 : 차선이 없거나 사실상 진입 불가  ← "꽉 찬 것"처럼 취급해서
                  에이전트가 자연스럽게 회피하게 만든다
            0.0 : 텅 빈 차선
        일반화: 차선이 시야보다 짧으면 link를 따라 다음 차선까지 이어서 센다.
        곧 끊기는 차선(link 없음)은 남은 거리만큼만 세므로 분모가 줄어
        같은 대수여도 밀도가 높게 나온다 → "끊기는 차선에 차가 몰려있음" 신호.
        """
        try:
            edge = traci.vehicle.getRoadID(self.ego_id)
            if edge.startswith(":"):          # 교차로 내부: 중립값
                return 0.0
            lane_index = traci.vehicle.getLaneIndex(self.ego_id) + offset
            if lane_index < 0 or lane_index >= traci.edge.getLaneNumber(edge):
                return 1.0                    # 차선 없음 = 꽉 참 취급
            lane_id = f"{edge}_{lane_index}"
            # [12팀] 보도 · 버스전용처럼 승용차가 못 쓰는 차로도 "없는 차로" (_lane_connectivity · 차로 변경과 같게, 11/15)
            _allowed = traci.lane.getAllowed(lane_id)
            if _allowed and "passenger" not in _allowed:
                return 1.0
            ego_pos = traci.vehicle.getLanePosition(self.ego_id)
            remaining = traci.lane.getLength(lane_id) - ego_pos
            links = traci.lane.getLinks(lane_id)

            # 이 차선에서 앞으로 실제로 쓸 수 있는 거리
            usable = self.visibility if links else min(self.visibility, remaining)
            if usable <= 0:
                return 1.0

            count = 0
            # (a) 현재 차선 구간의 전방 차량
            for vid in traci.lane.getLastStepVehicleIDs(lane_id):
                if vid == self.ego_id:
                    continue
                d = traci.vehicle.getLanePosition(vid) - ego_pos
                if 0.0 < d <= min(self.visibility, remaining):
                    count += 1
            # (b) 시야가 edge 끝을 넘고 다음 차선으로 이어지면 이어서 센다
            if links and remaining < self.visibility:
                next_lane_id = links[0][0]    # 첫 번째 연결 차선
                for vid in traci.lane.getLastStepVehicleIDs(next_lane_id):
                    if traci.vehicle.getLanePosition(vid) <= \
                            self.visibility - remaining:
                        count += 1

            max_count = usable / self.density_veh_len
            return float(np.clip(count / max_count, 0.0, 1.0))
        except traci.TraCIException:
            return 0.0

    def get_detected_vehicles(self):
        """현재 관측 시야에 들어온(=감지된) 배경차들의 좌표 목록을 반환.

        데이터셋 수집용 부가 정보 — 관측 벡터에는 들어가지 않는다.
        "감지"의 정의는 관측과 동일: 관측 대상 차선(obs_offsets 범위)에서
        ego 기준 종방향 ±visibility(W) 이내에 있는 차량.

        Returns: list[dict]  각 원소 =
            {"id": 차량 id, "x": 절대 x, "y": 절대 y,
             "rel_pos": ego 기준 종방향 상대거리(m, 앞 +), "speed": m/s,
             "lane_offset": ego 차선 기준 오프셋(+왼쪽)}
        ego가 없으면 빈 리스트.
        """
        out = []
        if self.ego_id not in traci.vehicle.getIDList():
            return out
        try:
            edge = traci.vehicle.getRoadID(self.ego_id)
            if edge.startswith(":"):
                return out
            ego_lane = traci.vehicle.getLaneIndex(self.ego_id)
            ego_pos = traci.vehicle.getLanePosition(self.ego_id)
            n_lanes = traci.edge.getLaneNumber(edge)
            for offset in self.obs_offsets:
                li = ego_lane + offset
                if li < 0 or li >= n_lanes:
                    continue
                for vid in traci.lane.getLastStepVehicleIDs(f"{edge}_{li}"):
                    if vid == self.ego_id:
                        continue
                    rel = traci.vehicle.getLanePosition(vid) - ego_pos
                    if abs(rel) > self.visibility:
                        continue
                    x, y = traci.vehicle.getPosition(vid)
                    out.append({"id": vid, "x": float(x), "y": float(y),
                                "rel_pos": float(rel),
                                "speed": float(traci.vehicle.getSpeed(vid)),
                                "lane_offset": int(offset)})
        except traci.TraCIException:
            pass
        return out

    def _get_obs(self):
        # ego가 이미 사라졌다면(충돌/도착) 0벡터 반환 — 종료 직전 마지막 관측용
        if self.ego_id not in traci.vehicle.getIDList():
            return np.zeros(self.obs_dim, dtype=np.float32)

        v = traci.vehicle.getSpeed(self.ego_id)
        obs = [v / self.vmax]                      # [0] 내 절대속도

        # ----- 5개 차선(좌2→좌1→현재→우1→우2)의 선행/후행 상대거리·상대속도 -----
        # [12팀 11/19] lane_dir_obs 켬: 옆 차로 차량을 getNeighbors · 차로 직접 스캔 중 가까운 쪽으로, 나란히 있는 차는 거리 0 (_neighbor_seen)
        nb = self._neighbor_seen if self.lane_dir_obs else self._neighbor
        for offset in self.obs_offsets:
            lead = nb(offset, leader=True)
            if lead is None:
                obs += [1.0, 1.0]                  # 없음: 멀고 + 위협 없음
            else:
                gap, lv = lead
                obs += [gap / self.visibility,
                        float(np.clip((lv - v) / self.vmax, -1.0, 1.0))]

            foll = nb(offset, leader=False)
            if foll is None:
                obs += [-1.0, -1.0]                # 없음: 멀고 + 위협 없음
            else:
                gap, fv = foll
                obs += [-gap / self.visibility,    # 후행은 음수 부호
                        float(np.clip((fv - v) / self.vmax, -1.0, 1.0))]

        # ----- 5개 차선의 연결성 (같은 순서) -----
        for offset in self.obs_offsets:
            obs.append(self._lane_connectivity(offset))

        # ----- 5개 차선의 전방 차량 밀도 (같은 순서) -----
        for offset in self.obs_offsets:
            obs.append(self._lane_density(offset))

        # ----- [12팀] 경로 · 신호 · 선택 관측 — 블록 순서는 __init__의 self._obs_blocks (칸 번호표 self.obs_index와 같은 순서) -----
        #   route: 다음 분기 방향 · 거리 (11/19 lane_dir_obs 켬: 분기 2개 새 정의 — route2 · reach도 같은 계산에서 나온다)
        #   signal: 앞 신호색 · 정지선 거리 / sig_time: 신호 남은 시간 (C-ITS) / conflict: 경합 차량 / speed: 차로 제한속도
        if self.lane_dir_obs:
            route8 = self._route_obs2()
            parts = {"route": route8[0:2], "route2": route8[2:4], "reach": route8[4:8]}
        else:
            parts = {"route": self._route_obs()}
        for blk in self._obs_blocks:
            if blk in parts:
                obs += parts[blk]
            elif blk == "signal":
                obs += self._signal_obs()
            elif blk == "sig_time":
                obs += self._signal_time_obs()
            elif blk == "conflict":
                obs += self._conflict_obs()
            elif blk == "speed":
                obs += self._speed_limit_obs()

        return np.clip(np.array(obs, dtype=np.float32), -1.0, 1.0)

    # ==================================================================
    # [12팀 11/15] 경로 · 신호 관측 (mdp_config.OBSERVATION 주석 [31..38])
    # ==================================================================
    _SIG_OBS = {"green": 1.0, "yellow": 0.0, "red": -1.0, "off": 1.0}

    # SUMO 연결 방향 → 관측 값 (차로 변경과 같은 부호: 왼쪽 +1 · 오른쪽 -1). t = 유턴(왼쪽으로 돈다)
    _TURN_DIR = {"l": 1.0, "L": 1.0, "t": 1.0, "s": 0.0, "r": -1.0, "R": -1.0}

    def _route_obs(self):
        """[31] 다음 분기에서 경로가 가는 방향 (좌 +1 · 직진 0 · 우 -1, 분기 없음 0), [32] 다음 분기까지 거리/route_horizon.

        내비게이션의 "300 m 앞 우회전"과 같은 지도 수준 정보만 준다 — 어느 차로로 가야 하는지는 주지 않는다
        (11/17 조교 공지 "1차 컨설팅 후 자주 묻는 질문": 라우팅은 가능, 차선 같은 세밀한 주행 정보는 안 됨 →
        11/15~17의 차로별 경로 계속 거리 [31..35](getBestLanes)를 뺐다). 교차로 안(내부 차로)에서는 직전 값을 쓴다."""
        try:
            edge = traci.vehicle.getRoadID(self.ego_id)
            if edge.startswith(":"):
                return list(self._last_route_obs)
            dist, turn = self._next_branch(edge)
            out = [turn, dist / self.route_horizon]
            self._last_route_obs = out
            return list(out)
        except traci.TraCIException:
            return list(self._last_route_obs)

    def _is_branch(self, edge: str) -> bool:
        """엣지 끝에서 승용차가 둘 이상의 다른 도로로 나갈 수 있으면 True (분기 · 교차로 · 회전교차로 출구).

        나가는 차로도 승용차가 들어갈 수 있어야 센다 (11/16 검토 E: 보행화 후 잠수교처럼 보행자 전용으로 막은 도로를
        분기로 세고 있었다). 막는 도로가 시나리오마다 달라서 캐시는 (시나리오, 엣지)별."""
        return len(self._branch_outs(edge)) > 1

    def _branch_outs(self, edge: str) -> set:
        """엣지 끝에서 승용차가 나갈 수 있는 다른 도로들 (승용차 차로에서 승용차 차로로 가는 연결만). (시나리오, 엣지)별 캐시."""
        key = (self.scenario, edge)
        if key not in self._branch_cache:
            outs = set()
            for i in range(traci.edge.getLaneNumber(edge)):
                lid = f"{edge}_{i}"
                allowed = traci.lane.getAllowed(lid)
                if allowed and "passenger" not in allowed:
                    continue
                for lk in traci.lane.getLinks(lid):
                    to_allowed = traci.lane.getAllowed(lk[0])
                    if to_allowed and "passenger" not in to_allowed:
                        continue
                    outs.add(lk[0].rsplit("_", 1)[0])
            self._branch_cache[key] = outs
        return self._branch_cache[key]

    def _next_branch(self, edge: str):
        """지금 위치에서 경로를 따라 다음 분기 엣지 끝까지 거리 (m, route_horizon에서 자름)와 그 분기에서 경로가 가는 방향.
        route_horizon 안에 분기가 없으면 (route_horizon, 0)."""
        route = traci.vehicle.getRoute(self.ego_id)
        idx = traci.vehicle.getRouteIndex(self.ego_id)
        d = traci.lane.getLength(traci.vehicle.getLaneID(self.ego_id)) - traci.vehicle.getLanePosition(self.ego_id)
        while d < self.route_horizon:
            e = route[idx]
            if idx + 1 >= len(route):
                return self.route_horizon, 0.0           # 경로 끝 — 더 갈라질 곳 없음
            if self._is_branch(e):
                return max(d, 0.0), self._turn_dir(e, route[idx + 1])
            idx += 1
            d += traci.lane.getLength(f"{route[idx]}_0")
        return self.route_horizon, 0.0

    def _turn_dir(self, edge: str, nxt: str) -> float:
        """edge 끝에서 nxt 도로로 가는 연결의 방향 (SUMO 연결 방향 l · L · t / s / r · R → +1 / 0 / -1).
        차로마다 방향 표기가 다를 수 있어 가장 많은 값을 쓴다. 도로망은 고정이라 (edge, nxt)별로 한 번만 계산."""
        key = (edge, nxt)
        if key not in self._turn_cache:
            votes = []
            for i in range(traci.edge.getLaneNumber(edge)):
                for lk in traci.lane.getLinks(f"{edge}_{i}"):
                    if lk[0].rsplit("_", 1)[0] == nxt and len(lk) > 6:
                        votes.append(self._TURN_DIR.get(lk[6], 0.0))
            self._turn_cache[key] = max(set(votes), key=votes.count) if votes else 0.0
        return self._turn_cache[key]

    # ------------------------------------------------------------------
    # [12팀 11/19] 분기 2개 · 차로 진행 방향 (lane_dir_obs — mdp_config 주석 [31..38])
    # 설계 원본은 sumolib로 쓴 정적 검증 스크립트 (rank · lane_reach · route_dirs) — 같은 정의를 traci로 옮겼다.
    # ------------------------------------------------------------------
    _LANE_REACH_DEPTH = 12        # 차로 연결을 따라가는 최대 단계 (설계 원본과 같음)

    @staticmethod
    def _shape_point(shape, offset):
        """폴리라인 shape 위 시작에서 offset m 지점 (길이를 넘으면 끝점) — sumolib.geomhelper.positionAtShapeOffset과 같은 뜻."""
        rest = float(offset)
        for (xa, ya), (xb, yb) in zip(shape[:-1], shape[1:]):
            seg = float(np.hypot(xb - xa, yb - ya))
            if seg > 0.0 and rest <= seg:
                t = rest / seg
                return xa + t * (xb - xa), ya + t * (yb - ya)
            rest -= seg
        return shape[-1]

    def _car_lane(self, lane: str) -> bool:
        allowed = traci.lane.getAllowed(lane)
        return not allowed or "passenger" in allowed

    def _lane_outs(self, lane: str):
        """차로 lane 끝에서 승용차가 들어갈 수 있는 다음 차로들 [(차로 id, 길이)]. (시나리오, 차로)별 캐시."""
        key = (self.scenario, lane)
        if key not in self._lane_out_cache:
            self._lane_out_cache[key] = [(lk[0], float(traci.lane.getLength(lk[0])))
                                         for lk in traci.lane.getLinks(lane) if self._car_lane(lk[0])]
        return self._lane_out_cache[key]

    def _rank(self, edge: str) -> dict:
        """분기 엣지 edge에서 나가는 도로들을 진행 방향 기준 왼쪽 → 오른쪽으로 순위: {도로: +1 맨 왼쪽 · 0 사이 · -1 맨 오른쪽}.
        좌우 = 분기 엣지 끝의 진행 방향 벡터(0번 차로 모양의 마지막 구간)에 대해, 나가는 도로 0번 차로의 30 m 지점(짧으면 끝)이
        왼쪽으로 얼마나 있는가. (시나리오, 엣지)별 캐시 (나가는 도로 목록이 시나리오마다 다를 수 있다)."""
        key = (self.scenario, edge)
        if key not in self._rank_cache:
            sh = traci.lane.getShape(f"{edge}_0")
            (x0, y0), (x1, y1) = sh[-2], sh[-1]
            hx, hy = x1 - x0, y1 - y0
            n = max(float(np.hypot(hx, hy)), 1e-9)
            hx, hy = hx / n, hy / n

            def lat(o):
                lane0 = f"{o}_0"
                px, py = self._shape_point(traci.lane.getShape(lane0), min(30.0, float(traci.lane.getLength(lane0))))
                return -(px - x1) * hy + (py - y1) * hx

            outs = sorted(sorted(self._branch_outs(edge)), key=lat, reverse=True)
            self._rank_cache[key] = {o: (1.0 if i == 0 else -1.0 if i == len(outs) - 1 else 0.0) for i, o in enumerate(outs)}
        return self._rank_cache[key]

    def _lane_reach_terms(self, lane: str):
        """차로 lane을 차로 변경 없이 따라가면 만나는 첫째 · 둘째 분기: 분기마다 [(lane 끝에서 그 분기 엣지 끝까지 m, 갈 수 있는 쪽 집합)].
        연결이 갈라지면 갈래마다 한 항목 (합집합은 _lane_reach에서 지금 위치로 거리를 판정한 뒤). AV 경로는 쓰지 않는다.
        거리는 위치와 무관한 부분만 캐시하고 지금 위치의 남은 거리는 _lane_reach에서 더한다."""
        key = (self.scenario, lane)
        if key in self._lane_reach_cache:
            return self._lane_reach_cache[key]
        H = self.branch_horizon
        terms = ([], [])

        def go(l, k, off, depth):
            if off > H or depth > self._LANE_REACH_DEPTH:
                return
            e = l.rsplit("_", 1)[0]
            outs = self._lane_outs(l)
            if self._is_branch(e):
                rk = self._rank(e)
                mine = frozenset(rk[t.rsplit("_", 1)[0]] for t, _ in outs if t.rsplit("_", 1)[0] in rk)
                terms[1 - k].append((off, mine))      # k = 남은 분기 수 - 1 → 0이면 이 분기가 찾던 것
                if k == 0:
                    return
                for t, ln in outs:
                    go(t, k - 1, off + ln, depth + 1)
                return
            for t, ln in outs:
                go(t, k, off + ln, depth + 1)

        go(lane, 1, 0.0, 0)
        self._lane_reach_cache[key] = terms
        return terms

    def _lane_reach(self, lane: str, d0: float):
        """[35..38]: 지금 차로(lane, 차로 끝까지 d0 m)에서 첫째 · 둘째 분기의 (왼쪽 가능, 오른쪽 가능) — 안 보이면 (1, 1)."""
        out = []
        for k_terms in self._lane_reach_terms(lane):
            seen = [s for off, s in k_terms if d0 + off <= self.branch_horizon]
            if not seen:
                out += [1.0, 1.0]
                continue
            S = frozenset().union(*seen)
            out += [1.0 if (1.0 in S or 0.0 in S) else 0.0, 1.0 if (-1.0 in S or 0.0 in S) else 0.0]
        return out

    def _route_dirs(self):
        """경로의 앞 분기 2개 [(경로가 가는 쪽 +1/0/-1, 분기 엣지 끝까지 m)] — branch_horizon 안의 것만."""
        route = traci.vehicle.getRoute(self.ego_id)
        idx = traci.vehicle.getRouteIndex(self.ego_id)
        d = traci.lane.getLength(traci.vehicle.getLaneID(self.ego_id)) - traci.vehicle.getLanePosition(self.ego_id)
        out = []
        for j in range(idx, len(route) - 1):
            if d > self.branch_horizon:
                break
            e = route[j]
            if self._is_branch(e):
                out.append((self._rank(e).get(route[j + 1], 0.0), max(d, 0.0)))
                if len(out) == 2:
                    break
            d += traci.lane.getLength(f"{route[j + 1]}_0")
        return out

    def _route_obs2(self):
        """lane_dir_obs 켬: [31..38] 8개 (첫 분기 방향 · 거리, 둘째 분기 방향 · 거리, 내 차로의 진행 가능 방향 4개 — 칸 번호는 obs_index).
        교차로 안(내부 차로)에서는 8개 모두 직전 값."""
        last = list(self._last_route_obs) + list(self._last_lane_dir_obs)
        try:
            edge = traci.vehicle.getRoadID(self.ego_id)
            if edge.startswith(":"):
                return last
            H = self.branch_horizon
            dirs = self._route_dirs()
            r = []
            for k in range(2):
                r += [float(dirs[k][0]), float(np.clip(dirs[k][1] / H, 0.0, 1.0))] if k < len(dirs) else [0.0, 1.0]
            lane = traci.vehicle.getLaneID(self.ego_id)
            d0 = traci.lane.getLength(lane) - traci.vehicle.getLanePosition(self.ego_id)
            reach = self._lane_reach(lane, d0)
            self._last_route_obs = r[:2]
            self._last_lane_dir_obs = r[2:] + reach
            return r + reach
        except traci.TraCIException:
            return last

    def _wrong_lane(self) -> bool:
        """[12팀 11/19] 잘못된 차로: 일반 엣지 위에서 차로 끝까지 wrong_lane_end_m 이내이고, 지금 차로에서 차로 변경 없이
        경로를 따라 (짧은 엣지는 건너뛰어) 다음 긴 엣지까지 갈 수 없다 (_lane_continues)."""
        try:
            road = traci.vehicle.getRoadID(self.ego_id)
            if not road or road.startswith(":"):
                return False
            route = traci.vehicle.getRoute(self.ego_id)
            idx = traci.vehicle.getRouteIndex(self.ego_id)
            if idx + 1 >= len(route) or route[idx] != road:
                return False
            lane = traci.vehicle.getLaneID(self.ego_id)
            if traci.lane.getLength(lane) - traci.vehicle.getLanePosition(self.ego_id) > self.wrong_lane_end_m:
                return False
            return not self._lane_continues(lane, tuple(route), idx)
        except traci.TraCIException:
            return False

    SHORT_EDGE = 40.0    # m. [12팀 11/19] 이보다 짧은 엣지는 차로를 바꿀 수 없는 구간으로 보고 잘못된 차로 판정에서 건너뛴다

    def _lane_continues(self, lane: str, route: tuple, idx: int) -> bool:
        """지금 차로(route[idx] 위)에서 차로 변경 없이 승용차 연결을 따라가, 경로의 다음 엣지들 중 SHORT_EDGE 미만 엣지를 연달아 지나
        처음 만나는 긴 엣지(또는 경로 끝)까지 갈 수 있는가. (예: goga_nb 0번 → m1(6 m) 2번은 이어지지만 그 다음 und_n으로는 못 간다.)
        (시나리오, 차로, 경로 인덱스, 경로)로 캐시."""
        key = (self.scenario, lane, idx, route)
        if key in self._continue_cache:
            return self._continue_cache[key]
        frontier, j, ok = {lane}, idx, True
        while j + 1 < len(route):
            nxt = route[j + 1]
            frontier = {t for l in frontier for t, _ in self._lane_outs(l) if t.rsplit("_", 1)[0] == nxt}
            if not frontier:
                ok = False
                break
            j += 1
            if j + 1 >= len(route) or traci.lane.getLength(f"{nxt}_0") >= self.SHORT_EDGE:
                break                                   # 긴 엣지(또는 경로 끝)에 닿음
        self._continue_cache[key] = ok
        return ok

    def _conflict_obs(self):
        """[35..38] 교차로 · 합류에서 AV와 같은 곳을 지나려는 경합 차량 (회전교차로 진입 · 합류부 끼어들기 — 같은 차로 · 옆 차로
        관측으로는 바로 앞에 올 때까지 안 보이고 안전장치도 못 막는다, 11/16 검토).

        traci.vehicle.getJunctionFoes(ego, conflict_horizon): AV 앞 conflict_horizon m 안의 교차로에서 AV 경로와 겹치는 연결로
        다가오는 차들 (경합 지점까지 AV · 상대 거리, 각자 그 지점을 벗어나는 거리, 양보 여부). 그중 이미 지나간 차는 빼고,
        경합 지점 도착 시각이 AV와 가장 가까운 차 하나를 고른다 (멀리 있는 차까지 섞여 나오므로).
          [35] AV -> 경합 지점 거리 / conflict_horizon (0~1, 없으면 1)
          [36] 경합 차량 -> 같은 지점 거리 / conflict_horizon (0~1, 없으면 1)
          [37] 경합 차량 속도 / vmax (0~1, 없으면 0)
          [38] 우선권: AV 우선(상대가 양보) +1 · AV가 양보 -1 · 대등(합류 등)이거나 없음 0
        교차로 안에서도 그대로 계산한다 (지점까지 거리가 0에 가까워짐)."""
        none = [1.0, 1.0, 0.0, 0.0]
        H = self.conflict_horizon
        try:
            foes = self._junction_foes(H)
            v_ego = float(traci.vehicle.getSpeed(self.ego_id))
        except traci.TraCIException:
            return none
        best, best_gap, best_v = None, None, 0.0
        for f in foes:
            foe_id, ego_d, foe_d = f[0], float(f[1]), float(f[2])
            try:
                v_f = float(traci.vehicle.getSpeed(foe_id))
            except traci.TraCIException:
                continue
            gap = abs(max(ego_d, 0.0) / max(v_ego, 0.5) - max(foe_d, 0.0) / max(v_f, 0.5))   # 도착 시각 차 (s)
            if best is None or gap < best_gap:
                best, best_gap, best_v = f, gap, v_f
        if best is None:
            return none
        prio = -1.0 if best[7] else (1.0 if best[8] else 0.0)
        return [float(np.clip(float(best[1]) / H, 0.0, 1.0)), float(np.clip(float(best[2]) / H, 0.0, 1.0)),
                float(np.clip(best_v / self.vmax, 0.0, 1.0)), prio]

    def _junction_foes(self, H):
        """AV 앞 H m 안 교차로 · 합류의 경합 차량 (getJunctionFoes 중 AV · 상대 둘 다 아직 경합 지점을 벗어나지 않은 것).
        관측 [35..38]과 보상의 먼 정지 감점(경합 대기 제외)이 같이 쓴다 (11/18 함수로 뺌). TraCIException은 호출한 쪽이 처리."""
        return [f for f in traci.vehicle.getJunctionFoes(self.ego_id, H)
                if float(f[3]) > 0.0 and float(f[4]) > 0.0]          # 둘 중 하나가 이미 경합 지점을 벗어난 차는 뺀다

    def _speed_limit_obs(self):
        """[39..41] 차로 제한속도 (속도 보상의 기준이자 속도 명령 상한인데 관측에 없었다 — 30 km/h 지하차도 · 회전교차로 앞에서
        환경이 강제로 줄이는 것을 미리 대비할 수 없었다, 11/16 검토).
          [39] 지금 차로 제한속도 / vmax (교차로 안이면 그 내부 차로 값 — 속도 상한과 같은 기준)
          [40] 경로를 따라 다음에 바뀌는 제한속도 / vmax (route_horizon 안에 없으면 지금 값)
          [41] 그 지점까지 거리 / route_horizon (0~1, 없으면 1)"""
        H = self.route_horizon
        try:
            lane = traci.vehicle.getLaneID(self.ego_id)
            cur = float(traci.lane.getMaxSpeed(lane))
            route = traci.vehicle.getRoute(self.ego_id)
            i = int(traci.vehicle.getRouteIndex(self.ego_id)) + 1
            d = float(traci.lane.getLength(lane)) - float(traci.vehicle.getLanePosition(self.ego_id))
            nxt, dist = cur, None
            while d < H and 0 < i < len(route):
                e = route[i]
                if e not in self._edge_speed:
                    self._edge_speed[e] = (float(traci.lane.getMaxSpeed(f"{e}_0")), float(traci.lane.getLength(f"{e}_0")))
                lim, length = self._edge_speed[e]
                if abs(lim - cur) > 0.01:
                    nxt, dist = lim, max(d, 0.0)
                    break
                d += length
                i += 1
        except traci.TraCIException:
            return [1.0, 1.0, 1.0]
        return [float(np.clip(cur / self.vmax, 0.0, 1.0)), float(np.clip(nxt / self.vmax, 0.0, 1.0)),
                1.0 if dist is None else float(np.clip(dist / H, 0.0, 1.0))]

    def _signal_obs(self):
        """[33] 경로상 앞 신호의 AV 방향 색(녹 1 · 황 0 · 적 -1, 없음 · 꺼짐 1), [34] 정지선까지 거리/signal_horizon (없으면 1)."""
        try:
            nxt = traci.vehicle.getNextTLS(self.ego_id)
        except traci.TraCIException:
            nxt = None
        if not nxt:
            return [1.0, 1.0]
        _tl, _link, dist, st = nxt[0]
        grp = _SIG_GROUP.get(st, "red")
        if grp == "off":
            return [1.0, 1.0]
        return [self._SIG_OBS[grp], float(np.clip(dist / self.signal_horizon, 0.0, 1.0))]

    def _signal_time_obs(self):
        """[42] 앞 신호의 AV 방향 색이 바뀔 때까지 남은 시간 / signal_time_horizon (0~1).
        앞에 신호 없음 · 꺼짐 · 한 주기 내내 같은 색이면 1. 정의는 데이터 칼럼 signal.remaining_s와 같다 (_tls_remaining)."""
        try:
            nxt = traci.vehicle.getNextTLS(self.ego_id)
        except traci.TraCIException:
            nxt = None
        if not nxt:
            return [1.0]
        tl, link, _dist, st = nxt[0]
        rem = self._tls_remaining(tl, link, st)
        return [1.0 if rem is None else float(np.clip(rem / self.signal_time_horizon, 0.0, 1.0))]

    # ==================================================================
    # 1 스텝 진행: 행동 적용 → 시뮬레이션 1스텝 → 보상/종료 판정
    # ==================================================================
    def _quantize_lane_change(self, raw: float) -> int:
        """연속 policy 출력을 실제 차선 명령 {-1,0,+1}로 변환한다.

        +1 = 왼쪽, 0 = 유지, -1 = 오른쪽. SUMO 차선 index가 왼쪽으로
        갈수록 증가하는 규약과 동일해서 해석이 직관적이다.
        """
        raw = float(np.clip(raw, -1.0, 1.0))
        if raw >= self.lane_change_threshold:
            return 1
        if raw <= -self.lane_change_threshold:
            return -1
        return 0

    def _scan_lane_neighbors(self, direction: int):
        """목표 차선을 직접 스캔해 (가장 가까운 앞차 gap·속도, 뒷차 gap·속도) 반환.

        getNeighbors의 백업. getNeighbors는 SUMO 차선변경모델의 이웃 캐시를
        읽는데, "방금 스폰된 차량"은 다음 스텝까지 이 캐시에 없어서
        도로 입구 근처에서 사각지대가 생긴다. 차선의 차량 목록을 직접
        훑으면 이미 삽입된 차량은 전부 잡힌다.
        """
        try:
            edge = traci.vehicle.getRoadID(self.ego_id)
            li = traci.vehicle.getLaneIndex(self.ego_id) + direction
            # 존재하지 않는 차선이면 조회하지 않는다.
            # (없는 lane id를 traci.lane.*로 조회하면 SUMO가 매번
            #  에러 로그를 찍으므로, 범위 검사를 먼저 해서 조용히 건너뛴다)
            if edge.startswith(":") or li < 0 \
                    or li >= traci.edge.getLaneNumber(edge):
                return None, None
            lane_id = f"{edge}_{li}"
            my_pos = traci.vehicle.getLanePosition(self.ego_id)
            front, rear = None, None
            for vid in traci.lane.getLastStepVehicleIDs(lane_id):
                d = traci.vehicle.getLanePosition(vid) - my_pos
                if d >= 0:
                    g = d - traci.vehicle.getLength(vid)
                    if front is None or g < front[0]:
                        front = (g, traci.vehicle.getSpeed(vid))
                else:
                    g = -d - traci.vehicle.getLength(self.ego_id)
                    if rear is None or g < rear[0]:
                        rear = (g, traci.vehicle.getSpeed(vid))
            return front, rear
        except traci.TraCIException:
            return None, None

    def _lane_change_is_safe(self, direction: int) -> bool:
        """목표 차선으로 지금 이동해도 측면충돌이 나지 않는지 검사한다.

        3중 안전 규칙:
          (0) 입구 보호구간: ego가 도로 시작 spawn_protect(m) 이내면 변경 금지.
              flow가 입구(0m)에서 차를 새로 만드는데, 생성 "직전" 차량은
              어떤 API로도 미리 볼 수 없어 원천적으로 사각지대이기 때문.
          (1) 앞: gap > min_front_gap + front_tau × 내속도
              + 접근 여유: gap > (내속도 − 앞차속도)⁺ × 1s   (내가 더 빠를 때)
          (2) 뒤: gap > min_rear_gap + rear_tau × 뒤차속도
              + 접근 여유: gap > (뒤차속도 − 내속도)⁺ × 1s   (뒤가 더 빠를 때)
        이웃은 getNeighbors 결과와 차선 직접 스캔 결과 중 "더 가까운 쪽"을
        쓴다 (스폰 직후 차량 사각지대 보완).
        """
        # 출발 뒤 달린 거리 기준 (직선 도로에서는 차선 위치와 같다. 실제 도로망은
        # 엣지가 짧아 차선 위치로 판정하면 짧은 엣지마다 차선변경이 막힌다)
        if traci.vehicle.getDistance(self.ego_id) < self.lc_spawn_protect:
            return False

        v = traci.vehicle.getSpeed(self.ego_id)
        scan_front, scan_rear = self._scan_lane_neighbors(direction)

        def closer(a, b):
            if a is None:
                return b
            if b is None:
                return a
            return a if a[0] <= b[0] else b

        lead = closer(self._neighbor(direction, leader=True), scan_front)
        if lead is not None:
            gap, lv = lead
            if gap <= self.lc_front_min + self.lc_front_tau * v:
                return False
            if gap <= max(v - lv, 0.0) * self.lc_horizon:  # 시간여유 내 접촉 위험
                return False

        foll = closer(self._neighbor(direction, leader=False), scan_rear)
        if foll is not None:
            gap, fv = foll
            if gap <= self.lc_rear_min + self.lc_rear_tau * fv:
                return False
            if gap <= max(fv - v, 0.0) * self.lc_horizon:  # 시간여유 내 접촉 위험
                return False
        return True

    def _apply_lane_change(self, lane_change: int) -> bool:
        """가능하고 "안전한" 경우에만 차선변경을 적용하고 성공 여부를 반환.

        안전 게이트(_lane_change_is_safe)를 통과하지 못한 명령은 실행하지
        않고 False를 반환한다 → step()에서 "불가능한 명령"과 동일하게
        invalid_action_penalty가 부과된다. 즉 에이전트 입장에서
        "위험한 변경 = 실행도 안 되고 감점"이라, 측면충돌로 즉사하며
        배우는 대신 안전한 타이밍을 고르는 쪽으로 학습이 유도된다.
        """
        if self.ego_id not in traci.vehicle.getIDList():
            return False
        try:
            edge = traci.vehicle.getRoadID(self.ego_id)
            if edge.startswith(":"):
                return False
            current = traci.vehicle.getLaneIndex(self.ego_id)
            n_lanes = traci.edge.getLaneNumber(edge)
            target = current + lane_change
            if target < 0 or target >= n_lanes:
                return False
            # 보도 · 버스전용 차로처럼 승용차가 못 쓰는 차로로는 바꾸지 않는다 (11/12 보도 추가)
            _allowed = traci.lane.getAllowed(f"{edge}_{target}")
            if lane_change != 0 and _allowed and "passenger" not in _allowed:
                return False
            # ★ 측면충돌 방지: 실제 이동 명령(0이 아닌 변경)에만 안전 검사
            if (lane_change != 0 and self.lc_safety_enabled
                    and not self._lane_change_is_safe(lane_change)):
                return False
            traci.vehicle.changeLane(
                self.ego_id, target, self.lane_change_duration)
            return True
        except traci.TraCIException:
            return False

    def _lane_limit(self) -> float:
        """지금 차로 제한속도 (m/s). 조회 실패 시 vmax."""
        try:
            return max(float(traci.lane.getMaxSpeed(traci.vehicle.getLaneID(self.ego_id))), 0.1)
        except traci.TraCIException:
            return self.vmax

    def _safe_speed(self):
        """[12팀] 안전장치: 같은 차로 앞차가 최대 감속(EGO decel)으로 서도 min_gap을 남기고 설 수 있는 이번 스텝 최고 속도.
        Krauss 안전 속도식 v = -b·τ + sqrt((b·τ)² + v_l² + 2·b·(gap - min_gap)), τ = 스텝 길이. 앞차가 없으면 None."""
        res = traci.vehicle.getLeader(self.ego_id, self.safety_lookahead)
        if res is None or res[0] == "":
            return None
        gap, vl = float(res[1]), float(traci.vehicle.getSpeed(res[0]))
        b, tau = self.max_decel, self.step_length
        return max(0.0, -b * tau + float(np.sqrt((b * tau) ** 2 + vl ** 2 + 2.0 * b * max(gap - self.safety_min_gap, 0.0))))

    RED_WAIT_RANGE = 150.0   # 이 거리(m) 안에 빨간불·황색 신호가 있으면 정지를 비정상으로 세지 않음

    def _waiting_at_red(self) -> bool:
        """ego 앞 신호(같은 경로 위)가 빨간불·황색이고 RED_WAIT_RANGE m 안이면 True (신호 대기)."""
        try:
            nxt = traci.vehicle.getNextTLS(self.ego_id)
        except traci.TraCIException:
            return False
        return bool(nxt) and nxt[0][2] <= self.RED_WAIT_RANGE and _SIG_GROUP.get(nxt[0][3]) in ("red", "yellow")

    def _far_stop_at_red(self, ego_speed, ego_gap) -> bool:
        """[12팀 11/18] 적색 신호 앞 먼 정지: 서 있고(1 m/s 미만) 경로상 앞 신호가 적색(황색 제외) · 정지선이 RED_WAIT_RANGE m 안인데
        정지선까지 red_far_stop_gap m 넘게 남았고, 같은 차로 앞차도 없거나 그만큼 멀다. 단 정지선보다 앞에 경합 차량이 있는
        교차로 · 합류 지점(회전교차로 진입 등)이 있으면 그 양보 대기일 수 있으므로 세지 않는다."""
        far = float(self.rw.get("red_far_stop_gap", 15.0))
        if ego_speed >= 1.0 or (ego_gap is not None and ego_gap <= far):
            return False
        try:
            nxt = traci.vehicle.getNextTLS(self.ego_id)
            if not nxt or _SIG_GROUP.get(nxt[0][3]) != "red":
                return False
            d_stop = float(nxt[0][2])
            if not (far < d_stop <= self.RED_WAIT_RANGE):
                return False
            return not any(float(f[1]) < d_stop for f in self._junction_foes(d_stop))
        except traci.TraCIException:
            return False

    def get_signal_state(self):
        """AV가 다음에 만날 신호 (데이터 칼럼 signal, privileged state — 정책 입력 아님). 앞에 신호가 없으면 None.

        tls: 신호 id — ix 신반포사거리 · xw 반포대로 북행 횡단보도 · s4032 반포대교남단 · jx1 잠수교남단 · jx2 잠수교
             (road_spec.SIGNALS, 5곳 모두 C-ITS 실측). 경로마다 지나는 신호가 다르다
             (side-banpo: ix → xw → s4032, goga-banpo: xw → s4032, goga-jamsu: jx1 → jx2)
        state: AV가 가는 방향(연결 차로)의 지금 신호 — "G" 녹색 · "g" 양보 녹색(비보호 · 보행 신호 중 우회전 등) ·
               "y" 황색 · "r" 적색 · "O" 꺼짐(보행화 후 잠수교)
        remaining_s: 그 신호색(녹색 G·g / 황색 / 적색)이 실제로 바뀔 때까지 남은 시간 (s). 꺼진 신호는 None.
             AV(정책)는 이 값을 모른다 — 신호 변화 예측(belief) 학습 · 빨간불 대기 분석의 정답으로 쓴다.
        (11/15 바꿈: 이전에는 신반포사거리 고정 {phase, remaining_s} — 신호가 1곳이던 11/12 설계가 남아 있었다)
        """
        try:
            nxt = traci.vehicle.getNextTLS(self.ego_id)
        except traci.TraCIException:
            return None
        if not nxt:
            return None
        tl, link, _dist, st = nxt[0]
        rem = self._tls_remaining(tl, link, st)
        return {"tls": tl, "state": st, "remaining_s": None if rem is None else round(rem, 2)}

    def _tls_remaining(self, tl, link, st):
        """신호 tl의 연결 link가 지금 색(st)에서 다른 색으로 바뀔 때까지 남은 시간 (s).
        꺼짐 · 현시 정보 없음 · 한 주기 내내 같은 색이면 None. 데이터 칼럼 signal과 관측 [42]이 같이 쓴다 (11/17 함수로 뺌)."""
        phases = self._tls_phases.get(tl)
        grp = _SIG_GROUP.get(st)
        if not phases or grp == "off":
            return None
        try:
            cur = int(traci.trafficlight.getPhase(tl))
            rem = float(traci.trafficlight.getNextSwitch(tl)) - float(traci.simulation.getTime())
            n = len(phases)
            for k in range(1, n + 1):                       # 다음 현시부터 한 주기를 돌며 같은 색이면 더한다
                d, s = phases[(cur + k) % n]
                if _SIG_GROUP.get(s[link]) != grp:
                    break
                rem += d
            else:
                return None                                 # 한 주기 내내 같은 색 (바뀌지 않음)
            return rem
        except (traci.TraCIException, IndexError):
            return None

    def get_sim_time(self) -> float:
        """[12팀 11/19] 지금 시뮬레이션 시각 (s) — 데이터 칼럼 sim_time. 신호 5곳은 고정 주기라 이 값으로 모든 신호 상태를 계산할 수 있다."""
        return float(traci.simulation.getTime())

    def get_privileged_state(self) -> dict:
        """POMDP 데이터셋용 simulator-side state.

        agent가 실제로 받는 입력은 _get_obs()의 부분관측이고, 이 메서드는
        분석/지도신호용 privileged state만 수집한다. 정책 입력으로 쓰지 않는다.
        """
        ids = traci.vehicle.getIDList()
        if self.ego_id not in ids:
            return {"ego_alive": False}
        try:
            x, y = traci.vehicle.getPosition(self.ego_id)
            # [12팀 수집 최적화] 차량마다 값 7개를 하나씩 묻던 것(스텝당 약 3,000번 왕복)을 TraCI 구독으로 바꿈:
            # 처음 보는 차량만 구독을 걸고, 이후에는 SUMO가 매 스텝 보내 주는 결과를 한 번에 읽는다.
            # 저장되는 값·순서는 이전과 같다 (getIDList 순서, 같은 변수).
            subs = traci.vehicle.getAllSubscriptionResults()
            vehicles = []
            for vid in ids:
                try:
                    r = subs.get(vid)
                    if r is None:
                        traci.vehicle.subscribe(vid, _STATE_VARS)
                        r = traci.vehicle.getSubscriptionResults(vid)
                    vx, vy = r[_tc.VAR_POSITION]
                    vehicles.append({
                        "id": vid,
                        "x": float(vx),
                        "y": float(vy),
                        "speed": float(r[_tc.VAR_SPEED]),
                        "lane": int(r[_tc.VAR_LANE_INDEX]),
                        "lane_id": r[_tc.VAR_LANE_ID],
                        "road_id": r[_tc.VAR_ROAD_ID],
                        "lane_pos": float(r[_tc.VAR_LANEPOSITION]),
                    })
                except (traci.TraCIException, KeyError):
                    continue
            return {
                "ego_alive": True,
                "ego_x": float(x),
                "ego_y": float(y),
                "ego_speed": float(traci.vehicle.getSpeed(self.ego_id)),
                "ego_lane": int(traci.vehicle.getLaneIndex(self.ego_id)),
                "ego_lane_id": traci.vehicle.getLaneID(self.ego_id),
                "ego_road_id": traci.vehicle.getRoadID(self.ego_id),
                "ego_lane_pos": float(traci.vehicle.getLanePosition(self.ego_id)),
                "vehicles": vehicles,
            }
        except traci.TraCIException:
            return {"ego_alive": False}

    def step(self, action):
        # action = [가감속 raw, 차선변경 raw]
        action_arr = np.asarray(action, dtype=np.float32).reshape(-1)
        if action_arr.size < self.act_dim:
            raise ValueError(
                f"action은 {self.act_dim}차원이어야 합니다. 받은 shape={np.asarray(action).shape}")

        accel_raw = float(np.clip(action_arr[self.accel_index], -1.0, 1.0))
        lane_raw = float(np.clip(action_arr[self.lane_change_index], -1.0, 1.0))
        lane_change = self._quantize_lane_change(lane_raw)
        lane_change_applied = False
        lane_before = None

        # ----- (1) 종방향 + 횡방향 행동 적용 -----
        v_before, road_before = None, None
        tls_before = None            # [12팀] 이번 스텝 직전 앞 신호 (tl, link, 정지선까지 m) — 신호 위반 판정
        safety_brake = False         # [12팀] 이번 스텝에 안전장치가 속도 명령을 낮췄는가
        if self.ego_id in traci.vehicle.getIDList():
            v = traci.vehicle.getSpeed(self.ego_id)
            v_before, road_before = v, traci.vehicle.getRoadID(self.ego_id)
            lane_before = int(traci.vehicle.getLaneIndex(self.ego_id))
            try:
                _n = traci.vehicle.getNextTLS(self.ego_id)
                tls_before = (_n[0][0], _n[0][1], float(_n[0][2])) if _n else None
            except traci.TraCIException:
                tls_before = None

            if not self.sumo_driver:
                accel_cmd = (accel_raw * self.max_accel
                             if accel_raw >= 0.0 else accel_raw * self.max_decel)
                dv = accel_cmd * self.step_length
                # [12팀 11/15] 속도 명령 상한 = 지금 차로 제한속도 (수업 원본은 vmax 22.2 m/s = AV 경로에 없는 올림픽대로 값).
                # 제한속도가 낮은 차로에 막 들어섰을 때 순간 감속하지 않게, 상한은 최대 감속으로 내려갈 수 있는 속도 아래로는 안 내린다.
                cap = max(min(self.vmax, self._lane_limit()), v - self.max_decel * self.step_length)
                target = float(np.clip(v + dv, 0.0, cap))
                if self.safety_brake:
                    v_safe = self._safe_speed()
                    if v_safe is not None:
                        # 안전장치도 한 스텝에 최대 감속(EGO decel)보다 세게는 못 줄인다 (11/16 검토 E: 1,314회 중 1회
                        # 물리 한계 이상 제동). 그래도 못 서면 충돌로 끝난다 — 안전장치는 물리를 넘지 않는다.
                        v_safe = max(v_safe, v - self.max_decel * self.step_length, 0.0)
                    if v_safe is not None and target > v_safe:
                        target, safety_brake = v_safe, True
                        self.safety_interventions += 1
                traci.vehicle.setSpeed(self.ego_id, target)

                lane_change_applied = self._apply_lane_change(lane_change)

        # ----- (2) 시뮬레이션 1스텝 -----
        traci.simulationStep()
        self.step_count += 1

        # ----- (3) 이벤트/관측 -----
        collided = self.ego_id in traci.simulation.getCollidingVehiclesIDList()
        arrived = self.ego_id in traci.simulation.getArrivedIDList()
        # 주의: SUMO는 충돌로 제거된 차량도 "도착 목록"에 넣는다. 그래서 도착은
        #   (충돌 아님) AND (경로의 마지막 도로에서 사라짐) 일 때만 인정한다.
        #   그 밖의 곳에서 사라지면 "경로 이탈"(off_route)로 따로 기록한다.
        off_route = False
        if arrived and (collided or (self.network and self._ego_road != self._route_last)):
            arrived = False
            off_route = not collided
        teleported = self.ego_id in traci.simulation.getStartingTeleportIDList()
        self.teleports += traci.simulation.getStartingTeleportNumber()
        alive = self.ego_id in traci.vehicle.getIDList() and not teleported
        # [12팀 11/15] 신호 위반: 직전 스텝 앞 신호의 정지선을 이번 스텝에 넘었고, 지금 그 방향이 적색이면 위반.
        # (SUMO는 스텝 시작에 신호를 바꾸고 차를 움직이므로 지금 색 = 넘은 순간의 색. 황색 진입은 위반 아님)
        # 정지선을 실제로 넘었을 때만 센다 = 접근 도로를 벗어남(교차로 안 · 다음 도로). 차로마다 신호 번호가 달라서
        # 정지선 앞 차로 변경만으로도 앞 신호가 바뀐다 (11/16 검토: 대기 중 차로 변경이 위반으로 잡혔다).
        red_light = False
        if alive and tls_before is not None and tls_before[2] <= (v_before or 0.0) * self.step_length + 5.0:
            try:
                _n = traci.vehicle.getNextTLS(self.ego_id)
                crossed = traci.vehicle.getRoadID(self.ego_id) != road_before
                if crossed and (not _n or (_n[0][0], _n[0][1]) != tls_before[:2]):
                    st = traci.trafficlight.getRedYellowGreenState(tls_before[0])[tls_before[1]]
                    red_light = _SIG_GROUP.get(st) == "red"
            except (traci.TraCIException, IndexError):
                pass
        lane_after = int(traci.vehicle.getLaneIndex(self.ego_id)) if alive else None
        if alive:
            _r = traci.vehicle.getRoadID(self.ego_id)
            if not _r.startswith(":"):
                self._ego_road = _r          # 교차로 내부를 뺀 마지막 도로 (도착 판정용)
        # [12팀 11/19] 잘못된 차로 조기 종료 (RL 모드만, SIMULATION["wrong_lane_end_m"]): 경로의 다음 엣지로 못 가는 차로에서
        # 차로 끝 wrong_lane_end_m 안에 들면 경로 이탈로 끝낸다 (그전에는 차로 끝에 서서 120 s 뒤 stuck으로 끝났다)
        wrong_lane = False
        if alive and not collided and not self.sumo_driver and self.wrong_lane_end_m > 0.0:
            wrong_lane = self._wrong_lane()
            off_route = off_route or wrong_lane

        # SUMO 기본 운전자 모드: 실제 움직임에서 행동을 거꾸로 계산해 기록한다 (BC 전문가 행동)
        if self.sumo_driver and alive and v_before is not None:
            acc = (traci.vehicle.getSpeed(self.ego_id) - v_before) / self.step_length
            accel_raw = float(np.clip(acc / (self.max_accel if acc >= 0 else self.max_decel), -1.0, 1.0))
            same_road = traci.vehicle.getRoadID(self.ego_id) == road_before
            lane_change = int(np.sign(lane_after - lane_before)) if same_road else 0
            lane_raw = float(lane_change)
            lane_change_applied = lane_change != 0
            self._last_sumo_accel = accel_raw
        elif self.sumo_driver:
            # 도착·순간이동으로 AV가 사라진 마지막 스텝은 속도를 읽을 수 없다.
            # 정책 출력(0)이 그대로 남지 않게 직전 스텝의 가감속을 유지하고, 차로 변경은 0으로 둔다.
            accel_raw, lane_raw = self._last_sumo_accel, 0.0

        obs = self._get_obs()
        terminated, truncated = False, False
        ego_speed, ego_gap = None, None
        far_stop = False

        # ----- (4) 기존 보상 유지 -----
        if collided:
            reward = -float(self.rw["collision_penalty"])
            terminated = True
        elif arrived:
            reward = float(self.rw["arrival_bonus"])
            terminated = True
        elif not alive:
            reward = 0.0
            terminated = True
        else:
            ego_speed = traci.vehicle.getSpeed(self.ego_id)
            same_lane_leader = self._neighbor(0, leader=True)
            if same_lane_leader is not None:
                ego_gap = same_lane_leader[0]
            # [12팀 11/15] 속도 보상 = 지금 차로 제한속도 대비 (과속해도 더 받지 않음)
            lane_limit = self._lane_limit()
            reward = float(self.rw["speed_weight"]) * min(ego_speed / lane_limit, 1.0)
            # [12팀 11/15] 근접 감점은 달리는 중(close_gap_min_speed 이상)에만 — 신호 대기열에 2 m 간격으로 서는 것은 정상이다
            # (수업 원본은 정지 중에도 매 스텝 감점 → 빨간불에 바르게 서도 리턴이 크게 깎였다)
            # [12팀 11/18] close_gap_headway(초) > 0이면 고정 거리 대신 차간 시간으로 판정: 간격 < close_gap_headway × 내 속도
            # (6 m 고정 기준은 15 m/s에서 0.4초라 고속 근접을 못 잡고, 저속에서는 너무 엄했다). 0이면 위 거리 기준 그대로.
            hw = float(self.rw.get("close_gap_headway", 0.0))
            close_lim = hw * ego_speed if hw > 0.0 else float(self.rw["close_gap_threshold"])
            if (ego_gap is not None and
                    ego_gap < close_lim
                    and ego_speed >= float(self.rw.get("close_gap_min_speed", 0.0))):
                reward -= float(self.rw["close_gap_penalty"])
            # [12팀 11/18] 접근 감점: 같은 차로 앞차에 빠르게 다가가는 중(충돌까지 남은 시간 TTC = 간격 / 접근 속도가
            # ttc_threshold 초 미만)이면 속도와 상관없이 감점. RL 1차 충돌은 거의 전부 신호 대기 줄 · 정체 꼬리에서 저속으로
            # 앞차에 다가가다 닿은 것이었다 — 위 근접 감점은 5 m/s 미만에서 꺼져 있어 이 구간에 신호가 없었다.
            # [12팀 11/18 보상 3차] 감점 크기를 TTC에 비례: ttc_penalty × clip(ttc_threshold / TTC, 1, ttc_max_scale).
            # ttc_max_scale = 1이면 이전처럼 고정 감점.
            tp = float(self.rw.get("ttc_penalty", 0.0))
            if tp > 0.0 and same_lane_leader is not None:
                closing = ego_speed - float(same_lane_leader[1])
                ttc_thr = float(self.rw.get("ttc_threshold", 2.0))
                if closing > 0.1 and ego_gap / closing < ttc_thr:
                    scale = min(max(ttc_thr / max(ego_gap / closing, 1e-6), 1.0), float(self.rw.get("ttc_max_scale", 1.0)))
                    reward -= tp * scale
            # [12팀 11/18 보상 3차] 적색 신호 앞 먼 정지 감점 (조건은 _far_stop_at_red). 0이면 끔.
            fp = float(self.rw.get("red_far_stop_penalty", 0.0))
            if fp > 0.0 and self._far_stop_at_red(ego_speed, ego_gap):
                far_stop = True
                reward -= fp

            # ---- 막힘 페널티: 느린 앞차 뒤에 갇혀 있는 동안 매 스텝 감점 ----
            # "차선을 유지한 채 막혀 있기"와 "변경으로 탈출하기" 사이에
            # 즉각적인 보상 차등을 만들어 추월 학습 신호를 조밀하게 한다.
            # (자세한 설계 이유는 environment/env/mdp_config.py의 REWARD 주석 참고)
            bp = float(self.rw.get("blocked_penalty", 0.0))
            if (bp > 0.0 and ego_gap is not None
                    and ego_gap < float(self.rw.get("blocked_gap", 20.0))
                    and ego_speed < float(self.rw.get("blocked_speed_frac", 0.7))
                        * lane_limit
                    and not self._waiting_at_red()):     # [12팀 11/15] 신호 대기열은 추월로 벗어날 막힘이 아님
                reward -= bp

            # ---- 차선변경 보상 항 ----
            # lane_change        : 이번 스텝에 policy가 낸 양자화 명령 (-1/0/+1)
            # lane_change_applied: 그 명령이 실제로 실행됐는지
            # 주의: _apply_lane_change는 유지 명령(0)에도 현재 차선으로
            # changeLane을 호출해 True를 반환한다(차선 고정 용도).
            # 따라서 "실제 변경"은 lane_change != 0 과 함께 판정해야 한다.
            if lane_change != 0 and lane_change_applied:
                # 실행된 변경마다 소액 감점 → 이득 없는 지그재그 억제.
                # 변경 덕에 속도 보상이 그 이상 늘면 순이득이므로,
                # "필요할 때만 바꾸는" 정책으로 수렴하게 만드는 항이다.
                reward -= float(self.rw.get("lane_change_penalty", 0.0))
            elif lane_change != 0:
                # 명령은 냈는데 실행 불가(차선 없음/교차로)였던 경우 감점
                # → 의미 없는 행동 출력을 줄이는 정칙화 (rl_action_penalty 대응)
                reward -= float(self.rw.get("invalid_action_penalty", 0.0))

        # ---- 순간이동·비정상 정지: 시뮬레이션 이상으로 보고 에피소드 종료 ----
        #   순간이동(teleport): SUMO가 오래 막힌 차를 앞으로 옮겨 버리는 것. ego에 생기면 종료.
        #   비정상 정지: stuck_seconds 이상 계속 멈춤 (경로 차선을 놓쳐 차선 끝에 선 경우 등).
        #   빨간불(·황색) 신호 대기는 정상 정지라 세지 않는다 (앞 신호까지 RED_WAIT_RANGE m 안 — 대기열 포함).
        stuck = False
        if alive and not terminated and not off_route:
            if ego_speed >= 0.1:
                self._stopped_steps = 0
            elif not self._waiting_at_red():
                self._stopped_steps += 1
            if self.stuck_steps and self._stopped_steps >= self.stuck_steps:
                stuck = True
                terminated = True
                reward -= float(self.rw.get("stuck_penalty", 0.0))
        if teleported or off_route:
            terminated = True
        if off_route:
            # [12팀 11/19] 경로 이탈 감점 (기본 0 — training/train.py REWARD_OVERRIDES). 사라짐 · 잘못된 차로 조기 종료 모두
            reward -= float(self.rw.get("off_route_penalty", 0.0))
        if red_light and not collided:
            terminated = True
            reward -= float(self.rw.get("red_light_penalty", 0.0))

        if self.step_count >= self.max_steps:
            truncated = True

        # 에피소드가 끝난 이유 (끝나지 않았으면 None). 기존 event.collided / event.arrived 외에
        # 경로 이탈·비정상 정지·순간이동·시간 초과를 구분한다 (데이터 칼럼 end_reason).
        end_reason = None
        if terminated or truncated:
            end_reason = ("collided" if collided else "off_route" if off_route else "teleported" if teleported
                          else "red_light" if red_light else "stuck" if stuck else "arrived" if arrived else "timeout" if truncated else "other")

        info = {
            "route": self.route_name,
            "scenario": self.scenario,
            "end_reason": end_reason,
            "teleported": teleported,
            "stuck": stuck,
            "off_route": off_route,
            "wrong_lane": wrong_lane,                           # [12팀 11/19] off_route 중 잘못된 차로 조기 종료
            "teleports": self.teleports,
            "red_light": red_light,
            "red_far_stop": far_stop,                           # [12팀 11/18] 이번 스텝 적색 신호 앞 먼 정지 감점
            "safety_brake": safety_brake,                       # 이번 스텝 안전장치 개입
            "safety_interventions": self.safety_interventions,  # 에피소드 누적
            "collided": collided,
            "arrived": arrived,
            "speed": ego_speed,
            "gap": ego_gap,
            "accel_raw": accel_raw,
            "lane_change_raw": lane_raw,
            "lane_change": lane_change,
            "lane_change_applied": lane_change_applied,
            "lane_before": lane_before,
            "lane_after": lane_after,
        }
        return obs, float(reward), terminated, truncated, info

    # ==================================================================
    def close(self):
        """환경 정리. 학습이 끝나면 반드시 호출해서 SUMO 프로세스를 종료할 것."""
        if self._started:
            traci.close()
            self._started = False
