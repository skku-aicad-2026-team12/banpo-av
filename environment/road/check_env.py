"""수업 환경(environment/env/) + 반포 도로망 연결 검증

사용 (프로젝트 루트에서):
    python -X utf8 environment/road/check_env.py                      # 두 시나리오, 경로마다 2회, AV = SUMO 기본 운전자
    python -X utf8 environment/road/check_env.py --scenario after --episodes 5
    python -X utf8 environment/road/check_env.py --policy hold         # AV를 정책처럼 직접 제어 (가감속 0·차선 유지) — 파이프라인 오류 확인용
    python -X utf8 environment/road/check_env.py --policy rule         # 관측만 보고 운전하는 규칙 정책 (RL 모드) — 관측에 경로 · 신호 정보가 충분한지
    python -X utf8 environment/road/check_env.py --policy rule --safety  # 위와 같고 안전장치(충돌 방지 제동) 켬
    python -X utf8 environment/road/check_env.py --gui --episodes 1    # 화면으로 보기
    python -X utf8 environment/road/check_env.py --no-traffic-test     # 배경 교통 30분 테스트 생략

검사:
  1. AV 경로 주행: environment/env/road_config.py 설정 그대로 SumoHighwayEnv를 만들어 경로(side-banpo · goga-banpo · goga-jamsu)마다 주행 (시나리오는 같은 도로망에서 에피소드마다 지정)
     → 결과(도착·충돌·비정상 정지·순간이동·시간초과), 스텝 수, 시뮬레이션 속도
     통과 기준 (--policy sumo): 모든 에피소드 도착, 충돌·순간이동 0
     + 잠수교 통제(보행화 후 차량 차로 전부 보행자 전용 · 잠수교 신호 2곳 꺼짐, 보행화 전에는 하나도 없음)
     + 데이터 칼럼 signal(AV 앞 신호): 경로별 신호 순서, 남은 시간이 음수 · 빈 값이 아니고 색이 바뀌는 순간 0에 닿는지
     + 관측(11/17): 차원 · 범위 [-1, 1] · 신호색 [33] · 분기 방향 [31] 값 {-1, 0, 1}, 분기 방향이 0이 아닌(좌 · 우 분기 앞) 스텝 비율
     통과 기준 (--policy rule): 모든 에피소드 도착 · 신호 위반 0 — 관측만으로 경로 차로 · 신호를 지킬 수 있다는 뜻
  2. 배경 교통 (ego 없이 traffic_seconds초, 환경과 같은 스텝 0.5 s · 보행화 후 통제): 순간이동·충돌 수, 마지막 시각 차량 수
     통과 기준: 순간이동 0, 충돌 0
결과는 environment/road/build/check_env_report.txt 에도 저장한다.
"""
import argparse, os, re, sys, time
import numpy as np
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))   # 저장소 루트
import repo_paths as P  # noqa: E402
ROOT = P.ROOT
os.chdir(ROOT)

from env.road_config import ROAD, EGO, TRAFFIC
from env.mdp_config import SIMULATION, OBSERVATION, ACTION, REWARD
from env import road_builder
from env.sumo_env import SumoHighwayEnv, after_closure_targets, apply_after_closure
import env.sumo_env as _E
from sumolib import checkBinary


def rule_policy(env):
    """관측만 보고 운전하는 규칙 정책 — 학습 정책이 볼 수 있는 정보로 경로 · 신호를 지킬 수 있는지 확인용.
    가감속: IDM(앞차 = 관측 [9] · [10]) + 신호 정지 · 양보(경합 차량 conf_*) · 정지 유지 (아래 주석). BC · DAgger 선생님으로도 쓴다
    (11/17 조교 답: "IDM으로 데이터셋 생성해도 됩니다", DAgger 전문가로도 "가능합니다").
    차로 (11/19): lane_dir_obs를 켠 환경에서만 바꾼다 — 첫째 · 둘째 분기 순으로 경로가 가는 쪽(route_dir1 · route_dir2)으로 내 차로가 못 가면
    (reach1_* · reach2_*) 그쪽 옆 차로(연결성 [22] 좌1 · [24] 우1)로 맞는 차로에 들 때까지 매 스텝 변경 명령 (옮길 쪽 뒤차가 가깝게 빨리 오면
    명령 0, 옮길 쪽 앞차에 빠르게 다가가는 중이면 명령 0 — rear_ok, 실행 여부는 환경 안전 게이트).
    칸 번호는 env.obs_index(이름 → 번호)로 읽는다 (43 · 49차원 배치가 다르다). 둘째 분기 때문이면 첫 분기 LC_HOLD_BEFORE_BRANCH m 안에서는 미룬다.
    그 분기가 가까우면(LC_GAP_DIST m 또는 LC_GAP_TIME 초) LC_GAP_MIN_V 넘는 속도에서 감속해 틈을 만든다. lane_dir_obs가 꺼져 있으면 바꾸지 않는다 (출발 차로 departLane=best가 경로 차로라 그대로 도착할 수 있다)."""
    W, vmax, R, S = env.visibility, env.vmax, env.route_horizon, env.signal_horizon
    # [11/19] 차로 변경 (lane_dir_obs).
    # 명령: 옮겨야 하면(want) 옆 차로가 있고 뒤차 거부에 안 걸리는 한 매 스텝 명령 — 실행 여부는 환경 안전 게이트에 맡긴다
    # (11/19 4차: 2 · 3차의 선생님 앞 틈 기준 · 뒤로 빠지기가 명령 자체를 막아 off_route가 늘었다 → 1차 방식으로 되돌림).
    # 틈 만들기: 옮겨야 하고 그 분기까지 남은 거리 < max(LC_GAP_DIST, LC_GAP_TIME × 내 속도)이면, 내 속도 > LC_GAP_MIN_V일 때만
    # min(a, -LC_GAP_DECEL). 그 아래면 a 그대로 (1차의 min(a, 0)은 정지 유지와 겹쳐 서 버렸다 — sbr-banpo stuck)
    LC_GAP_DIST, LC_GAP_TIME, LC_GAP_DECEL, LC_GAP_MIN_V = 150.0, 12.0, 2.0, 3.0   # LC_GAP_DIST는 6차부터 안 씀, LC_GAP_DECEL 1.5 → 2.0 (상한)
    # 둘째 분기 때문에 옮기는 것은 첫 분기까지 LC_HOLD_BEFORE_BRANCH m 안이면 미룬다 (분기 직전 · 교차로 안 변경 충돌 — goga-jamsu).
    # 단 두 분기 사이가 LC_HOLD_MIN_SPACING m 이상일 때만 (그보다 가까우면 첫 분기 뒤에 옮길 거리가 없다 — 측도 side_a4 약 35 m)
    LC_HOLD_BEFORE_BRANCH, LC_HOLD_MIN_SPACING = 60.0, 120.0
    # 뒤차 거부 (저속 끼어들기 뒤 빠른 뒤차 충돌 — 1차 oly-jamsu · sbr-banpo): 옮길 쪽 뒤차 거리 ≤ LC_REAR_MIN + LC_REAR_TTC ×
    # max(뒤차 속도 − 내 속도, 0)이면 명령 0. 앞쪽 · 나란한 차는 환경 안전 게이트가 막는다
    LC_REAR_MIN, LC_REAR_TTC = 3.0, 2.0
    # [11/19 5차] 앞차 거부 (느린 앞차 바로 뒤로 끼어들기 방지 — 4차 oly-jamsu 충돌: 9 m 앞에서 14 m/s 느린 차 뒤로 들어가 추돌):
    # 옮길 쪽 앞차가 있고 내가 더 빠르면(접근 속도 > 0) 앞 거리 ≤ LC_FRONT_MIN + LC_FRONT_TTC × 접근 속도일 때 명령 0.
    # 접근 속도가 0 이하면 조건 없음 (3차의 고정 앞 틈 기준과 다르다). LC_FRONT_TTC = 0이면 꺼짐
    # (11/19 비교, (나) random · limit 5×40 같은 시드: 0 → 도착 194 · 충돌 1, 2.0 → 도착 195 · 충돌 0 → 2.0 채택)
    LC_FRONT_MIN, LC_FRONT_TTC = 3.0, 2.0
    IX = env.obs_index                               # 칸 이름 → 번호 (sumo_env.__init__ — 43 · 49차원 배치가 다르다)
    LDO = bool(getattr(env, "lane_dir_obs", False))  # 49차원: 둘째 분기 · 내 차로 진행 방향이 있다
    I_SC, I_SD = IX["sig_color"], IX["sig_dist"]
    I_ST, TH = IX.get("sig_time"), getattr(env, "signal_time_horizon", 170.0)   # 신호 남은 시간 칸 (없으면 None) · 정규화 기준 (s)
    SIG_ANT_S = 6.0    # [11/19 8차] 황색 미리 대비를 시작하는 녹색 남은 시간 (s). 0이면 끔 (7차와 같음)
    BH = getattr(env, "branch_horizon", 600.0)
    amax, b, T, s0 = env.max_accel * 0.5, 3.0, 1.5, 2.0
    # [11/17] 신호 정지 — BC가 따라 하다 조금 덜 감속해도 정지선 앞에 서도록 여유를 둔 계획 (BC 선생님용):
    #  · 적색이거나 서기로 한 황색이면 정지선을 향해 가속하지 않는다 (속도를 올렸다가 마지막에 세게 서면 BC가 못 따라 함)
    #  · 정지 목표(정지선 STOP_MARGIN m 앞)까지 필요한 감속 a_req가 A_START를 넘으면 그 BRAKE_SLACK배로 감속 — 일찍 · 여유 있게
    #  · 황색은 정지 쪽으로: a_req ≤ A_YELLOW_MAX면 선다 (못 서는 거리면 통과 — 그 거리에서는 황색 3 s 안에 정지선을 넘는다)
    # 지난 방식: ① 정지선을 앞차로 본 IDM → 마지막 40 m 최대 제동, BC가 약하게 따라 해 신호 위반 (r0) ② 편한 감속 1.5 m/s²
    # 속도 추종 + 남은 시간으로 통과 판단 → 계획에 여유가 없어 BC가 덜 감속하면 중간에 "이제 못 선다"로 뒤집혀 신호 위반 (s0 9건)
    STOP_MARGIN, A_START, BRAKE_SLACK, A_YELLOW_MAX = 3.0, 0.8, 1.3, 3.5
    A_RAMP0, YIELD_BLEND_S = 0.4, 1.0   # [11/19 6차] 신호 감속이 시작되는 a_req · 양보를 섞기 시작하는 여유(초)
    A_COMF, T_REACT = 1.5, 1.0     # 양보 정지 (아래)
    # [11/17] 양보: 경합 차량 관측 [35..38](conflict_obs)에서 AV가 양보해야 하거나(우선권 -1) 대등한 합류인데 상대가 먼저 오면
    # 경합 지점 YIELD_MARGIN m 앞을 목표로 신호와 같은 방식으로 선다 (처음에는 경합 차량을 보지 않아 회전교차로 진입에서 충돌했다).
    # [11/19 2차] YIELD_MARGIN 3 → 6: 합류 경합 지점이 도로 끝 2.8 m 뒤라 3 m로는 교차로 안에 들어가 섰다 (side-banpo :drop_0 교착)
    YIELD_MARGIN, YIELD_GAP_S = 6.0, 2.5
    V_HOLD, A_GO, A_HOLD = 0.5, 0.5, 3.0    # m/s · m/s² · m/s² — 정지 유지 (아래). 정답 -0.56: 탐험 흔들림 σ 0.2로 가속 쪽에 넘어갈 확률 < 1%
    c_idx = IX.get("conf_av_dist")                   # 경합 차량 4칸 시작 (conflict_obs 꺼짐이면 None)
    C_H = getattr(env, "conflict_horizon", 100.0)

    def rear_ok(obs, v, want):
        """명령해도 되는가: 옮길 쪽 차로(좌1 [5..8] · 우1 [13..16])의 앞차 거부(LC_FRONT_TTC > 0일 때, 접근 중인 앞차가 가까우면 False)와
        뒤차 거부(뒤차가 없거나 거리 > LC_REAR_MIN + LC_REAR_TTC × max(뒤차 속도 − 내 속도, 0)여야 True). 나란한 차는 뒤 거리 0이라 거부."""
        i = 5 if want > 0 else 13
        if LC_FRONT_TTC > 0.0 and obs[i] < 0.999:              # 앞차 거부 (접근 중일 때만)
            closing = -float(obs[i + 1]) * vmax                 # 내 속도 − 앞차 속도
            if closing > 0.0 and float(obs[i]) * W <= LC_FRONT_MIN + LC_FRONT_TTC * closing:
                return False
        return obs[i + 2] <= -0.999 or -float(obs[i + 2]) * W > LC_REAR_MIN + LC_REAR_TTC * max(float(obs[i + 3]) * vmax, 0.0)

    def act(obs):
        v = float(obs[0]) * vmax
        gap = float(obs[9]) * W if obs[9] < 0.999 else 1e3
        vl = v + float(obs[10]) * vmax if obs[9] < 0.999 else v
        s_star = s0 + max(0.0, v * T + v * (v - vl) / (2 * (amax * b) ** 0.5))
        a = amax * (1 - (v / vmax) ** 4 - (s_star / max(gap, 0.01)) ** 2)   # 앞차 따라가기 (IDM)
        if gap < 1.0 and vl < 0.5:
            a = -env.max_decel
        if obs[I_SD] < 0.999:                                      # 신호 앞 (적색 · 황색, 8차부터 곧 끝나는 녹색도)
            d_stop = float(obs[I_SD]) * S - STOP_MARGIN
            a_req = v * v / (2 * max(d_stop, 0.1))                 # 정지 목표에 서려면 필요한 감속
            if obs[I_SC] < 0.5:
                stop = obs[I_SC] < -0.5 or a_req <= A_YELLOW_MAX     # 적색, 또는 설 수 있는 황색
            else:
                # [11/19 8차] 황색 미리 대비: 녹색이 SIG_ANT_S 초 안에 끝나고(신호 남은 시간 관측 sig_time) 지금 속도로는 그 안에
                # 정지선을 못 넘으면, 녹색일 때부터 황색과 같은 방식으로 줄이기 시작한다. 선생님은 황색이 켜진 뒤에 줄여도 서지만,
                # BC는 그 순간 감속을 절반도 못 따라 해 신호 위반으로 끝났다 (h0 · g2 150판 중 8판, 전부 황색이 켜질 때 정지선에서
                # 25~70 m). 통과 판단(a_req > A_YELLOW_MAX면 통과)은 그대로 — 11/17의 "남은 시간으로 통과 판단"과 달리 서는 쪽으로만
                # 앞당긴다: 줄이면 v × 남은 시간이 더 작아져 판단이 뒤집히지 않는다.
                t_rem = float(obs[I_ST]) * TH if I_ST is not None else 1e3
                stop = SIG_ANT_S > 0.0 and t_rem <= SIG_ANT_S and d_stop > v * t_rem and a_req <= A_YELLOW_MAX
            if stop:
                a = min(a, 0.0)                                    # 정지선을 향해 가속하지 않는다
                # [11/19 6차] 계단 없애기: 전에는 a_req가 A_START를 넘는 순간 0 → -BRAKE_SLACK × A_START로 뛰었다 (BC가 이 순간을
                # 못 따라 해 먼 거리 황색에서 가속을 유지하다 신호 위반). A_RAMP0에서 0, A_START에서 전부가 되게 잇는다.
                ramp = min(max((a_req - A_RAMP0) / (A_START - A_RAMP0), 0.0), 1.0)
                if ramp > 0.0:
                    a = min(a, -BRAKE_SLACK * a_req * ramp)
        if c_idx is not None and obs[c_idx] < 0.999 and obs[c_idx + 1] < 0.999:   # 경합 차량 있음
            d_av, d_foe = float(obs[c_idx]) * C_H, float(obs[c_idx + 1]) * C_H
            v_foe, prio = float(obs[c_idx + 2]) * vmax, float(obs[c_idx + 3])
            t_av = d_av / max(v, 1.0)
            # 서 있는 상대는 지점에 붙어 있고 내가 아직 지점 3 m 밖일 때만 [11/19: 합류 교착 — 양보 정지가 목표를 지나쳐 AV 앞부분이
            # 교차로에 든 채 서면, 우선권 있는 상대가 AV를 앞차로 보고 바로 앞에 서서 서로 기다리다 stuck (측도 m2 → m3a · :drop_0).
            # 2차: "상대가 더 가깝거나 같을 때"로는 0.1 m 차이 교착이 남아, 내가 이미 3 m 안이면 서 있는 상대에게 양보하지 않는다]
            t_foe = d_foe / v_foe if v_foe > 0.5 else (0.0 if (d_foe < 3.0 and d_av >= 3.0) else 1e3)
            # [11/19 6차] 계단 없애기: 양보 조건의 여유(초)가 0 이상이면 전과 같이 전부 양보, -YIELD_BLEND_S ~ 0 사이는 비율로 섞는다
            # (전보다 조금 일찍 · 약하게 시작 — 더 조심하는 쪽. BC 가감속 오차의 52%가 경합 차량이 보이는 스텝이었다)
            margin = (t_av + YIELD_GAP_S - t_foe) if prio < -0.5 else ((t_av - t_foe) if abs(prio) < 0.5 else -1e3)
            w_y = min(max(margin / YIELD_BLEND_S + 1.0, 0.0), 1.0)
            if w_y > 0.0:
                v_target = (2 * A_COMF * max(d_av - YIELD_MARGIN, 0.0)) ** 0.5
                a = a + w_y * (min(a, (v_target - v) / T_REACT) - a)
        # [11/19] 차로 변경: 첫째 → 둘째 분기 순으로 경로 방향 d_k인데 내 차로가 그쪽으로 못 가면 원하는 쪽 = d_k (첫 번째로 걸린 분기에서 멈춤)
        lane = 0.0
        if LDO:
            for k, (d_i, r_i, left_i, right_i) in enumerate(((IX["route_dir1"], IX["route_dist1"], IX["reach1_left"], IX["reach1_right"]),
                                                             (IX["route_dir2"], IX["route_dist2"], IX["reach2_left"], IX["reach2_right"]))):
                d_k = float(obs[d_i])
                if (d_k > 0.5 and obs[left_i] < 0.5) or (d_k < -0.5 and obs[right_i] < 0.5):
                    if (k == 1 and float(obs[IX["route_dist1"]]) * BH < LC_HOLD_BEFORE_BRANCH
                            and (float(obs[IX["route_dist2"]]) - float(obs[IX["route_dist1"]])) * BH >= LC_HOLD_MIN_SPACING):
                        break                                      # 둘째 분기 때문이면 (두 분기가 멀 때만) 첫 분기를 지난 뒤에 옮긴다
                    want = 1.0 if d_k > 0 else -1.0
                    if obs[22 if want > 0 else 24] > -0.999 and rear_ok(obs, v, want):   # 옆 차로가 있고 뒤차 거부에 안 걸림
                        lane = want                                # 실행 여부(앞 · 나란한 차)는 환경 안전 게이트가 정한다
                    # 틈 만들기 [11/19 6차] 계단 없애기: 전에는 "남은 거리 < max(LC_GAP_DIST, LC_GAP_TIME × v)이면 -LC_GAP_DECEL"이라
                    # 감속하면 조건이 풀려 가속 ↔ 감속을 오갔다 (BC가 평균을 내 덜 감속 → 제때 못 옮김). 이제 분기까지 LC_GAP_TIME 초가
                    # 남도록 속도 상한(남은 거리 / LC_GAP_TIME, 최소 LC_GAP_MIN_V)을 두고 그 속도로 부드럽게 맞춘다 (감속은 LC_GAP_DECEL까지).
                    v_cap = max(LC_GAP_MIN_V, float(obs[r_i]) * BH / LC_GAP_TIME)
                    a = min(a, max((v_cap - v) / T_REACT, -LC_GAP_DECEL))
                    break
        # [11/17] 정지 유지: 서 있고(v < V_HOLD) 출발할 이유가 없으면(a < A_GO) 브레이크를 밟고 있는다 (사람이 신호 대기 중 브레이크를 밟듯).
        # IDM은 정지 상태에서 0 근처를 내서, BC + RL 탐험 흔들림(σ)이 서 있는 차를 조금씩 밀어 신호 대기 줄에서 앞차를 박았다
        # (r0 확률적 평가: 오래 서는 측도 경로 충돌 100%, σ 0.05로 줄여도 90%. 1.5 m/s²로는 흔들림을 못 이겨 3.0으로).
        # [11/19 7차] 서 있을 때의 정답을 -A_HOLD 하나로 (전: min(a, -A_HOLD) — 앞차가 1 m 안이면 -1.0, 아니면 -0.56으로 갈려
        # 같은 '서 있기'인데 정답이 오갔다. 경합 지점 앞 대기 줄에서 급변의 절반쯤이 이것)
        if v < V_HOLD and a < A_GO:
            a = -A_HOLD
        return [float(np.clip(a / (env.max_accel if a >= 0 else env.max_decel), -1, 1)), lane]
    return act


def run_routes(scn, args, cfg, say):
    road = dict(ROAD, scenario="mixed")
    ego = dict(EGO, sumo_driver=(args.policy == "sumo"))
    action = dict(ACTION, safety_brake=dict(ACTION.get("safety_brake") or {}, enabled=bool(args.safety)))
    env = SumoHighwayEnv(cfg_path=cfg, road=road, ego=ego, mdp_sim=SIMULATION, mdp_obs=OBSERVATION,
                         action=action, reward=REWARD, traffic=TRAFFIC, gui=args.gui, gui_autostart=True)
    act = rule_policy(env) if args.policy == "rule" else (lambda o: [0.0, 0.0])
    fails = 0
    try:
        env.reset(options={"scenario": scn})          # 경로 목록을 얻기 위한 첫 리셋
        routes = env._route_ids[scn]
        # 보행화 후 통제가 환경이 실제로 쓰는 함수(apply_after_closure)대로 걸렸는지 — 막을 차로 · 끌 신호 전부
        tc = _E.traci
        closed, tls_off = after_closure_targets()
        import road_spec
        # 차량 차로만 센다 (보도 차로는 원래 보행자 전용 — road_spec.SIDEWALKS 엣지의 0번)
        lanes = [f"{e}_{i}" for e in closed for i in range(tc.edge.getLaneNumber(e))
                 if not (e in road_spec.SIDEWALKS and i == 0)]
        shut = [l for l in lanes if tuple(tc.lane.getAllowed(l)) == ("pedestrian",)]
        off = [tl for tl in tls_off if tc.trafficlight.getProgram(tl) == "off"]
        want = (len(lanes), len(tls_off)) if scn == "after" else (0, 0)
        ok = (len(shut), len(off)) == want
        fails += not ok
        say(f"  {'✓' if ok else '✗'} 잠수교 통제: 막힌 차량 차로 {len(shut)}/{len(lanes)}, 꺼진 신호 {len(off)}/{len(tls_off)} "
            f"({'보행화 후 — 전부여야 함' if scn == 'after' else '보행화 전 — 0이어야 함'})")
        say(f"  AV 경로: {', '.join(routes)}  (AV = " + {"sumo": "SUMO 기본 운전자", "hold": "가감속 0 · 차선 유지",
            "rule": "관측만 보는 규칙 정책"}[args.policy] + (", 안전장치 켬" if args.safety else "") + ")")
        seqs, sig_bad = {}, []
        obs_bad, lost, total_steps, interventions = [], 0, 0, 0
        for r in routes:
            res, steps, wall = [], 0, 0.0
            for _ in range(args.episodes):
                t0 = time.time()
                obs, info0 = env.reset(options={"route": r, "scenario": scn})
                assert info0["scenario"] == scn
                assert obs.shape == env.observation_space.shape == (env.obs_dim,) and env.obs_dim >= 39, obs.shape
                done, n = False, 0
                seq, prev = [], None
                while not done:
                    sig = env.get_signal_state()          # 데이터 칼럼 signal: AV 앞 신호 {tls, state, remaining_s}
                    if sig:
                        if not seq or seq[-1] != sig["tls"]:
                            seq.append(sig["tls"])
                        grp = _E._SIG_GROUP.get(sig["state"])
                        if sig["remaining_s"] is None or sig["remaining_s"] < 0:
                            if grp != "off":
                                sig_bad.append(f"{r} {sig}")
                        elif prev and prev[0] == sig["tls"] and prev[1] != grp and prev[2] > env.step_length + 1e-6:
                            sig_bad.append(f"{r} {sig['tls']}: {prev[1]} → {grp}인데 직전 남은 시간 {prev[2]} s")
                        prev = (sig["tls"], grp, sig["remaining_s"] if sig["remaining_s"] is not None else 0.0)
                    obs, _, term, trunc, info = env.step(act(obs))
                    n += 1
                    done = term or trunc
                    if not done:                              # 관측 범위 · 신호색 값 (종료 스텝은 0벡터)
                        IX = env.obs_index
                        if obs.min() < -1 or obs.max() > 1 or obs[IX["sig_color"]] not in (-1.0, 0.0, 1.0) or obs[IX["route_dir1"]] not in (-1.0, 0.0, 1.0) or not (0 <= obs[IX["sig_dist"]] <= 1):
                            obs_bad.append(f"{r} t={n}: {np.round(obs[31:], 2).tolist()}")
                        lost += obs[IX["route_dir1"]] != 0.0
                        total_steps += 1
                seqs.setdefault(r, set()).add(" → ".join(seq))
                wall += time.time() - t0
                steps += n
                interventions += info.get("safety_interventions", 0)
                end = ("충돌" if info["collided"] else "경로이탈" if info["off_route"] else "순간이동" if info["teleported"]
                       else "신호위반" if info.get("red_light") else "정지" if info["stuck"] else "도착" if info["arrived"] else "시간초과" if trunc else "사라짐")
                res.append(f"{end}({n * env.step_length:.0f}s)")
                if args.policy in ("sumo", "rule") and end != "도착":
                    fails += 1
            ok = all(x.startswith("도착") for x in res)
            mark = "✓" if ok else ("✗" if args.policy in ("sumo", "rule") else "·")
            say(f"  {mark} {r:4s} {' '.join(res)}  | {steps / max(wall, 1e-9):.0f} 스텝/초 (리셋 포함)")
        # signal 칼럼: 경로마다 지나는 신호 순서, 남은 시간이 음수 · 빈 값이 아니고 색이 바뀌는 순간 0에 닿는지
        for r in routes:
            say(f"  · {r} 신호: {' / '.join(sorted(seqs.get(r, {'(없음)'})))}")
        fails += bool(sig_bad)
        say(f"  {'✓' if not sig_bad else '✗'} signal 칼럼 남은 시간 일관성" + (f" — {len(sig_bad)}건: {sig_bad[:3]}" if sig_bad else ""))
        fails += bool(obs_bad)
        say(f"  {'✓' if not obs_bad else '✗'} 관측 {env.obs_dim}차원 · 범위 [-1, 1] · 신호색 {{-1, 0, 1}}" + (f" — {len(obs_bad)}건: {obs_bad[:2]}" if obs_bad else ""))
        say(f"  · 좌 · 우 분기 앞 스텝([31] ≠ 0) {lost}/{total_steps}" + (f", 안전장치 개입 {interventions} 스텝" if args.safety else ""))
    finally:
        env.close()
    return fails


def run_traffic(cfg, scn, seconds, say):
    """배경 교통만 seconds초 — 환경과 같은 SUMO 옵션(스텝 0.5 s · 충돌 시 제거 · 교차로 안 충돌 검사), 보행화 후는 같은 통제 함수."""
    import tempfile
    import traci as tc                         # TraCI 소켓 (SUMO 종료 통계를 표준 출력으로 받기 위해 libsumo 대신)
    exe = checkBinary("sumo")
    rou = os.path.join(os.path.dirname(cfg), f"banpo.{scn}.rou.xml")
    with tempfile.TemporaryFile("w+", encoding="utf-8", errors="replace") as out:
        tc.start([exe, "-c", cfg, "--route-files", rou, "--end", str(seconds), "--step-length", str(SIMULATION["step_length"]),
                  "--no-step-log", "true", "--duration-log.statistics", "true",
                  "--collision.action", "remove", "--collision.check-junctions", "true", "--no-warnings", "false"],
                 label="traffic", stdout=out)
        try:
            if scn == "after":
                apply_after_closure(tc, *after_closure_targets())
            tc.simulationStep(float(seconds))
        finally:
            tc.close()
        out.seek(0)
        txt = out.read()
    def num(key):
        m = re.search(rf"{key}:\s*(\d+)", txt)
        return int(m.group(1)) if m else 0         # SUMO는 0건이면 그 줄을 출력하지 않는다
    tele, coll = num("Teleports"), num("Collisions")
    running, inserted = num("Running"), num("Inserted")
    waiting = num("Waiting")
    ok = inserted > 0 and tele == 0 and coll == 0
    say(f"  {'✓' if ok else '✗'} {seconds}s: 투입 {inserted}대, 마지막 시각 주행 중 {running}대, 투입 대기 {waiting}대, "
        f"순간이동 {tele}건, 충돌 {coll}건")
    if not ok:
        for l in txt.splitlines():
            if ("Warning" in l and ("teleport" in l.lower() or "collision" in l.lower())) or "Error" in l:
                say("     " + l[:170])
    return 0 if ok else 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", choices=["before", "after"])
    ap.add_argument("--episodes", type=int, default=2)
    ap.add_argument("--policy", choices=["sumo", "hold", "rule"], default="sumo")
    ap.add_argument("--safety", action="store_true", help="안전장치(충돌 방지 제동) 켬 — RL 모드(hold · rule)에서만 작동")
    ap.add_argument("--gui", action="store_true")
    ap.add_argument("--no-traffic-test", action="store_true")
    ap.add_argument("--traffic-seconds", type=int, default=1800)
    a = ap.parse_args()
    lines = []
    def say(s=""):
        lines.append(s); print(s, flush=True)
    fails = 0
    # 환경 파일은 한 번만 만든다 (두 시나리오가 같은 도로망 · 시나리오별 수요 파일을 쓴다)
    cfg = road_builder.build(dict(ROAD, scenario="mixed"), dict(EGO, sumo_driver=(a.policy == "sumo")), TRAFFIC,
                             P.SUMO_DIR)
    for scn in ([a.scenario] if a.scenario else ["before", "after"]):
        say(f"===== [{scn}] 교통량 배율 {TRAFFIC.get('demand_scale', 1.0)}"
            f"{' (잠수교 차량 통제)' if scn == 'after' else ''}")
        say("1. AV 경로 주행")
        f = run_routes(scn, a, cfg, say)
        fails += f
        if not a.no_traffic_test:
            say("2. 배경 교통 (AV 없이)")
            fails += run_traffic(cfg, scn, a.traffic_seconds, say)
    say(f"결과: {'통과' if fails == 0 else f'실패 {fails}건'}")
    os.makedirs(P.ROAD_BUILD, exist_ok=True)
    with open(os.path.join(P.ROAD_BUILD, "check_env_report.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
