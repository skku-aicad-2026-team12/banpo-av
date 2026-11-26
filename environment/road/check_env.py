"""수업 환경과 반포 도로망의 연결·주행·배경 교통을 검증한다.

실행: python -X utf8 environment/road/check_env.py
--policy sumo는 SUMO 운전자, --policy rule은 관측만 사용하는 규칙 정책이다.
--gui로 화면을 열고 --no-traffic-test로 배경 교통 검사를 생략할 수 있다.

모든 AV 경로의 주행 결과, 관측 범위, 신호 상태, 보행화 후 통제를 확인한다.
배경 교통은 AV 없이 실행해 충돌·순간이동을 센다.
결과는 environment/road/build/check_env_report.txt에도 저장한다.
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
    """관측만 사용하는 IDM 기반 선생님 정책을 만든다.
    앞차 추종에 신호 제동·경합 차량 양보·정지 유지를 결합한다.
    lane_dir_obs가 켜져 있으면 앞의 두 분기 방향과 차로 연결성을 보고 차로를 고른다.
    앞뒤 차량의 여유를 확인하고, 가까운 분기 전에는 속도를 낮춰 이동할 틈을 만든다.
    관측 칸은 43·49차원 호환을 위해 env.obs_index의 이름으로 읽는다.
"""
    W, vmax, R, S = env.visibility, env.vmax, env.route_horizon, env.signal_horizon
    # 경로 방향에 맞춰 옆 차로로 이동하되 앞뒤 차량의 접근 여유를 확인한다.
    # 실제 실행 여부는 환경의 차로 변경 안전 검사에 맡긴다.
    LC_GAP_DIST, LC_GAP_TIME, LC_GAP_DECEL, LC_GAP_MIN_V = 150.0, 12.0, 2.0, 3.0   # LC_GAP_DIST는 현재 계산에 쓰지 않는다. 최대 감속은 LC_GAP_DECEL이다.
    # 두 분기 사이가 충분히 멀면 첫 분기를 지난 뒤 둘째 분기에 맞춘다.
    # 첫 분기 직전의 불필요한 차로 변경을 피하기 위한 조건이다.
    LC_HOLD_BEFORE_BRANCH, LC_HOLD_MIN_SPACING = 60.0, 120.0
    # 옆 차로 뒤차가 빠르게 접근하면 차로 변경 명령을 보류한다.
    # 최소 거리와 접근 속도로 계산한 TTC 여유를 함께 쓴다.
    LC_REAR_MIN, LC_REAR_TTC = 3.0, 2.0
    # 옆 차로의 느린 앞차에 접근 중이면 최소 거리와 TTC 여유를 확인한다.
    # 접근 중이 아니거나 LC_FRONT_TTC가 0이면 앞차 거부 조건을 쓰지 않는다.
    LC_FRONT_MIN, LC_FRONT_TTC = 3.0, 2.0
    IX = env.obs_index                               # 관측 칸 이름과 번호 대응. 43·49차원의 배치 차이를 처리한다.
    LDO = bool(getattr(env, "lane_dir_obs", False))  # 49차원: 둘째 분기 · 내 차로 진행 방향이 있다
    I_SC, I_SD = IX["sig_color"], IX["sig_dist"]
    I_ST, TH = IX.get("sig_time"), getattr(env, "signal_time_horizon", 170.0)   # 신호 남은 시간 칸 (없으면 None) · 정규화 기준 (s)
    SIG_ANT_S = 6.0    # 황색 정지를 준비하는 녹색 남은 시간(s). 0이면 비활성화.
    BH = getattr(env, "branch_horizon", 600.0)
    amax, b, T, s0 = env.max_accel * 0.5, 3.0, 1.5, 2.0
    I_LN, I_LX, I_LD = IX.get("lim_now"), IX.get("lim_next"), IX.get("lim_next_dist")   # 차로 제한속도 관측 (11/24부터 목표 속도, 없으면 vmax)
    LIM_DECEL = 2.0    # 다음 제한속도가 낮으면 그 지점까지 이 감속(m/s²)으로 미리 줄이는 속도를 목표로
    DT = env.step_length
    # 정지선 앞 STOP_MARGIN만큼 여유를 두고 제동한다.
    # 필요 감속에 BRAKE_SLACK 여유를 적용하고, 황색에는 정지 가능한 경우 멈춘다.
    STOP_MARGIN, A_START, BRAKE_SLACK, A_YELLOW_MAX = 3.0, 0.8, 1.3, 3.5
    A_RAMP0, YIELD_BLEND_S = 0.4, 1.0   # 신호 감속이 시작되는 a_req · 양보를 섞기 시작하는 여유(초)
    A_COMF, T_REACT = 1.5, 1.0     # 양보 정지 (아래)
    # 양보 의무가 있거나 대등한 합류에서 상대가 먼저 도착하면 진입을 늦춘다.
    # 경합 지점 앞에 YIELD_MARGIN 여유를 두고 멈춘다.
    YIELD_MARGIN, YIELD_GAP_S = 6.0, 2.5
    V_HOLD, A_GO, A_HOLD = 0.5, 0.5, 3.0    # m/s · m/s² · m/s² : 정지 유지 (아래). 정답 -0.56: 탐험 흔들림 σ 0.2로 가속 쪽에 넘어갈 확률 < 1%
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
        # 현재 차로 제한속도를 목표로 삼고 다음 제한속도가 낮으면 미리 감속한다.
        # 환경은 차로 제한속도로 속도 명령을 자르지 않는다.
        if I_LN is not None:
            lim_now = float(obs[I_LN]) * vmax
            lim_next, d_next = float(obs[I_LX]) * vmax, float(obs[I_LD]) * R
            d_eff = max(d_next - v * DT, 0.0)                     # 이번 스텝에 갈 거리만큼 앞당겨 본다 (스텝이 0.5 s라 늦는다)
            v_target = max(min(lim_now, (lim_next ** 2 + 2.0 * LIM_DECEL * d_eff) ** 0.5), 0.1)
        else:
            v_target = vmax
        a = amax * (1 - (v / v_target) ** 4 - (s_star / max(gap, 0.01)) ** 2)   # 앞차 따라가기 (IDM)
        if v > v_target:                                           # 넘었으면 다음 스텝에 목표 속도로 (IDM만으로는 천천히 줄어 과속이 남는다)
            a = min(a, (v_target - v) / DT)
        if gap < 1.0 and vl < 0.5:
            a = -env.max_decel
        if obs[I_SD] < 0.999:                                      # 적색·황색과 곧 종료되는 녹색에 대비해 제동한다.
            d_stop = float(obs[I_SD]) * S - STOP_MARGIN
            a_req = v * v / (2 * max(d_stop, 0.1))                 # 정지 목표에 서려면 필요한 감속
            if obs[I_SC] < 0.5:
                stop = obs[I_SC] < -0.5 or a_req <= A_YELLOW_MAX     # 적색, 또는 설 수 있는 황색
            else:
                # 녹색 종료가 가까우면 황색 정지에 필요한 제동을 미리 준비한다.
                # 정지 불가능한 경우의 통과 판단은 유지한다.
                t_rem = float(obs[I_ST]) * TH if I_ST is not None else 1e3
                stop = SIG_ANT_S > 0.0 and t_rem <= SIG_ANT_S and d_stop > v * t_rem and a_req <= A_YELLOW_MAX
            if stop:
                a = min(a, 0.0)                                    # 정지선을 향해 가속하지 않는다
                # 제동 시작 경계에서 감속 명령이 급변하지 않도록 가중치를 부드럽게 섞는다.
                ramp = min(max((a_req - A_RAMP0) / (A_START - A_RAMP0), 0.0), 1.0)
                if ramp > 0.0:
                    a = min(a, -BRAKE_SLACK * a_req * ramp)
        if c_idx is not None and obs[c_idx] < 0.999 and obs[c_idx + 1] < 0.999:   # 경합 차량 있음
            d_av, d_foe = float(obs[c_idx]) * C_H, float(obs[c_idx + 1]) * C_H
            v_foe, prio = float(obs[c_idx + 2]) * vmax, float(obs[c_idx + 3])
            t_av = d_av / max(v, 1.0)
            # 경합 지점에 멈춘 상대에게 양보하되 AV가 이미 지점 안에 있으면 교착을 피한다.
            t_foe = d_foe / v_foe if v_foe > 0.5 else (0.0 if (d_foe < 3.0 and d_av >= 3.0) else 1e3)
            # 양보 여유가 -YIELD_BLEND_S에서 0 사이일 때 제동 비중을 부드럽게 늘린다.
            margin = (t_av + YIELD_GAP_S - t_foe) if prio < -0.5 else ((t_av - t_foe) if abs(prio) < 0.5 else -1e3)
            w_y = min(max(margin / YIELD_BLEND_S + 1.0, 0.0), 1.0)
            if w_y > 0.0:
                v_target = (2 * A_COMF * max(d_av - YIELD_MARGIN, 0.0)) ** 0.5
                a = a + w_y * (min(a, (v_target - v) / T_REACT) - a)
        # 가까운 분기부터 검사해 현재 차로가 못 가는 방향으로 이동한다.
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
                    # 분기까지 LC_GAP_TIME 이상의 여유가 생기도록 속도를 낮춰 틈을 만든다.
                    # 감속 상한은 LC_GAP_DECEL, 최소 목표 속도는 LC_GAP_MIN_V다.
                    v_cap = max(LC_GAP_MIN_V, float(obs[r_i]) * BH / LC_GAP_TIME)
                    a = min(a, max((v_cap - v) / T_REACT, -LC_GAP_DECEL))
                    break
        # 정지 중이며 출발할 조건이 아니면 일정한 제동 정답을 유지한다.
        # 모방 오차와 탐험 잡음으로 대기열 앞차에 접근하는 것을 막기 위한 조건이다.
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
        # 보행화 후 통제가 환경이 실제로 쓰는 함수(apply_after_closure)대로 걸렸는지 : 막을 차로 · 끌 신호 전부
        tc = _E.traci
        closed, tls_off = after_closure_targets()
        import road_spec
        # 차량 차로만 센다 (보도 차로는 원래 보행자 전용 : road_spec.SIDEWALKS 엣지의 0번)
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
    """배경 교통만 seconds초 : 환경과 같은 SUMO 옵션(스텝 0.5 s · 충돌 시 제거 · 교차로 안 충돌 검사), 보행화 후는 같은 통제 함수."""
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
