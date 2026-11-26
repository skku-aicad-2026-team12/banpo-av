"""부분관측 정책의 SUMO 주행 데이터를 JSONL.gz와 NPZ로 수집한다.

한 행은 관측·행동·보상·다음 관측·종료 여부와 관측 시점의 시뮬레이터 상태다.
49차원 관측과 정책 설정은 environment/env/mdp_config.py, 보상은 training/train.py를 따른다.
시뮬레이터 전체 상태는 정책 입력으로 쓰지 않고 분석용으로 저장한다.

JSONL의 실수는 --ndigits 자리로 반올림하고 NPZ는 float32 값을 유지한다.
JSONL 중간 행의 next_observation·next_state는 다음 행으로 복원한다.
종료 행에는 두 값을 직접 기록하며 iter_transitions()가 읽을 때 채운다.
NPZ의 detected_xy는 좌표를 이어 붙인 배열이고 detected_offsets로 행을 구분한다.

--label-policy rule은 실행 행동 대신 선생님 정답을 action_raw에 기록하는 DAgger용이다.
JSONL action 필드에는 실제 실행 행동이 남는다. 최종 수집에는 이 옵션을 쓰지 않는다.
경로·신호·종료 원인 코드표는 아래 상수에 정의한다. scenario는 ego_route로 구한다.
sim_time은 예열과 AV 투입 대기를 포함한 시뮬레이션 시각으로, 판별 t와 다르다.

TraCI 구독으로 차량 상태를 묶어 읽고, 화면 없는 실행에는 libsumo를 사용한다.
--workers로 SUMO 인스턴스를 병렬 실행한다. --paired-seeds는 판마다 교통 시드를 고정한다.

예시:
    python -X utf8 data_collection/collect_pomdp_data.py --episodes 100 --workers 8
    python -X utf8 data_collection/collect_pomdp_data.py --model results/run/model.pt --episodes 100

출력: data/<name>.jsonl.gz, data/<name>.npz
읽기: iter_transitions(path) 또는 np.load(path, allow_pickle=False)
"""
import argparse
import gzip
import json
import multiprocessing
import os
import sys
import shutil
import time
from datetime import datetime

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))   # 저장소 루트
import repo_paths as P  # noqa: E402
from env.road_config import ROAD, EGO, TRAFFIC
from env.mdp_config import SIMULATION, OBSERVATION, ACTION
from env import road_builder
from env.sumo_env import SumoHighwayEnv
from train import HPARAMS, REWARD  # 학습과 같은 보상 조건 + PPO 네트워크 구조

BASE = P.ROOT   # 저장소 루트 (data/는 루트에)

# 코드표 : JSONL · NPZ 모두 이 숫자로 저장 (11/17부터. 그 전 JSONL은 문자열)
SCENARIO_CODES = {"before": 0, "after": 1}                       # 잠수교 보행화(차량 통제) 전 / 후 (11/19부터 칼럼 없음 : 옛 파일 읽기용)
ROUTE_CODES = {"side-banpo": 0, "goga-banpo": 1, "goga-jamsu": 2, "oly-jamsu": 3, "sbr-banpo": 4}   # environment/road/road_spec.py AV_ROUTES
SIGNAL_TLS_CODES = {"ix": 0, "xw": 1, "s4032": 2, "jx1": 3, "jx2": 4}            # environment/road/road_spec.py SIGNALS (-1 = 앞에 신호 없음)
# 신호색 (SUMO 신호 문자  /  값): 녹 1 · 황 0.5 · 적 0 · 꺼짐 -1, 앞에 신호 없음도 -1 (11/17 조교 공지 예시 {R, Y, G}  /  {0, 0.5, 1})
SIGNAL_STATE_VALUES = {"G": 1.0, "g": 1.0, "y": 0.5, "Y": 0.5, "r": 0.0, "R": 0.0, "s": 0.0, "u": 0.0, "O": -1.0, "o": -1.0}
END_REASON_CODES = {None: 0, "arrived": 1, "collided": 2, "off_route": 3,
                    "stuck": 4, "teleported": 5, "timeout": 6, "other": 7,
                    "red_light": 8}   # 0 = 진행 중. red_light = 적색 신호에 정지선 통과 (11/15 추가 : 기존 값은 그대로)


def route_scenario(route):
    """AV 경로(이름 또는 ROUTE_CODES 숫자)  /  시나리오 이름 (road_spec.AV_ROUTES[...]["scenarios"]가 하나일 때만, 아니면 None).
    scenario 칼럼을 뺀 뒤 파일을 읽는 쪽이 시나리오가 필요하면 이것으로 구한다."""
    if isinstance(route, (int, np.integer)) and not isinstance(route, bool):
        route = {v: k for k, v in ROUTE_CODES.items()}.get(int(route))
    road_dir = os.path.join(P.ROOT, "environment", "road")
    if road_dir not in sys.path:
        sys.path.insert(0, road_dir)
    import road_spec
    sc = tuple((road_spec.AV_ROUTES.get(route) or {}).get("scenarios") or ())
    return sc[0] if len(sc) == 1 else None


def episode_seed(base: int, ep: int) -> int:
    """에피소드 번호로 정하는 난수 시드 (--paired-seeds), 워커 수 · 배분과 무관하다.
    수집에서는 이 값 % 2**31을 SUMO 시드로 직접 쓴다. 평가는 기존 10만 가지 추첨을
    유지하므로 같은 판 번호여도 수집과 평가의 교통 시드가 같지는 않다."""
    return (int(base) * 1000 + int(ep)) % 2**32


def detect_algorithm(model_path: str) -> str:
    """model.pt가 어느 학습 스크립트 출력인지 파일 내용으로 판별한다.

    training/train.py · training/train_team.py(PPO, BC 포함) : policy.state_dict()를 그대로 저장  /  텐서만 있는 dict
    수업 train_bc.py(BC, 11/16 지움): {"algorithm": "bc", "state_dim", ..., "state_dict"} dict : 예전 파일 호환용
"""
    import torch
    ck = torch.load(model_path, map_location="cpu", weights_only=True)
    return "bc" if isinstance(ck, dict) and ck.get("algorithm") == "bc" else "ppo"


def make_policy(args, env, worker_seed: int):
    """args.policy에 맞는 obs -> action_raw(np.float32, shape (2,)) 함수를 만든다.

    ppo / bc 출력은 [-1, 1]로 클리핑해 저장한다. 환경의 입력 범위와 같아서
    이 데이터로 바로 BC(training/train_team.py bc)를 돌릴 수 있다.
"""
    if args.policy == "ppo":
        # 수업 test.py(11/16 지움)와 같은 방식: training/train.py의 HPARAMS로 같은 구조를 만든 뒤 가중치만 덮어쓴다.
        # seed는 워커마다 다르게 (--stochastic 샘플링이 워커 간에 겹치지 않도록).
        from algorithms.ppo import PPO
        agent = PPO(obs_dim=env.observation_space.shape[0],
                    act_dim=env.action_space.shape[0],
                    **{**HPARAMS, "device": args.device, "seed": worker_seed})
        agent.load(args.model)
        deterministic = not args.stochastic   # hybrid: (μ, argmax 차선) / 샘플링

        def act(obs):
            a = agent.predict(obs, deterministic=deterministic)
            return np.clip(a, -1.0, 1.0).astype(np.float32)
        return act

    if args.policy == "bc":
        from algorithms.bc import BCPolicy
        model = BCPolicy.load(args.model, device=args.device)

        def act(obs):
            return np.clip(model.predict(obs), -1.0, 1.0).astype(np.float32)
        return act

    if args.policy == "rule":
        # 관측만 보고 운전하는 규칙 정책 (environment/road/check_env.py rule_policy) : BC 시범 · DAgger 정답.
        # 5조합 × 30회 안전장치 끔: 도착 99.3% · 충돌 0.7% · 신호 위반 0, 앞차 최소 간격 약 12 m
        # (SUMO 운전자는 앞차 뒤 2.5 m = 충돌 판정 경계에 서서 BC 시범으로 부적합).
        return _rule_policy(env)

    if args.policy == "sumo":
        # SUMO 기본 운전자가 AV를 운전 (신호·양보·차선 선택을 지키는 전문가 시연).
        # 정책 출력은 쓰이지 않고, 기록하는 action_raw는 환경이 실제 움직임에서 역산한 값이다.
        return lambda obs: np.zeros(2, dtype=np.float32)

    if args.policy == "keep-lane":
        # 완만한 가속 + 차선 유지. GUI/파이프라인 점검용.
        return lambda obs: np.asarray([0.25, 0.0], dtype=np.float32)

    # random: 가감속은 연속 uniform[-1,1], 차선변경은 {-1,0,+1} 직접 샘플링 (유지를 더 자주)
    rng = np.random.default_rng(worker_seed)

    def act(obs):
        accel = rng.uniform(-1.0, 1.0)
        lane = rng.choice(np.asarray([-1.0, 0.0, 1.0]), p=[0.15, 0.70, 0.15])
        return np.asarray([accel, lane], dtype=np.float32)
    return act


def _rule_policy(env):
    """check_env.rule_policy(env) : environment/road를 경로에 넣고 불러온다 (evaluation/baseline.py와 같은 정책)."""
    road_dir = os.path.join(P.ROOT, "environment", "road")
    if road_dir not in sys.path:
        sys.path.insert(0, road_dir)
    import check_env
    act = check_env.rule_policy(env)
    return lambda obs: np.asarray(act(obs), dtype=np.float32)


def compact(obj, ndigits: int):
    """중첩 dict/list/ndarray를 float가 ndigits 자리로 반올림된 순수 Python 객체로 바꾼다.

    float32 값을 그대로 json.dumps하면 0.10000000149011612처럼 17자리로 찍히는데,
    반올림하면 0.1이 되어 텍스트 길이가 크게 준다. numpy 스칼라/배열도 함께 변환한다.
"""
    if isinstance(obj, (float, np.floating)):
        return round(float(obj), ndigits)
    if isinstance(obj, (bool, np.bool_)):
        return bool(obj)
    if isinstance(obj, (int, np.integer)):
        return int(obj)
    if isinstance(obj, np.ndarray):
        return compact(obj.tolist(), ndigits)
    if isinstance(obj, dict):
        return {k: compact(v, ndigits) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [compact(v, ndigits) for v in obj]
    return obj


def iter_transitions(jsonl_path: str):
    """저장된 JSONL(.gz 또는 평문)을 읽어, 생략된 next_observation / next_state를
    다음 레코드에서 채운 완전한 transition dict를 순서대로 yield한다.

    (수집이 중간에 끊긴 파일이면 마지막 레코드에만 next_*가 없을 수 있다.)
"""
    opener = gzip.open if jsonl_path.endswith(".gz") else open
    with opener(jsonl_path, "rt", encoding="utf-8") as f:
        prev = None
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            if prev is not None:
                # done 레코드는 next_*를 이미 갖고 있으므로 setdefault가 건드리지 않는다.
                prev.setdefault("next_observation", rec["observation"])
                prev.setdefault("next_state", rec["state"])
                yield prev
            prev = rec
        if prev is not None:
            yield prev


def collect(args, sumocfg, episodes, worker_id, jsonl_path, npz_path, tag=""):
    """SUMO 인스턴스 하나로 episodes(전역 에피소드 번호 목록)를 수집해 두 파일로 저장한다.

    --workers > 1이면 이 함수가 프로세스마다 하나씩 돌고, main()이 결과를 합친다.
    반환값: 수집한 transition 수.
"""
    worker_seed = (args.seed * 1000 + worker_id) % 2**32
    if args.workers > 1:
        # 프로세스마다 torch가 코어 수만큼 스레드를 띄우면 서로 뺏는다. 관측 1개씩
        # 추론이라 1스레드면 충분.
        try:
            import torch
            torch.set_num_threads(1)
        except ImportError:
            pass

    # 안전장치(충돌 방지 제동) 켬/끔 : 학습(train_team.py --safety) · 평가(baseline.py --safety)와 같은 방식으로 덮어쓴다.
    action = dict(ACTION, safety_brake=dict(ACTION.get("safety_brake") or {}, enabled=(args.safety == "on")))
    env = SumoHighwayEnv(
        cfg_path=sumocfg,
        road=ROAD, ego=dict(EGO, sumo_driver=(args.policy == "sumo"), sumo_min_gap=getattr(args, "demo_min_gap", None)),
        mdp_sim=SIMULATION, mdp_obs=OBSERVATION,
        action=action, reward=REWARD,
        traffic=TRAFFIC,
        gui=args.gui,
        gui_autostart=True,
    )
    act = make_policy(args, env, worker_seed)
    label = _rule_policy(env) if getattr(args, "label_policy", None) == "rule" else None   # DAgger 정답
    # sumo_env가 SUMO --seed를 np.random(전역)에서 뽑는다. PPO 생성자가 전역 시드를
    # HPARAMS["seed"]로 되돌려 놓으므로 정책을 만든 "뒤에" 워커별로 다시 시드해야
    # 워커들이 같은 교통 패턴을 반복하지 않는다.
    np.random.seed(worker_seed)

    # NPZ는 학습에서 바로 쓰기 좋은 고정 크기 값만 모은다.
    obs_buf, next_obs_buf, action_raw_buf = [], [], []
    lane_cmd_buf, reward_buf = [], []
    term_buf, trunc_buf, episode_buf, step_buf = [], [], [], []
    # 스텝별 감지 차량 (x,y)는 한 줄로 이어 붙이고, 스텝 경계는 offsets로 기록한다.
    detected_xy_buf, detected_offsets = [], [0]
    #
    sim_time_buf, route_buf, end_reason_buf = [], [], []
    sig_tls_buf, sig_state_buf, sig_remain_buf = [], [], []

    total = 0
    try:
        # compresslevel 6: 9와 크기 차이는 거의 없고 훨씬 빠르다.
        # 더 줄이고 싶으면 gzip.open 대신 lzma.open(..., "wt") (.jsonl.xz) :
        # 느린 대신 보통 20~30% 추가 절감.
        with gzip.open(jsonl_path, "wt", encoding="utf-8", compresslevel=6) as jf:
            for ep in episodes:
                opts = {k: v for k, v in (("scenario", args.scenario), ("route", args.route)) if v}
                dl_opt = getattr(args, "depart_lane", "best")               # 출발 옵션 (기본값이면 안 넣음)
                if dl_opt == "random":
                    opts["depart_lane"] = "random"
                elif dl_opt != "best":                                       # 차로 번호 고정 (영상 고르기용)
                    opts["depart_lane"] = int(dl_opt)
                if getattr(args, "depart_speed", "default") == "limit":
                    opts["depart_speed"] = "limit"
                if getattr(args, "paired_seeds", False):
                    paired_seed = episode_seed(args.seed, ep)
                    np.random.seed(paired_seed)
                    # 수집 판마다 고유한 교통 시드를 사용한다. 환경은 기존 난수 추첨도 소비한다.
                    opts["sumo_seed"] = paired_seed % 2**31
                try:
                    obs, _ = env.reset(seed=args.seed + ep, options=opts or None)
                except RuntimeError as e:
                    # ego 삽입 실패(출발 차로 입구가 60스텝 넘게 막힘)는 그 판만 건너뛴다.
                    # 전에는 예외가 워커를 죽여 그 워커의 데이터 전체와 합치기 단계를 잃었다 (11/19 시범 · DAgger 수집에서 두 번).
                    # 건너뛴 판은 기록에 없으므로 episode 번호가 비어 있다 (평가 baseline.py도 같은 오류를 판 단위로 넘긴다).
                    print(f"{tag}episode {ep + 1:4d}/{args.episodes} | 건너뜀 (reset 실패: {e})", flush=True)
                    continue
                state_t = env.get_privileged_state()
                done = False
                t = 0
                ep_return = 0.0

                while not done:
                    # 이번 관측 시점에 시야(W) 안에 들어온 차량들의 좌표.
                    # 관측 벡터와 별도로 저장하는 부가 데이터 (분석/시각화용).
                    detected = env.get_detected_vehicles()
                    signal_t = env.get_signal_state()   # 관측 시점 t에 AV가 다음에 만날 신호
                    sim_time_t = env.get_sim_time()     # 관측 시점 t의 시뮬레이션 시각 (s)
                    sig_tls = SIGNAL_TLS_CODES.get(signal_t["tls"], -1) if signal_t else -1
                    sig_state = SIGNAL_STATE_VALUES.get(signal_t["state"], -1.0) if signal_t else -1.0
                    sig_rem = signal_t["remaining_s"] if signal_t else None
                    action = act(obs)
                    # 저장할 행동 라벨: 보통은 실행한 행동(sumo는 실제 움직임 역산), DAgger면 규칙 정책의 행동
                    if label is not None:
                        a_label = label(obs)
                    next_obs, reward, terminated, truncated, info = env.step(action)
                    if label is None:
                        a_label = (np.asarray(action, dtype=np.float32) if args.policy != "sumo"
                                   else np.asarray([info["accel_raw"], info["lane_change_raw"]], dtype=np.float32))
                    state_tp1 = env.get_privileged_state()
                    done = terminated or truncated

                    record = {
                        "episode": int(ep),
                        "t": t,
                        "observation": compact(obs, args.ndigits),
                        # 스칼라 몇 개는 용량에 영향이 없으니 반올림하지 않는다.
                        "action_raw": [float(x) for x in a_label],
                        "action": {
                            "accel_raw": float(info["accel_raw"]),
                            "lane_change_raw": float(info["lane_change_raw"]),
                            "lane_change": int(info["lane_change"]),
                            "lane_change_applied": bool(info["lane_change_applied"]),
                        },
                        "reward": float(reward),
                        "terminated": bool(terminated),
                        "truncated": bool(truncated),
                        "state": compact(state_t, args.ndigits),
                        # 감지 차량 좌표 (관측과 같은 시점 t 기준)
                        "detected_vehicles": compact(detected, args.ndigits),
                        "event": {
                            "collided": bool(info["collided"]),
                            "arrived": bool(info["arrived"]),
                            "lane_before": info["lane_before"],
                            "lane_after": info["lane_after"],
                        },
                        # ── 12팀 추가 칼럼 (기존 칼럼으로 계산할 수 없는 것만) ──
                        "sim_time": sim_time_t,                                         # 관측 시점 시뮬레이션 시각 (s)
                        "ego_route": ROUTE_CODES.get(info.get("route"), -1),           # AV 경로 = 목적지
                        "signal_tls": sig_tls,                                          # 앞 신호 (-1 없음)
                        "signal_state": sig_state,                                      # 녹 1 · 황 0.5 · 적 0 · 꺼짐/없음 -1
                        "signal_remaining_s": sig_rem,                                  # 색이 바뀔 때까지 s (없으면 null)
                        "end_reason": END_REASON_CODES.get(info.get("end_reason"), 7),  # 종료 원인 (진행 중 0)
                    }
                    if done:
                        # 중간 스텝의 next_*는 다음 레코드의 observation/state와 같으므로
                        # 에피소드 마지막 스텝에만 기록한다 (iter_transitions가 복원).
                        record["next_observation"] = compact(next_obs, args.ndigits)
                        record["next_state"] = compact(state_tp1, args.ndigits)
                    jf.write(json.dumps(record, ensure_ascii=False,
                                        separators=(",", ":")) + "\n")

                    obs_buf.append(np.asarray(obs, dtype=np.float32))
                    next_obs_buf.append(np.asarray(next_obs, dtype=np.float32))
                    action_raw_buf.append(np.asarray(a_label, dtype=np.float32))
                    lane_cmd_buf.append(int(info["lane_change"]))
                    reward_buf.append(float(reward))
                    term_buf.append(bool(terminated))
                    trunc_buf.append(bool(truncated))
                    episode_buf.append(int(ep))
                    step_buf.append(t)
                    detected_xy_buf.extend((d["x"], d["y"]) for d in detected)
                    detected_offsets.append(len(detected_xy_buf))
                    # 정수 코드로 (뜻은 아래 *_CODES)
                    sim_time_buf.append(sim_time_t)
                    route_buf.append(ROUTE_CODES.get(info.get("route"), -1))
                    sig_tls_buf.append(sig_tls)
                    sig_state_buf.append(sig_state)
                    sig_remain_buf.append(signal_t["remaining_s"] if signal_t and signal_t["remaining_s"] is not None else np.nan)
                    end_reason_buf.append(END_REASON_CODES.get(info.get("end_reason"), 7))

                    total += 1
                    ep_return += reward
                    t += 1
                    obs = next_obs
                    state_t = state_tp1  # 스텝 끝의 상태 = 다음 스텝 시작의 상태

                print(f"{tag}episode {ep + 1:4d}/{args.episodes} | "
                      f"steps={t:4d} return={ep_return:8.3f}", flush=True)
    finally:
        env.close()

    np.savez_compressed(
        npz_path,
        observation=np.asarray(obs_buf, dtype=np.float32).reshape(total, -1),
        action_raw=np.asarray(action_raw_buf, dtype=np.float32).reshape(total, -1),
        lane_change=np.asarray(lane_cmd_buf, dtype=np.int8),
        reward=np.asarray(reward_buf, dtype=np.float32),
        next_observation=np.asarray(next_obs_buf, dtype=np.float32).reshape(total, -1),
        terminated=np.asarray(term_buf, dtype=np.bool_),
        truncated=np.asarray(trunc_buf, dtype=np.bool_),
        episode=np.asarray(episode_buf, dtype=np.int32),
        t=np.asarray(step_buf, dtype=np.int32),
        # 감지 차량 (x,y)를 전 스텝에 걸쳐 이어 붙인 (M, 2) float32.
        # 스텝 i의 좌표: detected_xy[detected_offsets[i]:detected_offsets[i+1]]
        detected_xy=np.asarray(detected_xy_buf, dtype=np.float32).reshape(-1, 2),
        detected_offsets=np.asarray(detected_offsets, dtype=np.int64),
        # 코드표는 SCENARIO_CODES / ROUTE_CODES / SIGNAL_TLS_CODES / SIGNAL_STATE_VALUES / END_REASON_CODES
        sim_time=np.asarray(sim_time_buf, dtype=np.float64),
        ego_route=np.asarray(route_buf, dtype=np.int8),
        signal_tls=np.asarray(sig_tls_buf, dtype=np.int8),
        signal_state=np.asarray(sig_state_buf, dtype=np.float32),
        signal_remaining_s=np.asarray(sig_remain_buf, dtype=np.float32),
        end_reason=np.asarray(end_reason_buf, dtype=np.int8),
    )
    return total


def merge_parts(parts, jsonl_path, npz_path):
    """워커별 부분 파일들을 하나로 합치고 부분 파일은 지운다."""
    # gzip은 멤버를 이어 붙인 파일도 유효하므로 바이트 그대로 연결한다.
    with open(jsonl_path, "wb") as out:
        for part_jsonl, _ in parts:
            with open(part_jsonl, "rb") as f:
                shutil.copyfileobj(f, out)
            os.remove(part_jsonl)

    arrays, xy, offsets, base = {}, [], [], 0
    for _, part_npz in parts:
        with np.load(part_npz) as d:
            for key in d.files:
                if key not in ("detected_xy", "detected_offsets"):
                    arrays.setdefault(key, []).append(d[key])
            xy.append(d["detected_xy"])
            off = d["detected_offsets"]
            offsets.append(off[:-1] + base)   # 부분 파일의 offset을 전체 기준으로 옮김
            base += int(off[-1])
        os.remove(part_npz)
    merged = {key: np.concatenate(v) for key, v in arrays.items()}
    merged["detected_xy"] = np.concatenate(xy).reshape(-1, 2)
    merged["detected_offsets"] = np.concatenate(
        offsets + [np.asarray([base], dtype=np.int64)]).astype(np.int64)
    np.savez_compressed(npz_path, **merged)


def main():
    parser = argparse.ArgumentParser(description="SUMO POMDP trajectory collector")
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--policy", choices=["random", "keep-lane", "ppo", "bc", "sumo", "rule"], default=None,
                        help="행동 정책. 생략 시 --model이 있으면 파일을 보고 ppo/bc 자동 판별, "
                             "없으면 random")
    parser.add_argument("--model", default=None,
                        help="training/train.py · training/train_team.py(PPO · BC)가 저장한 model.pt."
                             "주면 그 학습된 정책으로 수집한다")
    parser.add_argument("--demo-min-gap", type=float, default=None,
                        help="[12팀 11/17] --policy sumo의 앞차 정지 간격 (m). 기본은 차종 값 2.5 m = 충돌 판정 경계에 정확히 선다 — "
                             "BC 시범이면 4.5처럼 여유를 둔다 (규칙 정책 시범을 못 쓸 때의 대안)")
    parser.add_argument("--label-policy", choices=["rule"], default=None,
                        help="[12팀 11/17] DAgger: --model 정책으로 운전하고 action_raw에는 규칙 정책이 그 관측에서 냈을 행동을 "
                             "저장한다 (BC 시범 전용)")
    parser.add_argument("--stochastic", action="store_true",
                        help="ppo: 결정적(μ, argmax 차선) 대신 분포에서 샘플링해서 행동. "
                             "상태 다양성이 필요할 때")
    parser.add_argument("--safety", choices=["off", "on"], default="off",
                        help="[12팀] 안전장치(충돌 방지 제동, mdp_config.ACTION[\"safety_brake\"]) 끔/켬. "
                             "데이터에 칼럼으로 남지 않으므로 --name과 데이터 카드에 적는다 (예: final_200000_safetyon)")
    parser.add_argument("--device", default="cpu",
                        help="정책 추론 장치 (cpu/cuda). 관측 1개씩 추론이라 cpu로 충분")
    parser.add_argument("--seed", type=int, default=0)
    # 시나리오·AV 경로 지정 (생략하면 에피소드마다 무작위 : environment/env/road_config.py ROAD["scenario"]="mixed")
    parser.add_argument("--scenario", choices=["before", "after"], default=None,
                        help="잠수교 보행화 전(before) / 후(after) 고정")
    parser.add_argument("--route", choices=list(ROUTE_CODES), default=None,
                        help="AV 경로 고정 (경로마다 쓸 수 있는 시나리오는 road_spec.AV_ROUTES의 scenarios — "
                             "한 시나리오에만 있는 경로를 --scenario 없이 주면 그 시나리오로 고정)")
    # AV 출발 차로 · 속도 (env.reset options : 기본값이면 옵션을 주지 않아 기존 동작과 같다)
    parser.add_argument("--depart-lane", type=lambda v: v if v in ("best", "random") else str(int(v)), default="best",
                        metavar="{best,random,차로 번호}",
                        help="AV 출발 차로: best = 옵션 안 줌(EGO 설정) · random = 출발 엣지 승용차 차로 중 무작위")
    parser.add_argument("--depart-speed", choices=["default", "limit"], default="default",
                        help="AV 출발 속도: default = 옵션 안 줌(EGO 설정) · limit = 출발 차로 제한속도 × 0.8")
    parser.add_argument("--gui", action="store_true",
                        help="SUMO GUI로 수집. 기본은 headless")
    parser.add_argument("--workers", type=int, default=1,
                        help="병렬 SUMO 인스턴스 수. 코어 수만큼 주면 거의 그 배수로 빨라진다. "
                             "(--gui면 1로 고정)")
    parser.add_argument("--paired-seeds", action="store_true",
                        help="[12팀] 에피소드 번호로 SUMO 시드를 직접 정한다 (워커 수와 무관하게 수집 재현; 평가의 기존 교통 시드와는 다름)")
    parser.add_argument("--out-dir", default=os.path.join(BASE, "data"))
    parser.add_argument("--name", default=None,
                        help="출력 basename. 생략 시 날짜/시각 자동 생성")
    parser.add_argument("--ndigits", type=int, default=4,
                        help="JSONL의 observation/state/detected float 소수 자릿수 "
                             "(NPZ에는 영향 없음)")
    args = parser.parse_args()
    if args.policy in ("ppo", "bc") and args.model is None:
        parser.error(f"--policy {args.policy}에는 --model이 필요합니다.")
    if args.model is not None and not os.path.exists(args.model):
        parser.error(f"모델 파일이 없습니다: {args.model}")
    if args.policy is None:
        args.policy = detect_algorithm(args.model) if args.model else "random"
    if args.route is not None:               # 경로별 시나리오 제한 (road_spec.AV_ROUTES[...]["scenarios"])
        road_dir = os.path.join(P.ROOT, "environment", "road")
        if road_dir not in sys.path:
            sys.path.insert(0, road_dir)
        import road_spec
        allowed = road_spec.AV_ROUTES[args.route]["scenarios"]
        if allowed:
            if args.scenario is not None and args.scenario not in allowed:
                parser.error(f"--route {args.route}는 {'/'.join(allowed)} 시나리오에만 있습니다.")
            if args.scenario is None and len(allowed) == 1:
                args.scenario = allowed[0]
    if args.model:
        print(f"정책: {args.policy} ({args.model})"
              + (" [stochastic]" if args.stochastic and args.policy == "ppo" else ""))
    if args.label_policy:
        if args.policy in ("sumo", args.label_policy):
            parser.error("--label-policy는 학습한 모델(--model)이나 다른 정책으로 운전할 때만 의미가 있습니다.")
        print(f"행동 라벨: {args.label_policy} (DAgger — action_raw = 규칙 정책 행동, 운전은 {args.policy})")
    print(f"안전장치: {'켬' if args.safety == 'on' else '끔'} (--safety {args.safety})")
    workers = 1 if args.gui else max(1, min(args.workers, args.episodes))

    os.makedirs(args.out_dir, exist_ok=True)
    name = args.name or datetime.now().strftime("pomdp_%Y%m%d_%H%M%S")
    jsonl_path = os.path.join(args.out_dir, name + ".jsonl.gz")
    npz_path = os.path.join(args.out_dir, name + ".npz")

    # 도로 파일은 부모에서 한 번만 생성한다 (워커들이 동시에 쓰면 파일이 깨진다).
    sumocfg = road_builder.build(ROAD, EGO, TRAFFIC,
                                 P.SUMO_DIR)

    t0 = time.time()
    if workers == 1:
        total = collect(args, sumocfg, list(range(args.episodes)), 0,
                        jsonl_path, npz_path)
    else:
        chunks = [c.tolist() for c in np.array_split(np.arange(args.episodes), workers)]
        parts = [(os.path.join(args.out_dir, f".{name}.w{w}.jsonl.gz"),
                  os.path.join(args.out_dir, f".{name}.w{w}.npz"))
                 for w in range(workers)]
        jobs = [(args, sumocfg, chunks[w], w, parts[w][0], parts[w][1], f"[w{w}] ")
                for w in range(workers)]
        # spawn: 워커마다 깨끗한 프로세스에서 SUMO(libsumo)를 하나씩 띄운다.
        with multiprocessing.get_context("spawn").Pool(workers) as pool:
            total = sum(pool.starmap(collect, jobs))
        merge_parts(parts, jsonl_path, npz_path)
    elapsed = time.time() - t0

    mb = lambda p: os.path.getsize(p) / 2**20  # noqa: E731
    print("\n수집 완료")
    print(f"  transitions: {total}  ({elapsed:.1f}s, {total / max(elapsed, 1e-9):.0f} step/s, "
          f"workers={workers})")
    print(f"  JSONL(full POMDP + privileged state): {jsonl_path} ({mb(jsonl_path):.1f} MB)")
    print(f"  NPZ(training-friendly):               {npz_path} ({mb(npz_path):.1f} MB)")


if __name__ == "__main__":
    main()
