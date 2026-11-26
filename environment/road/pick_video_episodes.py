"""영상으로 찍을 판 고르기: 경로마다 후보 판을 화면 없이 먼저 돌려 상호작용 점수를 매기고, 점수가 높은 판 번호를 알려 준다.

record_video.py --episode N 과 **같은 시드 · 같은 출발 조건**으로 돌리므로, 여기서 고른 번호를 그대로 녹화하면 같은 판이 나온다
(시드 = evaluation/baseline.py SEED[경로] 기준 collect_pomdp_data.episode_seed, 출발 옵션도 같게 넘긴다).
기본 출발 = **경로마다 가장 어려운 차로**(HARD_LANE — 목적지까지 차로를 가장 많이 바꿔야 하는 차로) · 제한속도 × 0.8.
그래야 모델의 차로 선택과 끼어들기가 영상에 확실히 나온다 (record_video 기본값 best 차로로 출발하면 이미 맞는 차로라
차로 변경이 거의 없다). --depart-lane random(평가 · 최종 수집 조건) · best · 차로 번호도 된다.

사용 (저장소 루트에서):
    python -X utf8 environment/road/pick_video_episodes.py --model results/team_rl_1125_VA2/model_261632.pt
    python -X utf8 environment/road/pick_video_episodes.py --model … --routes goga-jamsu side-banpo --episodes 60 --top 5

점수 (도착한 판만, 시간은 초):
    경합 s       회전교차로 진입 · 합류 · 교차로에서 AV와 같은 지점으로 오는 차가 관측에 보인 시간 (observation[42..43])
    양보 s       그중 AV가 양보해야 하는 경합 (observation[45] = -1) — 경합에도 이미 들어가 있어 두 번 센다
    나란히 s     바로 옆 차로에 나란히 달리는 차가 있던 시간 (observation[5] · [13] = 0)
    차로 변경    실제로 실행된 차로 변경 횟수 (evaluation/metrics.py lane_changes)
    점수 = 경합 s + 양보 s + 0.5 × 나란히 s + 10 × 차로 변경
신호 대기(빨간불 앞 정지)는 상호작용으로 세지 않는다.

출력: 경로별 상위 판 표와 바로 붙여 쓸 record_video 명령. 후보 데이터는 data/video_pick/ (git 제외, 다 쓰면 지워도 된다).
"""
import argparse
import os
import subprocess
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))   # 저장소 루트
import repo_paths as P  # noqa: E402

import numpy as np  # noqa: E402
import metrics as M  # noqa: E402
import road_spec  # noqa: E402

SEED = {"side-banpo": 1000, "goga-banpo": 2000, "goga-jamsu": 3000, "oly-jamsu": 4000, "sbr-banpo": 5000}   # record_video.py · baseline.py와 같게
DT = M.DT


# 경로별 가장 어려운 출발 차로 (확정 평가 evaluation/eval/final_VA2_261632 1,000판에서 출발 차로별 평균 차로 변경이 가장 많은 차로):
#   goga-jamsu 0번 5회 · goga-banpo 0번 4회 · side-banpo 5번 2.9회 · oly-jamsu 2번 2.9회 · sbr-banpo 1번 1회 (모두 도착률 90% 이상)
HARD_LANE = {"goga-jamsu": 0, "goga-banpo": 0, "side-banpo": 5, "oly-jamsu": 2, "sbr-banpo": 1}


def depart_lane_of(a, route):
    return str(HARD_LANE[route]) if a.depart_lane == "hard" else a.depart_lane


def depart_args(a, route):
    lane = depart_lane_of(a, route)
    return ((["--depart-lane", lane] if lane != "best" else [])
            + (["--depart-speed", "limit"] if a.depart_speed == "limit" else []))


def run_candidates(a, route, scenario, out_dir, name):
    cmd = [sys.executable, "-X", "utf8", os.path.join(P.ROOT, "data_collection", "collect_pomdp_data.py"),
           "--model", a.model, "--safety", "off", "--paired-seeds", "--seed", str(SEED[route]),
           "--route", route, "--scenario", scenario, "--episodes", str(a.episodes), "--workers", str(a.workers),
           "--out-dir", out_dir, "--name", name] + depart_args(a, route)
    print(f"[{route}] 후보 {a.episodes}판 수집 중 …", flush=True)
    r = subprocess.run(cmd, cwd=P.ROOT, capture_output=True, text=True, encoding="utf-8", errors="replace")
    if r.returncode != 0:
        print(r.stdout[-2000:], r.stderr[-2000:])
        raise SystemExit(f"[{route}] 수집 실패")


def score_route(npz_path, jsonl_path):
    d = np.load(npz_path)
    o, ep = d["observation"], d["episode"]
    foe = (o[:, 43] < 1) & (o[:, 42] < 1)
    yld = foe & (o[:, 45] < -0.5)
    side = (np.abs(o[:, 5]) < 1e-6) | (np.abs(o[:, 13]) < 1e-6)
    lc = {int(m["episode"]): int(m["lane_changes"]) for m in M.file_metrics(jsonl_path)}
    rows = []
    for e in np.unique(ep):
        m = ep == e
        last = np.flatnonzero(m)[-1]
        arrived = int(d["end_reason"][last]) == 1
        r = dict(episode=int(e), arrived=arrived, steps=int(m.sum()),
                 conflict_s=foe[m].sum() * DT, yield_s=yld[m].sum() * DT, side_s=side[m].sum() * DT,
                 lane_changes=lc.get(int(e), 0))
        r["score"] = r["conflict_s"] + r["yield_s"] + 0.5 * r["side_s"] + 10 * r["lane_changes"]
        rows.append(r)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--routes", nargs="+", choices=list(SEED), default=["goga-jamsu", "side-banpo", "sbr-banpo", "oly-jamsu", "goga-banpo"])
    ap.add_argument("--episodes", type=int, default=40, help="경로당 후보 판 수 (판 번호 0 ~ N-1)")
    ap.add_argument("--workers", type=int, default=7)
    ap.add_argument("--top", type=int, default=3)
    ap.add_argument("--speed", type=float, default=3.0, help="영상 배속 (record_video 기본 3) — 영상 길이 계산용")
    ap.add_argument("--depart-lane", default="hard", help="hard(기본, 경로별 가장 어려운 차로) · random · best · 차로 번호")
    ap.add_argument("--depart-speed", choices=["default", "limit"], default="limit")
    ap.add_argument("--reuse", action="store_true", help="data/video_pick/에 같은 후보가 있으면 다시 돌리지 않고 점수만")
    a = ap.parse_args()
    out_dir = os.path.join(P.ROOT, "data", "video_pick")
    os.makedirs(out_dir, exist_ok=True)
    tag = os.path.splitext(os.path.basename(os.path.dirname(os.path.abspath(a.model))))[0]
    cmds = []
    for route in a.routes:
        scenario = (road_spec.AV_ROUTES[route]["scenarios"] or ("before", "after"))[0]   # record_video.py와 같은 기본값
        name = f"{route}_{tag}_{a.episodes}_lane{depart_lane_of(a, route)}_{a.depart_speed}"
        npz, jsonl = (os.path.join(out_dir, name + ext) for ext in (".npz", ".jsonl.gz"))
        if not (a.reuse and os.path.exists(npz) and os.path.exists(jsonl)):
            run_candidates(a, route, scenario, out_dir, name)
        rows = score_route(npz, jsonl)
        ok = sorted((r for r in rows if r["arrived"] and r["steps"] * DT / a.speed >= 30), key=lambda r: -r["score"])
        print(f"\n[{route}] ({scenario}, 출발 차로 {depart_lane_of(a, route)}) 도착 {sum(r['arrived'] for r in rows)}/{len(rows)}판 — 상위 {a.top}")
        print(f"  {'판':>4} {'점수':>6} {'경합s':>6} {'양보s':>6} {'나란히s':>7} {'차로변경':>6} {'영상s':>5}")
        for r in ok[:a.top]:
            print(f"  {r['episode']:4d} {r['score']:6.1f} {r['conflict_s']:6.1f} {r['yield_s']:6.1f} {r['side_s']:7.1f} "
                  f"{r['lane_changes']:6d} {r['steps'] * DT / a.speed:5.0f}")
        if ok:
            b = ok[0]["episode"]
            cmds.append(f"python -X utf8 environment/road/record_video.py --model {a.model} --route {route} --episode {b} "
                        f"{' '.join(depart_args(a, route))} --name final_{tag}_{route}_ep{b}")
    print("\n가장 높은 판 녹화 명령:")
    for c in cmds:
        print("  " + c)


if __name__ == "__main__":
    main()
