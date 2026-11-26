"""시범·DAgger NPZ의 action_raw를 현재 규칙 정책의 정답으로 다시 매긴다.

SUMO를 실행하지 않고 저장된 관측을 check_env.rule_policy에 넣는다.
다른 배열은 보존하고 --out-dir에 같은 파일명으로 저장한다. JSONL은 수정하지 않는다.
"""
import argparse
import glob
import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import repo_paths as P  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+", help="다시 매길 NPZ (경로 · 패턴)")
    ap.add_argument("--out-dir", required=True)
    a = ap.parse_args()
    paths = sorted({os.path.abspath(p) for pat in a.files for p in glob.glob(pat)})
    if not paths:
        raise SystemExit("파일이 없습니다.")
    out_dir = os.path.abspath(a.out_dir)

    import collect_pomdp_data as C
    from env import road_builder
    from env.sumo_env import SumoHighwayEnv
    sumocfg = road_builder.build(C.ROAD, C.EGO, C.TRAFFIC, P.SUMO_DIR)
    env = SumoHighwayEnv(cfg_path=sumocfg, road=C.ROAD, ego=dict(C.EGO, sumo_driver=False), mdp_sim=C.SIMULATION,
                         mdp_obs=C.OBSERVATION, action=C.ACTION, reward=C.REWARD, traffic=C.TRAFFIC, gui=False)   # SUMO는 띄우지 않는다
    label = C._rule_policy(env)          # check_env를 불러오며 작업 폴더가 저장소 루트로 바뀐다  /  경로는 위에서 절대 경로로
    os.makedirs(out_dir, exist_ok=True)
    for p in paths:
        d = dict(np.load(p))
        obs, old = d["observation"], d["action_raw"]
        if obs.shape[1] != env.observation_space.shape[0]:
            raise SystemExit(f"{p}: 관측 {obs.shape[1]}차원 — 지금 환경({env.observation_space.shape[0]}차원)과 다릅니다.")
        new = np.asarray([label(o) for o in obs], dtype=np.float32).reshape(len(obs), -1)
        d["action_raw"] = new
        out = os.path.join(out_dir, os.path.basename(p))
        np.savez_compressed(out, **d)
        da = np.abs(new[:, 0] - old[:, 0])
        lane_old, lane_new = np.round(old[:, 1]), np.round(new[:, 1])
        print(f"{os.path.basename(p)}: {len(obs):,} 스텝 | 가감속 정답이 0.05 넘게 바뀐 스텝 {100 * (da > 0.05).mean():.1f}% "
              f"(평균 변화 {da.mean():.3f}) | 차로 정답이 바뀐 스텝 {100 * (lane_old != lane_new).mean():.2f}%", flush=True)


if __name__ == "__main__":
    main()
