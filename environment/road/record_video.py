"""주행 영상 녹화: sumo-gui로 한 에피소드를 달리며 매 스텝 화면을 저장하고 ffmpeg로 mp4를 만든다.

사용 (프로젝트 루트에서):
    python -X utf8 environment/road/record_video.py                                   # SUMO 운전자 · 측도 경로 · 통제 전
    python -X utf8 environment/road/record_video.py --route goga-jamsu --speed 3
    python -X utf8 environment/road/record_video.py --model results/<run>/model.pt   # 학습한 AV (PPO · BC 자동 판별)
    python -X utf8 environment/road/record_video.py --policy rule --safety           # 관측만 보는 규칙 정책, 안전장치 켬

- 화면: AV를 따라가며 --zoom 확대 (기본 1500 — 시야 약 100 m). sumo-gui 창이 녹화하는 동안 떠 있다.
- 시드는 evaluation/baseline.py 평가와 같은 식 (collect_pomdp_data.episode_seed) — 평가 에피소드 번호 그대로 다시 볼 수 있다.
- 출력: environment/road/videos/<이름>.mp4 (프레임 PNG는 임시 폴더, 끝나면 지움). 영상 속도 = 실제 시간 × --speed.
- 필요: ffmpeg (PATH). 수업 요구: 30초 이상, 도로 · HV · AV · 상호작용 · 목적지 도착이 보여야 한다.
"""
import argparse
import os
import shutil
import subprocess
import sys
import tempfile

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))   # 저장소 루트
import repo_paths as P  # noqa: E402
ROOT = P.ROOT
os.chdir(ROOT)
os.environ["SUMO_LIBSUMO"] = "0"            # 화면은 TraCI 소켓으로만

import numpy as np  # noqa: E402
import collect_pomdp_data as C  # noqa: E402
from env import road_builder  # noqa: E402
from env.sumo_env import SumoHighwayEnv  # noqa: E402
import env.sumo_env as E  # noqa: E402

SEED = {"side-banpo": 1000, "goga-banpo": 2000, "goga-jamsu": 3000, "oly-jamsu": 4000, "sbr-banpo": 5000}   # evaluation/baseline.py와 같게


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", choices=["sumo", "rule", "ppo", "bc"], default=None)
    ap.add_argument("--model", default=None)
    ap.add_argument("--scenario", choices=["before", "after"], default=None,
                    help="생략하면 경로가 있는 시나리오 (두 시나리오 모두면 before)")
    ap.add_argument("--route", choices=list(SEED), default="side-banpo")
    ap.add_argument("--episode", type=int, default=0, help="에피소드 번호 (평가 · 기준 지표와 같은 시드)")
    ap.add_argument("--safety", action="store_true", help="안전장치 켬 (RL 모드)")
    ap.add_argument("--depart-lane", type=lambda v: v if v in ("best", "random") else str(int(v)), default="best",
                    metavar="{best,random,차로 번호}",
                    help="AV 출발 차로: best = EGO 설정 · random = 출발 엣지 승용차 차로 중 무작위 · 번호 = 그 차로 (11/21 — pick_video_episodes.py가 고른 조건 그대로)")
    ap.add_argument("--depart-speed", choices=["default", "limit"], default="default",
                    help="AV 출발 속도: limit = 출발 차로 제한속도 × 0.8 (평가 · 최종 수집과 같은 조건)")
    ap.add_argument("--speed", type=float, default=3.0, help="영상 배속 (실제 시간 대비)")
    ap.add_argument("--size", default="1600x900", help="화면 저장 크기 (가로x세로)")
    ap.add_argument("--zoom", type=float, default=1500.0,
                    help="sumo-gui 확대 %% (환경 기본 gui_zoom 4000은 시야 약 40 m, 600이면 약 250 m지만 차가 점처럼 작음 — 기본 1500 ≈ 100 m)")
    ap.add_argument("--name", default=None)
    a = ap.parse_args()
    pol = a.policy or (C.detect_algorithm(a.model) if a.model else "sumo")
    import road_spec                          # 경로별 시나리오 제한 (AV_ROUTES[...]["scenarios"])
    allowed = road_spec.AV_ROUTES[a.route]["scenarios"] or ("before", "after")
    if a.scenario is None:
        a.scenario = allowed[0]
    elif a.scenario not in allowed:
        ap.error(f"{a.route}는 {'/'.join(allowed)} 시나리오에만 있다")
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        ap.error("ffmpeg가 PATH에 없다")
    w, h = (int(x) for x in a.size.split("x"))
    name = a.name or f"{pol}_{a.scenario}_{a.route}_ep{a.episode}" + ("_safety" if a.safety else "")
    out_dir = P.VIDEOS
    os.makedirs(out_dir, exist_ok=True)
    out = os.path.join(out_dir, name + ".mp4")

    sumocfg = road_builder.build(C.ROAD, C.EGO, C.TRAFFIC, P.SUMO_DIR)
    action = dict(C.ACTION, safety_brake=dict(C.ACTION.get("safety_brake") or {}, enabled=a.safety))
    env = SumoHighwayEnv(cfg_path=sumocfg, road=C.ROAD, ego=dict(C.EGO, sumo_driver=(pol == "sumo")),
                         mdp_sim=dict(C.SIMULATION, gui_delay=0), mdp_obs=C.OBSERVATION, action=action,
                         reward=C.REWARD, traffic=C.TRAFFIC, gui=True, gui_autostart=True)
    if pol == "rule":
        import check_env
        act = check_env.rule_policy(env)
    else:
        act = C.make_policy(argparse.Namespace(policy=pol, model=a.model, device="cpu", stochastic=False), env, SEED[a.route])
    frames = tempfile.mkdtemp(prefix="frames_")
    tc = E.traci
    try:
        np.random.seed(C.episode_seed(SEED[a.route], a.episode))
        opts = {"scenario": a.scenario, "route": a.route}
        if a.depart_lane == "random":           # [12팀 11/21] collect_pomdp_data.py와 같은 옵션 · 같은 순서 → 같은 판
            opts["depart_lane"] = "random"
        elif a.depart_lane != "best":
            opts["depart_lane"] = int(a.depart_lane)
        if a.depart_speed == "limit":
            opts["depart_speed"] = "limit"
        obs, _ = env.reset(seed=SEED[a.route] + a.episode, options=opts)
        view = tc.gui.getIDList()[0]
        tc.gui.trackVehicle(view, env.ego_id)
        tc.gui.setZoom(view, a.zoom)
        k, done, info = 0, False, {}
        while not done:
            tc.gui.screenshot(view, os.path.join(frames, f"f{k:05d}.png"), w, h)   # 다음 스텝 화면이 그려진 뒤 저장됨
            obs, _, term, trunc, info = env.step(act(obs))
            k += 1
            done = term or trunc
        tc.simulationStep()                          # 마지막 화면 저장
    finally:
        env.close()
    fps = a.speed / env.step_length
    n = len([f for f in os.listdir(frames) if f.endswith(".png")])
    r = subprocess.run([ffmpeg, "-y", "-loglevel", "error", "-framerate", f"{fps:g}", "-i", os.path.join(frames, "f%05d.png"),
                        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2", "-r", "30", out],
                       capture_output=True, text=True)
    shutil.rmtree(frames, ignore_errors=True)
    if r.returncode != 0:
        raise SystemExit(f"ffmpeg 실패: {r.stderr[-800:]}")
    print(f"{out}: 프레임 {n}개, 실제 {k * env.step_length:.0f} s → 영상 {n / fps:.0f} s ({a.speed:g}배속), "
          f"종료 {info.get('end_reason')}")


if __name__ == "__main__":
    main()
