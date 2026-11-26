"""기준 지표: SUMO 기본 운전자로 시나리오 × AV 경로마다 여러 번 주행  /  잠수교 통제 전후 비교.
+ 평가 (11/15 §12.0 7번): 학습 모델 · 규칙 정책을 같은 5조합 · 같은 시드 · 같은 지표로 돌려 실패 원인별로 센다.

사용 (프로젝트 루트에서):
    python -X utf8 evaluation/baseline.py --episodes 30 --workers 7      # 기준 지표: 수집 + 집계
    python -X utf8 evaluation/baseline.py --skip-collect                 # 이미 모은 데이터로 집계만
    python -X utf8 evaluation/baseline.py --model results/<run>/model.pt --episodes 100 --workers 7   # 학습 모델 평가
    python -X utf8 evaluation/baseline.py --policy rule --safety both --episodes 10    # 관측만 보는 규칙 정책 (도구 점검 · 참고 기준선)
    python -X utf8 evaluation/baseline.py --policy sumo --eval --episodes 30           # SUMO 운전자를 평가 경로로 (데이터 파일 없이)
    python -X utf8 evaluation/baseline.py --policy rule --depart-lane random --depart-speed limit --episodes 40   # 출발 차로 무작위 (11/19)
    SUMO_LANE_DIR_OBS=0 python -X utf8 evaluation/baseline.py --model <43차원 모델> ...   # 11/19 전 43차원 관측으로 (mdp_config lane_dir_obs 덮어쓰기)

평가 (--model / --policy rule / --eval):
- 수집 파일 없이 워커 프로세스 안에서 주행하고 바로 지표를 계산한다 (1,000회를 JSONL로 남기면 약 5 GB). 지표는
  metrics.episode_metrics 그대로라 기준 지표와 같은 정의다. 실패는 end_reason으로 나눠 센다: 충돌 collided · 경로 이탈 off_route ·
  비정상 정지 stuck · 신호 위반 red_light · 순간이동 teleported · 시간초과 timeout · 기타 other(end_reason 없음) ·
  오류 error(SUMO 오류로 중단 : 11/16 검토 C3: 그 에피소드만 error로 적고 SUMO를 다시 띄워 나머지를 잇는다. 작업은
  에피소드 하나씩이라 워커 프로세스가 죽어도 그 에피소드만 다시 돌리고(_run_jobs), 그래도 죽으면 error로 적는다).
  error는 정책의 결과가 아니므로 도착률 · 충돌률 · 리턴 평균의 분모에서 뺀다 (보고서 "오류" 칸에 수만 적음).
  수업 test.py(11/16 지움) · training/utils/evaluator.py는
  앞의 넷을 모두 "시간초과"로 세므로 보고에 쓰지 않는다 (수업 원본은 고치지 않음).
- --safety off | on | both: 안전장치(mdp_config.ACTION["safety_brake"], 충돌 방지 제동) 끔(기본, 11/17 컨설팅에서 불허) / 켬 / 둘 다. 켬이면 개입 횟수도 센다.
  SUMO 운전자에는 안전장치가 없다 (끔만).
- 시드: 에피소드 번호마다 collect_pomdp_data.episode_seed(경로 시드, 번호) : 워커 수와 무관. --stochastic이면 정책 샘플링
  난수(torch)도 에피소드마다 같은 값으로 다시 정한다 (11/16: 켬/끔 · 다시 돌리기끼리 짝이 맞게). 기준 지표 수집도 --paired-seeds로
  같은 식을 쓰므로 같은 경로 · 같은 번호 = 같은 교통 조건 (AV가 다르게 움직이면 그 뒤 교통은 달라진다).
- 출력: evaluation/eval/<이름>/metrics.csv (에피소드별) · report.md (요약 + 기준 지표 대비). 이름 = --name 또는 모델 폴더 이름
  (파일이 model.pt가 아니면 뒤에 _<파일명>, 예: team_rl_x_model_200000 : 11/16 검토 C4: 체크포인트끼리 덮어쓰지 않게).

- 수집은 data_collection/collect_pomdp_data.py --policy sumo를 그대로 쓴다 (data/baseline/<시나리오>_<경로>.jsonl.gz · .npz).
- 지표 계산은 evaluation/metrics.py (기존 칼럼만). 나중에 학습한 AV 데이터도 같은 지표로 비교한다.
- 같은 경로의 통제 전·후는 같은 시드를 쓴다: --paired-seeds(11/15)로 에피소드 번호마다 전역 난수를 다시 정하므로 워커 수와
  무관하게 같은 번호 = 같은 SUMO 시드 (sumo_env는 SUMO 시드를 전역 난수 np.random에서 뽑는다). 평가 모드도 같은 식.
- 출력: evaluation/baseline_metrics.csv (에피소드별), evaluation/baseline_report.md (요약)
"""
import argparse
import csv
import os
import statistics as stt
import subprocess
import sys
from collections import Counter
from datetime import datetime

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))   # 저장소 루트
import repo_paths as P  # noqa: E402
ROOT = P.ROOT
import metrics  # noqa: E402

# (시나리오, 경로) : 경로별 시나리오는 road_spec.AV_ROUTES의 scenarios (통제 전 잠수교 2 · 통제 후 반포대교 3)
COMBOS = [("before", "goga-jamsu"), ("before", "oly-jamsu"),
          ("after", "side-banpo"), ("after", "goga-banpo"), ("after", "sbr-banpo")]
SEED = {"side-banpo": 1000, "goga-banpo": 2000, "goga-jamsu": 3000, "oly-jamsu": 4000, "sbr-banpo": 5000}
DATA = os.path.join(ROOT, "data", "baseline")
OUT_CSV = os.path.join(P.EVALUATION, "baseline_metrics.csv")
# B 갈래 기준 지표: SUMO 운전자, 무작위 출발 차로 · 출발 속도 제한속도 × 0.8 (평가 모드 --policy sumo --eval 결과의 metrics.csv 사본)
OUT_CSV_RANDOM = os.path.join(P.EVALUATION, "baseline_metrics_random.csv")
OUT_MD = os.path.join(P.EVALUATION, "baseline_report.md")

# 요약에 넣을 지표: (열 이름, 표시 이름, 도착한 에피소드만 볼지)
SUMMARY = [("travel_s", "주행 시간 (s)", True), ("stops", "정지 횟수", True), ("stop_s", "정지 시간 (s)", True),
           ("stop_red_s", "그중 빨간불 대기 (s)", True), ("lane_changes", "차로 변경", True),
           ("lc_split", "· 합류 전 구간", True), ("lc_merge", "· 합류 뒤 구간", True), ("merge3_s", "③ 합류 구간 통과 (s)", True),
           ("min_gap_m", "앞차 최소 간격 (m)", True), ("min_gap_merge3_m", "· ③ 합류 구간 (m)", True),
           ("bridge_q_mean", "반포대교 진입부 저속 차량 (평균 대)", False),
           ("bridge_q_max", "· 최대 (대)", False)]


def collect(scn, route, episodes, workers):
    name = f"{scn}_{route}"
    cmd = [sys.executable, "-X", "utf8", os.path.join(P.DATA_COLLECTION, "collect_pomdp_data.py"), "--policy", "sumo",
           "--scenario", scn, "--route", route, "--episodes", str(episodes), "--workers", str(workers),
           "--seed", str(SEED[route]), "--paired-seeds", "--out-dir", DATA, "--name", name]
    print(f"[수집] {name}: {episodes}회, workers={workers}", flush=True)
    r = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, encoding="utf-8", errors="replace")
    tail = "\n".join(r.stdout.strip().splitlines()[-4:])
    print(tail, flush=True)
    if r.returncode != 0:
        raise SystemExit(f"수집 실패 {name}\n{r.stderr[-2000:]}")


def num(rows, key):
    return [float(r[key]) for r in rows if r.get(key) not in ("", None)]


def fmt(xs):
    if not xs:
        return "—"
    med = stt.median(xs)
    if len(xs) == 1:
        return f"{med:.1f}"
    return f"{stt.mean(xs):.1f} ± {stt.stdev(xs):.1f} (중앙 {med:.1f})"


def _overspeed_report(rows, safeties=None):
    """실패 판도 포함하고 전체·경로별 스텝 가중 비율 및 최대값을 표시한다."""
    lines = ["", "## 과속", "",
             "> 차로별 제한속도는 생성된 SUMO net.xml의 lane speed(m/s), 교차로 내부 차로도 포함한다. "
             f"초과 스텝 판정 허용 오차 {metrics.OVERSPEED_EPS_MPS:g} m/s. "
             "도착·실패 판 모두 포함하고 error 판은 제외한다.",
             "> 초과 스텝 % = 초과 스텝 합 / 제한속도 확인 스텝 합 × 100 (판별 %의 단순 평균이 아님). "
             "최대 초과폭은 전체 스텝의 최댓값. 미확인 스텝이 있으면 전체 구간을 검증한 값으로 볼 수 없다.", "",
             "| 안전장치 | 구분 | 제한속도 확인 스텝 | 초과 스텝 | 초과 스텝 % | > 1 km/h % | > 2 km/h % | > 5 km/h % | 최대 초과 (km/h) | 미확인 스텝 |",
             "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for sf in safeties or [None]:
        group = [r for r in rows if sf is None or r.get("safety") == sf]
        groups = [("전체", group)] + [(f"{scn} · {route}", [r for r in group
                   if (r.get("scenario"), r.get("ego_route")) == (scn, route)]) for scn, route in COMBOS]
        for label, selected in groups:
            if not selected:
                continue
            stat = metrics.overspeed_summary(selected)
            pct = "—" if stat["overspeed_pct"] == "" else f"{stat['overspeed_pct']:.3f}"
            maximum = "—" if stat["max_overspeed_kmh"] == "" else f"{stat['max_overspeed_kmh']:.3f}"
            bands = " | ".join("—" if stat.get(f"overspeed_gt{k}_pct", "") == "" else f"{stat[f'overspeed_gt{k}_pct']:.3f}"
                               for k in metrics.OVERSPEED_BANDS_KMH)
            lines.append(f"| {sf or '—'} | {label} | {stat['speed_limit_steps']} | {stat['overspeed_steps']} | "
                         f"{pct} | {bands} | {maximum} | {stat['speed_limit_unknown_steps']} |")
    return lines


def report(rows, episodes_each):
    groups = {c: [r for r in rows if (r["scenario"], r["ego_route"]) == c] for c in COMBOS}
    L = [f"# 기준 지표 — SUMO 기본 운전자, 잠수교 통제 전후 ({datetime.now():%Y-%m-%d %H:%M})", "",
         "> `evaluation/baseline.py`가 만든 파일 (다시 만들면 덮어씀). 지표 정의는 `evaluation/metrics.py` 머리말.",
         (f"> 조합마다 {episodes_each}회 (아래 표의 횟수가 실제 값). " if episodes_each else "> 이미 모은 데이터로 집계 (조합별 횟수는 아래 표). ")
         + "값은 평균 ± 표준편차 (중앙값). 주행 지표는 도착한 에피소드만.",
         "> 차로 변경의 합류 전 / 뒤는 구간별 집계다 (원인별 아님 — `metrics.py` 머리말).", "",
         "## 결과 요약", "", "| 시나리오 | 경로 | 횟수 | 도착 | 종료 원인 |", "|---|---|---|---|---|"]
    for (scn, route), g in groups.items():
        if not g:
            continue
        ends = Counter(r["end_reason"] or "?" for r in g)
        arr = ends.get("arrived", 0)
        L.append(f"| {scn} | {route} | {len(g)} | {arr} ({arr / len(g):.0%}) | "
                 + ", ".join(f"{k} {v}" for k, v in ends.most_common()) + " |")
    L += ["", "## 지표", "", "| 지표 | " + " | ".join(f"{s} · {r}" for s, r in COMBOS) + " |",
          "|---|" + "---|" * len(COMBOS)]
    for key, label, arrived_only in SUMMARY:
        cells = []
        for c in COMBOS:
            g = [r for r in groups[c] if not arrived_only or r["end_reason"] == "arrived"]
            cells.append(fmt(num(g, key)))
        L.append(f"| {label} | " + " | ".join(cells) + " |")
    L += ["", "## 통제 전 → 후 (같은 경로)", "",
          "> 전 · 후는 각 조합의 평균(값이 있는 에피소드 수 n). 짝 비교는 같은 에피소드 번호(같은 시드)끼리의 차이 평균과 짝지은 t다.",
          "> |t|가 대략 2 이상(30쌍이면 2.05)이면 p < 0.05. 비율(%)은 전 값이 1 이상일 때만 표시한다.", "",
          "| 경로 | 지표 | 전 (n) | 후 (n) | 차이 | 짝 차이 (쌍) | 짝지은 t |", "|---|---|---|---|---|---|---|"]
    # 두 시나리오 모두에 있는 경로만 (지금 구성에서는 없음  /  표 머리만 남는다)
    for route in [r for s, r in COMBOS if s == "before" and ("after", r) in groups]:
        for key, label, arrived_only in SUMMARY:
            pick = (lambda g: [r for r in g if r["end_reason"] == "arrived"]) if arrived_only else (lambda g: g)
            gb, ga = pick(groups[("before", route)]), pick(groups[("after", route)])
            xb, xa = num(gb, key), num(ga, key)
            if not (xb and xa):
                continue
            mb, ma = stt.mean(xb), stt.mean(xa)
            d = f"{ma - mb:+.1f}" + (f" ({(ma - mb) / mb:+.0%})" if abs(mb) >= 1 else "")
            ab = {r["episode"]: r[key] for r in ga if r.get(key) not in ("", None)}
            diffs = [float(ab[r["episode"]]) - float(r[key]) for r in gb
                     if r.get(key) not in ("", None) and r["episode"] in ab]
            pd, pt = "—", "—"
            if len(diffs) >= 2:
                pd = f"{stt.mean(diffs):+.1f} ({len(diffs)})"
                sd = stt.stdev(diffs)
                pt = f"{stt.mean(diffs) / (sd / len(diffs) ** 0.5):+.1f}" if sd > 0 else "—"
            L.append(f"| {route} | {label} | {mb:.1f} ({len(xb)}) | {ma:.1f} ({len(xa)}) | {d} | {pd} | {pt} |")
    L += _overspeed_report(rows)
    with open(OUT_MD, "w", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")
    print("\n".join(L))



# 평가 모드 (11/15 §12.0 7번)

FAIL_REASONS = ["collided", "off_route", "stuck", "red_light", "teleported", "timeout", "other", "error"]
REASON_KO = {"arrived": "도착", "collided": "충돌", "off_route": "경로 이탈", "stuck": "비정상 정지", "red_light": "신호 위반",
             "teleported": "순간이동", "timeout": "시간초과", "other": "기타", "error": "오류"}
EVAL_COLS = metrics.COLS + ["safety", "return", "safety_interventions", "depart_lane"]
POOL_ROUNDS = 3          # 워커 프로세스가 죽었을 때 남은 작업을 풀로 다시 돌리는 횟수 (_run_jobs)


def _eval_worker(job):
    """워커 하나: (정책, 모델, 안전장치, 시나리오, 경로, 에피소드 번호 목록, sumocfg, stochastic)  /  에피소드별 지표 행 목록.
    collect_pomdp_data.collect와 같은 환경 설정 · 같은 정책 생성(make_policy) · 같은 시드 식. 기록은 메모리에서만 (지표 계산에
    필요한 칼럼만: state · signal · event 차로 · end_reason)."""
    import argparse as _ap
    import numpy as np
    try:
        import torch
        torch.set_num_threads(1)          # 워커마다 torch가 코어 수만큼 스레드를 띄우면 서로 뺏는다
    except ImportError:
        pass
    import collect_pomdp_data as C
    from env.sumo_env import SumoHighwayEnv
    pol, model, safety, scn, route, eps, sumocfg, stochastic, depart = job
    action = dict(C.ACTION, safety_brake=dict(C.ACTION.get("safety_brake") or {}, enabled=(safety == "on")))

    def make():
        e = SumoHighwayEnv(cfg_path=sumocfg, road=C.ROAD, ego=dict(C.EGO, sumo_driver=(pol == "sumo")),
                           mdp_sim=C.SIMULATION, mdp_obs=C.OBSERVATION, action=action, reward=C.REWARD,
                           traffic=C.TRAFFIC, gui=False)
        if pol == "rule":
            import check_env
            return e, check_env.rule_policy(e)
        args = _ap.Namespace(policy=pol, model=model, device="cpu", stochastic=stochastic)
        return e, C.make_policy(args, e, SEED[route])

    def run(env, act, ep):
        np.random.seed(C.episode_seed(SEED[route], ep))      # 기준 지표 수집(--paired-seeds)과 같은 식
        if stochastic:
            import torch
            torch.manual_seed(C.episode_seed(SEED[route], ep))   # 샘플링 행동도 에피소드 번호로 고정
        obs, info0 = env.reset(seed=SEED[route] + ep, options={"scenario": scn, "route": route, **depart})
        st = env.get_privileged_state()
        recs, ret, done = [], 0.0, False
        while not done:
            sig = env.get_signal_state()
            obs, r, term, trunc, info = env.step(act(obs))
            nst = env.get_privileged_state()
            done = term or trunc
            ret += r
            recs.append({"episode": ep, "scenario": info.get("scenario"), "ego_route": info.get("route"),
                         "end_reason": info.get("end_reason"), "state": st, "signal": sig,
                         "event": {"lane_before": info["lane_before"], "lane_after": info["lane_after"]},
                         # 차로 변경 집계용 (metrics: 명령 ≠ 0 이고 실행됨 : 도로 경계 스텝도 센다)
                         "action": {"lane_change": int(info["lane_change"]), "lane_change_applied": bool(info["lane_change_applied"])},
                         "next_state": {"ego_road_id": nst.get("ego_road_id")}})
            st = nst
        m = metrics.episode_metrics(recs)
        m.update({"file": f"{scn}_{route}", "safety": safety, "return": round(ret, 3),
                  "depart_lane": info0.get("depart_lane"),        # 실제 출발 차로 (차로별 집계용)
                  "safety_interventions": info.get("safety_interventions", 0)})
        return m

    env, act = make()
    rows = []
    try:
        for ep in eps:
            try:
                m = run(env, act, ep)
                print(f"  [{scn} {route} 안전장치 {safety}] ep {ep:3d}: {m['end_reason']} {m['travel_s']:.0f}s", flush=True)
            except Exception as e:          # noqa: BLE001 : SUMO 오류: 이 에피소드만 error로 적고 SUMO를 새로 띄워 잇는다
                m = _error_row(scn, route, safety, ep)
                print(f"  [{scn} {route} 안전장치 {safety}] ep {ep:3d}: error {e!r}", flush=True)
                try:
                    env.close()
                except Exception:           # noqa: BLE001
                    pass
                env, act = make()
            rows.append(m)
    finally:
        env.close()
    return rows


def _error_row(scn, route, safety, ep):
    """SUMO 오류로 끝까지 못 돈 에피소드 (지표는 빈칸)."""
    return {"file": f"{scn}_{route}", "episode": ep, "scenario": scn, "ego_route": route, "end_reason": "error",
            "safety": safety, "return": 0.0, "safety_interventions": 0}


def _rate_ci(k, n):
    """비율과 95% 윌슨 구간."""
    if n == 0:
        return "—"
    z, p = 1.96, k / n
    c = (p + z * z / (2 * n)) / (1 + z * z / n)
    h = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5) / (1 + z * z / n)
    return f"{p:.1%} [{max(c - h, 0):.1%}, {min(c + h, 1):.1%}]"


def eval_report(rows, title, out_md, safeties, base_csv=None):
    L = [f"# 평가 — {title} ({datetime.now():%Y-%m-%d %H:%M})", "",
         "> `evaluation/baseline.py` 평가 모드가 만든 파일 (다시 만들면 덮어씀). 지표 정의는 `evaluation/metrics.py` 머리말.",
         "> 실패는 end_reason별. 충돌률 괄호는 95% 윌슨 구간. 수업 통과 기준: 충돌률 < 5%. 충돌 = 앞차 minGap(2.5 m) 안으로"
         " 들어감 (SUMO 기본 판정, 수업 원본 설정).",
         "> 횟수 · 비율 · 평균은 오류(SUMO 오류로 못 끝낸 에피소드)를 뺀 값. 오류 칸은 그 수.",
         "> 주행 지표는 도착한 에피소드만, 평균 ± 표준편차 (중앙값).", "",
         "## 종합", "",
         "| 안전장치 | 횟수 | 도착 | 충돌률 | " + " | ".join(REASON_KO[k] for k in FAIL_REASONS) + " | 개입/에피소드 | 리턴 평균 |",
         "|---|---|---|---|" + "---|" * len(FAIL_REASONS) + "---|---|"]
    for sf in safeties:
        g = [r for r in rows if r["safety"] == sf]
        if not g:
            continue
        ends = Counter(r["end_reason"] for r in g)
        v = [r for r in g if r["end_reason"] != "error"]          # error는 분모 · 평균에서 뺀다 (머리말)
        n = len(v)
        if not n:
            L.append(f"| {sf} | 0 | — | — | " + " | ".join(str(ends.get(k, 0)) for k in FAIL_REASONS) + " | — | — |")
            continue
        L.append(f"| {sf} | {n} | {ends.get('arrived', 0) / n:.1%} | {_rate_ci(ends.get('collided', 0), n)} | "
                 + " | ".join(str(ends.get(k, 0)) for k in FAIL_REASONS)
                 + f" | {stt.mean(float(r['safety_interventions']) for r in v):.1f} | {stt.mean(float(r['return']) for r in v):.2f} |")
    L += ["", "## 조합별 종료 원인", "", "| 안전장치 | 시나리오 | 경로 | 횟수 | 도착 | 종료 원인 |", "|---|---|---|---|---|---|"]
    for sf in safeties:
        for scn, route in COMBOS:
            g = [r for r in rows if r["safety"] == sf and (r["scenario"], r["ego_route"]) == (scn, route)]
            if not g:
                continue
            ends = Counter(r["end_reason"] for r in g)
            nv = len(g) - ends.get("error", 0)
            L.append(f"| {sf} | {scn} | {route} | {nv} | {(ends.get('arrived', 0) / nv) if nv else 0:.0%} | "
                     + ", ".join(f"{REASON_KO.get(k, k)} {v}" for k, v in ends.most_common()) + " |")
    base = []                                 # 기준 지표 (SUMO 운전자 : base_csv, 기본 evaluation/baseline_metrics.csv)
    base_csv = base_csv or OUT_CSV
    if os.path.exists(base_csv):
        with open(base_csv, encoding="utf-8-sig") as f:
            base = list(csv.DictReader(f))
    for sf in safeties:
        L += ["", f"## 주행 지표 — 안전장치 {sf}" + (" (윗줄: 이 정책, '· 기준' 줄: SUMO 운전자 기준 지표)" if base else ""), "",
              "| 지표 | " + " | ".join(f"{s} · {r}" for s, r in COMBOS) + " |", "|---|" + "---|" * len(COMBOS)]
        for key, label, arrived_only in SUMMARY:
            cells, bcells = [], []
            for c in COMBOS:
                ok = lambda r: (r["scenario"], r["ego_route"]) == c and (not arrived_only or r["end_reason"] == "arrived")  # noqa: E731
                cells.append(fmt(num([r for r in rows if r["safety"] == sf and ok(r)], key)))
                bcells.append(fmt(num([r for r in base if ok(r)], key)))
            L.append(f"| {label} | " + " | ".join(cells) + " |")
            if base:
                L.append("| · 기준 | " + " | ".join(bcells) + " |")
    L += ["", f"> 기준 지표는 evaluation/{os.path.basename(base_csv)} 그대로다 (SUMO 운전자 — 이 평가와 출발 조건이 같은 측정)."
          if base else f"> 기준 지표 파일(evaluation/{os.path.basename(base_csv)})이 없어 비교를 생략했다."]
    L += _overspeed_report(rows, safeties)
    os.makedirs(os.path.dirname(out_md), exist_ok=True)
    with open(out_md, "w", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")
    print("\n".join(L))


def depart_options(a):
    """--depart-lane / --depart-speed  /  env.reset options (기본값 best · default면 빈 dict = 옵션 안 줌, 기존 동작)."""
    opts = {}
    dl = str(getattr(a, "depart_lane", "best"))
    if dl == "random":
        opts["depart_lane"] = "random"
    elif dl != "best":                                   # 차로 번호 고정 (오른쪽 끝이 0)
        opts["depart_lane"] = int(dl)
    if getattr(a, "depart_speed", "default") == "limit":
        opts["depart_speed"] = "limit"
    return opts


def _model_name(model):
    """평가 폴더 이름: 모델 폴더 이름, 파일이 model.pt가 아니면 뒤에 _<파일명> (체크포인트끼리 덮어쓰지 않게)."""
    path = os.path.abspath(model)
    run, stem = os.path.basename(os.path.dirname(path)), os.path.splitext(os.path.basename(path))[0]
    return run if stem == "model" else f"{run}_{stem}"


def _job_errors(job, e):
    pol, model, sf, scn, route, eps, *_ = job
    print(f"[평가] {sf} {scn} {route} 에피소드 {list(eps)}: {e!r} → error로 적음", flush=True)
    return [_error_row(scn, route, sf, ep) for ep in eps]


def _run_jobs(jobs, workers):
    """작업을 프로세스 풀로 돌린다 (multiprocessing.Pool은 워커가 죽으면 멈춰 버려서 이것을 쓴다).

    워커 프로세스가 죽으면(libsumo 치명 오류 등) 풀 전체가 깨져 끝나지 않은 작업이 모두 실패한다  /  남은 작업을 새 풀로
    POOL_ROUNDS번까지 다시 돌리고, 그래도 남으면 작업마다 따로 프로세스를 띄워 돌린다 (죽는 작업이 다른 작업을 끌고 가지
    않게). 거기서도 죽는 작업의 에피소드는 error 행으로 적는다."""
    from concurrent.futures import ProcessPoolExecutor, as_completed
    from concurrent.futures.process import BrokenProcessPool
    import multiprocessing
    ctx = multiprocessing.get_context("spawn")
    rows, pending = [], list(jobs)
    for _ in range(POOL_ROUNDS):
        failed = []
        with ProcessPoolExecutor(workers, mp_context=ctx) as ex:
            futs = {ex.submit(_eval_worker, j): j for j in pending}
            for f in as_completed(futs):
                try:
                    rows += f.result()
                except BrokenProcessPool:
                    failed.append(futs[f])
                except Exception as e:          # noqa: BLE001 : 그 밖의 예외: 그 작업만 error로 적고 나머지는 계속
                    rows += _job_errors(futs[f], e)
        if not failed:
            return rows
        print(f"[평가] 워커 프로세스가 죽어 작업 {len(failed)}개를 다시 돌림", flush=True)
        pending = failed
    print(f"[평가] 남은 작업 {len(pending)}개를 하나씩 따로 돌림", flush=True)
    for k in range(0, len(pending), workers):
        batch = pending[k:k + workers]
        exs = [ProcessPoolExecutor(1, mp_context=ctx) for _ in batch]
        try:
            for f, (pol, model, sf, scn, route, eps, *_) in [(ex.submit(_eval_worker, j), j) for ex, j in zip(exs, batch)]:
                try:
                    rows += f.result()
                except BrokenProcessPool as e:
                    rows += _job_errors((pol, model, sf, scn, route, eps), e)
                except Exception as e:          # noqa: BLE001
                    rows += _job_errors((pol, model, sf, scn, route, eps), e)
        finally:
            for ex in exs:
                ex.shutdown()
    return rows


def evaluate(a):
    import numpy as np
    from env import road_builder
    from env.road_config import ROAD, EGO, TRAFFIC
    pol = a.policy
    if a.model and pol not in ("ppo", "bc"):
        import collect_pomdp_data as C
        pol = C.detect_algorithm(a.model)
    safeties = ["off"] if pol == "sumo" else (["off", "on"] if a.safety == "both" else [a.safety])
    name = a.name or (_model_name(a.model) if a.model else pol)
    out_dir = os.path.join(P.EVALUATION, "eval", name)
    sumocfg = road_builder.build(ROAD, EGO, TRAFFIC, P.SUMO_DIR)   # 부모에서 한 번 (워커는 읽기만)
    eps = a.episodes or 100
    # 작업 = 에피소드 하나 (11/16: 워커 프로세스가 죽어도 그 에피소드만 다시 돌리게. 에피소드마다 SUMO를 새로 띄우므로
    # 작업마다 환경 · 정책을 새로 만드는 비용은 작다).
    depart = depart_options(a)
    combos = [c for c in COMBOS if a.route in (None, c[1])]          # --route: 그 경로만
    jobs = [(pol, a.model, sf, scn, route, [ep], sumocfg, a.stochastic, depart)
            for ep in range(a.episode_start, a.episode_start + eps) for sf in safeties for scn, route in combos]
    print(f"[평가] {name}: 정책 {pol}, 안전장치 {'/'.join(safeties)}, {len(combos)}조합 × {eps}회, workers={a.workers} (작업 {len(jobs)}개)"
          + (f", 출발 {depart}" if depart else ""), flush=True)
    t0 = datetime.now()
    rows = _run_jobs(jobs, a.workers)
    rows.sort(key=lambda r: (r["safety"], r["file"], r["episode"]))
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "metrics.csv"), "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=EVAL_COLS, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    shown = None
    if a.model:
        try:
            shown = os.path.relpath(a.model, ROOT)
        except ValueError:                        # 다른 드라이브 (Windows)
            shown = os.path.abspath(a.model)
    title = (pol + (f" `{shown}`" if shown else "") + f", {len(combos)}조합 × {eps}회" + (f" ({a.route})" if a.route else "")
             + (f", 판 번호 {a.episode_start}~{a.episode_start + eps - 1}" if a.episode_start else "")
             + (f", 출발 차로 {a.depart_lane} · 출발 속도 {a.depart_speed}" if depart else ""))
    # 기준 지표: 출발 조건이 같은 SUMO 운전자 측정과 비교한다 (무작위 차로 · 제한속도 × 0.8이면 baseline_metrics_random.csv)
    base_csv = a.base_metrics or (OUT_CSV_RANDOM if (depart.get("depart_lane") == "random" and os.path.exists(OUT_CSV_RANDOM)) else OUT_CSV)
    eval_report(rows, title, os.path.join(out_dir, "report.md"), safeties, base_csv)
    print(f"\n[평가] {len(rows)}회, {(datetime.now() - t0).total_seconds() / 60:.1f}분 → {out_dir}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=None, help="조합마다 횟수 (기본: 기준 지표 30, 평가 100)")
    ap.add_argument("--workers", type=int, default=7)
    ap.add_argument("--skip-collect", action="store_true")
    ap.add_argument("--model", default=None, help="평가할 학습 모델 (training/train.py · training/train_team.py 출력 — 종류 자동 판별)")
    ap.add_argument("--policy", choices=["sumo", "rule", "ppo", "bc"], default=None,
                    help="평가 정책 (rule = environment/road/check_env.py의 관측만 보는 규칙 정책)")
    ap.add_argument("--eval", action="store_true", help="--policy sumo를 평가 경로(데이터 파일 없이)로 돌린다")
    ap.add_argument("--safety", choices=["off", "on", "both"], default="off", help="안전장치 (평가 모드, 기본 끔 — 11/17 컨설팅: 안전장치 불허)")
    ap.add_argument("--stochastic", action="store_true", help="PPO 행동을 샘플링 (기본은 결정적)")
    ap.add_argument("--name", default=None, help="평가 결과 폴더 이름 (evaluation/eval/<이름>)")
    ap.add_argument("--depart-lane", default="best",
                    help="[평가 모드] AV 출발 차로: best = 옵션 안 줌(EGO 설정, 기존 동작) · random = 경로의 출발 차로 범위에서 무작위 · "
                         "숫자 = 그 차로 고정 (오른쪽 끝이 0, 11/19 — 보통 --route와 함께)")
    ap.add_argument("--route", choices=[r for _, r in COMBOS], default=None,
                    help="[평가 모드] 이 경로만 평가 (기본: 5조합 전부, 11/19)")
    ap.add_argument("--episode-start", type=int, default=0,
                    help="[평가 모드] 시작 판 번호 (기본 0). 판 번호가 시드를 정하므로, 모델을 고를 때 쓴 판(0~)과 겹치지 않는 "
                         "새 시드로 확인할 때 쓴다 (예: --episode-start 100 --episodes 200 → 100~299번, 11/19)")
    ap.add_argument("--base-metrics", default=None,
                    help="[평가 모드] 보고서 '· 기준' 줄에 쓸 SUMO 운전자 지표 CSV (기본: 출발 옵션이 random · limit이면 "
                         "evaluation/baseline_metrics_random.csv, 아니면 evaluation/baseline_metrics.csv)")
    ap.add_argument("--depart-speed", choices=["default", "limit"], default="default",
                    help="[평가 모드] AV 출발 속도: default = 옵션 안 줌(EGO 설정) · limit = 출발 차로 제한속도 × 0.8")
    a = ap.parse_args()
    if a.depart_lane not in ("best", "random") and not a.depart_lane.isdigit():
        ap.error("--depart-lane은 best · random · 차로 번호(0, 1, …) 중 하나입니다.")
    if a.model and not os.path.exists(a.model):
        ap.error(f"모델 파일이 없습니다: {a.model}")
    if a.policy in ("ppo", "bc") and not a.model:
        ap.error(f"--policy {a.policy}에는 --model이 필요합니다.")
    if a.model or a.policy == "rule" or a.eval:
        a.policy = a.policy or ("sumo" if not a.model else None)
        return evaluate(a)
    a.episodes = a.episodes or 30
    os.makedirs(DATA, exist_ok=True)
    if not a.skip_collect:
        for scn, route in COMBOS:
            collect(scn, route, a.episodes, a.workers)
    rows = []
    for scn, route in COMBOS:
        p = os.path.join(DATA, f"{scn}_{route}.jsonl.gz")
        if os.path.exists(p):
            rows += metrics.file_metrics(p)
    with open(OUT_CSV, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=metrics.COLS, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    report(rows, None if a.skip_collect else a.episodes)


if __name__ == "__main__":
    main()
