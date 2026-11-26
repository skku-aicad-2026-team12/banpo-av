"""장시간 안정성 점검 : 배경 교통만 (AV 없음) 시나리오 × 시드마다 몇 시간씩 돌려, 짧은 검증(check_env 30분)에서
안 보이는 문제를 찾는다: 서서히 자라는 대기열, 교착(gridlock), 순간이동, 충돌, 진입점 투입 부족, 측정 링크 재현.

사용 (프로젝트 루트에서, environment/env/sumo/banpo/ 파일이 이미 있어야 함 : check_env.py나 environment/view_road.py가 만든다):
    python -X utf8 environment/road/check_long.py                       # 두 시나리오 × 시드 3개 × 240분, 병렬 (약 6~8분)
    python -X utf8 environment/road/check_long.py --minutes 120 --seeds 11 22

- SUMO 옵션은 environment/env/sumo_env.py와 같다 (스텝 0.5 s, 충돌 시 제거, 교차로 안 충돌 검사). 보행화 후는 잠수교 차로를 막는다.
- 화면 없이(libsumo) 최대 속도로 돌려 시뮬레이션 4시간이 실제 수 분이다.
- 결과: environment/road/build/check_long_report.txt (콘솔에도 출력). 종료 코드는 충돌 · 순간이동이 있을 때만 1.
  투입 부족 · 투입 대기 누적 · 자라는 대기열은 "주의(△)"로 따로 모아 보여 준다 (실패는 아니지만 원인을 확인 · 기록).
- 11/12 이 점검으로 찾은 것: 회전교차로 고리 교착(2시간 뒤부터), 김포 방향 투입 −15~25%, 신반포로 동행 대기열 증가,
  측도 남행 포켓 충돌  /  모두 고침 (`docs/프로젝트_기록.md` §11 #26~#29).
"""
import argparse, collections, os, statistics as st, sys
import xml.etree.ElementTree as ET
from multiprocessing import Pool

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))   # 저장소 루트
import repo_paths as P  # noqa: E402
HERE = P.ROAD_DIR
ROOT = P.ROOT
D = os.path.join(P.SUMO_DIR, "banpo")
from env.sumo_env import after_closure_targets, apply_after_closure  # noqa: E402  (환경과 같은 보행화 후 통제 · SUMO_HOME 설정)
from sumolib import checkBinary  # noqa: E402

DT, WIN = 0.5, 600          # 스텝 (s), 집계 창 (s)
LINKS = {"bridge": "반포대교 북행", "jam_n": "잠수교 북행", "wb_a": "올림픽대로 김포 방향 진입", "wb_d": "올림픽대로 김포 방향 끝",
         "eb_a": "올림픽대로 하남 방향 진입", "eb_b": "올림픽대로 하남 방향 끝", "bp_sb": "반포대로 남행"}


def entry_of(vid):
    """배경차 f_<진입>__<출구>…, 버스 bus_<노선>__<진입>__<출구>  /  진입 도로."""
    p = vid.split("__")
    if vid.startswith("f_") and len(p) > 1:
        return p[0][2:]
    if vid.startswith("bus_") and len(p) > 2:
        return p[1]
    return "?"


def demand(scn):
    out = collections.Counter()
    for fl in ET.parse(os.path.join(D, f"banpo.{scn}.rou.xml")).getroot().iter("flow"):
        if fl.get("probability"):
            vph = float(fl.get("probability")) * 3600
        elif fl.get("period"):
            vph = 3600 / float(fl.get("period"))
        else:
            vph = float(fl.get("vehsPerHour", 0))
        out[entry_of(fl.get("id"))] += vph
    return out


def run(job):
    scn, seed, minutes = job
    try:
        import libsumo as traci
    except ImportError:
        import traci
    traci.start([checkBinary("sumo"), "-c", os.path.join(D, "banpo.sumocfg"), "--route-files", os.path.join(D, f"banpo.{scn}.rou.xml"),
                 "--step-length", str(DT), "--collision.action", "remove", "--collision.check-junctions", "true",
                 "--no-warnings", "true", "--no-step-log", "true", "--seed", str(seed)])
    if scn == "after":                          # 환경(environment/env/sumo_env.py reset)과 같은 함수로 잠수교 차로 통제 · 신호 끔
        apply_after_closure(traci, *after_closure_targets())
    edges = [e for e in traci.edge.getIDList() if not e.startswith(":")]
    nwin = max(1, int(minutes * 60 // WIN))     # 10분 창 단위로만 돈다 (남는 시간은 버림 : 마지막 창이 부풀지 않게)
    spw = int(WIN / DT)
    halt = {e: [0.0] * nwin for e in edges}
    maxwait = {e: [0.0] * nwin for e in edges}
    passed = {e: [0] * nwin for e in edges}
    prev = {e: set() for e in edges}
    inserted = collections.defaultdict(lambda: [0] * nwin)
    pending_end = collections.Counter()
    collisions, teleports, last_road = [], [], {}
    for s in range(nwin * spw):
        traci.simulationStep()
        t = (s + 1) * DT
        w = min(s // spw, nwin - 1)
        for v in traci.simulation.getDepartedIDList():
            inserted[entry_of(v)][w] += 1
        for c in traci.simulation.getCollisions():
            collisions.append((round(t / 60), c.lane, c.type, c.collider, c.victim))
        for v in traci.simulation.getStartingTeleportIDList():
            teleports.append((round(t / 60), last_road.get(v, "?")))
        for e in edges:
            halt[e][w] += traci.edge.getLastStepHaltingNumber(e)
            ids = set(traci.edge.getLastStepVehicleIDs(e))
            passed[e][w] += len(prev[e] - ids)
            prev[e] = ids
        if s % int(60 / DT) == 0:
            for v in traci.vehicle.getIDList():
                r = traci.vehicle.getRoadID(v)
                last_road[v] = r
                if not r.startswith(":"):
                    maxwait[r][w] = max(maxwait[r][w], traci.vehicle.getWaitingTime(v))
    for v in traci.simulation.getPendingVehicles():
        pending_end[entry_of(v)] += 1
    traci.close()
    return dict(scn=scn, seed=seed, nwin=nwin,
                inserted={k: [x * 3600 / WIN for x in v] for k, v in inserted.items()},
                pending_end=dict(pending_end),
                halt={e: [x / spw for x in v] for e, v in halt.items()},
                maxwait=maxwait, passed={e: [x * 3600 / WIN for x in v] for e, v in passed.items()},
                collisions=collisions, teleports=teleports)


def report(results, minutes):
    """판정: 충돌 · 순간이동이 있으면 실패(✗). 투입 부족(수요의 95% 미만) · 투입 대기 누적(끝날 때 대기 >
    수요 × 시간의 2%) · 자라는 대기열은 주의(△) : 실패는 아니지만 원인을 확인하고 기록한다."""
    L = []
    say = L.append
    ok = True
    warns = []
    for scn in ("before", "after"):
        R = [r for r in results if r["scn"] == scn]
        if not R:
            continue
        dem = demand(scn)
        say(f"===== [{scn}] 시드 {[r['seed'] for r in R]} × {minutes:.0f}분 (값은 처음 10분을 뺀 평균)")
        nc = sum(len(r["collisions"]) for r in R)
        nt = sum(len(r["teleports"]) for r in R)
        ok &= nc == 0 and nt == 0
        say(f"  {'✓' if nc == 0 else '✗'} 충돌 {nc}건" + "".join(f"\n      seed {r['seed']}: {c}" for r in R for c in r["collisions"][:3]))
        say(f"  {'✓' if nt == 0 else '✗'} 순간이동 {nt}건" + "".join(
            f"\n      seed {r['seed']}: 첫 {r['teleports'][0][0]}분, 도로 {collections.Counter(x[1] for x in r['teleports']).most_common(5)}"
            for r in R if r["teleports"]))
        say("  진입점: 수요 → 실제 투입 (대/시), 끝날 때 투입 대기")
        for k in sorted((k for k in dem if k != "?"), key=lambda k: -dem[k]):   # 버스는 entry_of가 진입 도로로 합쳐 둠
            ins = st.mean(st.mean(r["inserted"].get(k, [0])[1:] or [0]) for r in R)
            pend = [r["pending_end"].get(k, 0) for r in R]
            hours = R[0]["nwin"] * WIN / 3600
            short = ins < 0.95 * dem[k]
            backlog = st.mean(pend) > 0.02 * dem[k] * hours
            mark = "△" if short or backlog else "✓"
            if mark == "△":
                warns.append(f"[{scn}] {k} 투입 " + ("부족" if short else "대기 누적") + f" ({ins / dem[k] - 1:+.0%}, 끝날 때 대기 평균 {st.mean(pend):.0f}대)")
            say(f"    {mark} {k:9s} {dem[k]:6.0f} → {ins:6.0f} ({ins / dem[k] - 1:+.0%})  대기 {pend}")
        say("  측정 링크 통과량 (대/시)")
        for e, name in LINKS.items():
            if e in R[0]["passed"]:
                v = [st.mean(r["passed"][e][1:] or [0]) for r in R]
                say(f"    {name:18s} {st.mean(v):6.0f}  (시드별 {[round(x) for x in v]})")
        say("  대기열이 자라는 도로 (정지 대수 처음 30분 → 마지막 60분, 최장 대기 > 180 s 또는 +5대 이상)")
        flagged = 0
        for e in R[0]["halt"]:
            early = st.mean(st.mean(r["halt"][e][1:4] or [0]) for r in R)
            late = st.mean(st.mean(r["halt"][e][-6:]) for r in R)
            mw = max(max(r["maxwait"][e]) for r in R)
            if late > early + 5 or mw > 180:
                flagged += 1
                say(f"    △ {e:12s} {early:5.1f} → {late:5.1f}대, 최장 대기 {mw:.0f} s")
                warns.append(f"[{scn}] {e} 대기열 {early:.1f} → {late:.1f}대 (최장 대기 {mw:.0f} s)")
        if not flagged:
            say("    ✓ 없음")
    say("결과: " + ("통과 (충돌 · 순간이동 0)" if ok else "실패 (충돌 또는 순간이동)")
        + (f", 주의 {len(warns)}건 — 원인을 확인하고 기록할 것:" + "".join(f"\n  △ {w}" for w in warns) if warns else ", 주의 없음"))
    return "\n".join(L), ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--minutes", type=float, default=240)
    ap.add_argument("--seeds", type=int, nargs="+", default=[11, 22, 33])
    ap.add_argument("--scenario", choices=["before", "after"])
    a = ap.parse_args()
    scns = [a.scenario] if a.scenario else ["before", "after"]
    jobs = [(s, sd, a.minutes) for s in scns for sd in a.seeds]
    print(f"{len(jobs)}개 실행 (시뮬레이션 {a.minutes:.0f}분씩, 병렬)…", flush=True)
    with Pool(min(len(jobs), os.cpu_count() or 4)) as p:
        results = p.map(run, jobs)
    text, ok = report(results, a.minutes)
    print(text)
    os.makedirs(os.path.join(HERE, "build"), exist_ok=True)
    with open(os.path.join(HERE, "build", "check_long_report.txt"), "w", encoding="utf-8") as f:
        f.write(text + "\n")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
