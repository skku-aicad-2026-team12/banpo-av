"""build_net.py가 만든 도로망 자동 검증

사용:
    python -X utf8 environment/road/check_net.py                 # 두 시나리오 모두
    python -X utf8 environment/road/check_net.py --scenario after --no-sim
검사:
  1. 길이: road_spec.SPEC_LENGTHS (확정안) vs 도로망 엣지 길이
  2. 경로: 진입점 × 출구 조합별 주행 가능 여부 (차로 단위 : routing.lane_path)
  3. 규칙: ① 지하차도는 고가 쪽 왼쪽 차로에서만 · 차로 변경 금지 속성 · 신반포로 서행 좌회전 없음 · U턴 없음 · 잠수교(시나리오)
  4. 주행 테스트 (--no-sim 이면 생략): 주행 가능한 경로마다 승용차 1대  /  모두 도착 · 순간이동(teleport) 0
결과 요약은 environment/road/build/<scenario>/check_report.txt 에도 저장한다.
after는 build_net이 잠수교 엣지를 지워 만든 정적 도로망(environment/road/build/after)이다. 환경(학습 · 수집)의 보행화 후는 before 도로망에서 실행 중
잠수교를 막는 방식이라(environment/env/sumo_env.py apply_after_closure) environment/road/check_env.py · check_long.py가 검증한다.
"""
import argparse, os, subprocess, sys, xml.etree.ElementTree as ET
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import road_spec as S
import routing
sh = os.environ.get("SUMO_HOME", r"C:\Program Files (x86)\Eclipse\Sumo")
sys.path.append(os.path.join(sh, "tools"))
import sumolib

HERE = os.path.dirname(os.path.abspath(__file__))

def check(scn, build, run_sim):
    out = os.path.join(build, scn); netf = os.path.join(out, "banpo.net.xml")
    lines, fails = [], 0
    def say(s=""): lines.append(s); print(s)
    def bad(s):
        nonlocal fails; fails += 1; say("  ✗ " + s)
    net, lane_attr = routing.load(netf)
    say(f"===== [{scn}{' — build_net 정적 도로망' if scn == 'after' else ''}] {netf}")
    if scn == "after":   # 환경의 보행화 후는 before 도로망 + 실행 중 통제(environment/env/sumo_env.py apply_after_closure)  /  check_env · check_long이 검증
        say("  (환경이 학습 · 수집에 쓰는 보행화 후가 아님: 잠수교 엣지를 지운 environment/road/build/after — 도로 연결 규칙 확인용)")

    say("1. 길이 (도로망 엣지 + 그 사이 교차로 내부 길이 / 확정안)")
    def chain_len(eids):
        L = sum(net.getEdge(e).getLength() for e in eids)
        for a, b in zip(eids, eids[1:]):
            vias = [c.getViaLaneID() for c in net.getEdge(a).getConnections(net.getEdge(b)) if c.getViaLaneID()]
            if vias: L += net.getLane(vias[0]).getLength()
        return L
    for name, (eids, ref) in S.SPEC_LENGTHS.items():
        L = chain_len(eids)
        ok = abs(L - ref) <= max(0.1 * ref, 6)
        say(f"  {'✓' if ok else '✗'} {name}: {L:.0f} / {ref}")
        if not ok: fails += 1

    # 차로 단위 도달 탐색: 연결 + 같은 엣지 안의 차로 변경(금지 속성·버스전용 반영). 승용차 기준.

    say("2. 경로 (진입점 → 출구) ○ = 주행 가능 · △ = 도로는 이어지나 차로 규칙상 불가 · · = 연결 없음")
    entries = [e for e in S.ENTRIES if net.hasEdge(e)]
    exits = [e for e in S.EXITS if net.hasEdge(e)]
    routes, blocked = [], []
    say("  " + " " * 14 + " ".join(f"{x[:9]:>9}" for x in exits))
    for a in entries:
        row = []
        for b in exits:
            p, _ = net.getShortestPath(net.getEdge(a), net.getEdge(b), vClass="passenger")
            lp = routing.lane_path(net, lane_attr, a, b) if p else None
            row.append("○" if lp else ("△" if p else "·"))
            if lp: routes.append((a, b, lp))
            elif p: blocked.append((a, b, [e.getID() for e in p]))
        say(f"  {a[:14]:>14} " + " ".join(f"{c:>9}" for c in row))
        if "○" not in row: bad(f"{a}: 갈 수 있는 출구가 없음")
    for b in exits:
        if not any(r[1] == b for r in routes): bad(f"{b}: 들어올 수 있는 경로가 없음")
    say(f"  △ {len(blocked)}개는 측도·신반포로에서 ① 지하차도(→ 회전교차로·루프·주차장·잠수교)로 가는 조합이어야 한다")
    wrong = [(a, b) for a, b, p in blocked if "und_n" not in p]       # 도로상 경로가 ① 지하차도를 지나는 조합만 막혀야 정상
    if wrong: bad(f"예상 밖의 차로 규칙 차단: {wrong}")

    say("3. 규칙")
    def out_lanes(eid, li):
        return [(c.getTo().getID(), c.getToLane().getIndex()) for c in net.getEdge(eid).getLanes()[li].getOutgoing()]
    und_src = [(l.getEdge().getID(), l.getIndex()) for l in net.getEdge("und_n").getLanes()[0].getIncoming() if not l.getID().startswith(":")]
    if und_src == [("m1", 4)]: say("  ✓ 지하차도(①)로 들어가는 차로는 m1의 ① 전용 차로(4번)뿐")
    else: bad(f"지하차도 진입 차로가 예상과 다름: {und_src}")
    m1_src = sorted(set((l.getEdge().getID(), l.getIndex()) for l in net.getEdge("m1").getLanes()[4].getIncoming() if not l.getID().startswith(":")))
    if m1_src == [("goga_nb", 1)]: say("  ✓ ① 전용 차로는 고가 쪽 왼쪽 차로에서만 이어짐 (측도 차로와 연결 없음)")
    else: bad(f"① 전용 차로의 앞 차로가 예상과 다름: {m1_src}")
    cl = lane_attr.get("m1_3", {}).get("changeLeft"); cr = lane_attr.get("m1_4", {}).get("changeRight")
    if cl and "passenger" not in cl and cr and "passenger" not in cr:
        say(f"  ✓ 차로 변경 금지: m1_3 → 왼쪽 '{cl}' 만 · m1_4 → 오른쪽 '{cr}' 만")
    else: bad(f"차로 변경 금지 속성 누락: m1_3 changeLeft={cl}, m1_4 changeRight={cr}")
    # 차량 차로 0~2 (보도가 있으면 도로망 차로 번호는 1씩 밀린다 : road_spec.net_lane)
    a7 = {t for li in range(3) for t, _ in out_lanes("sbr_a7b", S.net_lane("sbr_a7b", li))}
    if a7 == {"sbr_e1a"}: say("  ✓ 신반포로 서행(A7)은 직진만 (좌회전·우회전은 우회전 도로)")
    else: bad(f"신반포로 서행 연결이 예상과 다름: {a7}")
    NBUS = sum(1 for e in S.EDGES for la in e["lane_attrs"].values() if la.get("allow") == "bus")
    bus = [lid for lid, a in lane_attr.items() if a.get("allow") == "bus"]
    say(f"  ✓ 버스전용 차로 {len(bus)}개: {', '.join(sorted(bus))}" if len(bus) == NBUS else f"  ✗ 버스전용 차로 수 {len(bus)} (예상 {NBUS})")
    if len(bus) != NBUS: fails += 1
    uturn = [(c.getFrom().getID(), c.getTo().getID()) for e in net.getEdges(withInternal=False) for to in e.getOutgoing() for c in e.getOutgoing()[to]
             if c.getFrom().getFromNode() == c.getTo().getToNode() and c.getFrom().getToNode() == c.getTo().getFromNode()]
    if not uturn: say("  ✓ U턴 연결 없음")
    else: bad(f"U턴 연결 있음: {uturn}")
    has_jam = net.hasEdge("jam_n")
    if (scn == "before") == has_jam: say(f"  ✓ 잠수교 {'열림' if has_jam else '닫힘'} ({scn})")
    else: bad("잠수교 시나리오가 예상과 다름")
    tls = [t.getID() for t in net.getTrafficLights()]
    say(f"  · 신호 교차로: {tls if tls else '없음'}")

    if run_sim:
        say("4. 주행 테스트 (경로마다 승용차 1대)")
        rou = os.path.join(out, "check_routes.rou.xml")
        with open(rou, "w", encoding="utf-8") as f:
            f.write('<routes>\n    <vType id="car" vClass="passenger"/>\n')
            for i, (a, b, eds) in enumerate(routes):
                f.write(f'    <vehicle id="{a}__{b}" type="car" depart="{i*4}" departLane="best" departSpeed="max">\n'
                        f'        <route edges="{" ".join(eds)}"/>\n    </vehicle>\n')
            f.write("</routes>\n")
        trip = os.path.join(out, "check_tripinfo.xml")
        exe = os.path.join(sh, "bin", "sumo.exe") if os.path.exists(os.path.join(sh, "bin", "sumo.exe")) else "sumo"
        r = subprocess.run([exe, "-n", netf, "-r", rou, "--tripinfo-output", trip, "--no-step-log", "true",
                            "--duration-log.statistics", "true", "--time-to-teleport", "120", "--end", "3000"],
                           capture_output=True, text=True, encoding="utf-8", errors="replace")
        arrived = len(ET.parse(trip).getroot().findall("tripinfo")) if os.path.exists(trip) else 0
        tele = 0
        for l in (r.stdout + r.stderr).splitlines():
            if l.strip().startswith("Teleports:"):
                tele = int(l.split(":")[1].split()[0])
        ok = arrived == len(routes) and tele == 0 and r.returncode == 0
        say(f"  {'✓' if ok else '✗'} 경로 {len(routes)}개 중 {arrived}대 도착, 순간이동 {tele}건")
        if not ok:
            fails += 1
            for l in (r.stdout + r.stderr).splitlines():
                if "Warning" in l or "Error" in l: say("    " + l[:160])
    say(f"결과: {'통과' if fails == 0 else f'실패 {fails}건'}")
    with open(os.path.join(out, "check_report.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    return fails

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", choices=S.SCENARIOS)
    ap.add_argument("--no-sim", action="store_true")
    ap.add_argument("--build", default=os.path.join(HERE, "build"))
    a = ap.parse_args()
    fails = sum(check(s, a.build, not a.no_sim) for s in ([a.scenario] if a.scenario else S.SCENARIOS))
    sys.exit(1 if fails else 0)

if __name__ == "__main__":
    main()
