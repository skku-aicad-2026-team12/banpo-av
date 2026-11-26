"""road_spec.py  /  SUMO 입력 XML(.nod/.edg/.con)  /  netconvert  /  .net.xml

사용:
    python -X utf8 environment/road/build_net.py                 # 두 시나리오(before, after) 모두, 신호 5곳(C-ITS 실측) 포함
    python -X utf8 environment/road/build_net.py --scenario after
결과: environment/road/build/<scenario>/banpo.{nod,edg,con,tll}.xml, banpo.net.xml, netconvert.log
(수업 환경은 environment/env/road_builder.py가 write_xml · netconvert · apply_signal을 그대로 불러 before 도로망 하나를 만든다)
"""
import argparse, os, subprocess, sys
from xml.sax.saxutils import escape
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import road_spec as S

HERE = os.path.dirname(os.path.abspath(__file__))

def shape_str(pts):
    return " ".join(f"{p[0]:.2f},{p[1]:.2f}" + (f",{p[2]:.2f}" if len(p) > 2 else "") for p in pts)

def densify(pts, step=10.0):
    """긴 직선 구간에 점을 끼워 넣는다 (층 높이 z가 구간 중간에서 바뀔 수 있게)."""
    out = [pts[0]]
    for a, b in zip(pts, pts[1:]):
        d = ((b[0] - a[0]) ** 2 + (b[1] - a[1]) ** 2) ** 0.5
        n = max(1, int(d // step))
        for i in range(1, n + 1):
            out.append((a[0] + (b[0] - a[0]) * i / n, a[1] + (b[1] - a[1]) * i / n))
    return out

def shape_3d(e):
    """엣지 형상에 z를 붙인다. 높이가 0인 엣지는 원래 2D 형상 그대로."""
    pts = densify(e["shape"])
    zs = [S.z_at(e["id"], x, y) for x, y in pts]
    zs[0] = S.Z_NODES.get(e["frm"], 0.0)      # 양 끝은 노드 높이와 맞춘다
    zs[-1] = S.Z_NODES.get(e["to"], 0.0)
    if not any(abs(z) > 1e-6 for z in zs):
        return e["shape"]
    return [(x, y, z) for (x, y), z in zip(pts, zs)]

def edges_for(scn):
    return [e for e in S.EDGES if e["scenarios"] is None or scn in e["scenarios"]]

def write_xml(outdir, scn):
    os.makedirs(outdir, exist_ok=True)
    edges = edges_for(scn)
    used = {e["frm"] for e in edges} | {e["to"] for e in edges}
    with open(os.path.join(outdir, "banpo.nod.xml"), "w", encoding="utf-8") as f:
        f.write('<?xml version="1.0" encoding="UTF-8"?>\n<nodes>\n')
        for nid, (x, y, typ) in S.NODES.items():
            if nid not in used: continue
            z = S.Z_NODES.get(nid, 0.0)
            zs = f' z="{z:.2f}"' if z else ""
            f.write(f'    <node id="{nid}" x="{x:.2f}" y="{y:.2f}"{zs} type="{typ}"/>\n')
        f.write("</nodes>\n")
    with open(os.path.join(outdir, "banpo.edg.xml"), "w", encoding="utf-8") as f:
        f.write('<?xml version="1.0" encoding="UTF-8"?>\n<edges>\n')
        for e in edges:
            attrs = (f'id="{e["id"]}" from="{e["frm"]}" to="{e["to"]}" numLanes="{e["lanes"]}" speed="{e["speed"]}" '
                     f'spreadType="{e["spread"]}" shape="{shape_str(shape_3d(e))}" name="{escape(e["name"])}"')
            if e.get("priority") is not None:
                attrs += f' priority="{e["priority"]}"'
            if e["id"] in S.SIDEWALKS:                      # 보도 (0번 차로로 붙고 차량 차로 번호가 1씩 밀린다 : 연결은 아래에서 맞춤)
                attrs += f' sidewalkWidth="{S.SIDEWALK_WIDTH}"'
            if e["lane_attrs"]:
                f.write(f"    <edge {attrs}>\n")
                for idx, la in sorted(e["lane_attrs"].items()):
                    extra = " ".join(f'{k}="{v}"' for k, v in la.items())
                    f.write(f'        <lane index="{idx}" {extra}/>\n')
                f.write("    </edge>\n")
            else:
                f.write(f"    <edge {attrs}/>\n")
        f.write(f'    <roundabout nodes="{" ".join(S.ROUNDABOUT["nodes"])}" edges="{" ".join(S.ROUNDABOUT["edges"])}"/>\n')
        f.write("</edges>\n")
    ids = {e["id"] for e in edges}
    conns = [c for c in S.CONNECTIONS if c[0] in ids and c[2] in ids]
    with open(os.path.join(outdir, "banpo.con.xml"), "w", encoding="utf-8") as f:
        f.write('<?xml version="1.0" encoding="UTF-8"?>\n<connections>\n')
        # changeLeft/changeRight: 교차로 안(내부 차로)에서 차로 변경 금지. 신반포사거리처럼 내부 차로가 길면(59 m) SUMO는
        # 나란한 내부 차로 사이 차로 변경을 허용하는데, 출구가 막혀 사거리 안에 갇힌 차가 옆 내부 차로로 옮기는 순간
        # 그 차를 기다리던 교차 방향 차가 출발해 충돌했다 (11/13 검증, §1-6의 :ix_5 충돌). 실제 차도 교차로 안에서는 차로를 바꾸지 않는다.
        no_lc = 'changeLeft="emergency authority" changeRight="emergency authority"'
        for fr, fl, to, tl in conns:
            f.write(f'    <connection from="{fr}" to="{to}" fromLane="{S.net_lane(fr, fl)}" toLane="{S.net_lane(to, tl)}" {no_lc}/>\n')
        # 신호 횡단보도 (road_spec.SIGNALS의 crossings). 신호 노드가 이 시나리오 도로망에 있고 건너는 엣지가 모두 있을 때만
        for tlid, sp in S.SIGNALS.items():
            if tlid not in used:
                continue
            for _name, cedges, _g0, _gl in sp["crossings"]:
                if all(c in ids for c in cedges):
                    f.write(f'    <crossing node="{tlid}" edges="{" ".join(cedges)}" priority="true"/>\n')
        f.write("</connections>\n")
    return len(edges), len(conns)

def netconvert(outdir):
    exe = "netconvert"
    sh = os.environ.get("SUMO_HOME", r"C:\Program Files (x86)\Eclipse\Sumo")   # routing · check_net과 같은 기본 경로
    if os.path.exists(os.path.join(sh, "bin", "netconvert.exe")):
        exe = os.path.join(sh, "bin", "netconvert.exe")
    cmd = [exe,
           "--node-files", os.path.join(outdir, "banpo.nod.xml"),
           "--edge-files", os.path.join(outdir, "banpo.edg.xml"),
           "--connection-files", os.path.join(outdir, "banpo.con.xml"),
           "--output-file", os.path.join(outdir, "banpo.net.xml"),
           "--no-turnarounds", "true",          # U턴 전부 금지
           "--junctions.join", "false",         # 겹치는 도로(고가·지하)를 교차로로 합치지 않음
           "--roundabouts.guess", "false",      # 회전교차로는 spec에서 지정
           "--tls.guess", "false",
           "--offset.disable-normalization", "true",   # spec 좌표 그대로
           "--default.lanewidth", str(S.W),
           "--junctions.corner-detail", "5",
           # 차도는 보행 금지 : 보행자는 보도(sidewalkWidth) · 보행자 전용 길 · 횡단보도로만. 이 옵션이 없으면 모든 차로가
           # 보행 가능이라 netconvert가 교차로마다 차도와 겹치는 보행 구역(walkingarea)을 만들어(52개) 보행자가 차도로 다닐 수
           # 있었고, 화면에서도 회전교차로 등에 덩어리 모양이 생겼다 (11/13). 보행화 후 잠수교는 sumo_env가 실행 중에 보행 허용.
           "--default.disallow", "pedestrian",
           "--lefthand", "false"]
    r = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
    with open(os.path.join(outdir, "netconvert.log"), "w", encoding="utf-8") as f:
        f.write(" ".join(cmd) + "\n\n" + r.stdout + "\n" + r.stderr)
    if r.returncode == 0:
        rc2, err2 = apply_signal(outdir, exe)
        if rc2 != 0:
            return rc2, err2
    return r.returncode, r.stderr


def apply_signal(outdir, exe):
    """road_spec.SIGNALS(실측 신호: 이동류 · 횡단보도별 녹색 시각표)를 신호 프로그램으로 바꿔 넣는다.
    netconvert가 정한 연결 번호(link index)를 읽어, 시각표가 바뀌는 시각마다 끊은 현시별 신호 문자열을 만들고
    도로망을 한 번 더 변환한다. 도로망에 없는 신호(보행화 후 도로망의 잠수교 신호 등)는 건너뛴다."""
    import xml.etree.ElementTree as ET
    netf = os.path.join(outdir, "banpo.net.xml")
    root = ET.parse(netf).getroot()
    crossing_edges = {e.get("id"): set((e.get("crossingEdges") or "").split())
                      for e in root.iter("edge") if e.get("function") == "crossing"}
    lines, problems = [], []
    for tl, sp in S.SIGNALS.items():
        links = {}
        for c in root.iter("connection"):
            if c.get("tl") != tl:
                continue
            idx = int(c.get("linkIndex"))
            cr = c.get("to") if c.get("to") in crossing_edges else (c.get("from") if c.get("from") in crossing_edges else None)
            links[idx] = ("crossing", frozenset(crossing_edges[cr])) if cr else ("veh", c.get("from"), c.get("to"), int(c.get("fromLane")))
        if not links:
            continue
        n = max(links) + 1
        owner = {}
        for i, lk in links.items():
            if lk[0] == "crossing":
                hit = [name for name, edges, _g0, _gl in sp["crossings"] if frozenset(edges) == lk[1]]
            else:
                _, fr, to, fl = lk
                hit = [name for name, _g0, _gl, _yl, mv, _y in sp["groups"]
                       if any(fr == a and to == b and (l is None or S.net_lane(a, l) == fl) for a, b, l in mv)]
            if len(hit) != 1:
                problems.append(f"{tl} 연결 {i} {lk}: 신호 그룹 {hit or '없음'}")
            else:
                owner[i] = hit[0]
        timeline = S.signal_timeline(tl)
        lines.append(f'    <tlLogic id="{tl}" type="static" programID="0" offset="{sp.get("offset", 0)}">')
        for dur, st in timeline:
            state = "".join(st.get(owner.get(i), "r") for i in range(n))
            lines.append(f'        <phase duration="{dur}" state="{state}"/>')
        lines.append("    </tlLogic>")
    if problems:
        return 1, "신호 연결 매핑 실패:\n  " + "\n  ".join(problems)
    if not lines:                                  # 이 시나리오 도로망에 신호 노드가 없음
        return 0, ""
    tll = os.path.join(outdir, "banpo.tll.xml")
    with open(tll, "w", encoding="utf-8") as f:
        f.write("<tlLogics>\n" + "\n".join(lines) + "\n</tlLogics>\n")
    cmd = [exe, "--sumo-net-file", netf, "--tllogic-files", tll, "--output-file", netf,
           "--offset.disable-normalization", "true"]
    r = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
    with open(os.path.join(outdir, "netconvert.log"), "a", encoding="utf-8") as f:
        f.write("\n\n# 신호 프로그램 적용 (road_spec.SIGNALS)\n" + " ".join(cmd) + "\n\n" + r.stdout + "\n" + r.stderr)
    return r.returncode, r.stderr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", choices=S.SCENARIOS)
    ap.add_argument("--out", default=os.path.join(HERE, "build"))
    a = ap.parse_args()
    ok = True
    for scn in ([a.scenario] if a.scenario else S.SCENARIOS):
        out = os.path.join(a.out, scn)
        ne, nc = write_xml(out, scn)
        rc, err = netconvert(out)
        warns = [l for l in err.splitlines() if l.strip()]
        print(f"[{scn}] 엣지 {ne}, 연결 {nc} → netconvert {'성공' if rc == 0 else '실패'} (경고 {len(warns)}줄, {os.path.join(out, 'netconvert.log')})")
        for l in warns[:15]: print("   ", l)
        ok &= rc == 0
    sys.exit(0 if ok else 1)

if __name__ == "__main__":
    main()
