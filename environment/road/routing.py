"""차로 단위 경로 탐색 (check_net.py · calibrate.py · environment/env/road_builder.py 공용)

엣지가 이어져 있어도 차로 규칙(연결 · 차로 변경 금지 · 버스전용) 때문에 못 가는 경로가 있다.
예: 측도에서 온 차는 ① 지하차도로 못 들어간다. 그래서 경로를 차로 단위로 탐색한다 (승용차 기준).
"""
import os, sys, xml.etree.ElementTree as ET
sh = os.environ.get("SUMO_HOME", r"C:\Program Files (x86)\Eclipse\Sumo")
sys.path.append(os.path.join(sh, "tools"))
import sumolib


def load(netf):
    """(sumolib net, 차로별 속성 dict) — 속성은 changeLeft/changeRight 확인용 (통행 허용은 sumolib lane.allows)."""
    net = sumolib.net.readNet(netf, withInternal=True)
    root = ET.parse(netf).getroot()
    lane_attr = {l.get("id"): l.attrib for e in root.iter("edge") if not e.get("id", "").startswith(":")
                 for l in e.iter("lane")}
    return net, lane_attr


def _can_change(lane_attr, lane, side):
    a = lane_attr.get(lane.getID(), {}).get("changeLeft" if side > 0 else "changeRight")
    return a is None or "passenger" in a.split()


def _next_lanes(net, lane_attr, l):
    nxt = [c.getToLane() for c in l.getOutgoing()]
    lanes = l.getEdge().getLanes()
    for side in (1, -1):
        j = l.getIndex() + side
        if 0 <= j < len(lanes) and _can_change(lane_attr, l, side):
            nxt.append(lanes[j])
    return [m for m in nxt if m.allows("passenger") and not m.getEdge().getID().startswith(":")]


def lane_path(net, lane_attr, a, b):
    """엣지 a → b 로 승용차가 실제로 갈 수 있는 엣지 순서 (없으면 None)."""
    start = [l for l in net.getEdge(a).getLanes() if l.allows("passenger")]
    par = {l.getID(): None for l in start}
    q = list(start)
    while q:
        l = q.pop(0)
        if l.getEdge().getID() == b:
            seq, k = [], l.getID()
            while k:
                seq.append(net.getLane(k).getEdge().getID()); k = par[k]
            out = []
            for e in reversed(seq):
                if not out or out[-1] != e:
                    out.append(e)
            return out
        for m in _next_lanes(net, lane_attr, l):
            if m.getID() not in par:
                par[m.getID()] = l.getID(); q.append(m)
    return None


def route_feasible(net, lane_attr, edges):
    """주어진 엣지 순서를 그대로 따라 끝까지 갈 수 있는 차로 경로가 있는지."""
    if not all(net.hasEdge(e) for e in edges):
        return False
    idx = {e: i for i, e in enumerate(edges)}
    start = [l for l in net.getEdge(edges[0]).getLanes() if l.allows("passenger")]
    seen = {l.getID() for l in start}
    q = list(start)
    while q:
        l = q.pop(0)
        i = idx[l.getEdge().getID()]
        if i == len(edges) - 1:
            return True
        for m in _next_lanes(net, lane_attr, l):
            j = idx.get(m.getEdge().getID())
            if j in (i, i + 1) and m.getID() not in seen:
                seen.add(m.getID()); q.append(m)
    return False


def od_paths(net, lane_attr, entry, exits, closed=()):
    """진입점 → 출구 비율 표를 경로로 바꾼다 → [(출구 키, 비율, 엣지 순서)].

    출구 키는 "출구" 또는 "출구|경유엣지" (예: "bridge|side_a4" = 측도를 거쳐 반포대교).
    경유를 안 주면 차로 단위 최단 경로. 막힌 엣지(closed)를 지나거나 차로 규칙상 못 가는 조합은 뺀다."""
    out = []
    closed = set(closed)
    for key, w in exits.items():
        ex, _, via = key.partition("|")
        if w <= 0 or ex in closed or not net.hasEdge(ex):
            continue
        if via:
            p1 = lane_path(net, lane_attr, entry, via) if net.hasEdge(via) else None
            p2 = lane_path(net, lane_attr, via, ex) if p1 else None
            p = p1 + p2[1:] if p1 and p2 else None
            if p and not route_feasible(net, lane_attr, p):
                p = None
        else:
            p = lane_path(net, lane_attr, entry, ex)
        if p and not closed & set(p):
            out.append((key, w, p))
    return out
