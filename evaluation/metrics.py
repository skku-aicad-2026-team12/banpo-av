"""수집 데이터(JSONL)에서 에피소드별 주행 지표를 계산한다 : 수집된 칼럼만 사용 (지표용 새 칼럼을 만들지 않음).
기본 칼럼(state · event 등)과 12팀 추가 칼럼(scenario · ego_route · signal · end_reason)을 읽는다.
signal 칼럼이 없거나 옛 형식({phase, remaining_s} : 11/15 전 수집분)인 데이터는 stop_red_s를 빈 값으로 둔다.

SUMO 기본 운전자 기준값(evaluation/baseline.py)과, 나중에 학습한 AV로 수집한 데이터에 똑같이 쓴다.

사용:
    python -X utf8 evaluation/metrics.py data/xxx.jsonl.gz [더 많은 파일 ...] [--csv out.csv]

지표 (에피소드 하나 = AV 한 번 주행):
    scenario, ego_route, end_reason        추가 칼럼 그대로
    travel_s                               주행 시간 = 스텝 수 × 0.5 s (도착한 에피소드에서 의미)
    stops                                  정지 횟수 (속도 ≥ 1 m/s로 달리다가 0.1 m/s 미만이 된 횟수)
    stop_s                                 정지 시간 합 (속도 < 0.1 m/s)
    stop_red_s                             그중 빨간불 · 황색 신호 앞에 선 시간 (신호 5곳 모두)
                                           (signal.state가 r · y + 그 신호 정지선까지 경로 거리 150 m 이내)
    lane_changes                           차로 변경 횟수 : 기록에 action.lane_change · lane_change_applied가 있으면 "명령 ≠ 0 이고 실행됨"
                                           (11/19), 없으면 event.lane_before ≠ lane_after (스텝 전후 같은 도로일 때 : 실제 결과)
    lc_split                               그중 합류 전 구간(반포대로 · 고가 · 측도)에서 : 구간별 집계, 원인별 아님
                                           (고가/측도 분기 앞 준비와 측도 A8 3  /  2 차로 감소가 함께 들어감)
    lc_merge                               그중 합류 뒤 구간(m1  /  ③  /  반포대교)에서 : 구간별 집계, 원인별 아님
                                           (합류 직후 정리, ① 지하차도 · ② 램프 분기 피하기, 바깥 차로 끝 m2  /  m3가 함께 들어감)
    merge3_s                               ③ 합류 구간(m5) 통과 시간
    min_gap_m                              같은 차로 앞차와의 최소 간격 (범퍼 간, 경로 전체, AV 속도 ≥ 5 m/s일 때만)
    min_gap_merge3_m                       같은 값, ③ 합류 구간(m4 · m5)에서만
    bridge_q_mean / bridge_q_max           반포대교 진입부 저속 차량 수 평균 / 최대
                                           (m3 · m4a · m4 · m5 · ③ 램프 r3 위, 속도 < 5 m/s, AV 제외, 매 스텝)
    overspeed_pct / max_overspeed_kmh      실제 차로 제한속도 초과 스텝 % / 최대 초과폭(km/h), 교차로 내부 차로 포함
    overspeed_gt{1,2,5}_steps              그중 초과폭이 1 · 2 · 5 km/h를 넘은 스텝 수 (11/25 §8 개정: 초과폭별로 보고)
                                           비교 오차 0.001 m/s, 분모는 제한속도를 확인한 스텝. 알 수 없는 스텝은 따로 센다.

한계: 앞차가 다음 도로(엣지)에 있으면 간격에서 놓친다 (짧은 m4 · m5 경계에서 최소 간격이 크게 나올 수 있음).
stop_red_s는 정지선 150 m 안에서 선 시간만 센다 (대기열이 그보다 길게 밀리면 그 부분은 빠짐).

간격은 SUMO lane_pos(앞 범퍼 위치) 차이에서 앞차 길이(승용 5 m, 버스 12 m)를 뺀 값이다.
"""
import argparse
import csv
import math
import os
import sys
from collections import defaultdict
from functools import lru_cache

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))   # 저장소 루트
import repo_paths as P  # noqa: E402
ROOT = P.ROOT
from collect_pomdp_data import (iter_transitions, SCENARIO_CODES, ROUTE_CODES, SIGNAL_TLS_CODES,  # noqa: E402
                                END_REASON_CODES, route_scenario)

# 11/17부터 JSONL의 12팀 칼럼은 숫자 코드다 (조교 공지) : 지표 계산은 문자열 기준이라 읽을 때 되돌린다 (옛 문자열 파일도 그대로 읽힘)
_INV = {k: {v: n for n, v in d.items()} for k, d in
        (("scenario", SCENARIO_CODES), ("ego_route", ROUTE_CODES), ("end_reason", END_REASON_CODES))}
_INV_TLS = {v: n for n, v in SIGNAL_TLS_CODES.items()}
_STATE_LETTER = {1.0: "G", 0.5: "y", 0.0: "r"}


def _normalize(r):
    """숫자 코드 레코드  /  지표 계산이 쓰는 문자열 형식 (scenario · ego_route · end_reason 문자열, signal dict)."""
    for k, inv in _INV.items():
        if isinstance(r.get(k), int) and not isinstance(r.get(k), bool):
            r[k] = inv.get(r[k])
    if r.get("scenario") is None and r.get("ego_route") is not None:   # scenario 칼럼 없는 파일: 경로에서 구한다
        r["scenario"] = route_scenario(r["ego_route"])
    if "signal" not in r and "signal_tls" in r:
        tls = _INV_TLS.get(r.get("signal_tls"))
        r["signal"] = None if tls is None else {"tls": tls, "state": _STATE_LETTER.get(r.get("signal_state"), "O"),
                                                "remaining_s": r.get("signal_remaining_s")}
    return r

DT = 0.5                                   # environment/env/mdp_config.py SIMULATION["step_length"]
STOP_V, MOVE_V, SLOW_V = 0.1, 1.0, 5.0
RED_STATES = set("rRsyYu")                 # 적색 · 황색 (environment/env/sumo_env.py _SIG_GROUP의 red · yellow와 같게)
RED_RANGE = 150.0                          # 신호 정지선까지 이 거리(m) 안에서 선 것만 "빨간불 대기" (environment/env/sumo_env.py RED_WAIT_RANGE와 같게)
OVERSPEED_BANDS_KMH = (1, 2, 5)            # 초과폭별 스텝 수도 센다 (> 1 · > 2 · > 5 km/h)
OVERSPEED_EPS_MPS = 0.001                  # JSONL 소수 4자리 반올림·부동소수점 오차보다 큰 비교 허용 오차 (0.0036 km/h)


@lru_cache(maxsize=4)
def _load_lane_speed_limits(net, _mtime_ns):
    """생성된 SUMO 네트워크의 lane speed(m/s). 내부 차로(:...)도 제외하지 않는다."""
    import xml.etree.ElementTree as ET
    try:
        root = ET.parse(net).getroot()
        return {lane.get("id"): float(lane.get("speed")) for lane in root.iter("lane")
                if lane.get("id") and lane.get("speed") is not None and float(lane.get("speed")) > 0}
    except (OSError, ET.ParseError):
        return {}


def lane_speed_limits(net=None):
    """빌드가 import 뒤에 일어나도 읽고, 도로 파일이 바뀌면 캐시를 갱신한다. 없으면 알 수 없음."""
    net = os.fspath(net or os.path.join(P.SUMO_DIR, "banpo", "banpo.net.xml"))
    try:
        return _load_lane_speed_limits(net, os.stat(net).st_mtime_ns)
    except OSError:
        return {}


def overspeed_summary(rows):
    """판별 %의 평균 대신 전체 초과 스텝/측정 스텝으로 집계한다. 오류 판은 제외한다."""
    rows = [r for r in rows if r.get("end_reason") != "error"]
    known = sum(int(r.get("speed_limit_steps") or 0) for r in rows)
    exceeded = sum(int(r.get("overspeed_steps") or 0) for r in rows)
    unknown = sum(int(r.get("speed_limit_unknown_steps") or 0) if r.get("speed_limit_steps") not in ("", None)
                  else int(r.get("steps") or 0) for r in rows)
    maxima = [float(r["max_overspeed_kmh"]) for r in rows if r.get("max_overspeed_kmh") not in ("", None)]
    out = {f"overspeed_gt{b}_pct": (100 * sum(int(r.get(f"overspeed_gt{b}_steps") or 0) for r in rows) / known
                                    if known and all(r.get(f"overspeed_gt{b}_steps") not in ("", None) for r in rows) else "")
           for b in OVERSPEED_BANDS_KMH}
    return {**out, "speed_limit_steps": known, "overspeed_steps": exceeded, "speed_limit_unknown_steps": unknown,
            "overspeed_pct": 100 * exceeded / known if known else "",
            "max_overspeed_kmh": max(maxima) if known and maxima else ""}


def _signal_geometry():
    """AV 경로별 신호 정지선 위치 : 수업 환경 도로망(environment/env/sumo/banpo/banpo.net.xml)과 road_spec.AV_ROUTES에서 구한다.

    반환: (경로별 엣지 목록, 엣지 길이 {엣지: m}, {경로: {신호 id: 경로상 정지선 앞 엣지 번호}}). 환경 파일이 없으면 빈 dict.
"""
    import xml.etree.ElementTree as ET
    import road_spec
    net = os.path.join(P.SUMO_DIR, "banpo", "banpo.net.xml")
    try:
        root = ET.parse(net).getroot()
    except (OSError, ET.ParseError):
        return {}, {}, {}
    length = {e.get("id"): float(e.find("lane").get("length")) for e in root.iter("edge") if not e.get("function")}
    inc = defaultdict(set)                           # 신호  /  정지선 앞 엣지 (신호가 거는 연결의 출발 엣지)
    for c in root.iter("connection"):
        if c.get("tl") and not c.get("from").startswith(":"):
            inc[c.get("tl")].add(c.get("from"))
    stop_idx = {name: {tl: i for i, e in enumerate(r["edges"]) for tl, es in inc.items() if e in es}
                for name, r in road_spec.AV_ROUTES.items()}
    return {n: r["edges"] for n, r in road_spec.AV_ROUTES.items()}, length, stop_idx


ROUTE_EDGES, EDGE_LEN, STOP_IDX = _signal_geometry()


def dist_to_stop(route, road, lane_pos, tl):
    """AV 경로를 따라 신호 tl의 정지선까지 남은 거리 (m). 경로 밖이거나 이미 지났으면 None.
    (교차점 내부 길이는 빼고 엣지 길이만 더한다 : 몇 m 짧게 나올 수 있음)"""
    edges = ROUTE_EDGES.get(route) or []
    j = STOP_IDX.get(route, {}).get(tl)
    if road not in edges or j is None:
        return None
    i = edges.index(road)
    if i > j:
        return None
    return EDGE_LEN.get(road, 0.0) - lane_pos + sum(EDGE_LEN.get(e, 0.0) for e in edges[i + 1:j + 1])


SPLIT_EDGES = {"bp_nb", "goga_nb", "side_a4u", "side_a4", "side_a8a", "side_a8b", "side_a8c"}
MERGE_EDGES = {"m1", "m2", "m3a", "m3", "m4a", "m4", "m5", "bridge"}
MERGE3_EDGES = {"m5"}
MERGE3_GAP_EDGES = {"m4", "m5"}
BRIDGE_Q_EDGES = {"m3", "m4a", "m4", "m5", "r3"}   # 11/12 m3 · m4를 신호 위치에서 나눔 (m3 = 24032 횡단보도 뒤, m4a = 4032 정지선 앞)
COLS = ["file", "episode", "scenario", "ego_route", "end_reason", "steps", "travel_s", "stops", "stop_s",
        "stop_red_s", "lane_changes", "lc_split", "lc_merge", "merge3_s", "min_gap_m", "min_gap_merge3_m",
        "bridge_q_mean", "bridge_q_max", "speed_limit_steps", "speed_limit_unknown_steps",
        "overspeed_steps", "overspeed_pct", "max_overspeed_kmh",
        "overspeed_gt1_steps", "overspeed_gt2_steps", "overspeed_gt5_steps"]


def _veh_len(vid):
    return 12.0 if vid.startswith("bus_") else 5.0


def front_gap(st):
    """state 하나에서 AV와 같은 차로 바로 앞차의 범퍼 간 간격 (없으면 None)."""
    lane, pos = st.get("ego_lane_id"), st.get("ego_lane_pos")
    best = None
    for v in st.get("vehicles", ()):
        if v["id"] == "ego" or v["lane_id"] != lane or v["lane_pos"] <= pos:
            continue
        g = v["lane_pos"] - _veh_len(v["id"]) - pos
        if best is None or g < best:
            best = g
    return best


def episode_metrics(recs):
    m = defaultdict(float)
    first = recs[0]
    m.update(episode=first["episode"], scenario=first.get("scenario"), ego_route=first.get("ego_route"),
             end_reason=recs[-1].get("end_reason"), steps=len(recs))
    m["travel_s"] = len(recs) * DT
    limits = lane_speed_limits()
    known, unknown, exceeded, max_excess = 0, 0, 0, 0.0
    bands = {b: 0 for b in OVERSPEED_BANDS_KMH}
    moving, q, gaps, gaps3 = False, [], [], []
    for r in recs:
        st = r.get("state") or {}
        if not st.get("ego_alive"):
            unknown += 1
            continue
        v, road = st["ego_speed"], st["ego_road_id"]
        limit = limits.get(st.get("ego_lane_id"))
        if limit is not None and math.isfinite(v):
            known += 1
            excess = max(0.0, v - limit)
            exceeded += excess > OVERSPEED_EPS_MPS
            max_excess = max(max_excess, excess)
            for b in bands:
                bands[b] += 3.6 * excess > b
        else:
            unknown += 1
        if v >= MOVE_V:
            moving = True
        if v < STOP_V:
            m["stop_s"] += DT
            if moving:
                m["stops"] += 1
                moving = False
            sig = r.get("signal") or {}
            if sig.get("state") in RED_STATES:
                d = dist_to_stop(first.get("ego_route"), road, st.get("ego_lane_pos", 0.0), sig.get("tls"))
                if d is not None and d <= RED_RANGE:
                    m["stop_red_s"] += DT
        if road in MERGE3_EDGES:
            m["merge3_s"] += DT
        # 실제로 바뀐 차로만 센다 (명령이 아니라 결과).
        # 기록에 action.lane_change(명령)와 lane_change_applied가 있으면(RL · 규칙 정책) "명령 ≠ 0 이고 실행됨"으로 센다 :
        # 아래 옛 방식은 도로 경계를 넘는 스텝의 변경을 못 셌다. 없으면 (옛 기록. SUMO 운전자 모드 기록의 두 값은 실제 움직임에서 역산한 것이라 옛 방식과 같은 값)
        # 옛 방식: 스텝 전후 차로 번호가 다르고 같은 도로 위일 때. (거부된 차로 변경 명령은 어느 쪽도 세지 않는다)
        ev, nst = r.get("event") or {}, r.get("next_state") or {}
        lb, la = ev.get("lane_before"), ev.get("lane_after")
        act = r.get("action") if isinstance(r.get("action"), dict) else {}
        if "lane_change" in act and "lane_change_applied" in act:
            changed = act["lane_change"] != 0 and bool(act["lane_change_applied"])
        else:
            changed = lb is not None and la is not None and la != lb and nst.get("ego_road_id") == road
        if changed:
            m["lane_changes"] += 1
            m["lc_split"] += road in SPLIT_EDGES
            m["lc_merge"] += road in MERGE_EDGES
        g = front_gap(st) if v >= SLOW_V else None   # 신호 대기 중 정지 간격(최소 간격 2~2.5 m)은 빼고, 달리는 중만
        if g is not None:
            gaps.append(g)
            if road in MERGE3_GAP_EDGES:
                gaps3.append(g)
        q.append(sum(1 for x in st.get("vehicles", ())
                     if x["id"] != "ego" and x["road_id"] in BRIDGE_Q_EDGES and x["speed"] < SLOW_V))   # AV 자신은 뺌
    m["min_gap_m"] = round(min(gaps), 2) if gaps else ""
    m["min_gap_merge3_m"] = round(min(gaps3), 2) if gaps3 else ""
    m["bridge_q_mean"] = round(sum(q) / len(q), 2) if q else ""
    m["bridge_q_max"] = max(q) if q else ""
    for k in ("stops", "lane_changes", "lc_split", "lc_merge"):
        m[k] = int(m[k])
    for k in ("stop_s", "stop_red_s", "merge3_s"):      # 0인 에피소드도 0으로 남긴다 (빈 값이면 평균에서 빠짐)
        m[k] = round(m[k], 2)
    # signal 칼럼이 없거나 11/15 이전 형식({phase}), 또는 도로망 파일이 없어 정지선 위치를 모름  /  알 수 없음 (0으로 두면 "빨간불 대기 없음"으로 읽힘)
    if not STOP_IDX or not any("state" in (r.get("signal") or {}) for r in recs):
        m["stop_red_s"] = ""
    m.update(speed_limit_steps=known, speed_limit_unknown_steps=unknown, overspeed_steps=exceeded,
             overspeed_pct=round(100 * exceeded / known, 6) if known else "",
             max_overspeed_kmh=round(3.6 * max_excess, 6) if known else "",
             **{f"overspeed_gt{b}_steps": (bands[b] if known else "") for b in bands})
    return dict(m)


def file_metrics(path):
    """파일 하나의 에피소드별 지표 목록 (에피소드 단위로 읽어 메모리를 아낀다)."""
    out, cur, ep = [], [], None
    for r in iter_transitions(path):
        r = _normalize(r)
        if ep is not None and r["episode"] != ep:
            out.append(episode_metrics(cur))
            cur = []
        ep = r["episode"]
        cur.append(r)
    if cur:
        out.append(episode_metrics(cur))
    for m in out:
        m["file"] = os.path.basename(path)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+")
    ap.add_argument("--csv", default=None)
    a = ap.parse_args()
    rows = [m for f in a.files for m in file_metrics(f)]
    if a.csv:
        with open(a.csv, "w", encoding="utf-8-sig", newline="") as f:
            w = csv.DictWriter(f, fieldnames=COLS, extrasaction="ignore")
            w.writeheader()
            w.writerows(rows)
    for m in rows:
        print({k: m.get(k) for k in COLS if k != "file"})


if __name__ == "__main__":
    main()
