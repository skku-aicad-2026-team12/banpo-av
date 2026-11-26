"""교통 수요 · 신호 주기 근거 계산 (실측 TOPIS 2026-09 평일 낮 평균 기준)

사용 (프로젝트 루트에서):
    python -X utf8 environment/road/calibrate.py

먼저 environment/road/build/before/banpo.net.xml이 있어야 한다 (python -X utf8 environment/road/build_net.py).

1. 실측 목표: docs/02.근거/교통_신호_버스/topis_2026_09_반포권역_시간대평균.csv 에서 평일 11~15시 평균 (대/시).
   신반포로(D-36 사평대로 대용)는 참고로만 출력 (측정 지점이 없어 비교하지 않음 : 진입량은 4절)
2. 수요 검사: environment/env/road_config.py TRAFFIC["od"]를 차로 단위 경로로 배정해, 측정 지점 8곳(반포대교 북행 · 남행,
   잠수교 북행 · 남행, 반포대로 북행 · 남행, 올림픽대로 양방향)에 흐르는 교통량을 실측과 비교 (보행화 전: TOPIS,
   후: t-data 10081 : after_target에 적어 둔 실측값)
3. 신호 주기 (기록용 : 신호가 추정이던 11/12의 계산): 신반포사거리 회전 교통량  /  Webster 주기 + 보행 최소 녹색
    /  현시별 녹색시간 (road_spec.SIGNAL_PLAN과 비교)
4. 실측 신호(road_spec.SIGNALS["ix"]) 기준 그룹별 포화도와 신반포로 권장 진입량 (지금 수요의 근거, 11/13)
결과는 environment/road/build/calibrate_report.txt 에도 저장한다.
"""
import csv, math, os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))   # 저장소 루트
import repo_paths as P  # noqa: E402
ROOT = P.ROOT
import road_spec as S
import routing
from env.road_config import TRAFFIC

TOPIS = os.path.join(ROOT, "docs", "02.근거", "교통_신호_버스", "topis_2026_09_반포권역_시간대평균.csv")
HOURS = range(11, 16)                  # 평일 낮 (11~15시)

# 측정 지점  /  도로망에서 같은 교통량이 지나는 엣지
POINTS = {
    ("C-11", "반포대교남단->반포대교북단"): ("bridge", "반포대교 상층 북행"),
    ("C-11", "반포대교북단->반포대교남단"): ("sb_up", "반포대교에서 내려오는 남행"),
    ("C-12", "반포고가차도->반포대교남단"): ("jam_n", "잠수교 북행"),
    ("C-12", "반포대교남단->반포고가차도"): ("jam_s", "잠수교 남행"),
    ("D-37", "서초역->서울성모병원"): ("bp_nb", "반포대로 북행 (서초역 지점, 대용)"),
    ("D-37", "서울성모병원->서초역"): ("bp_sb", "반포대로 남행 (서초역 지점, 대용)"),
    ("F-01", "동작대교남단->한강대교남단"): ("wb_a", "올림픽대로 김포 방향 (동작~한강대교 지점, 대용)"),
    ("F-01", "한강대교남단->동작대교남단"): ("eb_a", "올림픽대로 하남 방향 (동작~한강대교 지점, 대용)"),
    ("D-36", "삼호가든사거리->서울성모병원"): ("sbr_a6", "신반포로 (사평대로 고속터미널역 지점, 대용)"),
    ("D-36", "서울성모병원->삼호가든사거리"): ("sbr_a7a", "신반포로 (사평대로 고속터미널역 지점, 대용)"),
}


def targets():
    out = {}
    with open(TOPIS, encoding="utf-8-sig") as f:
        rows = list(csv.reader(f))
    h = rows[0]
    idx = [h.index(f"{k}시") for k in HOURS]
    for r in rows[1:]:
        if r[4] != "평일":
            continue
        key = (r[0], r[3])
        if key in POINTS:
            out[POINTS[key][0]] = (sum(float(r[i]) for i in idx) / len(idx), POINTS[key][1])
    return out


def assign(net, lane_attr, scn):
    """시나리오별 OD를 경로로 배정  /  {엣지: 대/시}, {(from, to) at ix: 대/시}."""
    closed = set(S.CLOSED_AFTER) if scn == "after" else set()
    edge_flow, paths = {}, []
    for entry, spec in TRAFFIC["od"].items():
        if entry in closed or not net.hasEdge(entry):
            continue
        vph = float(spec.get(f"vph_{scn}", spec["vph"]))
        exits = spec.get(f"exits_{scn}", spec["exits"])
        ok = {k: (w, p) for k, w, p in routing.od_paths(net, lane_attr, entry, exits, closed)}
        tw = sum(w for w, _ in ok.values())
        for ex, (w, p) in ok.items():
            q = vph * w / tw
            paths.append((q, p))
            for e in p:
                edge_flow[e] = edge_flow.get(e, 0.0) + q
    return edge_flow, paths


def main():
    say_lines = []
    def say(s=""):
        say_lines.append(s); print(s)
    net, lane_attr = routing.load(os.path.join(P.ROAD_BUILD, "before", "banpo.net.xml"))
    tg = targets()
    say("1. 실측 목표 (TOPIS 2026-09 평일 11~15시 평균, 대/시)")
    for e, (v, name) in tg.items():
        say(f"   {e:8s} {v:7.0f}  {name}")

    say("\n2. 수요 배정 결과 vs 실측 (잠수교 폐쇄 전 = 9월 실측, 후 = 11/9~11 실측)")
    # 보행화 후 목표: 서울 t-data "1시간 지점별 교통량(차로별)" 11/9~11 평일 11~15시 평균 (docs/02.근거/교통_신호_버스/tdata_10081_*.csv, 11/12)
    after_target = {"bridge": 2858, "sb_up": 2384, "jam_n": 0.0, "jam_s": 0.0, "bp_nb": 2233, "bp_sb": 1974, "wb_a": 7190, "eb_a": 5219}
    for scn in ("before", "after"):
        flow, _ = assign(net, lane_attr, scn)
        say(f"  [{scn}]")
        for e in ("bridge", "jam_n", "sb_up", "jam_s", "bp_sb", "bp_nb", "wb_a", "eb_a"):
            if scn == "before":
                t = tg[e][0]
            else:
                t = after_target.get(e, tg[e][0])
            got = flow.get(e, 0.0)
            err = (got - t) / t * 100 if t else 0.0
            mark = "✓" if (got == 0 if t == 0 else abs(err) <= 10) else "✗"   # 목표 0(통제된 잠수교)은 배정도 0이어야 통과
            say(f"   {mark} {e:8s} 배정 {got:7.0f}   목표 {t:7.0f}   {err:+5.1f}%")

    # 3. 신호 (보행화 전 수요 기준 : 실제 신호도 폐쇄 전부터 운영돼 온 것)
    say("\n3. 신반포사거리 신호 (Webster + 보행 최소 녹색)")
    _, paths = assign(net, lane_attr, "before")
    mv = {}
    for q, p in paths:
        for a, b in zip(p, p[1:]):
            if net.getEdge(a).getToNode().getID() == "ix":
                mv[(a, b)] = mv.get((a, b), 0.0) + q
    for (a, b), q in sorted(mv.items()):
        say(f"   회전 교통량 {a:>10s} → {b:<10s} {q:6.0f} 대/시")
    plan = S.SIGNAL_PLAN
    ys = []
    for ph in plan["phases"]:
        crit = 0.0
        for (a, b) in ph["movements"]:
            q = mv.get((a, b), 0.0)
            lanes = len({c.getFromLane().getIndex() for c in net.getEdge(a).getOutgoing().get(net.getEdge(b), [])
                         if c.getFromLane().allows("passenger")}) or 1
            s = plan["sat_left"] if (a, b) in plan["left_movements"] else plan["sat_through"]
            crit = max(crit, q / lanes / s)
        ys.append(crit)
        say(f"   {ph['name']}: 임계 y = {crit:.3f}")
    Y = sum(ys)
    L = len(plan["phases"]) * plan["lost_per_phase"]
    # 신반포로 진입량 권장값: 신반포로에는 측정 지점이 없어, 신반포로 현시(φ1·φ2)가 목표 주기에서
    # Webster 식을 만족하는 크기로 정한다 (측도 현시 φ3·φ4는 실측 기반 수요 그대로)
    #   목표 주기에서 측도 현시(φ3·φ4)는 max(보행 최소 녹색, 포화도 목표를 맞추는 녹색)을 받고,
    #   남는 녹색을 신반포로 현시(φ1·φ2)가 포화도 목표(x_target)로 쓸 수 있는 만큼이 권장 진입량이다.
    c_t, x_t = plan["target_cycle"], plan["x_target"]
    eff_t = c_t - L
    side_g = sum(max(ph["min_green"], y * c_t / x_t) for ph, y in zip(plan["phases"][2:], ys[2:]))
    sbr_y = ys[0] + ys[1]
    if sbr_y > 0:
        k = x_t * (eff_t - side_g) / c_t / sbr_y
        sbr = {e: float(TRAFFIC["od"][e].get("vph_before", TRAFFIC["od"][e]["vph"])) for e in ("sbr_a6", "sbr_a7a")}
        say(f"   신반포로 권장 진입량 (목표 주기 {c_t} s, 포화도 {x_t}): 현재값의 {k:.2f}배 → "
            + ", ".join(f"{e} {v * k:.0f}" for e, v in sbr.items()) + " 대/시")
    if Y >= 1:
        say(f"   ✗ Y = {Y:.3f} ≥ 1: 이 현시로는 수요를 처리할 수 없음 (과포화)")
        Y = 0.999
    c_web = (1.5 * L + 5) / (1 - Y)
    gmin = [ph["min_green"] for ph in plan["phases"]]
    c_min = sum(gmin) + L

    def split(c):
        """주기 c에서 녹색 배분: 임계 y 비율로 나누되 최소 녹색보다 짧은 현시는 최소로 고정하고 나머지를 다시 나눔."""
        eff = c - L
        fixed = {}
        while True:
            free = [i for i in range(len(ys)) if i not in fixed]
            rest = eff - sum(fixed.values())
            ysum = sum(ys[i] for i in free) or 1.0
            g = [fixed.get(i, rest * ys[i] / ysum) for i in range(len(ys))]
            low = [i for i in free if g[i] < gmin[i]]
            if not low:
                break
            for i in low:
                fixed[i] = gmin[i]
        g = [round(x) for x in g]
        g[-1] += eff - sum(g)
        return g

    # 주기: Webster·최소 녹색 합 이상에서 10 s씩 늘려, 모든 현시 포화도가 x_target 이하가 되는 가장 짧은 주기
    lo, hi = plan["cycle_bounds"]
    c = max(lo, math.ceil(max(min(c_web, hi), c_min) / 10) * 10)
    while True:
        greens = split(c)
        if max(y * c / g for y, g in zip(ys, greens)) <= plan["x_target"] or c >= hi:
            break
        c += 10
    say(f"   Y = {Y:.3f}, 손실시간 L = {L} s → Webster 주기 {c_web:.0f} s, 최소 녹색 합 기준 {c_min} s → 주기 {c} s")
    for ph, g, g0, y in zip(plan["phases"], greens, [ph["green"] for ph in plan["phases"]], ys):
        x = y * c / g if g else 0
        say(f"   {ph['name']}: 녹색 {g} s (road_spec 값 {g0} s, 최소 {ph['min_green']} s)  + 황색 {plan['yellow']} + 전적색 {plan['all_red']}"
            f"  · 포화도 {x:.2f}")
    # 4. 실측 신호(road_spec.SIGNALS["ix"], 11/12) 기준 그룹별 포화도와 신반포로 진입량 (11/13)
    #   위 3절은 신호가 추정이던 때의 계산이다. 신반포로에는 측정 지점이 없어 진입량을 "사거리가 포화도 x_target으로
    #   처리할 수 있는 크기"로 추정했는데(사평대로 실측의 약 0.53배), 실측 신호에서는 서행 직진 녹색이 44 s뿐이라 그 값이 넘친다.
    #   같은 원칙으로 다시 정한다: 그룹 포화도 x = 회전 교통량 / (승용차 차로 수 × 포화교통류율 × 녹색 / 주기).
    #   녹색 = 실측 녹색 시간(황색 · 출발 손실은 서로 상쇄한다고 봄). 처리하지 못하는(x > x_target) 진입로만
    #   x_target에 맞춰 줄인다 : 늘리지는 않는다(늘리면 반포대교 등 실측 지점 교통량 맞춤이 깨짐).
    sp = S.SIGNALS["ix"]
    say(f"\n4. 실측 신호 기준 포화도 (주기 {sp['cycle']} s, 포화교통류율 직진 {plan['sat_through']} · 좌회전 {plan['sat_left']}, 목표 {x_t})")
    worst = {}
    for name, _g0, gl, _yl, mvs, _yield in sp["groups"]:
        q = sum(mv.get(ab, 0.0) for ab in {(a, b) for a, b, _ in mvs})   # 승용차 회전 교통량 (버스 제외)
        lanes = set()
        for a, b, fl in mvs:
            for c in net.getEdge(a).getOutgoing().get(net.getEdge(b), []):
                ln = c.getFromLane()
                if ln.allows("passenger") and (fl is None or ln.getIndex() == S.net_lane(a, fl)):
                    lanes.add((a, ln.getIndex()))
        if not lanes:
            continue   # 버스 전용 그룹
        s = plan["sat_left"] if any((a, b) in plan["left_movements"] for a, b, _ in mvs) else plan["sat_through"]
        x = q / (len(lanes) * s * gl / sp["cycle"])
        a0 = mvs[0][0]
        worst[a0] = max(worst.get(a0, 0.0), x)
        say(f"   {name:16s} 녹색 {gl:3d} s · 차로 {len(lanes)} · {q:5.0f} 대/시 → 포화도 {x:.2f}{'  ✗' if x > x_t else ''}")
    for e, a in (("sbr_a6", "sbr_a6"), ("sbr_a7a", "sbr_a7b")):
        x = worst.get(a, 0.0)
        v = float(TRAFFIC["od"][e].get("vph_before", TRAFFIC["od"][e]["vph"]))
        if x > x_t:
            say(f"   신반포로 {e}: 최대 포화도 {x:.2f} > {x_t} → 권장 진입량 {v * x_t / x:.0f} 대/시 (지금 {v:.0f})")
        else:
            say(f"   신반포로 {e}: 최대 포화도 {x:.2f} ≤ {x_t} → 지금 값 {v:.0f} 대/시 유지")

    os.makedirs(P.ROAD_BUILD, exist_ok=True)
    with open(os.path.join(P.ROAD_BUILD, "calibrate_report.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(say_lines) + "\n")


if __name__ == "__main__":
    main()
