"""반포대교 남단 ~ 신반포사거리 도로 정의 (단일 원천)

실제 도로에서 확인한 값 · 확정 가정 · 바뀐 과정은 docs/프로젝트_기록.md §2 · §11.
좌표: m, x = 동쪽, y = 북쪽. 축척 실제 (올림픽대로 두 방향 간격도 실제 65 m).
차로 번호: SUMO 규칙대로 0 = 가장 오른쪽 차로.
형상 규칙:
  spread="right"  : shape = 진행 방향 왼쪽 경계선, 차로는 오른쪽으로 쌓임 (일방통행 본선)
  spread="center" : shape = 중심선 (램프·회전교차로 등)
  양방향 도로는 같은 중심선을 쓰는 일방 엣지 두 개 (각각 spread="right")

이 파일만 고치면 build_net.py가 SUMO 입력(XML)과 도로망을 다시 만든다.
현재 상태·검증 결과·남은 작업: environment/road/README.md.
"""
import math

W = 3.2  # 차로 폭 (m)

# ---------------------------------------------------------------- 레이아웃 상수
Y_SPLIT = 100                         # 고가·측도 분기
Y_ENTRY = Y_SPLIT - 300               # 반포대로 진입 (분기 300 m 앞)
Y_IX = 350                            # 신반포사거리 중심
Y_BOX0, Y_BOX1 = Y_IX - 20, Y_IX + 18 # 사거리 남·북 가장자리
Y_MED_N, Y_MED_S = Y_IX + 2, Y_IX - 2 # 신반포로 중앙선 (서행 / 동행 왼쪽 경계)
Y_MERGE = Y_IX + 330                  # 고가·측도 합류
Y_SEP = Y_MERGE + 10                  # ① 지하차도 진입부
Y_DROP = Y_MERGE + 40                 # 측도 쪽 바깥 차로 끝
Y_R2 = Y_MERGE + 190                  # ② 분기
Y_EB = Y_MERGE + 245                  # 올림픽대로 하남 방향 왼쪽 경계 (차로는 남쪽)
GAP_OLY = 65                          # 올림픽대로 두 방향 사이 간격 (실제 약 65 m. 11/11 전까지 5 m로 압축했었음)
Y_WB = Y_EB + GAP_OLY                 # 올림픽대로 김포 방향 왼쪽 경계 (차로는 북쪽)
Y_R3J = Y_EB + 12                     # ③ 램프가 본선 옆 합류 차로로 붙는 곳 (두 방향 사이 중앙녹지)
Y_R3 = Y_R3J - 8                      # ③ 램프가 중앙녹지에서 본선 옆으로 다가오는 높이
Y_BR = Y_MERGE + 290                  # ③ 합류 끝 = 반포대교 상층 3차로 시작 (김포 방향 20 m 앞)
Y_TOP = Y_BR + 300                    # 반포대교 출구
Y_END = Y_TOP + 100                   # 지도 북쪽 끝 — 북행 다리(반포대교 · 잠수교)도 남행 진입점과 같은 높이에서 끝낸다 (11/13: 100 m 짧게 끝나
                                      # 화면용으로만 이어 그렸더니, sumo-gui가 막다른 차로 끝에 그리는 검은 막대가 다리 한가운데에 보였음)
WB4 = Y_WB + W                        # ③ 진출 뒤 김포 방향 왼쪽 경계
WB5C = WB4 + 4.5 * W                  # 김포 방향 5번째 차로 중심
X_R3 = 440                            # 김포 방향 ③ 진출점
X_WB_IN = X_R3 + 400                  # 김포 방향 진입
X_LOOP = 10                           # 루프 합류
X_PARK = X_LOOP - 260                 # 주차장 진출 (엮임 260 m)
R2_R = Y_EB - 4.5 * W - Y_R2          # ② 램프 원호 반경
X_EB5 = 8 + R2_R + (250 - math.pi / 2 * R2_R)   # ② 합류 → 하남 방향 5차로 시작 (② 램프 약 250 m)
X_W = -413                            # 올림픽대로 서쪽 끝
L_SBR = 200                           # 신반포로 진입·출구 길이
RR = 17                               # 회전교차로 반경
RC = (-35, Y_WB + 55)                 # 회전교차로 중심 (지하차도 약 340 m가 되는 위치)
RB_SE = 320                           # 지하차도 갈래가 회전교차로에 붙는 방향 (°, 동 = 0, 반시계) — 남동쪽에서 중심을 향해 진입 (11/14)
RB_E = 10                             # 루프 쪽 출구 방향 (°) — 지하차도 갈래(RB_SE)와 회전차로 15 m 띄우려고 동쪽에서 10° 올림 (11/14)
# 지하차도 북행 → 루프 우회로 갈래 높이 = 지하차도 직선(x = −6.4)이 회전교차로 중심에서 RB_SE 방향 직선과 만나는 점 (꺾이는 점).
# 갈래 → 회전교차로 구간을 곧은 왕복 도로로 (11/14 — 꺾이는 곡선을 이 구간에 두었더니 netconvert가 북행 차로를 3 m 어긋난
# 직선으로 잘라 남행과 0.95 m로 겹쳐 보이고 중앙선도 안 생겼음)
UND_J = RC[1] + (-6.4 - RC[0]) * math.tan(math.radians(RB_SE))
X_LOOP_TURN = 215                     # 루프가 되돌아 나오는 동쪽 끝 (루프 약 500 m)
WARM1 = RC[0] - RR - 45               # 회전교차로 서쪽 갈래 끝 (주차장 쪽)
POCKET = Y_BOX1 + 80                  # A5 좌회전 포켓 시작
A5D = Y_BOX1 + 17                     # A5 우회전 도로 갈라지는 곳
A8DROP = Y_MERGE - 60                 # A8 3 → 2
GSB_D = Y_MERGE - 2                   # 남행 측도 → 고가 남행 갈라지는 곳
GSB_J = Y_MERGE - 20                  # 고가 남행 시작
X_SB = -14.4                          # 측도 남행(반포대교에서) 왼쪽 경계 — 지하차도 남행(−9.6)과 중앙분리 4.8 m
Y_XW = Y_DROP + 50                    # 북행 횡단보도 신호 (C-ITS 24032) — 4 → 3차로 끝에서 약 50 m (11/12 로드뷰 확인)
Y_S4032 = Y_R2 + 15                   # 반포대교남단 신호 정지선 (C-ITS 4032) — ② 램프가 갈라진 직후 직진 3차로 (11/12 로드뷰)
JX1_D, JX2_D = 20, 190                # 잠수교 횡단보도 신호 (C-ITS 4616 · 24616): 회전교차로 북쪽 끝에서 잠수교 쪽으로 거리 (m)
                                      # (4616: 11/14 10 → 20 m — 횡단보도를 지나 우회전 우회로(byp_nw)가 남행 차로 옆으로 갈라질 자리, 팀 요청)

# 제한속도 (m/s)
KMH = lambda v: round(v / 3.6, 2)
V_OLY, V_RAMP, V_MAIN, V_SIDE, V_LOW = KMH(80), KMH(40), KMH(60), KMH(50), KMH(30)

SCENARIOS = ("before", "after")       # 보행화 전(잠수교 열림) / 후(잠수교 차량 통제 — 환경은 실행 중 통제, build_net은 잠수교 엣지를 뺀 검증 도로망)

# ---------------------------------------------------------------- 형상 도우미
def arc(cx, cy, r, a0, a1, n=12):
    return [(round(cx + r * math.cos(math.radians(a0 + (a1 - a0) * i / n)), 2),
             round(cy + r * math.sin(math.radians(a0 + (a1 - a0) * i / n)), 2)) for i in range(n + 1)]

def path(*items):
    pts = []
    for it in items:
        for p in (it if isinstance(it, list) else [it]):
            p = (round(p[0], 2), round(p[1], 2))
            if not pts or math.dist(pts[-1], p) > 1e-3:
                pts.append(p)
    return pts

def fillet(a, c, b, r, n=8):
    """꺾이는 점 c를 반경 약 r의 곡선으로 (a → c → b, 2차 베지어)."""
    da, db = math.dist(a, c), math.dist(b, c)
    p0 = (c[0] + (a[0] - c[0]) * r / da, c[1] + (a[1] - c[1]) * r / da)
    p2 = (c[0] + (b[0] - c[0]) * r / db, c[1] + (b[1] - c[1]) * r / db)
    return [((1 - t) ** 2 * p0[0] + 2 * (1 - t) * t * c[0] + t * t * p2[0],
             (1 - t) ** 2 * p0[1] + 2 * (1 - t) * t * c[1] + t * t * p2[1]) for t in (i / n for i in range(n + 1))]

def smooth(pts, it=3):
    """꺾인 점들을 부드러운 곡선으로 (Chaikin 모서리 깎기 — 양 끝점은 그대로)."""
    for _ in range(it):
        out = [pts[0]]
        for a, b in zip(pts, pts[1:]):
            out += [(0.75 * a[0] + 0.25 * b[0], 0.75 * a[1] + 0.25 * b[1]), (0.25 * a[0] + 0.75 * b[0], 0.25 * a[1] + 0.75 * b[1])]
        out.append(pts[-1])
        pts = out[:1] + out[2:-2] + out[-1:]
    return path(pts)

def plen(p):
    return sum(math.dist(a, b) for a, b in zip(p, p[1:]))

def _park_path(x_hp):
    return path((X_PARK, WB5C), (x_hp, WB5C), arc(x_hp, WB5C + 12, 12, 270, 90), (-160, WB5C + 24),
                (WARM1 - 15, RC[1] - 4.8), (WARM1, RC[1] - 4.8))
X_HP = X_PARK - 10
while plen(_park_path(X_HP)) < 480:   # 주차장 램프 약 480 m
    X_HP -= 1

# ---------------------------------------------------------------- 노드 (id: (x, y, type))
# type: priority(기본) / traffic_light / dead_end 는 netconvert가 판단
NODES = {
    # 반포대로 남쪽 · 고가 · 측도
    "bp_in":     (9.6, Y_ENTRY, "priority"),        # 반포대로 북행 진입
    "bp_out":    (-8.0, Y_ENTRY, "priority"),       # 반포대로 남행 출구
    "split_nb":  (9.6, Y_SPLIT, "priority"),        # 고가 / 측도 분기
    "a4s":       (16.0, Y_BOX0 - 50, "priority"),   # 측도 북행 우회전 도로 분기 (사거리 50 m 앞 — 11/14 팀 확인: 실제는 사거리 거의 다 와서 갈라짐)
    "merge_sb":  (-8.0, Y_SPLIT, "priority"),       # 측도 남행 + 고가 남행 합류
    "ix":        (0.0, Y_IX, "traffic_light"),      # 신반포사거리
    "e2m":       (60.0, Y_MED_S - 6, "priority"),   # A4 우회전 도로 → E2 합류
    "a7d":       (60.0, Y_MED_N + 6, "priority"),   # A7 우회전 도로 분기
    "a8m":       (20.0, Y_BOX1 + 22, "priority"),   # A7 우회전 도로 → A8 합류
    "a8drop":    (12.0, A8DROP, "priority"),        # A8 3 → 2
    "e1m":       (-55.0, Y_MED_N + 6, "priority"),  # A5 우회전 도로 → E1 합류
    "sbr_w_in":  (-227.0, Y_MED_S - 8, "priority"), # 신반포로 서쪽 진입 (동행)
    "sbr_w_out": (-227.0, Y_MED_N + 6, "priority"), # 신반포로 서쪽 출구 (서행)
    "sbr_e_in":  (227.0, Y_MED_N + 6, "priority"),  # 신반포로 동쪽 진입 (서행)
    "sbr_e_out": (227.0, Y_MED_S - 6, "priority"),  # 신반포로 동쪽 출구 (동행)
    "sb_in":     (-17.6, Y_EB - 25, "priority"),    # 남행 측도 진입 (반포대교에서, 범위 밖)
    "gsb_d":     (-17.6, GSB_D, "priority"),        # 남행 측도 → 고가 남행 분기
    "gsb_j":     (-3.2, GSB_J, "priority"),         # 고가 남행 시작
    "pocket":    (-19.2, POCKET, "priority"),       # A5 좌회전 포켓 시작 (여기부터 sb_2 5차로)
    "a5d":       (-17.6, A5D, "priority"),          # A5 우회전 도로 분기
    # 합류 ~ 반포대교
    "merge_nb":  (4.8, Y_MERGE, "priority"),        # 고가 + 측도 합류 (+ ① 차로 생김)
    "sep":       (4.8, Y_SEP, "priority"),          # ① 지하차도 진입부
    "drop":      (4.8, Y_DROP, "priority"),         # 측도 쪽 바깥 차로 끝
    "xw":        (4.8, Y_XW, "traffic_light"),      # 북행 횡단보도 신호 (24032)
    "xwp_s":     (4.8, Y_XW - 25, "priority"),      # 횡단보도 왼쪽(서쪽) 보도 끝 (보행자 전용 보도 — 일방통행 도로를 건너는 횡단보도용)
    "xwp_n":     (4.8, Y_XW + 25, "priority"),
    "xwpe_s":    (4.8, Y_XW - 26, "priority"),      # 횡단보도 오른쪽(동쪽) 보도 끝 (11/13: m3a · m3 보도 대신 — 보도가 ② 램프 분기까지 이어져 램프가 보도에서 갈라져 나오는 것처럼 보였음)
    "xwpe_n":    (4.8, Y_XW + 26, "priority"),
    "r2d":       (4.8, Y_R2, "priority"),           # ② 분기
    "s4032":     (4.8, Y_S4032, "traffic_light"),   # 반포대교남단 신호 정지선 (4032)
    "r3m":       (6.4, Y_R3J, "priority"),          # ③ 합류 차로 시작
    "br":        (4.8, Y_BR, "priority"),           # ③ 합류 끝 (반포대교 상층 3차로)
    "br_out":    (4.8, Y_END, "priority"),          # 반포대교 출구 (지도 북쪽 끝)
    # 올림픽대로
    "eb_in":     (X_W, Y_EB - 6.4, "priority"),     # 하남 방향 진입 (서쪽)
    "ebm":       (X_EB5, Y_EB - 8, "priority"),     # ② 합류
    "eb_out":    (X_WB_IN, Y_EB - 8, "priority"),   # 하남 방향 출구 (동쪽)
    "wb_in":     (X_WB_IN, Y_WB + 8, "priority"),   # 김포 방향 진입 (동쪽)
    "wbr3":      (X_R3, Y_WB + 8, "priority"),      # ③ 진출
    "wblm":      (X_LOOP, WB4 + 8, "priority"),     # 루프 합류
    "wbpk":      (X_PARK, WB4 + 8, "priority"),     # 주차장 진출
    "wb_out":    (X_W, WB4 + 6.4, "priority"),      # 김포 방향 출구 (서쪽)
    # 잠수교 남단 회전교차로 · 주차장
    "rb_se":     (RC[0] + RR * math.cos(math.radians(RB_SE)), RC[1] + RR * math.sin(math.radians(RB_SE)), "priority"),   # 지하차도 갈래 (남동쪽)
    "rb_e":      (RC[0] + RR * math.cos(math.radians(RB_E)), RC[1] + RR * math.sin(math.radians(RB_E)), "priority"),   # 루프 쪽 출구
    "rb_n":      (RC[0], RC[1] + RR, "priority"),
    "rb_w":      (RC[0] - RR, RC[1], "priority"),
    "jam_out":   (RC[0], Y_END, "priority"),        # 잠수교 북쪽 끝 (보행화 전에만, 지도 북쪽 끝)
    "jam_in":    (RC[0], Y_END, "priority"),  # 잠수교 남행 진입 (11/13: jx2 신호 앞 투입 공간 16 m → 116 m)
    "jx1":       (RC[0], RC[1] + RR + JX1_D, "traffic_light"),   # 잠수교남단 횡단보도 신호 (4616, 보행화 전에만 작동)
    "jx2":       (RC[0], RC[1] + RR + JX2_D, "traffic_light"),   # 잠수교 횡단보도 신호 (24616, 보행화 전에만 작동)
    "arm_w":     (WARM1, RC[1], "priority"),        # 회전교차로 서쪽 갈래 / 주차장 쪽 분기·합류
    # 회전교차로 우회전 우회로 (11/12 로드뷰: 진입로마다 옆 출구로 바로 빠지는 우회전 길. 동쪽은 모델에 진입로가 없어 3개)
    "und_j":     (-6.4, UND_J, "priority"),                # 지하차도 북행 → 루프 우회로 갈래 (북행만)
    "und_m":     (-6.4, RC[1] - RR - 13.8, "priority"),      # 서쪽 → 지하차도 남행 우회로 합류 (남행만) — 11/13: und_j 하나에 북행 갈래 · 남행 합류를
                                                           # 같이 두면 교차로가 올림픽대로 밑까지 늘어나 북행 정지선(sumo-gui 막대)이 올림픽대로 위에 보였음
    "lp_m":      (2.0, RC[1], "priority"),                  # 루프 쪽 합류 (남 → 동 우회로)
    "arm_j":     (WARM1 + 22, RC[1], "priority"),          # 서쪽 진입로 갈래 (서 → 남 우회로) · 서쪽 진출로 합류 (북 → 서 우회로)
    "lot_in":    (-170.0, RC[1] + 30, "priority"),  # 주차장 들어감 (목적지)
    "lot_out":   (-200.0, RC[1] + 30, "priority"),  # 주차장 나옴 (출발지)
}

# ---------------------------------------------------------------- 엣지
# 필드: id, from, to, lanes, speed, shape, spread, name, lane_attrs({index: {attr: value}}), scenarios(None=모두), priority(합류 우선순위, None=기본)
BUS = {"allow": "bus"}
# 차로 변경 금지 (긴급차만) — 신반포사거리 출구의 횡단보도 구간(짧은 출구 도로 전체). 사거리를 빠져나가자마자 횡단보도 위에서
# 차로를 바꾼 차가 옮겨 간 차로로 건너던 보행자와 충돌했다 (11/13 장시간 점검, SUMO는 원래 차로의 보행자만 확인).
# 실제로도 교차로 · 횡단보도 직후는 차로 변경 금지(백색 실선) 구간
NO_LC = {"changeLeft": "emergency authority", "changeRight": "emergency authority"}
EDGES = []
def E(eid, fr, to, lanes, speed, shape, spread="right", name="", lane_attrs=None, scenarios=None, priority=None):
    EDGES.append(dict(id=eid, frm=fr, to=to, lanes=lanes, speed=speed, shape=shape, spread=spread,
                      name=name, lane_attrs=lane_attrs or {}, scenarios=scenarios, priority=priority))

# 반포대로 남쪽 · 고가
E("bp_nb", "bp_in", "split_nb", 6, V_MAIN, [(0, Y_ENTRY), (0, Y_SPLIT)], name="반포대로 북행")
E("bp_sb", "merge_sb", "bp_out", 5, V_MAIN, [(0, Y_SPLIT), (0, Y_ENTRY)], name="반포대로 남행")
E("goga_nb", "split_nb", "merge_nb", 2, V_MAIN, [(0, Y_SPLIT), (0, Y_MERGE)], name="강남터미널고가 북행")
E("goga_sb_ramp", "gsb_d", "gsb_j", 2, V_MAIN, path((-17.6, GSB_D), (-3.2, GSB_J + 2), (-3.2, GSB_J)), "center", "강남터미널고가 남행 진입")
E("goga_sb", "gsb_j", "merge_sb", 2, V_MAIN, [(0, GSB_J), (0, Y_SPLIT)], name="강남터미널고가 남행")
# 측도 북행 · 신반포사거리 북쪽
# 측도 북행: 분기에서 4차로(맨 오른쪽 = 우회전 차로) → 사거리 50 m 앞(a4s)에서 우회전 도로가 갈라지고 직진·좌회전 3차로 + 교통섬 쪽 보도.
# (11/13까지는 고가 분기에서부터 우회전 도로가 따로 230 m를 가고 그 사이에 보도가 있어 처음부터 갈라진 길로 보였음)
E("side_a4u", "split_nb", "a4s", 4, V_SIDE, path((6.4, Y_SPLIT), (16, Y_SPLIT + 30), (16, Y_BOX0 - 50)), name="측도 북행 (A4)")
E("side_a4", "a4s", "ix", 3, V_SIDE, [(16, Y_BOX0 - 50), (16, Y_BOX0)], name="측도 북행 (A4)")
# 우회전 도로 3곳은 사거리 갈래의 보도 바깥에 0.3 m 띄워 지난다 (보도 = 교통섬 쪽 보도). 보도를 붙이기 전 자리(차로에 바로 붙음)는
# 보도와 겹쳐 차와 보행자가 같은 자리를 지나는 것처럼 보였다 (11/13). 합류 · 분기 교차로 안에서 바깥 차로로 옮겨 간다.
E("slip_a4", "a4s", "e2m", 1, V_LOW, path((27.2, Y_BOX0 - 50), (29.9, Y_BOX0 - 38), (29.9, Y_BOX0 - 21),
                                              arc(51.9, Y_BOX0 - 21, 22.0, 180, 90), (60, Y_MED_S - 3.5 * W - 5.8)), "center", "A4 우회전 도로")
E("side_a8a", "ix", "a8m", 3, V_SIDE, [(16, Y_BOX1), (16, Y_BOX1 + 22)], name="측도 북행 (A8)", lane_attrs={i: NO_LC for i in range(3)})
E("side_a8b", "a8m", "a8drop", 3, V_SIDE, [(16, Y_BOX1 + 22), (16, A8DROP)], name="측도 북행 (A8)")
E("side_a8c", "a8drop", "merge_nb", 2, V_SIDE, path((16, A8DROP), (16, A8DROP + 20), (6.4, Y_MERGE - 8), (6.4, Y_MERGE)), name="측도 북행 (A8)")
# 신반포로 (중앙 버스전용 = 각 방향 가장 왼쪽 차로)
E("sbr_a6", "sbr_w_in", "ix", 5, V_SIDE, [(-27 - L_SBR, Y_MED_S), (-27, Y_MED_S)], name="신반포로 동행 (A6)", lane_attrs={4: BUS})
E("sbr_e2a", "ix", "e2m", 4, V_SIDE, [(27, Y_MED_S), (60, Y_MED_S)], name="신반포로 동행 (E2)",
  lane_attrs={0: NO_LC, 1: NO_LC, 2: NO_LC, 3: {**BUS, **NO_LC}})
E("sbr_e2b", "e2m", "sbr_e_out", 4, V_SIDE, [(60, Y_MED_S), (27 + L_SBR, Y_MED_S)], name="신반포로 동행 (E2)", lane_attrs={3: BUS})
E("sbr_a7a", "sbr_e_in", "a7d", 4, V_SIDE, [(27 + L_SBR, Y_MED_N), (60, Y_MED_N)], name="신반포로 서행 (A7)", lane_attrs={3: BUS})
E("sbr_a7b", "a7d", "ix", 3, V_SIDE, [(60, Y_MED_N), (27, Y_MED_N)], name="신반포로 서행 (A7)", lane_attrs={2: BUS})
E("slip_a7", "a7d", "a8m", 1, V_LOW, path((60, Y_MED_N + 3.5 * W + 2.8), (41, Y_MED_N + 3.5 * W + 2.8),
                                         arc(41, Y_MED_N + 3.5 * W + 13.8, 11, 270, 180), (30, Y_BOX1 + 22)), "center", "A7 우회전 도로")
E("sbr_e1a", "ix", "e1m", 3, V_SIDE, [(-27, Y_MED_N), (-55, Y_MED_N)], name="신반포로 서행 (E1)",
  lane_attrs={0: NO_LC, 1: NO_LC, 2: {**BUS, **NO_LC}})
E("sbr_e1b", "e1m", "sbr_w_out", 4, V_SIDE, [(-55, Y_MED_N), (-27 - L_SBR, Y_MED_N)], name="신반포로 서행 (E1)", lane_attrs={3: BUS})
# 측도 남행
E("sb_up", "sb_in", "gsb_d", 5, V_SIDE, [(X_SB, Y_EB - 25), (X_SB, GSB_D)], name="측도 남행 (반포대교에서)")   # 지하차도와 4.8 m 띄움
E("sb_1", "gsb_d", "pocket", 3, V_SIDE, path((X_SB - 2 * W, GSB_D), (X_SB - 2 * W, GSB_D - 15), (-16, GSB_D - 45), (-16, POCKET)), name="측도 남행")   # 3개 차로가 포켓 구간(sb_2)의 0 · 1 · 2번 차로와 일직선
# 측도 남행 포켓 구간 (11/12 로드뷰 — 본선 0: 직진·우회전, 1: 직진, 2 · 3: 좌회전, 정지선 직전에서 가장 왼쪽 좌회전 차로가
# 둘로 갈라져 정지선(sb_a5)에서는 직진 2 + 좌회전 3). 모델은 포켓 구간을 처음부터 5차로로 두고(3 · 4번은 포켓에서 새로 생김)
# a5d에서 5 → 5 일직선으로 잇는다 — 짧은 교차점(8 m) 안에서 한 차로가 둘로 부채꼴로 갈라지면 갈라지는 두 차가
# SUMO 안에서 옆면 충돌한다 (11/13 검증: 143번 버스 ↔ 승용차, §11 #29와 같은 유형). 갈라지는 위치가 정지선 약 68 m 앞으로 당겨진 단순화.
# 교량 · 유턴 차로로 차로 수가 잠깐씩 바뀌는 구간도 단순화
E("sb_2", "pocket", "a5d", 5, V_SIDE, [(-9.6, POCKET), (-9.6, A5D)], name="측도 남행 (A5 포켓)")
E("sb_a5", "a5d", "ix", 5, V_SIDE, [(-9.6, A5D), (-9.6, Y_BOX1)], name="측도 남행 (A5)")
# 우회전 도로는 직진 차로(−24)에서 6 m 바깥(−30, 보도 바깥). 4.8 m 이상 띄워야 한다 — 3.2 m(한 차로 폭)면 8 m 교차점 안에서 꺾이는 12 m 버스 차체가
# 정지선 앞(sb_a5, 13 m)에 꽉 차게 선 직진 버스 뒤끝에 닿았다 (11/13 장시간 점검, 직진·우회전 공용 차로에서 갈라질 때)
E("slip_a5", "a5d", "e1m", 1, V_LOW, path((-30.0, A5D), (-30.0, Y_BOX1 + 10), arc(-42.0, Y_BOX1 + 10, 12, 0, -90),
                                         (-55, Y_MED_N + 3.5 * W + 2.8)), "center", "A5 우회전 도로")
E("sb_e3", "ix", "merge_sb", 3, V_SIDE, path((-16, Y_BOX0), (-16, Y_SPLIT + 30), (-6.4, Y_SPLIT)), name="측도 남행 (E3)")
# 합류 ~ 반포대교 (m1의 3번 차로 → ① 차로 변경 금지 · ① 차로 → 오른쪽 변경 금지)
E("m1", "merge_nb", "sep", 5, V_MAIN, [(-W, Y_MERGE), (-W, Y_SEP)], name="반포대로 북행 (합류)",
  lane_attrs={4: {"changeRight": "emergency authority"}, 3: {"changeLeft": "emergency authority"}})
E("m2", "sep", "drop", 4, V_MAIN, [(0, Y_SEP), (0, Y_DROP)], name="반포대로 북행")
E("m3a", "drop", "xw", 3, V_MAIN, [(0, Y_DROP), (0, Y_XW)], name="반포대로 북행")
E("m3", "xw", "r2d", 3, V_MAIN, [(0, Y_XW), (0, Y_R2)], name="반포대로 북행")
E("m4a", "r2d", "s4032", 3, V_MAIN, [(0, Y_R2), (0, Y_S4032)], name="반포대로 북행 (반포대교남단 신호 앞)")
E("m4", "s4032", "r3m", 3, V_MAIN, [(0, Y_S4032), (0, Y_R3J)], name="반포대로 북행")
# 북행 횡단보도(24032) 왼쪽 보도: 일방통행 도로를 건너는 횡단보도는 양쪽에 보도가 있어야 만들어져, 반대 방향 보행자 전용 보도를 붙인다
E("xw_pw_s", "xw", "xwp_s", 1, V_LOW, [(0, Y_XW), (0, Y_XW - 25)], name="보도", lane_attrs={0: {"allow": "pedestrian", "width": 2.5}})
E("xw_pw_n", "xwp_n", "xw", 1, V_LOW, [(0, Y_XW + 25), (0, Y_XW)], name="보도", lane_attrs={0: {"allow": "pedestrian", "width": 2.5}})
E("xw_pe_s", "xwpe_s", "xw", 1, V_LOW, [(9.6, Y_XW - 25), (9.6, Y_XW)], name="보도", lane_attrs={0: {"allow": "pedestrian", "width": 2.5}})
E("xw_pe_n", "xw", "xwpe_n", 1, V_LOW, [(9.6, Y_XW), (9.6, Y_XW + 25)], name="보도", lane_attrs={0: {"allow": "pedestrian", "width": 2.5}})
E("m5", "r3m", "br", 4, V_MAIN, [(0, Y_R3J), (0, Y_BR)], name="반포대교 (③ 합류 구간)")
E("bridge", "br", "br_out", 3, V_MAIN, [(0, Y_BR), (0, Y_END)], name="반포대교 상층 북행")
E("r2", "r2d", "ebm", 1, V_RAMP, path((8, Y_R2), arc(8 + R2_R, Y_R2, R2_R, 180, 90, 16), (X_EB5, Y_EB - 4.5 * W)), "center", "② 램프")
E("r3", "wbr3", "r3m", 1, V_RAMP, path((X_R3, Y_WB + 1.6), (X_R3 - 12, Y_WB + 1.6), (X_R3 - 150, Y_R3), (19.2, Y_R3),
                                     arc(19.2, Y_R3J, 8, 270, 180)), "center", "③ 램프")   # 왼쪽 진출 → 중앙녹지를 따라 내려옴
# 올림픽대로
E("eb_a", "eb_in", "ebm", 4, V_OLY, [(X_W, Y_EB), (X_EB5, Y_EB)], name="올림픽대로 하남 방향")
E("eb_b", "ebm", "eb_out", 5, V_OLY, [(X_EB5, Y_EB), (X_WB_IN, Y_EB)], name="올림픽대로 하남 방향")
E("wb_a", "wb_in", "wbr3", 5, V_OLY, [(X_WB_IN, Y_WB), (X_R3, Y_WB)], name="올림픽대로 김포 방향")
E("wb_b", "wbr3", "wblm", 4, V_OLY, [(X_R3, WB4), (X_LOOP, WB4)], name="올림픽대로 김포 방향")
E("wb_c", "wblm", "wbpk", 5, V_OLY, [(X_LOOP, WB4), (X_PARK, WB4)], name="올림픽대로 김포 방향 (엮임)")
E("wb_d", "wbpk", "wb_out", 4, V_OLY, [(X_PARK, WB4), (X_W, WB4)], name="올림픽대로 김포 방향")
# 지하차도 (양방향 1/1)
_und1 = path((-1.6, Y_SEP), (-6.4, Y_SEP + 10), (-6.4, RC[1] - RR - 13))                      # 지하차도 (올림픽대로 김포 방향 북쪽까지)
# 갈래 → 회전교차로: 지하차도 직선(x = −6.4)이 회전교차로 중심에서 RB_SE 방향으로 뻗은 직선과 만나는 점(UND_J)에서 꺾여
# 중심을 향해 정면으로 들어간다 (11/14 — 그 전에는 동쪽에서 휘어 들어와 회전 방향과 반대로 진입, 진입 내부 차로가 U턴 모양이었음)
_und2 = path((-6.4, UND_J), NODES["rb_se"][:2])
E("und_n", "sep", "und_j", 1, V_LOW, path(_und1, (-6.4, UND_J)), name="잠수교 지하차도 북행")
E("und_n2", "und_j", "rb_se", 1, V_LOW, _und2, name="잠수교 지하차도 북행")
# 지하차도 남행은 반포대로 가운데로 올라와 강남터미널고가 남행 왼쪽 차로로 합류 (합류점에서 지하차도 남행이 우선)
E("und_s1", "rb_se", "und_m", 1, V_LOW, path(NODES["rb_se"][:2], fillet(NODES["rb_se"][:2], (-6.4, UND_J), (-6.4, RC[1] - RR - 13.8), 3),
                                            (-6.4, RC[1] - RR - 13.8)), name="잠수교 지하차도 남행")
E("und_s", "und_m", "gsb_j", 1, V_LOW, path((-6.4, RC[1] - RR - 13.8), _und1[::-1][1:-1], (-3.2, Y_MERGE + 4), (-3.2, GSB_J + 12), (0, GSB_J)),
  name="잠수교 지하차도 남행", priority=10)   # 합류점에서 우선 (고가 남행 진입로 왼쪽 차로가 양보) — 양보하게 두면 회전교차로까지 막힘
# 회전교차로 (반시계 방향 1차로)
E("ring_se", "rb_se", "rb_e", 1, V_LOW, arc(RC[0], RC[1], RR, RB_SE, 360 + RB_E, 6), "center", "회전교차로")
E("ring_en", "rb_e", "rb_n", 1, V_LOW, arc(RC[0], RC[1], RR, RB_E, 90, 10), "center", "회전교차로")
E("ring_nw", "rb_n", "rb_w", 1, V_LOW, arc(RC[0], RC[1], RR, 90, 180), "center", "회전교차로")
E("ring_ws", "rb_w", "rb_se", 1, V_LOW, arc(RC[0], RC[1], RR, 180, RB_SE, 18), "center", "회전교차로")
E("loop1", "rb_e", "lp_m", 1, V_LOW, path(NODES["rb_e"][:2], (RC[0] + RR + 6, RC[1]), (2.0, RC[1])), "center", "루프 (→ 김포 방향)")
E("loop", "lp_m", "wblm", 1, V_LOW, path((2.0, RC[1]), (X_LOOP_TURN, RC[1]), arc(X_LOOP_TURN, (RC[1] + WB5C) / 2, (RC[1] - WB5C) / 2, 90, -90, 20),
                                       (X_LOOP, WB5C)), "center", "루프 (→ 김포 방향)")
_JY0, _JY1, _JY2 = RC[1] + RR, RC[1] + RR + JX1_D, RC[1] + RR + JX2_D   # 회전교차로 북쪽 끝 · 4616 · 24616
E("jam_n1", "rb_n", "jx1", 1, V_LOW, [(RC[0], _JY0), (RC[0], _JY1)], name="잠수교 북행", scenarios=("before",))
E("jam_n2", "jx1", "jx2", 1, V_LOW, [(RC[0], _JY1), (RC[0], _JY2)], name="잠수교 북행", scenarios=("before",))
E("jam_n", "jx2", "jam_out", 1, V_LOW, [(RC[0], _JY2), (RC[0], Y_END)], name="잠수교 북행", scenarios=("before",))
# 잠수교 남행 진입은 jx2에서 116 m 북쪽(지도 끝) — 16 m였을 때 jx2 횡단보도 신호(주기 75 s) 대기열이 투입 지점까지 차서
# 출발하는 차 바로 뒤에 새 차가 투입돼 추돌했다 (11/13 장시간 점검). 북행 끝도 같은 높이(Y_END)
E("jam_s", "jam_in", "jx2", 1, V_LOW, [(RC[0], Y_END), (RC[0], _JY2)], name="잠수교 남행", scenarios=("before",))
E("jam_s2", "jx2", "jx1", 1, V_LOW, [(RC[0], _JY2), (RC[0], _JY1)], name="잠수교 남행", scenarios=("before",))
E("jam_s1", "jx1", "rb_n", 1, V_LOW, [(RC[0], _JY1), (RC[0], _JY0)], name="잠수교 남행", scenarios=("before",))
E("arm_in1", "arm_w", "arm_j", 2, V_LOW, [(WARM1, RC[1]), (WARM1 + 22, RC[1])], name="회전교차로 서쪽 진입")
E("arm_in", "arm_j", "rb_w", 2, V_LOW, [(WARM1 + 22, RC[1]), (RC[0] - RR, RC[1])], name="회전교차로 서쪽 진입")
E("arm_out1", "rb_w", "arm_j", 2, V_LOW, [(RC[0] - RR, RC[1]), (WARM1 + 22, RC[1])], name="회전교차로 서쪽 진출")
E("arm_out", "arm_j", "arm_w", 2, V_LOW, [(WARM1 + 22, RC[1]), (WARM1, RC[1])], name="회전교차로 서쪽 진출")
# 우회전 우회로 (회전교차로에 들어가지 않고 바로 옆 출구로)
RB_BYP = RR + 4.0                     # 우회로 중심 반경 (회전차로 바깥 가장자리와 0.8 m)
# 지하차도 → 루프: 지하차도가 꺾이는 점(und_j)에서 갈라져 북쪽으로 곧장 루프 합류(lp_m)로 간다. 회전교차로에 붙여 돌리면
# und_j 교차로가 지하차도 곡선을 삼켜 갈고리 모양이 됐다 (11/13 시도, 당시 지하차도는 동쪽에서 휘어 들어왔음).
E("byp_se", "und_j", "lp_m", 1, V_LOW, path((-4.8, UND_J), fillet((-4.8, UND_J), (-4.8, RC[1]), (2.0, RC[1]), 6), (2.0, RC[1])),
  "center", "회전교차로 우회로 (지하차도 → 루프)")
# 서쪽 · 북쪽 우회로는 회전차로 바깥에 0.8 m 띄워(RB_BYP = 회전차로 중심 반경 + 4 m) 나란히 돈다 — 회전교차로가 2차로처럼 보이게 (11/13 팀 결정).
#   서쪽 진입로 옆(0.6 m) → 회전교차로 남서쪽 → 지하차도 남행 바로 밑 → und_m 합류 / 잠수교 남행 옆 → 북서쪽 → 서쪽 진출로 바깥(0.4 m) → arm_j 합류
E("byp_ws", "arm_j", "und_m", 1, V_LOW, smooth(path((WARM1 + 22, RC[1] - 4.8), (WARM1 + 28, RC[1] - 8.4), (RC[0] - 23, RC[1] - 8.4),
                                                  arc(RC[0], RC[1], RB_BYP, 204, 252, 6), (RC[0], RC[1] - RR - 5.4), (-28.0, RC[1] - RR - 5.8),
                                                  (-21.0, RC[1] - RR - 7.0), (-16.5, RC[1] - RR - 10.5), (-14.2, RC[1] - RR - 14.5),
                                                  (-12.6, RC[1] - RR - 18.0)), 3),   # und_m 교차로 모양 안으로 들어가면 netconvert가 중간 형상을 잘라 버림
  "center", "회전교차로 우회로 (서쪽 → 지하차도 남행)")
E("byp_nw", "jx1", "arm_j", 1, V_LOW, smooth(path((RC[0] - 1.6, RC[1] + RR + JX1_D), (RC[0] - 5.1, RC[1] + RR + JX1_D - 6),
                                                  arc(RC[0], RC[1], RB_BYP, 104, 148, 6), (RC[0] - 20.5, RC[1] + 8.4), (WARM1 + 22, RC[1] + 8.4)), 3),
  "center", "회전교차로 우회로 (잠수교 남행 → 서쪽)", scenarios=("before",))
E("park_ramp", "wbpk", "arm_w", 1, V_LOW, _park_path(X_HP), "center", "주차장 램프")
E("lot_exit", "lot_out", "arm_w", 1, V_LOW, path((-200, RC[1] + 30), fillet((-200, RC[1] + 30), (-200, RC[1] + 15), (WARM1 - 25, RC[1] - 1.6), 10), (WARM1 - 25, RC[1] - 1.6), (WARM1, RC[1] - 1.6)), "center", "주차장 출구")
E("lot_entry", "arm_w", "lot_in", 2, V_LOW, path((WARM1, RC[1] + 3.2), (WARM1 - 30, RC[1] + 3.2), fillet((WARM1 - 30, RC[1] + 3.2), (-170, RC[1] + 18), (-170, RC[1] + 30), 8), (-170, RC[1] + 30)), "center", "주차장 진입")

# ---------------------------------------------------------------- 신반포사거리 신호 — 옛 추정 (11/12, 기록용 · 도로망에 쓰지 않음)
# 도로망 신호는 아래 SIGNALS(실측)다. 이 SIGNAL_PLAN은 실측 자료를 찾기 전 추정이며, calibrate.py 3절(Webster 기록)과
# 4절(포화교통류율 sat_* · 좌회전 이동류 left_movements)만 읽는다. 아래는 당시 추정 근거:
# 서울시 운영 방식을 따른 4현시 + 실측 교통량(TOPIS 2026-09 평일 낮)으로 계산한 추정값. 근거 계산: environment/road/calibrate.py
#   - 현시: 중앙버스전용차로 교차로는 버스 직진과 일반 좌회전을 분리 (서울시 2007 자료), 측도는 직좌 공용 차로가
#     있어 방향별 분리 현시. 우회전 도로 3곳은 비신호(양보).
#   - 황색 3 s + 전적색 2 s (서울시: 차량 신호 뒤 1~2 s 전적색), 손실시간 현시당 5 s, 포화교통류율 직진 1,800 · 좌회전 1,600 대/시/차로
#   - 보행 최소 녹색 = 7 s + 횡단 길이 m당 1 s (서울시): φ1 측도 6차로 19 m → 26 s, φ3 신반포로 동쪽 27 m → 34 s,
#     φ4 신반포로 서쪽 30 m → 37 s (보행자는 모델에 없지만 실제 주기를 정하는 제약)
#   - 녹색은 임계 교통량 비율로 배분하고 보행 최소 녹색을 보장. 주기는 모든 현시 포화도가 0.9 이하가 되는 가장 짧은
#     주기(10 s 단위) → **170 s** (녹색 56 · 23 · 34 · 37 s, 현시마다 황색 3 + 전적색 2, 포화도 0.69~0.85)
SIGNAL_PLAN = dict(
    yellow=3, all_red=2, lost_per_phase=5, sat_through=1800, sat_left=1600, cycle_bounds=(60, 180),
    target_cycle=160, x_target=0.9,   # 신반포로 진입량을 정할 때 쓰는 목표 주기·포화도 (서울 대형 교차로 낮 시간대 수준 — 가정)
    left_movements=[("sbr_a6", "side_a8a"), ("side_a4", "sbr_e1a"), ("sb_a5", "sbr_e2a")],
    phases=[
        dict(name="φ1 신반포로 직진 (버스 포함)", green=56, min_green=26,
             movements=[("sbr_a6", "sbr_e2a"), ("sbr_a7b", "sbr_e1a"), ("sbr_a6", "sb_e3")]),
        dict(name="φ2 신반포로 동행 좌회전", green=23, min_green=10,
             movements=[("sbr_a6", "side_a8a"), ("sbr_a6", "sb_e3")]),
        dict(name="φ3 측도 북행 직진·좌회전", green=34, min_green=34,
             movements=[("side_a4", "side_a8a"), ("side_a4", "sbr_e1a")]),
        dict(name="φ4 측도 남행 직진·좌회전", green=37, min_green=37,
             movements=[("sb_a5", "sb_e3"), ("sb_a5", "sbr_e2a")]),
    ],
)

# ---------------------------------------------------------------- 신호 — 실측 (11/12, 서울 교통빅데이터플랫폼 C-ITS)
# 원자료: t-data.seoul.go.kr 파일 데이터 10384 "신호제어기 신호 정보"(전일, 1초 간격, 서울 전역 C-ITS 교차로)와
#         10144 "교차로 MAP"(교차로 ID · 이름 · 좌표). 평일 10~16시, 통제 전 10/21 · 통제 후 11/11을 분석했다
#         (environment/road/measured/README.md). 신반포사거리 · 24032 · 4032는 두 날의 운영이 같다.
# 시각은 주기 안 위치(s)이고, 기준 0 = 신반포사거리 동행 직진 녹색 시작. 주기 170 s인 세 신호는 이 기준으로 맞물려(연동) 돈다.
# 이동류: (from, to, fromLane) — fromLane이 None이면 그 두 엣지 사이의 모든 연결.
# groups: (이름, 녹색 시작, 녹색 길이, 황색 길이, 이동류 목록, 양보 여부). 양보=True면 녹색이 'g'(횡단보도 보행자 등에 양보).
# crossings: (이름, 건너는 엣지 목록, 보행 녹색 시작, 보행 녹색 길이) — 보행 녹색은 "걸어도 됨"(점멸 전)만. 점멸 동안은 새로 건너지 않는다.
# SIGNAL_PLAN(위, 추정)은 비교 기록용으로 남긴다 (주기 170 s는 실측과 같았고, 현시 구성 · 순서 · 녹색 시간은 달랐다).
SIGNALS = {
    "ix": dict(name="신반포사거리 (C-ITS 4031 강남터미널고가밑)", cycle=170, offset=0, groups=[
        ("신반포로 서행 직진", 0, 44, 4, [("sbr_a7b", "sbr_e1a", None)], False),
        ("신반포로 동행 버스 직진", 0, 44, 4, [("sbr_a6", "sbr_e2a", 4)], False),
        ("신반포로 동행 직진", 0, 85, 4, [("sbr_a6", "sbr_e2a", 1), ("sbr_a6", "sbr_e2a", 2)], False),
        # 우회전은 남측 횡단보도 보행 신호(점멸 포함 3–43 s) 동안 멈춘다 (도로교통법: 보행 신호 중 우회전 차량 일시정지) → 44–84 s만 양보 녹색
        ("신반포로 동행 우회전", 44, 41, 4, [("sbr_a6", "sb_e3", None)], True),
        ("신반포로 동행 좌회전", 49, 36, 4, [("sbr_a6", "side_a8a", None)], False),
        ("측도 남행 직진·좌회전", 90, 40, 3, [("sb_a5", "sb_e3", None), ("sb_a5", "sbr_e2a", None)], False),
        ("측도 북행 직진·좌회전", 134, 32, 3, [("side_a4", "side_a8a", None), ("side_a4", "sbr_e1a", None)], False),
    ], crossings=[
        ("북측 횡단보도", ["side_a8a", "sb_a5"], 3, 13),
        ("남측 횡단보도", ["side_a4", "sb_e3"], 3, 14),
        ("서측 횡단보도", ["sbr_a6", "sbr_e1a"], 92, 11),
        ("동측 횡단보도", ["sbr_a7b", "sbr_e2a"], 137, 10),
    ]),
    "xw": dict(name="반포대로 북행 횡단보도 (C-ITS 24032 반포대교남단(연등)1)", cycle=170, offset=0, groups=[
        ("북행", 54, 143, 3, [("m3a", "m3", None)], False),
    ], crossings=[("횡단보도", ["m3"], 33, 13)]),
    # 4032: 남행 → 올림픽대로 하남 방향 좌회전 (28–74 s)은 모델에 없는 이동류라 그 시간은 북행 적색으로만 비워 둔다 (11/12 팀 결정)
    "s4032": dict(name="반포대교남단 (C-ITS 4032)", cycle=170, offset=0, groups=[
        ("북행 직진", 78, 117, 3, [("m4a", "m4", None)], False),
    ], crossings=[]),
    # 잠수교 두 신호는 주기 75 s로 사거리와 연동되지 않는다 (둘 사이 시차는 기록이 부족해 0으로 둠). 보행화 후에는 점멸 → 끔
    "jx1": dict(name="잠수교남단 (C-ITS 4616)", cycle=75, offset=0, scenarios=("before",), groups=[
        ("잠수교 양방향", 0, 40, 3, [("jam_n1", "jam_n2", None), ("jam_s2", "jam_s1", None), ("jam_s2", "byp_nw", None)], False),
    ], crossings=[("횡단보도", ["jam_n2", "jam_s2"], 45, 10)]),
    "jx2": dict(name="잠수교 (C-ITS 24616 잠수교남단(연등))", cycle=75, offset=0, scenarios=("before",), groups=[
        ("잠수교 양방향", 0, 46, 3, [("jam_n2", "jam_n", None), ("jam_s", "jam_s2", None)], False),
    ], crossings=[("횡단보도", ["jam_n", "jam_s"], 51, 13)]),
}


# 보도(보행자 차로)를 붙인 엣지: netconvert가 0번에 보도를 넣어 차량 차로 번호가 1씩 밀린다.
# 이 파일의 차로 번호(conn · lane_attrs · SIGNALS · 버스 정류소)는 모두 "차량 차로" 기준이다. conn · SIGNALS · 정류소는 build_net ·
# road_builder가 net_lane()으로 바꾸고, lane_attrs(<lane index>)는 netconvert가 보도를 넣으며 스스로 밀어 준다.
SIDEWALKS = {
    # 신반포사거리 네 갈래 (횡단보도 4곳)
    "sbr_a6", "sbr_e1a", "sbr_a7b", "sbr_e2a", "side_a4", "sb_e3", "side_a8a", "sb_a5",
    # 반포대로 북행 횡단보도 (24032) — 양쪽 모두 보행자 전용 보도 xw_pw_* · xw_pe_* (25 m)
    # 잠수교 (횡단보도 4616 · 24616, 보행화 후에는 보행전용교)
    "jam_n2", "jam_n", "jam_s", "jam_s2",   # 보행자 경로는 횡단보도 북쪽에서만. 횡단보도 ~ 회전교차로 20 m(jam_n1 · jam_s1)에는 보도 없음
                                            # — 남행 우회로가 바로 옆으로 붙어 갈라지는 자리 (11/13)
}
SIDEWALK_WIDTH = 2.5

# 보행자 (11/12) — 실측 보행량 자료가 없어 추정값 (C-ITS "횡단 보행자 정보" 파일은 2024년 표본 1,000건뿐이고 이 교차로가 없음).
#   (이름, 출발 엣지, 도착 엣지, 시간당 인원, 시나리오) — 출발 · 도착은 보도가 있는 엣지. SUMO가 횡단보도를 거쳐 걷는다.
#   신반포사거리: 고속터미널 · 신반포역 사이 생활권이라 횡단보도마다 시간당 양방향 200명(방향별 100)으로 가정
#   반포대로 북행 횡단보도: 버스정류소(반포대교남단.한강시민공원입구)와 한강공원 입구 → 100명
#   잠수교: 보행화 전 횡단보도 2곳 각 100명, 보행화 후 보행전용교를 걷는 사람 양방향 400명 (보행화로 늘어난 보행 — 가정)
PEDESTRIANS = [
    ("북측 횡단보도 동→서", "side_a8a", "sb_a5", 100, None), ("북측 횡단보도 서→동", "sb_a5", "side_a8a", 100, None),
    ("남측 횡단보도 동→서", "side_a4", "sb_e3", 100, None), ("남측 횡단보도 서→동", "sb_e3", "side_a4", 100, None),
    ("서측 횡단보도 남→북", "sbr_a6", "sbr_e1a", 100, None), ("서측 횡단보도 북→남", "sbr_e1a", "sbr_a6", 100, None),
    ("동측 횡단보도 북→남", "sbr_a7b", "sbr_e2a", 100, None), ("동측 횡단보도 남→북", "sbr_e2a", "sbr_a7b", 100, None),
    ("반포대로 북행 횡단보도 동→서", "xw_pe_n", "xw_pw_s", 50, None), ("반포대로 북행 횡단보도 서→동", "xw_pw_n", "xw_pe_s", 50, None),
    ("잠수교남단 횡단보도", "jam_n2", "jam_s2", 50, ("before",)), ("잠수교남단 횡단보도 반대", "jam_s2", "jam_n2", 50, ("before",)),
    ("잠수교 횡단보도", "jam_n", "jam_s", 50, ("before",)), ("잠수교 횡단보도 반대", "jam_s", "jam_n", 50, ("before",)),
    ("잠수교 보행 (북쪽으로)", "jam_n2", "jam_n", 200, ("after",)), ("잠수교 보행 (남쪽으로)", "jam_s", "jam_s2", 200, ("after",)),
]
def net_lane(edge, lane):
    """road_spec 차량 차로 번호 → 실제 도로망 차로 번호 (보도가 있으면 +1)."""
    return lane + (1 if edge in SIDEWALKS else 0)


def signal_timeline(tl):
    """SIGNALS[tl] → [(길이 s, {그룹 또는 횡단보도 이름: 'G'|'g'|'y'|'r'})] — 상태가 바뀌는 시각마다 끊은 주기 한 바퀴.
    build_net이 SUMO 현시로 바꾼다 (환경은 SUMO에서 그 현시를 읽어 데이터 칼럼 signal의 남은 시간을 계산한다)."""
    sp = SIGNALS[tl]; cyc = sp["cycle"]
    items = []
    for name, g0, gl, yl, _mv, yield_ in sp["groups"]:
        items.append((name, g0, gl, yl, "g" if yield_ else "G"))
    for name, _edges, g0, gl in sp["crossings"]:
        items.append((name, g0, gl, 0, "G"))
    def state(name, g0, gl, yl, gch, t):
        d = (t - g0) % cyc
        return gch if d < gl else ("y" if d < gl + yl else "r")
    cuts = sorted({c % cyc for _n, g0, gl, yl, _g in items for c in (g0, g0 + gl, g0 + gl + yl)} | {0})
    out = []
    for a, b in zip(cuts, cuts[1:] + [cyc]):
        if b > a:
            out.append((b - a, {n: state(n, g0, gl, yl, gc, a) for n, g0, gl, yl, gc in items}))
    return out


ROUNDABOUT = dict(nodes=["rb_se", "rb_e", "rb_n", "rb_w"], edges=["ring_se", "ring_en", "ring_nw", "ring_ws"])

# ---------------------------------------------------------------- 층 높이 z (m) — 화면 구분용
# 겹치는 도로는 원래부터 노드를 공유하지 않아 서로 간섭하지 않는다 (주행·충돌에는 영향 없음).
# z는 sumo-gui에서 층을 구분해 그리는 값이다 (environment/env/road_builder.py가 높이 대역별로 같은 색을 칠함).
# 실측값이 아니라 층 순서만 실제와 맞춘다:
#   고가(+6: 강남터미널고가·반포대교 상층) > 지상(0: 반포대로·신반포로·올림픽대로 김포 방향)
#   > 한강 둔치 층(−10: 잠수교 지하차도·회전교차로·잠수교·주차장). 올림픽대로는 양방향 모두 지상(11/12: 하남 방향 터널 없앰)
# 지하차도는 둔치 높이로 내려가 그대로 회전교차로·잠수교로 이어지고, 루프·주차장 램프가 둔치 ↔ 김포 방향을 오르내린다.
Z_UP, Z_DEEP = 6.0, -10.0                   # 고가 · 한강 둔치 층(잠수교)
RIVER_EDGES = ("ring_se", "ring_en", "ring_nw", "ring_ws", "jam_n1", "jam_n2", "jam_n", "jam_s", "jam_s2", "jam_s1", "arm_in1", "arm_in",
               "arm_out1", "arm_out", "lot_exit", "lot_entry", "loop1", "byp_se", "byp_ws", "byp_nw")
def _ramp(d, length):
    return max(0.0, min(1.0, d / length))
def _interp(v, pts):                                        # 구간별 직선 보간 (pts: (위치, 높이) 오름차순)
    if v <= pts[0][0]: return pts[0][1]
    for (a, za), (b, zb) in zip(pts, pts[1:]):
        if v <= b: return za + (zb - za) * (v - a) / (b - a)
    return pts[-1][1]
def z_at(eid, x, y):
    if eid in ("goga_nb", "goga_sb"):                       # 강남터미널고가 (사거리 위), 경사 5%
        return Z_UP * _ramp(min(y - Y_SPLIT, GSB_J - y), 120)   # 북행·남행 같은 지점에서 오르내림
    if eid in ("und_n", "und_n2", "und_s1", "und_s"):                           # 잠수교 지하차도: 하남 방향 앞에서 둔치 높이까지 내려가 그대로
        return _interp(y, [(Y_SEP + 20, 0), (Y_EB - 30, Z_DEEP)])   # 진입부 20 m 뒤부터 둔치 높이까지, 5.4%
    if eid in RIVER_EDGES:                                  # 회전교차로 · 잠수교 · 주차장 (둔치)
        return Z_DEEP
    if eid == "loop":                                       # 둔치 → 동쪽으로 가며 올라가(반포대교 밑 통과) 김포 방향 높이로, 5%
        return Z_DEEP * (1 - _ramp(x - (RC[0] + RR + 15), 200)) if y > RC[1] - 0.5 and x <= X_LOOP_TURN else 0.0
    if eid == "park_ramp":                                  # 김포 방향 → 되돌아 나온 뒤 동쪽으로 가며 둔치로 내려감, 3.5%
        return Z_DEEP * _ramp(x - X_HP, (WARM1 - 15) - X_HP) if y > WB5C + 12 and x >= X_HP else 0.0
    if eid in ("m3a", "m3", "xw_pw_s", "xw_pw_n", "xw_pe_s", "xw_pe_n", "m4a", "m4", "m5", "bridge"):   # 반포대교 상층으로 올라감 (김포 방향 위), 4.4%
        return _z_bridge(y)
    if eid == "r2":                                         # ② 램프는 분기 높이에서 하남 방향 높이로 내려감
        return Z_R2 * (1 - _ramp(math.dist((x, y), (8, Y_R2)) - 20, 150))   # 교차로 구간(20 m)은 평평
    if eid == "r3":                                         # ③ 램프도 상층 높이로 올라가 합류, 2%
        return Z_UP * _ramp(300 - x, 280)
    return 0.0
Z_R2 = 4.5                                                  # ② 분기 교차로 높이 (교차로 안은 평평해야 경사가 튀지 않음)
def _z_bridge(y):                                           # 측도 쪽 차로 끝 뒤부터 상승, ② 분기 교차로(약 ±15 m)는 평평, 하남 방향 위에서 상층 높이
    return _interp(y, [(Y_DROP + 10, 0.0), (Y_R2 - 15, Z_R2), (Y_R2 + 25, Z_R2), (Y_EB - 5, Z_UP)])
Z_NODES = {"r2d": Z_R2, "r3m": Z_UP, "br": Z_UP, "br_out": Z_UP,   # 땅 위가 아닌 노드
           **{n: Z_DEEP for n in ("rb_se", "rb_e", "rb_n", "rb_w", "jam_out", "jam_in", "jx1", "jx2", "arm_w", "arm_j", "und_j", "und_m", "lp_m", "lot_in", "lot_out")}}
Z_NODES.update({n: _z_bridge(y) for n, y in (("xw", Y_XW), ("xwp_s", Y_XW - 25), ("xwp_n", Y_XW + 25), ("xwpe_s", Y_XW - 26), ("xwpe_n", Y_XW + 26), ("s4032", Y_S4032))})

# ---------------------------------------------------------------- AV 경로
# 프로젝트 목적: 다리를 타고 강북으로 건너가는 차량의 상호작용, 특히 잠수교 차량 통제 전후의 변화.
# AV는 반포대로 북행(신반포사거리 남쪽) · 올림픽대로 김포 방향 · 신반포로 서행에서 출발해 반포대교 또는 잠수교로 간다.
# 통제 전(before)은 잠수교로 가는 두 경로, 통제 후(after)는 반포대교로 가는 세 경로만 쓴다.
# 이름 → 엣지 순서. scenarios=None이면 두 시나리오 모두. road_builder가 차로 단위로 주행 가능한지 검사한다.
AV_ROUTES = {
    # 측도 → 신반포사거리(신호) 직진 → 고가·측도 합류(측도 쪽 차로 끝) → ② 분기 → ③ 합류 → 반포대교
    "side-banpo": dict(edges=["bp_nb", "side_a4u", "side_a4", "side_a8a", "side_a8b", "side_a8c", "m1", "m2", "m3a", "m3", "m4a", "m4", "m5", "bridge"],
                 scenarios=("after",), desc="측도 → 신반포사거리 → 합류 → 반포대교"),
    # 고가 직진 → 합류(① 지하차도 차로는 지나침) → ② 분기 → ③ 합류 → 반포대교. 잠수교 통제 뒤 잠수교 수요가 몰리는 길
    "goga-banpo": dict(edges=["bp_nb", "goga_nb", "m1", "m2", "m3a", "m3", "m4a", "m4", "m5", "bridge"],
                 scenarios=("after",), desc="고가 직진 → 반포대교"),
    # 신반포로 서행 → 우회전 연결로(A7) → 측도 쪽 차로 합류 → 고가·측도 합류 → ② 분기 → ③ 합류 → 반포대교. 통제 후에만
    "sbr-banpo": dict(edges=["sbr_a7a", "slip_a7", "side_a8b", "side_a8c", "m1", "m2", "m3a", "m3", "m4a", "m4", "m5", "bridge"],
                 # depart_lanes [0, 1] (11/19 저녁, 팀 결정): 2번(가장 왼쪽) 차로 출발은 뺀다. 짧은 거리 안에 오른쪽으로 두 번 옮겨
                 # 우회전해야 하는 조건이라 모델의 비정상 정지가 여기에 몰렸다 (PPO L2 500판, 조기 종료 끔: 2번 차로 39판 중 정지 7 —
                 # 전체 정지 7판의 전부. 선생님도 16판 중 1판, 무작위 200판 중 2판을 같은 자리에서 선다). SUMO 운전자는 이 경로를
                 # 100판 모두 도착하므로 "모델 · 선생님의 한계로 뺀 출발 조건"으로 발표 · 데이터 설명에 밝힌다.
                 scenarios=("after",), depart_lanes=[0, 1], desc="신반포로 → 우회전 연결로 → 합류 → 반포대교"),
    # 고가 왼쪽 차로 → ① 지하차도 → 회전교차로(양보) → 잠수교. 보행화 전에만
    "goga-jamsu": dict(edges=["bp_nb", "goga_nb", "m1", "und_n", "und_n2", "ring_se", "ring_en", "jam_n1", "jam_n2", "jam_n"],
                  scenarios=("before",), desc="고가 → ① 지하차도 → 회전교차로 → 잠수교"),
    # 올림픽대로 김포 방향 → 주차장 램프 → 회전교차로 서쪽 진입(양보) → 잠수교. 보행화 전에만
    # depart_lanes (선택 필드, 없으면 출발 엣지 승용 차로 전부): 환경 reset의 depart_lane="random"이 뽑는 출발 차로 범위.
    #   11/19 선생님(check_env.rule_policy 4차) 차로별 평가 16판: 16판 중 14판 이하인 차로는 뺀다 — oly-jamsu 3번 11/16 · 4번 13/16
    #   (나머지 경로 · 차로는 모두 15~16판)
    "oly-jamsu": dict(edges=["wb_a", "wb_b", "wb_c", "park_ramp", "arm_in1", "arm_in", "ring_ws", "ring_se", "ring_en", "jam_n1", "jam_n2", "jam_n"],
                  scenarios=("before",), depart_lanes=[0, 1, 2], desc="올림픽대로 → 주차장 램프 → 회전교차로 → 잠수교"),
}
# 보행화 후 차량 통제되는 엣지 (도로는 남아 있고 차만 못 다님 — 보행전용교)
CLOSED_AFTER = [e["id"] for e in EDGES if e["scenarios"] and "after" not in e["scenarios"]]

# ---------------------------------------------------------------- 차로 단위 연결 (from, fromLane, to, toLane)
C = []
def conn(fr, to, pairs):
    for a, b in pairs:
        C.append((fr, a, to, b))

# 고가·측도 분기: 북행 6 = [5,4 → 고가] [3..0 → 측도 side_a4u] (0 = 우회전 차로, 사거리 50 m 앞 a4s에서 slip_a4로)
conn("bp_nb", "goga_nb", [(5, 1), (4, 0)]); conn("bp_nb", "side_a4u", [(3, 3), (2, 2), (1, 1), (0, 0)])
conn("side_a4u", "side_a4", [(3, 2), (2, 1), (1, 0)]); conn("side_a4u", "slip_a4", [(0, 0)])
# 신반포사거리 (차로 구성표)
conn("side_a4", "sbr_e1a", [(2, 1), (1, 0)])                 # A4 좌·직좌 → 서쪽
conn("side_a4", "side_a8a", [(1, 2), (0, 1)])                # A4 직좌·직 → 북쪽
conn("sb_a5", "sbr_e2a", [(4, 2), (3, 1), (2, 0)])           # A5 좌 3 → 동쪽
conn("sb_a5", "sb_e3", [(1, 2), (0, 1)])                     # A5 직 2 → 남쪽
conn("sbr_a6", "sbr_e2a", [(4, 3), (2, 2), (1, 1)])          # A6 버스·직 2 → 동쪽
conn("sbr_a6", "side_a8a", [(3, 0)])                         # A6 좌 → 북쪽
conn("sbr_a6", "sb_e3", [(0, 0)])                            # A6 우 → 남쪽
conn("sbr_a7b", "sbr_e1a", [(2, 2), (1, 1), (0, 0)])         # A7 버스·직 2 → 서쪽 (좌회전 없음)
# 우회전 도로 · 신반포로 분기·합류
conn("sbr_e2a", "sbr_e2b", [(i, i) for i in range(4)]); conn("slip_a4", "sbr_e2b", [(0, 0)])
conn("sbr_a7a", "sbr_a7b", [(3, 2), (2, 1), (1, 0)]); conn("sbr_a7a", "slip_a7", [(0, 0)])
conn("side_a8a", "side_a8b", [(i, i) for i in range(3)]); conn("slip_a7", "side_a8b", [(0, 0)])
conn("side_a8b", "side_a8c", [(2, 1), (1, 0), (0, 0)])       # A8 3 → 2
conn("sbr_e1a", "sbr_e1b", [(2, 3), (1, 2), (0, 1)]); conn("slip_a5", "sbr_e1b", [(0, 0)])
# 측도 남행 · 고가 남행
conn("sb_up", "goga_sb_ramp", [(4, 1), (3, 0)]); conn("sb_up", "sb_1", [(2, 2), (1, 1), (0, 0)])
conn("goga_sb_ramp", "goga_sb", [(1, 1), (0, 0)]); conn("und_s", "goga_sb", [(0, 1)])   # 지하차도 남행 → 고가 남행 왼쪽 차로
conn("sb_1", "sb_2", [(2, 2), (1, 1), (0, 0)])   # 일직선 1:1. 포켓 구간에서 왼쪽 좌회전 차로(3 · 4)가 새로 생기고 차로 변경으로 들어간다 (11/13 포켓 5차로로)
# (한 차로를 짧은 교차점 안에서 여러 차로로 부채꼴로 가르거나 옆으로 꺾으면, 앞뒤 · 옆 차가 갈라지는 순간 SUMO 안에서 옆면 충돌이 난다 —
#  11/12 #21 버스 · 승용차, #29 장시간 점검에서 4시간에 약 1건. 일직선 1:1로 바꾼 뒤 24시간 0건, 직진 차도 2개 차로로 나뉘어 대기열이 자라지 않음)
conn("sb_2", "sb_a5", [(4, 4), (3, 3), (2, 2), (1, 1), (0, 0)]); conn("sb_2", "slip_a5", [(0, 0)])   # 일직선 1:1 (정지선 앞 좌회전 3 = 포켓 2 · 3 · 4)
conn("goga_sb", "bp_sb", [(1, 4), (0, 3)]); conn("sb_e3", "bp_sb", [(2, 2), (1, 1), (0, 0)])
# 합류 → ① → 측도 쪽 차로 끝 → ② → ③ → 반포대교
conn("goga_nb", "m1", [(1, 4), (1, 3), (0, 2)])              # 고가 쪽 왼쪽 차로가 둘로 (① 전용 + 직진)
conn("side_a8c", "m1", [(1, 1), (0, 0)])                     # 측도 차로는 ①과 연결 없음
conn("m1", "und_n", [(4, 0)]); conn("m1", "m2", [(3, 3), (2, 2), (1, 1), (0, 0)])
conn("m2", "m3a", [(3, 2), (2, 1), (1, 0), (0, 0)])          # 측도 쪽 바깥 차로 끝
conn("m3a", "m3", [(i, i) for i in range(3)])                # 북행 횡단보도 신호 (24032)
conn("m3", "m4a", [(2, 2), (1, 1), (0, 0)]); conn("m3", "r2", [(0, 0)])   # ② 오른쪽 차로가 둘로
conn("m4a", "m4", [(i, i) for i in range(3)])                # 반포대교남단 신호 (4032)
conn("m4", "m5", [(2, 3), (1, 2), (0, 1)]); conn("r3", "m5", [(0, 0)])
conn("m5", "bridge", [(3, 2), (2, 1), (1, 0), (0, 0)])       # ③ 차로가 오른쪽 차로와 합쳐짐
# 올림픽대로
conn("eb_a", "eb_b", [(3, 4), (2, 3), (1, 2), (0, 1)]); conn("r2", "eb_b", [(0, 0)])
conn("wb_a", "r3", [(4, 0)]); conn("wb_a", "wb_b", [(3, 3), (2, 2), (1, 1), (0, 0)])   # ③ 왼쪽 진출
conn("wb_b", "wb_c", [(3, 4), (2, 3), (1, 2), (0, 1)]); conn("loop", "wb_c", [(0, 0)])
conn("wb_c", "wb_d", [(4, 3), (3, 2), (2, 1), (1, 0)]); conn("wb_c", "park_ramp", [(0, 0)])   # 엮임 → 주차장 진출
# 회전교차로 · 주차장
conn("und_n", "und_n2", [(0, 0)]); conn("und_n", "byp_se", [(0, 0)]); conn("und_s1", "und_s", [(0, 0)]); conn("byp_ws", "und_s", [(0, 0)])
conn("byp_se", "loop", [(0, 0)]); conn("loop1", "loop", [(0, 0)])
conn("und_n2", "ring_se", [(0, 0)]); conn("ring_ws", "ring_se", [(0, 0)]); conn("ring_ws", "und_s1", [(0, 0)])
conn("ring_se", "ring_en", [(0, 0)]); conn("ring_se", "loop1", [(0, 0)])
conn("ring_en", "ring_nw", [(0, 0)])
conn("ring_nw", "ring_ws", [(0, 0)]); conn("ring_nw", "arm_out1", [(0, 0), (0, 1)])
conn("arm_out1", "arm_out", [(1, 1), (0, 0)]); conn("byp_nw", "arm_out", [(0, 0)])
conn("arm_in", "ring_ws", [(1, 0), (0, 0)])                  # 서쪽 진입 2 → 1
conn("arm_in1", "arm_in", [(1, 1), (0, 0)]); conn("arm_in1", "byp_ws", [(0, 0)])
conn("park_ramp", "arm_in1", [(0, 0)]); conn("lot_exit", "arm_in1", [(0, 1)])
conn("arm_out", "lot_entry", [(1, 1), (0, 0)])
# 잠수교 (보행화 전에만 있는 엣지 — environment/road/build/after 검증 도로망에서는 build_net이 거른다. 수업 환경은 before 도로망 하나에
# 보행화 후에는 sumo_env가 실행 중에 차량을 막는다)
conn("jam_n1", "jam_n2", [(0, 0)]); conn("jam_n2", "jam_n", [(0, 0)]); conn("jam_s", "jam_s2", [(0, 0)]); conn("jam_s2", "jam_s1", [(0, 0)])
conn("jam_s2", "byp_nw", [(0, 0)])                            # 잠수교 남행 → 서쪽 우회로 (잠수교남단 신호 뒤)
conn("ring_en", "jam_n1", [(0, 0)]); conn("jam_s1", "ring_nw", [(0, 0)])   # 회전교차로 ↔ 잠수교
CONNECTIONS = C

# ---------------------------------------------------------------- 진입점 · 출구 (경로 검사 · 교통량 단계용)
ENTRIES = {
    "bp_nb": "반포대로 북행 (남쪽)", "sbr_a6": "신반포로 동행 (이수 쪽)", "sbr_a7a": "신반포로 서행 (강남구청 쪽)",
    "sb_up": "남행 측도 (반포대교에서)", "eb_a": "올림픽대로 하남 방향 (김포 쪽에서)", "wb_a": "올림픽대로 김포 방향 (하남 쪽에서)",
    "lot_exit": "반포한강공원 주차장", "jam_s": "잠수교 남행 (보행화 전)",
}
EXITS = {
    "bridge": "반포대교 북행", "bp_sb": "반포대로 남행 (남쪽)", "sbr_e1b": "신반포로 서쪽 (이수)", "sbr_e2b": "신반포로 동쪽 (강남구청)",
    "eb_b": "올림픽대로 하남 방향", "wb_d": "올림픽대로 김포 방향", "lot_entry": "반포한강공원 주차장",
    "jam_n": "잠수교 북행 (보행화 전)",
}

# 확정안 길이 (check_net.py가 도로망의 실제 엣지 길이와 비교)
SPEC_LENGTHS = {
    "반포대로 진입 → 분기": (["bp_nb"], 300),
    "고가 (분기 → 합류)": (["goga_nb"], 580),
    "측도 분기 → 사거리 → 합류 (사거리 내부 포함)": (["side_a4u", "side_a4", "side_a8a", "side_a8b", "side_a8c"], 250 + 330),
    "합류 → ① 진입부": (["m1"], 10),
    "합류 → ②": (["m1", "m2", "m3a", "m3"], 190),
    "합류 → ③ 합류 끝": (["m1", "m2", "m3a", "m3", "m4a", "m4", "m5"], 290),
    "반포대교 (③ 뒤)": (["bridge"], 400),   # 확정안 300 m + 지도 북쪽 끝(Y_END)까지 100 m (11/13)
    "김포 방향 진입 → ③": (["wb_a"], 400),
    "엮임": (["wb_c"], 260),
    "② 램프": (["r2"], 250),
    "③ 램프": (["r3"], 445),
    "루프": (["loop1", "loop"], 500),
    "주차장 램프": (["park_ramp"], 480),
    "지하차도": (["und_n", "und_n2"], 340),
}
