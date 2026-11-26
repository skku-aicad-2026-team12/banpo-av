"""SUMO 도로 파일 자동 생성 : 12팀 반포대교 남단 도로망.

environment/road/road_spec.py(도로 정의)와 environment/env/road_config.py(시나리오 · AV 경로 · 교통 수요 · 버스)를 읽어
environment/env/sumo/<network>/ 에 도로망 · 시나리오별 수요 · 정류소 · 화면 설정을 만들고 sumocfg 경로를 돌려준다.
학습 · 평가 · 수집 · 화면 스크립트가 시작할 때마다 build()를 호출한다 (수업 구조 그대로).
수업 원본의 직선 도로 생성 코드는 11/16 지웠다 : 12팀은 반포 도로망만 쓴다.
"""
import math
import os
import subprocess

from sumolib import checkBinary  # SUMO 실행파일(sumo, netconvert 등) 경로를 찾아주는 유틸
from env.mdp_config import SIMULATION


def build(road: dict, ego: dict, traffic: dict, out_dir: str) -> str:
    """SUMO 파일들을 out_dir/<road["network"]>/ 에 생성하고, sumocfg 파일 경로를 반환한다.

    Args:
        road:    env.road_config.ROAD    (도로망 이름 · 시나리오 · AV 경로)
        ego:     env.road_config.EGO     (AV 차량 속성)
        traffic: env.road_config.TRAFFIC (배경 교통 · 버스)
        out_dir: 생성 위치 (environment/env/sumo)
"""
    if not road.get("network"):
        raise ValueError('ROAD["network"]가 없다 — 12팀 환경은 반포 도로망("banpo")만 지원한다 (environment/env/road_config.py)')
    return _build_network(road, ego, traffic, out_dir)


def _vtypes_block(traffic: dict):
    """배경차 vType(분포) XML 블록과 flow가 쓸 type id를 반환한다."""
    controllers = traffic.get("controllers")
    if controllers:
        # probability 합이 1이 아니어도 되도록 자동 정규화
        total_p = sum(float(c.get("probability", 1.0))
                      for c in controllers.values())
        vtype_lines = []
        for name, c in controllers.items():
            prob = float(c.get("probability", 1.0)) / total_p
            attrs = [
                f'id="car_{name}"',
                f'probability="{prob:.4f}"',
                f'carFollowModel="{c["carFollowModel"]}"',
                f'accel="{c.get("accel", 2.5)}"',
                f'decel="{c.get("decel", 4.5)}"',
                f'tau="{c.get("tau", 1.0)}"',
                f'minGap="{c.get("min_gap", 2.5)}"',
                # 컨트롤러별 희망속도 오버라이드 (없으면 공통 max_speed).
                # 컨트롤러마다 속도를 다르게 주면 차선 간 속도차가 생겨
                # "추월할 이유"가 자연스럽게 만들어진다.
                f'maxSpeed="{c.get("max_speed", traffic["max_speed"])}"',
                # speedDev: 같은 타입 안에서도 개체별 희망속도 편차 (0~)
                f'speedDev="{c.get("speed_dev", 0.1)}"',
                f'color="{c.get("color", "1,1,0")}"',
                'vClass="passenger"', 'guiShape="passenger/sedan"',
            ]
            # sigma는 Krauss 계열에만 의미가 있으므로 지정된 경우에만 넣는다
            if "sigma" in c:
                attrs.append(f'sigma="{c["sigma"]}"')
            # 양보 진입(회전교차로 · 합류)에서 기다린 만큼 과감해지는 정도 (impatience, SUMO 기본 0).
            # 0이면 회전 차량 흐름에 틈이 날 때까지 계속 양보해 회전교차로 진입부가 빠지지 않는다 (12팀 11/12 확인)
            if "impatience" in c:
                attrs.append(f'impatience="{c["impatience"]}"')
            # 뒤차가 이 차의 최대 감속을 얼마로 가정할지 (apparentDecel). ACC·EIDM은 정체 속 정지-출발에서
            # 선언한 decel보다 세게 서기도 해서, 뒤따르는 Krauss 차가 간격을 좁게 잡으면 추돌한다 (12팀 11/12 확인)
            if "apparent_decel" in c:
                attrs.append(f'apparentDecel="{c["apparent_decel"]}"')
            # ── 배경차 차선 유지 (keep_lane, 기본 True) ──
            # 배경차의 자율 차선변경(전략/협조/속도이득/우측유지)을 전부 꺼서
            # 각자 자기 차선을 지키게 한다. 이유:
            #   배경차가 자유롭게 돌아다니면 (측정: 150스텝에 477회 변경)
            #   차선 간 틈이 수시로 닫혀서, ego가 틈을 노려 파고드는
            #   "칼치기 추월"을 학습할 안정적인 기회 자체가 사라진다.
            #   배경차가 차선을 유지하면 "느린 차 뒤  /  옆 틈으로 변경  /
            #   가속"이라는 추월 패턴이 재현 가능한 형태로 나타난다.
            # 특정 컨트롤러만 돌아다니게 하려면 그 항목에 "keep_lane": False.
            if c.get("keep_lane", True):
                attrs += ['lcStrategic="0"', 'lcCooperative="0"',
                          'lcSpeedGain="0"', 'lcKeepRight="0"']
            vtype_lines.append("        <vType " + " ".join(attrs) + "/>")
        vtypes_block = ('    <vTypeDistribution id="carDist">\n'
                        + "\n".join(vtype_lines)
                        + "\n    </vTypeDistribution>")
        flow_type = "carDist"
    else:
        # controllers 설정이 없으면 예전 방식(단일 Krauss)으로 동작 (하위호환)
        vtypes_block = (f'    <vType id="car" accel="2.5" decel="4.5" '
                        f'maxSpeed="{traffic["max_speed"]}" '
                        f'sigma="{traffic.get("sigma", 0.5)}" '
                        f'vClass="passenger" guiShape="passenger/sedan"/>')
        flow_type = "car"
    return vtypes_block, flow_type



# 실제 도로망 (road["network"]이 있을 때) : 반포대교 남단 접속부
#
# 도로 정의의 단일 원천은 environment/road/road_spec.py 이다 (직선·원호 + 실측 길이·차로).
# 도로망은 하나다 (잠수교 포함). 보행화 전·후는 같은 도로에서
#   - 교통 수요 파일만 다르고 (banpo.before.rou.xml / banpo.after.rou.xml)
#   - 보행화 후에는 sumo_env가 에피소드 시작 때 잠수교 차로를 보행자 전용으로 바꾼다 (차량 통제).
# 수요 파일에 들어가는 것:
#   - AV 경로 (<route id="av_side-banpo"> …) : road_spec.AV_ROUTES 중 road["av_routes"], 그 시나리오에서 갈 수 있는 것
#   - 배경 교통 (진입점  /  출구별 flow)  : traffic["od"]
# 화면: 높이 대역마다 같은 색 다각형으로 다시 그려 위아래를 고정 (둔치 층 < 지하 1층 < 지상 < 고가).
# 결과: out_dir/<network>/

# 높이 대역 : 비슷한 높이의 도로는 같은 색 (environment/road/road_spec.py 층 높이 참고)
#   (이름, 하한 z, 상한 z, 면 색, 가장자리 색, layer)  · 지상(GROUND_COLOR)은 sumo-gui가 직접 그린다
#   layer: sumo-gui 차로는 layer 5 밑에 그려진다  /  지하·둔치 다각형도 차로 위(5.2~5.4)에 그려 sumo-gui가
#   투명한 지하 차로에 남기는 선(차선 경계 점선의 검은 틈 등)을 가리고, 그 위를 지나는 도로는 _write_layers가
#   겹치는 구간만 더 위(6.x)에 덧칠해 위아래를 맞춘다.
# 색은 지도의 높이별 색(등고선 채색, hypsometric tint) 관례: 낮을수록 초록  /  노랑  /  높을수록 갈색.
GROUND_COLOR = "0,0,0"                                            # 지상 0 m: sumo-gui 기본 검정
DECK_Z = 2.0                   # 이 높이(m) 이상 = 고가
UNDER_Z = -3.0                 # 이 높이(m) 이하 = 지하·둔치 (원래 차로는 숨기고 다각형으로 다시 그림)
BANDS = [
    ("deck",  DECK_Z, 1e9,     "160,90,45",   "90,45,15",  10.0),   # 고가: 강남터미널고가 · 반포대교 상층 · ③ 램프 윗부분
    ("tunnel", -8.0, UNDER_Z,  "150,190,90",  "75,110,35", 5.4),   # 지하 경사로 (−3 ~ −8 m): 지하차도 진입부 · 루프 · 주차장 램프
    ("river", -1e9, -8.0,      "45,125,70",   "15,65,30",  5.2),   # 한강 둔치 층: 잠수교 지하차도 · 회전교차로 · 잠수교 · 주차장
]


NET_STAMP = "banpo.net.stamp"   # 도로망 입력 지문 (같으면 netconvert를 건너뜀)


def _build_network(road: dict, ego: dict, traffic: dict, out_dir: str) -> str:
    """덮어쓰기 방지 (11/15 §12.0 6번): view_road · train · test · 수집이 실행마다 build()를 부르므로, 장시간 수집 ·
    학습 중에 다른 스크립트를 띄우면 실행 중인 워커가 쓰다 만 파일을 읽을 수 있었다. 이제
      1) 모든 파일을 out_dir 안 임시 폴더에 만들고 (도로망 입력 지문이 지금 것과 같으면 netconvert는 건너뛰고 지금 도로망을 복사),
      2) 지금 파일과 내용이 같으면 하나도 건드리지 않고, 다른 파일만 하나씩 os.replace로 통째 교체한다
         (파일 단위 원자적 교체 : 읽는 쪽은 옛 파일이나 새 파일 전체 중 하나만 본다)."""
    import shutil
    import tempfile
    net_dir = os.path.join(out_dir, road["network"])
    os.makedirs(out_dir, exist_ok=True)
    work = tempfile.mkdtemp(prefix=f".{road['network']}_build_", dir=out_dir)   # 같은 드라이브여야 os.replace가 원자적
    try:
        _write_network(road, ego, traffic, work, net_dir)
        changed = _commit_dir(work, net_dir)
        if changed:
            print(f"[road_builder] {net_dir}: 바뀐 파일 {len(changed)}개 교체 — {', '.join(changed)}")
    finally:
        shutil.rmtree(work, ignore_errors=True)
    return os.path.join(net_dir, "banpo.sumocfg")


def _net_stamp(work: str, build_net, road_spec) -> str:
    """netconvert 결과를 정하는 입력의 지문: 생성한 nod · edg · con + 변환 옵션 · 신호 적용 코드(build_net.py) +
    신호 시각표(road_spec.py). 코드 파일은 통째로 넣어 보수적으로 (주석만 바뀌어도 다시 변환)."""
    import hashlib
    h = hashlib.sha256()
    for p in [os.path.join(work, n) for n in ("banpo.nod.xml", "banpo.edg.xml", "banpo.con.xml")] + \
             [build_net.__file__, road_spec.__file__]:
        with open(p, "rb") as f:
            h.update(os.path.basename(p).encode() + b"\0" + f.read() + b"\0")
    return h.hexdigest()


def _same_file(a: str, b: str) -> bool:
    if not os.path.exists(b) or os.path.getsize(a) != os.path.getsize(b):
        return False
    with open(a, "rb") as fa, open(b, "rb") as fb:
        return fa.read() == fb.read()


def _commit_dir(work: str, net_dir: str) -> list:
    """work의 파일 중 net_dir과 내용이 다른 것만 os.replace로 옮긴다. 바뀐 파일 이름 목록을 돌려준다.
    Windows는 다른 프로세스가 대상 파일을 열고 있는 순간 교체가 거부되므로(PermissionError) 잠깐씩 다시 시도한다.
    도로망(net.xml)을 맨 나중에 바꾼다 : 새 수요 · 화면 파일이 옛 도로망과 섞이는 틈보다, 수요 파일의 엣지가 새 도로망에
    없을 때 SUMO가 시작에서 바로 오류를 내는 쪽이 안전하다 (도로망이 바뀌는 건 도로를 고칠 때뿐)."""
    import time
    os.makedirs(net_dir, exist_ok=True)
    names = sorted(os.listdir(work), key=lambda n: (n == NET_STAMP, n == "banpo.net.xml", n))   # 지문은 도로망 뒤 맨 마지막
    changed = [n for n in names if not _same_file(os.path.join(work, n), os.path.join(net_dir, n))]
    # 전부 바꾸거나 하나도 안 바꾼다 (11/16 검토 E): SUMO는 시뮬레이션 내내 수요 파일(.rou.xml)을 열어 두므로, 학습 · 수집 중에
    # 도로 설정을 바꿔 build()를 부르면 그 파일만 교체가 거부된다. 앞 파일만 바뀐 채 멈추면 돌고 있는 워커가 다음 에피소드에서
    # 새 · 옛 파일이 섞인 도로를 읽는다  /  먼저 바꿀 파일이 모두 풀려 있는지 보고, 잠긴 게 있으면 아무것도 안 바꾸고 알린다.
    holders = _holders(net_dir) if changed else []
    if holders:
        # 학습(training/train_team.py rl)이 도는 동안: 수요 파일이 안 바뀌고 도로망만 바뀌는 설정(신호 시각표 등)은 잠김 검사로 못
        # 잡으므로(SUMO는 도로망을 시작에만 읽고 닫는다), 학습이 표시한 동안에는 아무 파일도 바꾸지 않는다 (11/16 검토).
        raise RuntimeError(
            f"[road_builder] 학습이 {net_dir}을(를) 쓰는 중이라 (PID {', '.join(map(str, holders))}) 도로 파일을 하나도 바꾸지 "
            f"않았다. 도로 · 교통 설정이 학습과 다르다 — 학습이 끝난 뒤 다시 실행하거나 설정을 되돌린다. "
            f"(바뀔 파일: {', '.join(changed)}. 학습이 이미 끝났는데 이 오류가 나면 {net_dir}의 {HOLD_PREFIX}* 파일을 지운다)")
    locked = [n for n in changed if _locked(os.path.join(net_dir, n))]
    if locked:
        raise RuntimeError(
            f"[road_builder] {net_dir}의 {', '.join(locked)}을(를) 다른 프로세스(학습 · 수집 · 평가 중인 SUMO)가 쓰고 있어 "
            f"도로 파일을 하나도 바꾸지 않았다. 도로 · 교통 설정(road_config 등)이 지금 돌고 있는 것과 다르다는 뜻이다 — "
            f"그 작업이 끝난 뒤 다시 실행하거나, 설정을 돌고 있는 것과 같게 되돌린다. (바뀔 파일: {', '.join(changed)})")
    for n in changed:
        src, dst = os.path.join(work, n), os.path.join(net_dir, n)
        for k in range(50):
            try:
                os.replace(src, dst)
                break
            except PermissionError:
                if k == 49:
                    raise
                time.sleep(0.1)
    return changed


def _locked(path: str) -> bool:
    """다른 프로세스가 열고 있어 교체할 수 없는 파일이면 True (Windows: 열린 파일은 이름을 바꿀 수 없다). 없는 파일은 False.
    잠깐 이름을 바꿨다 되돌려 본다 : 잠깐 읽기를 겹친 경우를 위해 0.1 s 간격으로 몇 번 다시 본다."""
    import time
    if not os.path.exists(path):
        return False
    probe = f"{path}.lockcheck"
    for _ in range(10):
        try:
            os.replace(path, probe)
        except PermissionError:
            time.sleep(0.1)
            continue
        # 되돌리기는 반드시 성공해야 한다 (실패하면 원래 파일이 .lockcheck 이름으로 남아 SUMO가 못 연다 : 11/16 검토).
        # 백신 등이 막 바뀐 이름의 파일을 잠깐 잡을 수 있어 넉넉히 다시 시도한다.
        for k in range(100):
            try:
                os.replace(probe, path)
                break
            except PermissionError:
                if k == 99:
                    raise RuntimeError(f"[road_builder] {probe}를 {path}로 되돌리지 못했다 — 직접 이름을 되돌린다")
                time.sleep(0.1)
        return False
    return True


HOLD_PREFIX = ".in_use_"


class _Hold:
    def __init__(self, path):
        self.path = path

    def release(self):
        try:
            os.remove(self.path)
        except OSError:
            pass


def hold(net_dir: str) -> "_Hold":
    """학습처럼 오래 도는 작업이 이 도로 폴더를 쓰고 있다고 표시한다 (net_dir/.in_use_<PID>). 표시가 있는 동안 다른
    프로세스의 build()는 내용이 다른 파일이 하나라도 있으면 아무것도 바꾸지 않고 오류를 낸다 (같은 설정이면 그대로 통과).
    끝나면 .release(). 프로세스가 강제 종료돼 표시가 남아도, 그 PID가 살아 있지 않으면 무시한다."""
    path = os.path.join(net_dir, f"{HOLD_PREFIX}{os.getpid()}")
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"{os.getpid()}\n")
    return _Hold(path)


def _holders(net_dir: str) -> list:
    """이 폴더를 쓰고 있다고 표시한 살아 있는 다른 프로세스 PID 목록 (죽은 PID의 표시는 지운다)."""
    out = []
    for n in os.listdir(net_dir) if os.path.isdir(net_dir) else []:
        if not n.startswith(HOLD_PREFIX):
            continue
        try:
            pid = int(n[len(HOLD_PREFIX):])
        except ValueError:
            continue
        if pid == os.getpid():
            continue
        if _pid_alive(pid):
            out.append(pid)
        else:
            try:
                os.remove(os.path.join(net_dir, n))
            except OSError:
                pass
    return out


def _pid_alive(pid: int) -> bool:
    """PID가 살아 있는가. Windows에서 os.kill(pid, 0)은 프로세스를 끝내 버리므로 쓰지 않는다."""
    if os.name == "nt":
        import ctypes
        k32 = ctypes.windll.kernel32
        h = k32.OpenProcess(0x1000, False, pid)            # PROCESS_QUERY_LIMITED_INFORMATION
        if not h:
            return False
        try:
            code = ctypes.c_ulong()
            return bool(k32.GetExitCodeProcess(h, ctypes.byref(code))) and code.value == 259   # STILL_ACTIVE
        finally:
            k32.CloseHandle(h)
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True


def _write_network(road: dict, ego: dict, traffic: dict, net_dir: str, live_dir: str):
    """반포 도로망 파일 전부를 net_dir(임시 폴더)에 만든다. live_dir은 지금 쓰이는 폴더 : 도로망 지문이 같으면
    netconvert 대신 거기 있는 도로망을 복사한다."""
    import sys
    import shutil
    road_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "road")
    if road_dir not in sys.path:
        sys.path.insert(0, road_dir)
    import road_spec as S
    import build_net
    import routing

    # 1) 도로망 하나 (잠수교 포함 = environment/road/build_net.py의 before 도로망과 같음)
    build_net.write_xml(net_dir, "before")
    stamp = _net_stamp(net_dir, build_net, S)
    live_stamp = os.path.join(live_dir, NET_STAMP)
    reuse = (os.path.exists(live_stamp) and open(live_stamp, encoding="utf-8").read().strip() == stamp
             and all(os.path.exists(os.path.join(live_dir, n)) for n in ("banpo.net.xml", "banpo.tll.xml", "netconvert.log")))
    if reuse:
        for n in ("banpo.net.xml", "banpo.tll.xml", "netconvert.log"):
            shutil.copyfile(os.path.join(live_dir, n), os.path.join(net_dir, n))
    else:
        rc, err = build_net.netconvert(net_dir)
        if rc != 0:
            # 임시 폴더는 지워지므로 로그 내용을 오류에 싣는다
            log = open(os.path.join(net_dir, "netconvert.log"), encoding="utf-8", errors="replace").read()
            raise RuntimeError(f"netconvert 실패 (기존 {live_dir}는 그대로):\n{log[-1500:]}\n{err[-800:]}")
    with open(os.path.join(net_dir, NET_STAMP), "w", encoding="utf-8") as f:
        f.write(stamp + "\n")
    netf = os.path.join(net_dir, "banpo.net.xml")
    net, lane_attr = routing.load(netf)
    vtypes_block, flow_type = _vtypes_block(traffic)
    scale = float(traffic.get("demand_scale", 1.0))

    # 2) 시나리오별 수요 파일
    for scn in S.SCENARIOS:
        closed = set(S.CLOSED_AFTER) if scn == "after" else set()
        av_lines = []
        for name in road.get("av_routes") or list(S.AV_ROUTES):
            r = S.AV_ROUTES[name]
            if (r["scenarios"] and scn not in r["scenarios"]) or closed & set(r["edges"]):
                continue
            if not routing.route_feasible(net, lane_attr, r["edges"]):
                raise RuntimeError(f"AV 경로 {name}이(가) 차로 규칙상 끝까지 갈 수 없습니다: {r['edges']}")
            av_lines.append(f'    <route id="av_{name}" edges="{" ".join(r["edges"])}"/>  <!-- {r["desc"]} -->')
        if not av_lines:
            raise RuntimeError(f"시나리오 {scn}에서 쓸 수 있는 AV 경로가 없습니다: {road.get('av_routes')}")

        # 배경 교통: 진입점마다 시간당 대수(vph) × 출구 비율  /  출구별 flow (무작위 도착)
        bg_lines = []
        for entry, spec in (traffic.get("od") or {}).items():
            if not net.hasEdge(entry) or entry in closed:
                continue
            exits = spec.get(f"exits_{scn}", spec["exits"])
            ok = {k: (w, p) for k, w, p in routing.od_paths(net, lane_attr, entry, exits, closed)}
            total_w = sum(w for w, _ in ok.values())
            for key, (w, p) in ok.items():
                vph = float(spec.get(f"vph_{scn}", spec["vph"])) * scale * w / total_w
                ex = key.replace("|", "_via_")
                rid = f"bg_{entry}__{ex}"
                bg_lines.append(f'    <route id="{rid}" edges="{" ".join(p)}"/>')
                # flow 하나는 초당 최대 1대(3,600 대/시)까지만 투입되므로, 큰 수요는 900 대/시 이하 flow 여러 개로 나눈다.
                # 진입 차로는 "best"(경로를 차로 변경 없이 가장 멀리 갈 수 있는 차로 중 빈 곳) : 실제 차는 범위 밖에서 이미
                # 갈 차로에 서 있다. "free"(아무 빈 차로)로 넣으면 짧은 진입 도로에서 차로 변경이 몰려 입구가 막힌다
                # (12팀 11/12: 김포 방향 투입 −15~25%, 신반포로 동행 우회전 전용 차로가 매 주기 막힘  /  best로 해결).
                k = max(1, math.ceil(vph / 900.0))
                for j in range(k):
                    bg_lines.append(f'    <flow id="f_{entry}__{ex}_{j}" type="{flow_type}" route="{rid}" begin="0" end="1000000" '
                                    f'probability="{vph / k / 3600.0:.4f}" departLane="best" departSpeed="max"/>')
        # 버스: 노선마다 공식 배차간격(period)으로 투입. 경로는 bus_paths의 방향별 경로 키
        #   (path = 전후 공통, before_path / after_path = 보행화 전후가 다름 : road_config._bus_lines)
        bus_lines, bus_flows = [], []   # 노선 경로 정의 / (출발 시각, flow)
        bus_stops = traffic.get("bus_stops") or {}
        bus_paths = traffic.get("bus_paths") or {}
        for b in traffic.get("buses") or []:
            edges = bus_paths[b.get(f"{scn}_path") or b["path"]]
            if not edges or closed & set(edges):
                continue
            if not all(net.hasEdge(e) for e in edges):
                raise RuntimeError(f"버스 {b['id']} 경로에 없는 엣지: {edges}")
            stops = b.get("stops", [])
            for sid in stops:
                if sid not in bus_stops or bus_stops[sid][0] not in edges:
                    raise RuntimeError(f"버스 {b['id']} 정류소 {sid}가 경로 위에 없음")
            # 차량 id = 노선 + 진입·출구 (배경차 f_<진입>__<출구>와 같이 id만으로 목적지를 안다 : 따로 칼럼 없음)
            bid = f'bus_{b["id"]}__{edges[0]}__{edges[-1]}'
            bus_lines.append(f'    <route id="{bid}" edges="{" ".join(edges)}"/>')
            stop_xml = "".join(f'<stop busStop="{sid}" duration="20"/>' for sid in stops)
            # 화면: 노선 번호를 버스에 띄운다 (banpo.gui.xml vehicleText : 승용차에는 이 값이 없어 글자가 안 뜸)
            stop_xml = f'<param key="busLine" value="{b["id"]}"/>' + stop_xml
            begin = b.get("offset", 0)
            color = f' color="{b["color"]}"' if b.get("color") else ""   # 노선 유형 색 (road_config.BUS_COLORS)
            bus_flows.append((begin, f'    <flow id="{bid}" type="bus" route="{bid}"{color} begin="{begin}" end="1000000" '
                                     f'period="{b["period"]}" departLane="best" departSpeed="max">{stop_xml}</flow>'))
        if bus_lines:   # 버스 색은 노선 종류별로 flow에 붙인다 (_bus_lines)
            bg_lines = ['    <vType id="bus" vClass="bus" length="12" minGap="2.5" accel="1.2" decel="4.0" maxSpeed="16.7" '
                        'guiShape="bus"/>'] + bus_lines + bg_lines
        # 보행자 (road_spec.PEDESTRIANS, 11/12): 보도  /  신호 횡단보도  /  보도. SUMO가 횡단보도 · 보도 경로를 찾는다
        ped_lines = []
        for k, (pname, pfrom, pto, per_h, pscn) in enumerate(S.PEDESTRIANS):
            if (pscn and scn not in pscn) or not (net.hasEdge(pfrom) and net.hasEdge(pto)) or per_h <= 0:
                continue
            # 출발 · 도착 보도가 한 교차로에 붙어 있으면 경로를 그 둘로 못박아 그 교차로 안(횡단보도)에서만 건너게 한다.
            # from/to만 주면 SUMO가 가까운 다른 노드의 보행 구역(차도와 겹침)으로 돌아가는 경로를 고를 수 있다 :
            # 잠수교남단(jx1)에서 4 m 옆 회전교차로 노드(rb_n)로 돌아가 차량과 충돌 (11/13 검증).
            ef, et = net.getEdge(pfrom), net.getEdge(pto)
            shared = {ef.getFromNode().getID(), ef.getToNode().getID()} & {et.getFromNode().getID(), et.getToNode().getID()}
            walk = f'<walk edges="{pfrom} {pto}"/>' if shared else f'<walk from="{pfrom}" to="{pto}"/>'
            # end는 PED_END_S로 끊는다 : SUMO는 personFlow를 차량 flow와 달리 읽을 때 한꺼번에 펼쳐 만든다.
            # end=1000000이면 매 시작마다 보행자 30만 명을 만들어 시작에 34 s · 메모리 0.9 GB (11/14 측정, 6시간이면 1 s)
            ped_lines.append(f'    <personFlow id="ped_{k}" begin="0" end="{PED_END_S}" period="{3600.0 / per_h:.1f}" color="{PERSON_COLOR}">'
                             f'{walk}</personFlow>   <!-- {pname} -->')
        # SUMO는 수요 파일을 시간순으로 조금씩 읽어(route-steps) 출발 시각이 앞 항목보다 이른 것을 버린다.
        # 출발 0초인 배경차 · 보행자를 먼저, 노선마다 출발을 흩어 둔 버스 flow는 맨 뒤에 출발 순으로 쓴다.
        bg_lines = bg_lines + ped_lines + [ln for _, ln in sorted(bus_flows, key=lambda t: t[0])]
        with open(os.path.join(net_dir, f"banpo.{scn}.rou.xml"), "w", encoding="utf-8") as f:
            f.write(f"<routes>  <!-- 시나리오: {scn} -->\n")
            f.write(f'    <vType id="ego_type" accel="{ego["accel"]}" decel="{ego["decel"]}" maxSpeed="{ego["max_speed"]}" '
                    f'color="1,0,0" vClass="passenger" guiShape="passenger/sedan"/>\n')
            # SUMO 운전자 모드 AV (sumo_env가 sumo_driver일 때 이 차종으로 넣는다). 두 차종을 늘 함께 써 두어
            # 수요 파일이 운전 모드에 따라 바뀌지 않게 한다 (모드가 다른 스크립트를 동시에 띄워도 파일이 같음)
            sv = ego.get("sumo_vtype") or {"accel": 2.6, "decel": 4.5, "sigma": 0.0}
            f.write(f'    <vType id="ego_sumo" accel="{sv["accel"]}" decel="{sv["decel"]}" sigma="{sv["sigma"]}" '
                    f'maxSpeed="{ego["max_speed"]}" color="1,0,0" vClass="passenger" guiShape="passenger/sedan"/>\n')
            f.write(vtypes_block + "\n")
            f.write("\n".join(av_lines) + "\n")
            f.write("\n".join(bg_lines) + "\n")
            f.write("</routes>\n")

    # 3) 화면: 층별 다각형 (지하는 깊이 순으로 지상 밑, 고가는 상판으로 지상 위)
    _write_layers(netf, os.path.join(net_dir, "banpo.layers.xml"))
    _write_background(S, os.path.join(net_dir, "banpo.background.xml"))
    # 버스 정류소
    with open(os.path.join(net_dir, "banpo.stops.xml"), "w", encoding="utf-8") as f:
        f.write("<additional>\n")
        for sid, (edge, lane, p0, p1, name) in (traffic.get("bus_stops") or {}).items():
            # 버스전용 차선 = 파랑, 정류소 = 연한 파랑 (버스 차량 색은 노선 종류별 : 간선 파랑 · 지선 초록 · 광역 빨강 등, _bus_lines)
            f.write(f'    <busStop id="{sid}" lane="{edge}_{S.net_lane(edge, lane)}" startPos="{p0}" endPos="{p1}" name="{name}" lines="bus" '
                    f'color="{BUS_STOP_COLOR}"/>\n')
        f.write("</additional>\n")
    gui_delay = float(SIMULATION.get("gui_delay", 100.0))
    with open(os.path.join(net_dir, "banpo.gui.xml"), "w", encoding="utf-8") as f:
        # laneEdgeMode 22 = "by height at geometry-segment start" (SUMO 1.27 기준): 높이로 색칠.
        # 지하 1층(−8 < z ≤ UNDER_Z)은 투명 : 대신 _write_layers가 대역 색으로 지상 도로 밑에 다시 그린다.
        # 둔치 층(z ≤ −8)은 불투명(둔치 색)으로 그리게 둔다: sumo-gui는 투명한 차로의 신호 막대(정지선 · 횡단보도 끝의
        # 빨강/초록)를 안 그려서 잠수교 횡단보도 신호가 안 보였다 (11/13). 차로 면은 위의 둔치 다각형(5.2)이 덮고,
        # 신호 막대는 다각형보다 위에 그려진다. 둔치 차로가 지상 도로 밑을 지나는 곳은 덮개(cover)가 지상 도로를 다시 칠한다.
        # 고가는 상판 다각형이 덮는다. 교차로도 높이로 칠해 지하·둔치 교차로는 투명하게.
        f.write(f"""<viewsettings>
    <scheme name="banpo-layers">
        <edges laneEdgeMode="22" laneShowBorders="1" showLinkDecals="1">
            <colorScheme name="by height at geometry-segment start" interpolated="1">
                <entry color="{BANDS[2][3]}" threshold="-1000"/>
                <entry color="{BANDS[2][3]}" threshold="-8"/>
                <entry color="{BANDS[1][3]},0" threshold="-7.999"/>
                <entry color="{BANDS[1][3]},0" threshold="{UNDER_Z}"/>
                <entry color="{GROUND_COLOR}" threshold="{UNDER_Z + 0.001}"/>
                <entry color="{GROUND_COLOR}" threshold="1000"/>
            </colorScheme>
        </edges>
        <junctions junctionMode="3">
            <colorScheme name="by height" interpolated="1">
                <entry color="0,0,0,0" threshold="-1000"/>
                <entry color="0,0,0,0" threshold="{UNDER_Z}"/>
                <entry color="{GROUND_COLOR}" threshold="{UNDER_Z + 0.001}"/>
                <entry color="{GROUND_COLOR}" threshold="1000"/>
            </colorScheme>
        </junctions>
        <pois poiName_show="1" poiName_size="120" poiName_color="0,0,0" poiName_constantSize="0"/>
        <vehicles vehicleQuality="2" vehicleText_show="1" vehicleTextParam="busLine" vehicleText_size="28" vehicleText_color="255,255,255" vehicleText_bgColor="30,30,30" vehicleText_constantSize="0"/>  <!-- 버스 노선 번호 -->
        <persons personQuality="2" person_exaggeration="3.0" person_constantSize="0"/>  <!-- 보행자를 3배로 (실제 크기로는 점처럼 보임) -->
    </scheme>
    <delay value="{gui_delay}"/>
</viewsettings>
""")

    # 4) sumocfg : 수요 파일 기본값은 보행화 전 (sumo_env가 에피소드마다 --route-files로 바꿈).
    #    교차로 안 충돌도 검사한다 (기본값은 교차로 내부 충돌을 무시함).
    #    차로 변경 시간 LC_DURATION (0 = SUMO 기본 순간 이동 : 11/14 팀 결정, LC_DURATION 주석). AV(ego)의 차로 변경도 같다.
    cfg_path = os.path.join(net_dir, "banpo.sumocfg")
    with open(cfg_path, "w", encoding="utf-8") as f:
        f.write(f"""<configuration>
    <input>
        <net-file value="banpo.net.xml"/>
        <route-files value="banpo.before.rou.xml"/>
        <additional-files value="banpo.background.xml,banpo.layers.xml,banpo.stops.xml"/>
    </input>
    <processing>
        <collision.check-junctions value="true"/>
        <lanechange.duration value="{LC_DURATION}"/>
    </processing>
    <gui_only>
        <gui-settings-file value="banpo.gui.xml"/>
    </gui_only>
</configuration>
""")
    return cfg_path


def _offset(pts, d):
    """2D 폴리라인을 진행 방향 왼쪽(+)/오른쪽(-)으로 d만큼 평행이동."""
    out = []
    n = len(pts)
    for i in range(n):
        nx = ny = 0.0
        for a, b in ((pts[i - 1], pts[i]) if i > 0 else (None, None),
                     (pts[i], pts[i + 1]) if i < n - 1 else (None, None)):
            if a is None:
                continue
            dx, dy = b[0] - a[0], b[1] - a[1]
            L = math.hypot(dx, dy) or 1.0
            nx += -dy / L
            ny += dx / L
        L = math.hypot(nx, ny) or 1.0
        out.append((pts[i][0] + d * nx / L, pts[i][1] + d * ny / L))
    return out


def _band(z):
    for bd in BANDS:
        if bd[1] <= z < bd[2] or (bd[0] != "deck" and z == bd[2]):
            return bd
    return None                                   # 지상


GROUND_BAND = ("ground", UNDER_Z, DECK_Z, GROUND_COLOR, GROUND_COLOR, 5.6)
LC_DURATION = 0.0               # 차로 변경 시간 (s). 0 = SUMO 기본(한 스텝에 옆 차로로 이동 : 화면에서 평행 이동처럼 보임).
                                # 팀 결정(11/14): 기본값 유지. 실측 평균 4.6 s(Toledo & Zohar 2007)는 반포대교 ③ 합류 뒤
                                # 측면 충돌(4시간 × 3시드 약 30건)이 생겨 취소, sublane 모델은 더 나빴음(충돌 · 순간이동 36건 · 4배 느림)
BUS_LINE_COLOR = "0,110,255"    # 버스전용 차로 차선 (파랑)
BUS_STOP_COLOR = "130,180,255"  # 버스 정류소 (연한 파랑 : 정차한 버스가 정류소 색에 묻히지 않게)
SIDEWALK_COLOR = "196,188,172"  # 보도 · 교차로 모퉁이 보행 구역 (보도블록 회베이지 : 검은 차도 · 초록 둔치 · 연녹 배경과 구분)
CROSSING_COLOR = "255,255,255"  # 횡단보도 줄무늬
CENTER_LINE_COLOR = "255,200,0" # 중앙선 (노랑, 반대 방향 도로가 붙은 곳)
PED_END_S = 6 * 3600            # 보행자 flow 끝 시각 (s) : 에피소드(웜업 포함 약 11분) · check_long(4시간)보다 길게. 더 오래 돌리면 늘린다
PERSON_COLOR = "120,30,150"     # 보행자 (진보라 : 밝은 보도 · 흰 줄무늬 위에서 보이고, 차량 색(노랑·하늘·주황·분홍)과 다름)
DASH = (6.0, 2.5, 5.5)          # sumo-gui 차선 경계 점선: 6 m 주기, 차로 시작부터 2.5~5.5 m 구간을 칠함 (화면 실측)


def _runs(shape):
    """(x,y,z) 점들을 높이 대역별 연속 구간으로 나눈다. 대역 경계(고가 2 m · 지하 −3 m · 둔치 −8 m)를
    지나는 정확한 지점에서 자르므로, 방향이 반대인 차로(북행·남행)도 같은 지점에서 색이 바뀐다.

    반환: [(2D 폴리라인, 대역 : 지상은 GROUND_BAND, 차로 시작부터 거리, 앞 구간 대역,
           폴리라인에서 첫 원래 꼭짓점 위치)]"""
    pts, d = [], 0.0                              # (x, y, z, 차로 시작부터 거리, 잘린 점인가)
    for i, p in enumerate(shape):
        z = p[2] if len(p) > 2 else 0.0
        if i:
            q = shape[i - 1]
            zq = q[2] if len(q) > 2 else 0.0
            L = math.dist(q[:2], p[:2])
            cuts = sorted(((c - zq) / (z - zq), c) for c in (DECK_Z, UNDER_Z, -8.0) if (zq - c) * (z - c) < 0)
            for t, c in cuts:
                pts.append((q[0] + (p[0] - q[0]) * t, q[1] + (p[1] - q[1]) * t, c, d + L * t, True))
            d += L
        pts.append((p[0], p[1], z, d, False))
    # 선분마다 가운데 높이로 대역을 정해 묶는다 (꼭짓점이 경계값과 정확히 같아도 방향과 상관없이 같은 결과)
    runs, cur, cb, prev_b = [], [pts[0]], None, None
    for i in range(len(pts) - 1):
        bd = _band((pts[i][2] + pts[i + 1][2]) / 2)
        if i and bd is not cb:
            k = next((j for j, c in enumerate(cur) if j and not c[4]), len(cur) - 1)
            runs.append(([(c[0], c[1]) for c in cur], cb or GROUND_BAND, cur[0][3], prev_b, k))
            prev_b, cur = cb or GROUND_BAND, [pts[i]]
        cb = bd
        cur.append(pts[i + 1])
    if len(cur) >= 2:
        k = next((j for j, c in enumerate(cur) if j and not c[4]), len(cur) - 1)
        runs.append(([(c[0], c[1]) for c in cur], cb or GROUND_BAND, cur[0][3], prev_b, k))
    return runs


def _dashes(line, d0):
    """폴리라인을 sumo-gui와 같은 위상의 점선 조각들로 (d0 = 선 시작점의 차로 시작부터 거리)."""
    period, on0, on1 = DASH
    out, cur, d = [], [], d0
    for a, b in zip(line, line[1:]):
        L = math.dist(a, b)
        if L <= 0:
            continue
        cuts = [0.0]                                  # 선분 안에서 점선 경계를 지나는 지점마다 끊는다
        k = math.floor(d / period)
        while k * period <= d + L:
            for c in (k * period + on0, k * period + on1):
                if d < c < d + L:
                    cuts.append(c - d)
            k += 1
        cuts = sorted(cuts) + [L]
        for t0, t1 in zip(cuts, cuts[1:]):
            mid = (d + (t0 + t1) / 2) % period
            p0 = (a[0] + (b[0] - a[0]) * t0 / L, a[1] + (b[1] - a[1]) * t0 / L)
            p1 = (a[0] + (b[0] - a[0]) * t1 / L, a[1] + (b[1] - a[1]) * t1 / L)
            if on0 <= mid <= on1:
                if not cur:
                    cur = [p0]
                cur.append(p1)
            elif cur:
                out.append(cur)
                cur = []
        d += L
    if cur:
        out.append(cur)
    return [c for c in out if len(c) >= 2]


def _center_lines(root, pts3, add):
    """중앙선(노란 실선): 반대 방향 도로가 바로 붙어 있는 구간(잠수교 등)의 경계에 그린다.
    각 도로의 가장 왼쪽 차량 차로 왼쪽 경계를 1 m마다 보고, 0.6 m 안에 반대 방향 도로의 왼쪽 경계가 있으면 중앙선."""
    STEP = 1.0
    bounds = {}                                   # 엣지  /  [(x, y, z, 진행 방향 단위벡터)]
    for e in root.iter("edge"):
        if e.get("function"):
            continue
        lanes = [ln for ln in e.findall("lane") if ln.get("allow") != "pedestrian"]
        if not lanes:
            continue
        ln = lanes[-1]
        sh = pts3(ln.get("shape"))
        left = _offset([(p[0], p[1]) for p in sh], float(ln.get("width", 3.2)) / 2)
        pts, d, nxt = [], 0.0, 0.5
        for i, (a, b) in enumerate(zip(left, left[1:])):
            L = math.dist(a, b)
            za, zb = (sh[i][2] if len(sh[i]) > 2 else 0.0), (sh[i + 1][2] if len(sh[i + 1]) > 2 else 0.0)
            while L > 0 and nxt <= d + L:
                t = (nxt - d) / L
                pts.append((a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t, za + (zb - za) * t,
                            ((b[0] - a[0]) / L, (b[1] - a[1]) / L)))
                nxt += STEP
            d += L
        bounds[e.get("id")] = pts
    grid = {}
    for eid, pts in bounds.items():
        for x, y, z, u in pts:
            grid.setdefault((int(x // 4), int(y // 4)), []).append((x, y, z, u, eid))
    for eid, pts in bounds.items():
        hit = []
        for x, y, z, u in pts:
            ok = False
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    for qx, qy, qz, qu, qid in grid.get((int(x // 4) + dx, int(y // 4) + dy), []):
                        if qid != eid and abs(qz - z) < 1.0 and u[0] * qu[0] + u[1] * qu[1] < -0.9 and math.dist((x, y), (qx, qy)) < 0.6:
                            ok = True
            hit.append(ok)
        run = []
        for (x, y, z, u), h in zip(pts, hit):
            if h:
                run.append((x, y, z))
                continue
            if len(run) >= 3:
                _emit_center(run, add)
            run = []
        if len(run) >= 3:
            _emit_center(run, add)


def _emit_center(run, add):
    zm = run[len(run) // 2][2]
    L = (_band(zm) or GROUND_BAND)[5]
    add("centerline", CENTER_LINE_COLOR, [(x, y) for x, y, _ in run], False, L + 0.007, 0.15)


def _write_layers(netf: str, out_path: str):
    """층 위아래를 화면에 맞게 그리는 다각형 additional 파일 (고가 상판 + 지하 도로).

    sumo-gui 2D 화면은 도로를 높이와 상관없이 정해진 순서로 그려서, 지하 도로가 고가 위에
    그려지기도 한다. 다각형은 layer 값 순서대로 그려지고(지상 도로는 layer 1~5 사이), 차량은
    항상 맨 위에 그려진다. 그래서 층마다 다각형으로 다시 그려 위아래를 고정한다.
      - 비슷한 높이의 도로는 같은 색 (BANDS) : 지도 높이별 색(hypsometric tint) 관례대로 낮을수록 초록,
        높을수록 갈색: 한강 둔치 층(≤ −8 m, 진초록) < 지하 1층(−8 ~ −3 m, 연두) < 지상(sumo-gui 기본 검정)
        < 고가(≥ 2 m, 갈색). 경사로는 지나는 높이에 따라 색이 바뀐다. 서쪽 빈 곳에 범례.
      - 지하·둔치 원래 차로·교차로는 다각형으로 덮는다 (지하 1층 차로 · 지하·둔치 교차로는 화면 설정에서 투명, 둔치 차로는
        신호 막대가 보이게 불투명 : 면은 다각형이 덮음). 다각형은 차로 위(layer 5.2~5.6)에
        그린다. 그 위를 지나는 지상·지하 1층 도로는 겹치는 구간만 6.x에 덧칠하고, 고가 상판은 10.
         /  지하차도가 올림픽대로 밑으로, 루프가 반포대교 밑으로 지나가는 것이 보인다.
    다각형 위에 가장자리선(실선)과 차선 경계선(sumo-gui와 같은 위상의 점선)을 다시 그린다.
"""
    import xml.etree.ElementTree as ET
    root = ET.parse(netf).getroot()

    def pts3(s):
        return [tuple(float(v) for v in t.split(",")) for t in s.split()]

    def shp(p):
        return " ".join(f"{x:.2f},{y:.2f}" for x, y in p)

    polys = []

    def add(kind, color, pts, fill, layer, width=None):
        extra = f' lineWidth="{width}"' if width else ""
        polys.append(f'    <poly id="layer_{len(polys)}" type="{kind}" color="{color}" fill="{1 if fill else 0}"{extra} '
                     f'layer="{layer:.3f}" shape="{shp(pts)}"/>')

    def lane_lines(kind, run, d0, w, li, nl, edge_c, L, edges=True, right_ped=False):
        """차로 다각형 위 선: 양 끝 차로의 가장자리선(실선)과 차로 사이 경계(sumo-gui 위상의 점선).
        edges=False면 가장자리선은 생략 (지하·둔치·고가 본 그림은 underlay() · outline()이 바깥 테두리를 그린다).
        right_ped: 오른쪽 차로가 보도  /  경계 점선 대신 색 차이로 구분."""
        if edges and kind != "ground" and li == nl - 1:   # 가장 왼쪽 차로: 왼쪽 가장자리선
            add(kind + "_edge", edge_c, _offset(run, w / 2), False, L + 0.005, 0.35)
        if edges and kind != "ground" and li == 0:        # 가장 오른쪽 차로: 오른쪽 가장자리선
            add(kind + "_edge", edge_c, _offset(run, -w / 2), False, L + 0.005, 0.35)
        if li > 0 and not right_ped:                      # 오른쪽 차선 경계선 (점선)
            for dash in _dashes(_offset(run, -w / 2), d0):
                add("lanemark", "255,255,255", dash, False, L + 0.006, 0.06)

    def outline(kind, edge_c, pts, L):
        """교차로 테두리: 면보다 살짝 아래(L − 0.01)에 굵은 닫힌 선(0.7 m)을 깐다. 같은 층의 면이 선의 안쪽 절반을 덮어
        바깥 둘레에만 0.35 m 테두리가 남는다."""
        add(kind + "_edge", edge_c, pts + pts[:1], False, L - 0.01, 0.7)

    def underlay(kind, edge_c, run, w, L):
        """차로 테두리: 차로보다 양옆으로 0.35 m 넓은 테두리색 띠를 면 바로 밑(L − 0.01)에 깐다. 같은 층의 차로 · 교차로
        면이 안쪽을 덮으므로 도로가 이어진 바깥 둘레에만 테두리가 남는다 (안쪽 이음매 · 층이 바뀌는 지점에는 선이 안 생기고,
        굵은 선과 달리 꺾이는 꼭짓점에 틈이 없다)."""
        add(kind + "_edge", edge_c, _offset(run, w / 2 + 0.4) + _offset(run, -(w / 2 + 0.4))[::-1], True, L - 0.01)

    ped_areas = []                                     # (보행 구역 · 횡단보도) : 차로 다음에 따로 그림
    for e in root.iter("edge"):
        lanes = e.findall("lane")
        fn = e.get("function")
        internal = fn == "internal"
        if fn in ("walkingarea", "crossing"):
            ped_areas += [(fn, ln) for ln in lanes]
            continue
        for li, ln in enumerate(lanes):
            w = float(ln.get("width", 3.2))
            shape = pts3(ln.get("shape"))
            ped = ln.get("allow") == "pedestrian"           # 보도 (sidewalk · 보행자 전용 길)
            right_ped = li > 0 and lanes[li - 1].get("allow") == "pedestrian"
            # 지상 구간은 sumo-gui가 그리지만 두 경우는 직접 그린다:
            #  - 시작점이 투명(지하)인 차로는 sumo-gui가 통째로 건너뛴다 (루프 · 지하차도 남행)  /  지상 구간 전부
            #  - sumo-gui는 꼭짓점 사이 선분을 시작점 높이 색 하나로 칠한다  /  지하에서 올라와 지상으로 넘어가는
            #    선분의 지상 쪽 나머지(경계 지점 ~ 다음 꼭짓점)가 투명하게 빈다  /  그 부분만
            starts_under = _band(shape[0][2] if len(shape[0]) > 2 else 0.0) not in (None, BANDS[0])
            for run, band, d0, prev_b, k in _runs(shape):
                kind, _, _, fill_c, edge_c, L = band
                if ped:                                    # 보도는 모든 층에서 직접 그린다 (차도와 다른 포장색)
                    # 차도 쪽(왼쪽)으로 0.15 m 더 덮어 sumo-gui가 지상 차로 경계에 그리는 점선을 가린다
                    left, right = _offset(run, w / 2 + 0.15), _offset(run, -(w / 2 + 0.05))
                    add("sidewalk", SIDEWALK_COLOR, left + right[::-1], True, L + 0.003)
                    if kind != "ground":
                        underlay(kind, edge_c, run, w, L)
                    continue
                if kind == "ground":
                    if not starts_under:
                        if prev_b is None or prev_b[0] not in ("tunnel", "river"):
                            continue
                        run = run[:k + 1]
                        if len(run) < 2:
                            continue
                left, right = _offset(run, w / 2 + 0.05), _offset(run, -(w / 2 + 0.05))
                add(kind, fill_c, left + right[::-1], True, L)
                if kind != "ground":
                    underlay(kind, edge_c, run, w, L)
                if not internal:
                    lane_lines(kind, run, d0, w, li, len(lanes), edge_c, L, edges=False, right_ped=right_ped)
            # 버스전용 차로: 파란 차선 (양쪽 실선, 우리나라 중앙버스전용차로 표시) : sumo-gui 차로 위에 그린다
            if not internal and "bus" in (ln.get("allow") or "").split() and "passenger" not in (ln.get("allow") or ""):
                for side in (w / 2, -w / 2):
                    add("buslane", BUS_LINE_COLOR, _offset([(q[0], q[1]) for q in shape], side), False, 5.7, 0.2)
    _center_lines(root, pts3, add)
    # 위층 도로 덧칠: 아래층 도로와 겹치는 위층(지상·지하 1층) 도로 구간을 아래층 다각형보다 위
    # layer(6.x)에 한 번 더 그린다  /  올림픽대로가 지하차도 위로 보인다.
    # (고가는 이미 layer 10 상판)
    import collections
    rank = lambda z: 0 if z <= -8.0 else 1 if z <= UNDER_Z else 2 if z < DECK_Z else 3
    STEP = 0.5
    samples, meta = {}, {}                        # 차로 id  /  [(x, y, 높이 순위)] (index × STEP = 차로 시작부터 거리)
    for e in root.iter("edge"):
        if e.get("function"):                  # 내부 차로 · 보행 구역 · 횡단보도 제외 (보행 구역 모양을 차로로 읽으면 검은 덩어리가 생김)
            continue
        lanes = e.findall("lane")
        for li, ln in enumerate(lanes):
            sh = pts3(ln.get("shape"))
            pts, d, nxt = [], 0.0, 0.0
            for a, b in zip(sh, sh[1:]):
                L = math.dist(a[:2], b[:2])
                while nxt <= d + L and L > 0:
                    t = (nxt - d) / L
                    z = (a[2] if len(a) > 2 else 0.0) * (1 - t) + (b[2] if len(b) > 2 else 0.0) * t
                    pts.append((a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t, rank(z)))
                    nxt += STEP
                d += L
            samples[ln.get("id")] = pts
            meta[ln.get("id")] = (float(ln.get("width", 3.2)), li, len(lanes), e.get("id"))
    grid = collections.defaultdict(list)
    for lid, pts in samples.items():
        for x, y, r in pts:
            grid[(int(x // 6), int(y // 6))].append((x, y, r, lid))
    # 아래층 교차로 모양도 겹침 판정에 넣는다 : 교차로가 위층 도로 밑까지 뻗으면 (갈래 · 합류가 비스듬하면 netconvert가 교차로를
    # 길게 늘림) 차로끼리는 안 겹쳐도 교차로 다각형이 위층 도로 위로 보인다 (11/13 und_j)
    low_juncs = []                                # (높이 순위, 꼭짓점, 경계 상자)
    for j in root.iter("junction"):
        s = j.get("shape")
        if s and j.get("type") != "internal":
            p = pts3(s)
            if len(p) >= 3:
                r = min(rank(q[2] if len(q) > 2 else 0.0) for q in p)
                if r < 2:
                    xs, ys = [q[0] for q in p], [q[1] for q in p]
                    low_juncs.append((r, [(q[0], q[1]) for q in p], (min(xs), min(ys), max(xs), max(ys))))

    def in_low_junction(x, y, r):
        for jr, poly, (x0, y0, x1, y1) in low_juncs:
            if jr < r and x0 <= x <= x1 and y0 <= y <= y1:
                inside = False
                for (ax, ay), (bx, by) in zip(poly, poly[1:] + poly[:1]):
                    if (ay > y) != (by > y) and x < ax + (y - ay) * (bx - ax) / (by - ay):
                        inside = not inside
                if inside:
                    return True
        return False
    for lid, pts in samples.items():
        w, li, nl, eid = meta[lid]
        hit = [False] * len(pts)
        for i, (x, y, r) in enumerate(pts):
            if r not in (1, 2):
                continue
            if in_low_junction(x, y, r):
                hit[i] = True
                continue
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    for qx, qy, qr, qid in grid[(int(x // 6) + dx, int(y // 6) + dy)]:
                        if qr < r and meta[qid][3] != eid and math.dist((x, y), (qx, qy)) < (w + meta[qid][0]) / 2 - 0.5:   # 실제로 겹칠 때만 (나란한 옆 차로 제외)
                            hit[i] = True
        idx = [i for i, h in enumerate(hit) if h]
        if not idx:
            continue
        groups, cur = [], [idx[0]]
        for i in idx[1:]:
            if i - cur[-1] <= 6:
                cur.append(i)
            else:
                groups.append(cur)
                cur = [i]
        groups.append(cur)
        for g in groups:
            a, b = max(0, g[0] - 6), min(len(pts) - 1, g[-1] + 6)
            run = [(x, y) for x, y, _ in pts[a:b + 1]]
            r = pts[g[0]][2]
            kind, fill_c, edge_c = ("ground", GROUND_COLOR, GROUND_COLOR) if r == 2 else ("tunnel", BANDS[1][3], BANDS[1][4])
            L = 6.0 + r * 0.1
            left, right = _offset(run, w / 2 + 0.05), _offset(run, -(w / 2 + 0.05))
            add("cover", fill_c, left + right[::-1], True, L)
            lane_lines(kind, run, a * STEP, w, li, nl, edge_c, L)
    # 교차로(합류·분기점·회전교차로) 모양: 모든 꼭짓점이 같은 대역이면 그 색으로 덮는다.
    # 지상 밖 교차로는 화면 설정에서 투명하게 숨긴다.
    for j in root.iter("junction"):
        s = j.get("shape")
        if s and j.get("type") != "internal":
            p = pts3(s)
            bds = {_band(q[2] if len(q) > 2 else 0.0) for q in p}
            if len(p) >= 3 and len(bds) == 1 and None not in bds:
                kind, _, _, fill_c, edge_c, L = bds.pop()
                p2 = [(q[0], q[1]) for q in p]
                add(kind, fill_c, p2, True, L - 0.002)
                if j.get("type") != "dead_end":       # 막다른 끝(지도 끝)은 테두리 없이 : 화면용 연장과 이어지게
                    outline(kind, edge_c, p2, L - 0.002)
    # 보행 구역(교차로 모퉁이 보도)은 보도 색, 횡단보도는 흰 줄무늬 (차도 방향과 나란한 0.5 m 폭 막대를 1 m 간격으로)
    for fn, ln in ped_areas:
        p = pts3(ln.get("shape"))
        z = sum(q[2] if len(q) > 2 else 0.0 for q in p) / len(p)
        kind, _, _, _, edge_c, L = _band(z) or GROUND_BAND
        p2 = [(q[0], q[1]) for q in p]
        if fn == "walkingarea":
            if len(p2) >= 3:
                add("walkingarea", SIDEWALK_COLOR, p2, True, L + 0.002)
            continue
        w = float(ln.get("width", 4.0))
        for a, b in zip(p2, p2[1:]):
            dx, dy = b[0] - a[0], b[1] - a[1]
            D = math.hypot(dx, dy)
            if D < 1.0:
                continue
            ux, uy, nx, ny = dx / D, dy / D, -dy / D * w / 2, dx / D * w / 2
            n = int((D + 0.5) // 1.0)                   # 0.5 m 막대 n개 (막대 사이 0.5 m), 차로 폭 전체를 덮게
            s0 = (D - (n * 1.0 - 0.5)) / 2              # 막대들을 가운데 정렬
            for i in range(n):
                t0, t1 = s0 + i * 1.0, s0 + i * 1.0 + 0.5
                c0 = (a[0] + ux * t0, a[1] + uy * t0)
                c1 = (a[0] + ux * t1, a[1] + uy * t1)
                add("crossing", CROSSING_COLOR, [(c0[0] + nx, c0[1] + ny), (c1[0] + nx, c1[1] + ny),
                                                 (c1[0] - nx, c1[1] - ny), (c0[0] - nx, c0[1] - ny)], True, L + 0.004)
    # 범례 (서쪽 빈 곳, 위에서부터 높은 층)
    LEG_X, LEG_Y, BOX = -560.0, 860.0, 22.0
    # 글자는 sumo-gui 내장 글꼴에 한글이 없어 숫자·영어로 (한글 범례는 environment/road/README.md)
    legend = [("+6 m  elevated", BANDS[0][3]), ("0 m  ground", GROUND_COLOR),
              ("-3 to -8 m  slope", BANDS[1][3]), ("-10 m  riverside (Jamsu)", BANDS[2][3])]
    NBSP = " "                   # POI ID에는 공백·쉼표를 못 쓴다  /  줄바꿈 없는 공백
    # 글자는 POI 위치에 가운데 정렬되므로 글자 폭(약 4.2 m/글자, poiName_size=120 기준)의 반만큼 오른쪽으로 민다
    pois = []
    for i, (label, c) in enumerate(legend):
        y = LEG_Y - i * (BOX + 10)
        sq = [(LEG_X, y), (LEG_X + BOX, y), (LEG_X + BOX, y - BOX), (LEG_X, y - BOX)]
        add("legend", c, sq, True, 20.0)
        add("legend_edge", "60,60,60", sq + sq[:1], False, 20.01, 0.6)
        pois.append(f'    <poi id="{label.replace(" ", NBSP)}" type="legend" color="0,0,0,0" x="{LEG_X + BOX + 9 + len(label) * 2.1:.1f}" '
                    f'y="{y - BOX / 2:.1f}" layer="20"/>')
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("<additional>\n" + "\n".join(polys + pois) + "\n</additional>\n")


# ── 배경 (도로 이외 지역) : 화면 표시용, 주행에는 영향 없음 ──
# 실제 지도와 모양이 다르다: 도로를 직선·원호로 단순화했으므로 배경도 우리 좌표에 맞춰 직접 그린다 (위치는 대략).
# 색은 도로 색과 겹치지 않게 옅게: 물 = 회색빛 하늘색, 녹지 = 아주 연한 초록, 시가지 = 연한 회색
# (파랑은 버스, 초록·연두는 둔치·경사로 도로, 갈색은 고가가 이미 쓴다)
BG_WATER = "175,208,225"
BG_GREEN = "205,232,195"
BG_CITY = "232,232,228"
BG_TERMINAL = "218,216,226"
BG_ISLAND = "240,225,170"
BG_PARKING = "205,205,200"


def _write_background(S, out_path: str):
    """한강 · 세빛섬 · 반포한강공원 · 올림픽대로 중앙녹지 · 시가지 블록 · 지명·도로명 (영어 : sumo-gui 글꼴에 한글 없음)."""
    x0, x1 = S.X_W, S.X_WB_IN                       # 올림픽대로 양 끝
    wb_top = S.WB4 + 5 * S.W                        # 김포 방향 가장 바깥 차로 끝
    river = S.RC[1] + S.RR + 33                     # 둔치 끝 = 강가 (회전교차로 북쪽 약 33 m)
    rect = lambda xa, ya, xb, yb: [(xa, ya), (xb, ya), (xb, yb), (xa, yb)]
    areas = [  # (이름, 색, 도형)
        # 지도 북쪽 끝 = 가장 북쪽 도로 끝(잠수교 남행 진입, 11/13 116 m로 늘림) = 반포대교 · 잠수교 북행 끝(Y_END)
        ("han_river", BG_WATER, rect(x0, river, x1, S.NODES["jam_in"][1])),
        ("hangang_park", BG_GREEN, rect(x0, wb_top + 3, x1, river)),
        ("olympic_median", BG_GREEN, rect(x0, S.Y_EB + 2, x1, S.Y_WB - 2)),
        # 신반포사거리 네 귀퉁이 (반포대로 x ±45, 신반포로 y 325~400 비움)
        ("block_nw", BG_CITY, rect(x0, 400, -45, S.Y_EB - 30)),
        ("block_ne", BG_CITY, rect(45, 400, x1, S.Y_R2 - 15)),
        ("block_sw", BG_CITY, rect(x0, S.Y_ENTRY, -45, 310)),
        ("bus_terminal", BG_TERMINAL, rect(45, 60, 330, 310)),
        ("block_se", BG_CITY, rect(45, S.Y_ENTRY, 330, 40)),
        # 반포한강공원 주차장 (주차장 진입·출구 도로 끝)
        ("parking", BG_PARKING, rect(-218, S.RC[1] + 26, -152, river - 3)),
        # 세빛섬 (반포대교 서쪽 강가의 인공섬 3개)
        ("sebit_1", BG_ISLAND, rect(-150, river + 17, -118, river + 45)),
        ("sebit_2", BG_ISLAND, rect(-110, river + 23, -84, river + 51)),
        ("sebit_3", BG_ISLAND, rect(-76, river + 15, -58, river + 37)),
    ]
    labels = [  # (글자, x, y) : 글자는 POI 위치에 가운데 정렬
        ("Han River", 330, river + 110),
        ("Sebitseom", -104, river + 66),
        ("Parking", -185, (S.RC[1] + 26 + river - 3) / 2),
        ("Banpo Hangang Park", 470, S.RC[1] + 15),
        ("Banpo Bridge", 70, S.Y_TOP - 40),
        ("Jamsu Bridge", -100, S.Y_TOP - 40),
        ("Olympic-daero", 650, (S.Y_EB + S.Y_WB) / 2),
        ("Banpo-daero", 80, S.Y_ENTRY + 60),
        ("Sinbanpo-ro", -160, 390),
        ("Express Bus Terminal", 190, 185),
        ("Residential (Banpo)", -230, 650),
        ("Residential (Banpo) ", 430, 640),
        ("Residential (Banpo)  ", -230, 60),
        ("Residential (Banpo)   ", 190, -80),
    ]
    nbsp = " "                                 # POI ID에는 공백을 못 쓴다
    lines = []
    for name, color, pts in areas:
        shape = " ".join(f"{x:.1f},{y:.1f}" for x, y in pts)
        layer = -4 if name.startswith("sebit") else -5      # 섬은 강 위에
        lines.append(f'    <poly id="bg_{name}" type="background" color="{color}" fill="1" layer="{layer}" shape="{shape}"/>')
    for text, x, y in labels:
        lines.append(f'    <poi id="{text.replace(" ", nbsp)}" type="label" color="0,0,0,0" x="{x:.1f}" y="{y:.1f}" layer="20"/>')
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("<additional>\n" + "\n".join(lines) + "\n</additional>\n")
