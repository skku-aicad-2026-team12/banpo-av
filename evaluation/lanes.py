"""SUMO 차로 id → 차량 기준 차로 번호 (분석용, 11/15).

데이터의 state.ego_lane · vehicles[].lane · event.lane_before/after는 SUMO 차로 번호 그대로다.
보도가 있는 엣지(road_spec.SIDEWALKS — 신반포사거리 네 갈래 · 잠수교 등)는 netconvert가 0번에 보도를 넣어
차량 차로 번호가 1씩 밀린다 (예: side_a4의 맨 오른쪽 차량 차로 = 1, 보도 없는 side_a4u에서는 0).
같은 기준으로 비교하려면 lane_id(state.ego_lane_id · vehicles[].lane_id)를 이 함수로 바꾼다.
이 값은 기존 칼럼 + 도로망으로 계산되므로 데이터 칼럼으로 따로 저장하지 않는다.

    from analysis.lanes import veh_lane
    veh_lane("side_a4_1")   # → 0 (맨 오른쪽 차량 차로)
    veh_lane("m1_3")        # → 3 (보도 없는 엣지는 그대로)

버스전용 차로는 차량 차로로 센다 (0 = 맨 오른쪽 차량 차로, 버스 차로든 아니든).
"""
import os
import xml.etree.ElementTree as ET
from functools import lru_cache

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NET = os.path.join(ROOT, "environment", "env", "sumo", "banpo", "banpo.net.xml")   # 수업 환경 도로망 (view_road · check_env가 만든다)


@lru_cache(maxsize=1)
def _sidewalk_count(net=NET):
    """엣지별 보행자 전용 차로 수 (오른쪽 끝부터 이어진 것만)."""
    out = {}
    for e in ET.parse(net).getroot().iter("edge"):
        if e.get("function"):
            continue
        n = 0
        for ln in sorted(e.iter("lane"), key=lambda l: int(l.get("index"))):
            if ln.get("allow") == "pedestrian":
                n += 1
            else:
                break
        out[e.get("id")] = n
    return out


def veh_lane(lane_id, net=NET):
    """'엣지_번호' → 차량 기준 차로 번호 (보도는 None). 교차점 내부(':'로 시작) · 모르는 엣지는 SUMO 번호 그대로."""
    edge, _, idx = lane_id.rpartition("_")
    i = int(idx)
    k = _sidewalk_count(net).get(edge, 0)
    if i < k:
        return None
    return i - k
