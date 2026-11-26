"""SUMO 차로 ID를 차량 기준 차로 번호로 변환한다.

보도가 0번 차로에 붙은 엣지에서는 차량 차로 번호가 하나씩 밀린다.
보도는 None, 내부 차로와 알 수 없는 엣지는 원래 SUMO 번호를 돌려준다.
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
    """'엣지_번호'  /  차량 기준 차로 번호 (보도는 None). 교차점 내부(':'로 시작) · 모르는 엣지는 SUMO 번호 그대로."""
    edge, _, idx = lane_id.rpartition("_")
    i = int(idx)
    k = _sidewalk_count(net).get(edge, 0)
    if i < k:
        return None
    return i - k
