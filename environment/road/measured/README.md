# 실측 자료

도로망의 교통 수요 · 신호 · 버스를 정할 때 쓴 실측 자료다. 값을 어떻게 정했는지는 [`docs/프로젝트_기록.md`](../../../docs/프로젝트_기록.md) §5, 설정 위치는 수요 · 버스 `environment/env/road_config.py`, 신호 `environment/road/road_spec.py` (`SIGNALS`).

| 파일 | 내용 | 출처 |
|---|---|---|
| `topis_2026_09_반포권역_시간대평균.csv` | 잠수교 통제 전 교통량 — 지점 C-11 · C-12 · F-01 · D-36 · D-37의 방향 · 평일/주말별 0~23시 평균 (2026년 9월, 평일 20일). `environment/road/calibrate.py`가 읽는다 | 서울시 TOPIS 자료실 "교통량 조사자료" <https://topis.seoul.go.kr/refRoom/openRefRoom_2.do> |
| `tdata_10081_지점차로별_1시간_20261107-11_반포권역.csv` | 잠수교 통제 후 교통량 — 지점 · 차로별 1시간 교통량 (11/7~11) | 서울 교통빅데이터플랫폼(t-data) 10081 |
| `tdata_10384_신호기록_20261021_10-16시_반포권역.csv.gz` · `…_20261111_…` | 범위 안 신호 5곳의 C-ITS 신호제어기 기록 (1초 간격) — 통제 전 10/21 · 후 11/11 평일 10~16시 | t-data 10384 |
| `tdata_signal_timing.py` | 위 신호 기록에서 이동류별 녹색 · 황색 길이, 주기, 시작 위치를 뽑는 스크립트 | (작성) |
| `tdata_bus_노선분류_반포권역.txt` | 범위 안 정류소 6곳을 지나는 버스 노선의 진입 → 진출 방향 · 공식 배차 · 정차 정류소 | t-data 버스정보시스템 10246 · 10251 · 10257 · 10261 · 10262 |

```bash
# 측정 지점 교통량 검산 (도로망을 먼저 만든다: environment/road/build_net.py)
python -X utf8 environment/road/calibrate.py
# 신호 시간 추출
python -X utf8 environment/road/measured/tdata_signal_timing.py environment/road/measured/tdata_10384_신호기록_20261111_10-16시_반포권역.csv.gz
```
