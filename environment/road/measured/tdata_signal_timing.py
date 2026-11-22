"""t-data 10384 신호 기록(1초 간격)에서 교차로별 이동류 녹색 · 황색 길이, 주기, 시작 위치를 뽑는다 (11/13 신호 실측 분석, 1회용 기록).

사용: python -X utf8 environment/road/measured/tdata_signal_timing.py environment/road/measured/tdata_10384_신호기록_<날짜>_10-16시_반포권역.csv.gz
결과를 road_spec.SIGNALS에 옮겨 적었다 (environment/road/measured/README.md).
- 시작 위치는 신반포사거리(4031) 동행 직진 녹색 시작을 0으로 한 주기(170 s) 안의 위치다. 주기가 75 s인 잠수교 두 신호
  (4616 · 24616)는 사거리와 연동되지 않아 이 값이 의미 없다 — 그 둘은 녹색 길이와 주기만 쓴다.
"""
import collections, csv, gzip, sys
fn = sys.argv[1]; CYC = 170
code = {'protected-Movement-Allowed': 'G', 'permissive-Movement-Allowed': 'g', 'protected-clearance': 'y',
        'permissive-clearance': 'f', 'stop-And-Remain': 'r', 'dark': 'd'}
data = collections.defaultdict(dict)
for r in csv.DictReader((gzip.open if fn.endswith('.gz') else open)(fn, 'rt', encoding='utf-8')):
    if not (10 <= int(r['TRSM_HR']) < 16): continue
    t = int(r['TRSM_UTC_TIME']) // 1000
    data[r['ITST_ID']][t] = {k[:-8]: code.get(v, '?') for k, v in r.items() if k.endswith('_STAT_NM') and v}
def segs(d, col, states):
    """완전히 관측된 구간만: (시작, 길이) — 앞뒤 전환이 1~2초 간격 안에서 보인 것."""
    ts = sorted(d); out = []; start = None
    for a, b in zip(ts, ts[1:]):
        if b - a > 2: start = None; continue
        ina, inb = d[a].get(col, '-') in states, d[b].get(col, '-') in states
        if not ina and inb: start = b
        if ina and not inb and start is not None: out.append((start, b - start)); start = None
    return out
ref = [s for s, _ in segs(data['4031'], 'WT_STSG', 'Gg')]
def pos(t):
    prev = [x for x in ref if x <= t]
    return (t - prev[-1]) if prev and t - prev[-1] < CYC else None
for iid in ('4031', '24032', '4032', '4616', '24616'):
    d = data.get(iid, {})
    if not d: continue
    print(f'== {iid}')
    for c in sorted({c for v in d.values() for c in v}):
        g = segs(d, c, 'Gg'); G = segs(d, c, 'G'); y = segs(d, c, 'yf')
        if not g: print(f'   {c:8s} 항상 {collections.Counter(v.get(c) for v in d.values()).most_common(1)}'); continue
        L = collections.Counter(l for _, l in g).most_common(2)
        LG = collections.Counter(l for _, l in G).most_common(1)
        Y = collections.Counter(l for _, l in y).most_common(1)
        P = collections.Counter(pos(s) for s, _ in g if pos(s) is not None).most_common(3)
        cyc = collections.Counter(b - a for (a, _), (b, _) in zip(g, g[1:])).most_common(2)
        print(f'   {c:8s} 녹색 {L} (G만 {LG})  황색 {Y}  시작위치(ix 동행직진 기준) {P}  주기 {cyc}  n={len(g)}')
