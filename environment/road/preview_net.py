"""netconvert가 만든 도로망(.net.xml)의 차로를 그대로 그려 기준도와 눈으로 비교한다.

사용: python -X utf8 environment/road/preview_net.py [--scenario before|after]
결과: environment/road/build/<scenario>/net_preview.png (전체), net_preview_merge.png (합류 ~ 반포대교 남단 확대), net_preview_ix.png (신반포사거리 확대)
"""
import argparse, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import road_spec as S
sh = os.environ.get("SUMO_HOME", r"C:\Program Files (x86)\Eclipse\Sumo")
sys.path.append(os.path.join(sh, "tools"))
import sumolib
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "Malgun Gothic"; plt.rcParams["axes.unicode_minus"] = False
HERE = os.path.dirname(os.path.abspath(__file__))

def draw(net, ax, box):
    ax.set_facecolor("#dfe8d6"); ax.set_aspect("equal"); ax.set_xlim(box[0], box[1]); ax.set_ylim(box[2], box[3])
    ax.set_xticks([]); ax.set_yticks([])
    for e in net.getEdges(withInternal=True):
        internal = e.getID().startswith(":")
        for l in e.getLanes():
            xs, ys = zip(*l.getShape())
            col = "#b5534f" if l.allows("bus") and not l.allows("passenger") else ("#777" if internal else "#4a4a4a")
            if e.getID() in ("m1",) and l.getIndex() == 4: col = "#9a7cc4"
            if e.getID().startswith(("und_",)): col = "#2f2f2f"
            if e.getID().startswith("jam_"): col = "#a9a2b8"
            ax.plot(xs, ys, color=col, lw=l.getWidth() * (0.9 if not internal else 0.6) * 72 / 25.4 * ax.figure.dpi / 100 / max(1, (box[1] - box[0]) / 250),
                    solid_capstyle="butt", zorder=1 if internal else 2)
    for t in net.getTrafficLights():
        n = net.getNode(t.getID()) if net.hasNode(t.getID()) else None
        if n: ax.plot(*n.getCoord(), "o", color="#d62728", ms=8, zorder=5)

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--scenario", choices=S.SCENARIOS, default="before")
    a = ap.parse_args()
    out = os.path.join(HERE, "build", a.scenario)
    net = sumolib.net.readNet(os.path.join(out, "banpo.net.xml"), withInternal=True)
    xmin, ymin, xmax, ymax = net.getBBoxXY()[0] + net.getBBoxXY()[1]
    fig, ax = plt.subplots(figsize=(12, 15)); draw(net, ax, (xmin - 10, xmax + 10, ymin - 10, ymax + 10))
    ax.set_title(f"netconvert 결과 도로망 ({a.scenario}) — 전체", fontsize=12)
    fig.savefig(os.path.join(out, "net_preview.png"), dpi=110, bbox_inches="tight"); plt.close(fig)
    fig, ax = plt.subplots(figsize=(14, 9)); draw(net, ax, (-140, 260, S.Y_MERGE - 20, S.RC[1] + 60))
    ax.set_title(f"netconvert 결과 도로망 ({a.scenario}) — 합류 ~ 반포대교 남단 확대 (보라 = ① 전용 차로)", fontsize=12)
    fig.savefig(os.path.join(out, "net_preview_merge.png"), dpi=130, bbox_inches="tight"); plt.close(fig)
    fig, ax = plt.subplots(figsize=(9, 9)); draw(net, ax, (-90, 90, S.Y_IX - 70, S.Y_IX + 70))
    ax.set_title(f"netconvert 결과 도로망 ({a.scenario}) — 신반포사거리 확대 (빨간 차로 = 버스전용)", fontsize=12)
    fig.savefig(os.path.join(out, "net_preview_ix.png"), dpi=130, bbox_inches="tight"); plt.close(fig)
    print("ok", out)

if __name__ == "__main__":
    main()
