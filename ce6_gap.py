# ce6_gap.py —— 实飞点 (05-03 17:27, fly=112.2h) 全φ域扫描, 判间隙真伪
import diyue6 as d
import numpy as np, math, datetime

d0 = (datetime.date(2024, 5, 3) - datetime.date(1858, 11, 17)).days + 2400000.5 - 8.0/24
t0 = d0 + (17.0 + 27.0/60.0)/24.0
FLY = 112.2/24.0
hits = []
for phid in np.arange(math.degrees(d.PHI_LO), math.degrees(d.PHI_HI) + 1e-9, 1.0):
    F, ev = d._probe(t0, FLY, 500.0, {}, math.radians(phid))   # 每档冷启, 防 warm 串扰
    if ev is not None:
        hits.append(phid)
        print("φ=%5.1f° 滑%5.1fmin 近月高%5.1fkm arc%4.0fkm → 到达 %s" % (
            phid, ev['coast']/60.0, ev['d']['r_p'] - d.r_moon, ev['arc_up'],
            d.to_bj(ev['tL'] + ev['fly'])), flush=True)
    else:
        print("φ=%5.1f° min|r_p−1937.4|=%6.0f km" % (phid, F), flush=True)
print(">>> 可行 φ: %s" % (["%.1f" % p for p in hits] if hits else "无 → 物理折叠"))
