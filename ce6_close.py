# ce6_close.py —— 决定性收尾: t0=17:27 固定, fly 112.0h+0~36min 自由, i 细扫
import diyue6 as d
import numpy as np, datetime

d0 = (datetime.date(2024, 5, 3) - datetime.date(1858, 11, 17)).days + 2400000.5 - 8.0/24
t0 = d0 + (17.0 + 27.0/60.0)/24.0
for inc in np.arange(23.15, 23.451, 0.05):
    d.INC_TGT = float(inc)
    best = None
    for fmin in np.arange(0.0, 36.1, 1.0):
        fh = (112.0*60.0 + fmin)/60.0/24.0
        F, ev = d._probe(t0, fh, 500.0, {})   # 每次冷启动, 防 warm 链漏解
        if ev is not None:
            r = (abs(ev['coast'] - 937.5), fh*24.0, ev)
            if best is None or r[0] < best[0]: best = r
    if best is None:
        print("i=%.2f  -- 无解(带未穿过实飞点火)" % inc, flush=True)
    else:
        ev = best[2]
        print("i=%.2f  fly=%.2fh 滑%.1fmin 近月高%.0fkm az%.1f° → 到达 %s" % (
            inc, best[1], ev['coast']/60.0, ev['d']['r_p'] - d.r_moon,
            ev['az'], d.to_bj(ev['tL'] + ev['fly'])), flush=True)
