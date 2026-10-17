# ce6_plane.py —— 判决实验一: 实飞两点平面直算, 反演 CE-6 停泊倾角
import diyue5 as d
import numpy as np, math, datetime

def jd(y, m, dd, h_bj, mi_bj):
    return ((datetime.date(y, m, dd) - datetime.date(1858, 11, 17)).days
            + 2400000.5 - 8.0/24 + (h_bj + mi_bj/60.0)/24.0)

t0 = jd(2024, 5, 3, 17, 27)      # 实飞点火(北京时)
tE = jd(2024, 5, 8, 10, 12)      # 实飞近月制动(北京时)
r_site = d.site_hat(t0)
rM = d._moon(tE)
n = np.cross(r_site, rM); n /= np.linalg.norm(n)
print("反演: i* = %.2f° | az = %.1f°" % (
    math.degrees(math.acos(abs(float(n[2])))), d.site_azimuth(n, r_site)))
for br in (+1, -1):
    nm = d.plane_from_site(r_site, 21.299, br)
    if nm is not None:
        print("与 CE-5 标定平面(分支%+d, 21.299°)夹角: %.2f°" % (
            br, math.degrees(math.acos(min(1.0, abs(float(np.dot(n, nm))))))))
