
import os, sys, math, datetime, functools, threading
from jplephem.spk import SPK
import numpy as np
from scipy.optimize import brentq

for _f in ('de405.bsp', 'de421.bsp'):
    if os.path.exists(_f):
        kernel = SPK.open(_f); KER = _f; break
else:
    raise SystemExit("当前目录需要 de405.bsp 或 de421.bsp")

muE, muM = 398600.4418, 4902.80
r_earth, r_moon = 6378.14, 1737.4
r_park = r_earth + 208.0
r_s = 66200.0
v_circ = math.sqrt(muE/r_park)
V_ORB = v_circ
N_ORB = math.sqrt(muE/r_park**3)
R_PL = r_moon + 400.0
lat0, lon0 = math.radians(19.61), math.radians(110.95)
OMEGA_E = 7.292115e-5
JD0 = 2459173.85
Y_LO, Y_HI = 10.70, 11.001
TT_UTC = 69.184
unit = lambda v: v/np.linalg.norm(v)
T_UP = 738.7
T_LI = 1676.2
COAST = T_LI - T_UP
INC_TGT = 21.299
Y_MEAS = 10.9124
C3_MEAS = -1.961
AZ_MEAS = 98.4
ARC_UP_LO, ARC_UP_HI = 500.0, 4000.0
AZ_LO, AZ_HI = 60.0, 150.0
LAM_LO, LAM_HI = 0.0, 135.0
TE_FR = np.arange(0.80, 0.9001, 0.01)
RP_TOL_DEF = 50.0
T_SEP = 2184.0
BURN = T_SEP - T_LI
T_IMP = T_LI + 0.5*BURN
PHI_S = N_ORB*(T_IMP - T_UP)
RP_FLOOR = r_moon + 100.0
T_PARK = 2*math.pi/N_ORB
COAST_IGN_LO, COAST_IGN_HI = 300.0, 1800.0
PHI_LO = N_ORB*(COAST_IGN_LO + 0.5*BURN)
PHI_HI = N_ORB*(COAST_IGN_HI + 0.5*BURN)
ARC_UP_TGT = 2000.0

def to_utc(jd):
    return (datetime.datetime(1858,11,17) + datetime.timedelta(days=jd-2400000.5)).strftime("%m-%d %H:%M:%S")

def to_bj(jd):
    return (datetime.datetime(1858,11,17) + datetime.timedelta(days=jd-2400000.5+8.0/24)).strftime("%m-%d %H:%M:%S")

def gmst_deg(jd):
    D = jd - 2451545.0; T = D/36525.0
    return (280.46061837 + 360.98564736629*D + 0.000387933*T*T - T*T*T/38710000.0) % 360.0

# v18.4: IAU1976 岁差 — site_hat 由"当日真赤道系"旋入 ICRF, 与 DE405 星历同帧
#       (J2000→2020 春分点漂移 0.268°; 实测使 ce5 射向残差 0.14°→0.03°)
_AS = 4.848136811e-6
def _prec_m(jd):                                   # IAU1976 岁差
    T = (jd - 2451545.0)/36525.0
    z1 = (2306.2181*T + 0.30188*T*T + 0.017998*T**3)*_AS
    z2 = (2306.2181*T + 1.09468*T*T + 0.018203*T**3)*_AS
    th = (2004.3109*T - 0.42665*T*T - 0.041833*T**3)*_AS
    def rz(a):
        c,s = math.cos(a), math.sin(a); return np.array([[c,-s,0],[s,c,0],[0,0,1]])
    def ry(a):
        c,s = math.cos(a), math.sin(a); return np.array([[c,0,s],[0,1,0],[-s,0,c]])
    return rz(-z2) @ ry(th) @ rz(-z1)

def site_hat(jd):
    th = math.radians(gmst_deg(jd)) + lon0; c = math.cos(lat0)
    r = np.array([c*math.cos(th), c*math.sin(th), math.sin(lat0)])
    return _prec_m(jd) @ r                         # date→ICRF

_KER_LOCK = threading.Lock()

@functools.lru_cache(maxsize=300000)
def _moon(jd):
    with _KER_LOCK:
        t = jd + TT_UTC/86400.0
        return kernel[3,301].compute(t) - kernel[3,399].compute(t)

@functools.lru_cache(maxsize=300000)
def _moonv(jd):
    with _KER_LOCK:
        t = jd + TT_UTC/86400.0
        _, v1 = kernel[3,301].compute_and_differentiate(t)
        _, v2 = kernel[3,399].compute_and_differentiate(t)
        return (v1 - v2)/86400.0

def plane_from_site(r_site, inc_deg, branch=+1):
    i = math.radians(inc_deg)
    sx, sy, sz = r_site
    A = sx*sx + sy*sy
    if A < 1e-12: return None
    d = -math.cos(i)*sz
    disc = math.sin(i)**2 - d*d/A
    if disc < 0: return None
    p = branch*math.sqrt(disc)
    sA = math.sqrt(A)
    n = np.array([d*sx/A - p*sy/sA, d*sy/A + p*sx/sA, math.cos(i)])
    return unit(n)

def site_azimuth(n_hat, r_site):
    v = np.cross(n_hat, r_site)
    th = math.atan2(r_site[1], r_site[0])
    east = np.array([-math.sin(th), math.cos(th), 0.0])
    north = unit(np.cross(r_site, east))
    return math.degrees(math.atan2(float(np.dot(v, east)), float(np.dot(v, north))))

def make_ctx(n_hat, M_E, v_M):
    h_M = float(np.dot(M_E, n_hat))
    if r_s*r_s - h_M*h_M <= 0.0: return None
    M_par = M_E - h_M*n_hat
    c_par = float(np.linalg.norm(M_par))
    r_sp = math.sqrt(r_s*r_s - h_M*h_M)
    xh = M_par/c_par
    yh = np.cross(n_hat, xh)
    return (c_par, r_sp, h_M, float(xh[0]), float(xh[1]), float(xh[2]),
            float(yh[0]), float(yh[1]), float(yh[2]),
            float(n_hat[0]), float(n_hat[1]), float(n_hat[2]),
            float(M_E[0]), float(M_E[1]), float(M_E[2]),
            float(v_M[0]), float(v_M[1]), float(v_M[2]))

try:
    import numba
except ImportError:
    raise SystemExit("需要 numba: pip install numba")

@numba.njit(cache=True, fastmath=False, nogil=True)
def core3_n(ctx, y, lam):
    c_par = ctx[0]; r_sp = ctx[1]; h_M = ctx[2]
    xx = ctx[3]; xy = ctx[4]; xz = ctx[5]
    yx = ctx[6]; yy = ctx[7]; yz = ctx[8]
    nx = ctx[9]; ny = ctx[10]; nz = ctx[11]
    Mx = ctx[12]; My = ctx[13]; Mz = ctx[14]
    vMx = ctx[15]; vMy = ctx[16]; vMz = ctx[17]
    lr = math.radians(lam)
    cl = math.cos(lr); sl = math.sin(lr)
    Sx = (Mx - h_M*nx) + r_sp*(-cl*xx + sl*yx)
    Sy = (My - h_M*ny) + r_sp*(-cl*xy + sl*yy)
    Sz = (Mz - h_M*nz) + r_sp*(-cl*xz + sl*yz)
    r_e2 = math.sqrt(Sx*Sx + Sy*Sy + Sz*Sz)
    eng_e = y*y*0.5 - muE/r_park
    if eng_e >= 0.0: return _C3F()
    a_e = -muE/(2.0*eng_e)
    h_e = r_park*y
    p_e = h_e*h_e/muE
    if p_e >= a_e: return _C3F()
    e_e = math.sqrt(1.0 - p_e/a_e)
    if r_e2 > a_e*(1.0+e_e)*0.999999: return _C3F()
    cf = (p_e - r_e2)/(e_e*r_e2)
    if cf > 1.0 or cf < -1.0: return _C3F()
    f2c = math.acos(cf)
    E2c = 2.0*math.atan(math.sqrt((1.0-e_e)/(1.0+e_e))*math.tan(f2c/2.0))
    t_e = (E2c - e_e*math.sin(E2c))*math.sqrt(a_e**3/muE)
    shx = Sx/r_e2; shy = Sy/r_e2; shz = Sz/r_e2
    tx = ny*shz - nz*shy; ty = nz*shx - nx*shz; tz = nx*shy - ny*shx
    vr = muE/h_e*e_e*math.sin(f2c)
    vt = muE/h_e*(1.0 + e_e*math.cos(f2c))
    vx = vr*shx + vt*tx; vy = vr*shy + vt*ty; vz = vr*shz + vt*tz
    Px = math.cos(f2c)*shx - math.sin(f2c)*tx
    Py = math.cos(f2c)*shy - math.sin(f2c)*ty
    Pz = math.cos(f2c)*shz - math.sin(f2c)*tz
    ux = vx - vMx; uy = vy - vMy; uz = vz - vMz
    rho_x = Sx - Mx; rho_y = Sy - My; rho_z = Sz - Mz
    if ux*rho_x + uy*rho_y + uz*rho_z >= 0.0: return _C3F()
    vinf2 = ux*ux + uy*uy + uz*uz
    hx = rho_y*uz - rho_z*uy; hy = rho_z*ux - rho_x*uz; hz = rho_x*uy - rho_y*ux
    hmag = math.sqrt(hx*hx + hy*hy + hz*hz)
    eng_m = vinf2*0.5 - muM/r_s
    if eng_m <= 0.0: return _C3F()
    a_m = -muM/(2.0*eng_m)
    p_m = hmag*hmag/muM
    e_m = math.sqrt(1.0 - p_m/a_m)
    r_p = p_m/(1.0 + e_m)
    cf2 = (p_m/r_s - 1.0)/e_m
    if cf2 > 1.0 or cf2 < -1.0: return _C3F()
    f_in = math.acos(cf2)
    coshH = (1.0 - r_s/a_m)/e_m
    if coshH < 1.0: return _C3F()
    H = math.acosh(coshH)
    t_hyp = math.sqrt((-a_m)**3/muM)*(e_m*math.sinh(H) - H)
    return (1.0, t_e, t_hyp, r_p, math.sqrt(vinf2), f_in,
            y*y - 2.0*muE/r_park, h_M,
            Sx, Sy, Sz, vx, vy, vz, Px, Py, Pz)

@numba.njit(cache=True, nogil=True)
def _C3F():
    return (0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0)

@functools.lru_cache(maxsize=200000)
def core3(ctx, y, lam):
    ok, t_e, t_hyp, r_p, vinf, f2, C3, hM, Sx,Sy,Sz, vx,vy,vz, Px,Py,Pz = core3_n(ctx, y, lam)
    if ok < 0.5: return None
    return dict(t_e=t_e, t_hyp=t_hyp, r_p=r_p, vinf=vinf, f2=f2, C3=C3, hM=hM,
                S=(Sx,Sy,Sz), vsc=(vx,vy,vz), P=(Px,Py,Pz))

def y_for_phase(ctx, lam, fly):
    def g(yv):
        d = core3(ctx, yv, lam)
        return np.nan if d is None else d['t_e'] + d['t_hyp'] - fly*86400.0
    ys = np.arange(Y_LO, Y_HI, 0.005)
    prev = g(ys[0])
    for y1, y2 in zip(ys[:-1], ys[1:]):
        cur = g(y2)
        if np.isfinite(prev) and np.isfinite(cur) and prev*cur < 0:
            return brentq(g, y1, y2, xtol=1e-6)
        prev = cur
    return None

def solve2_3d(ctx, fly, tol=0.0):
    pts = []
    for lam in np.arange(LAM_LO, LAM_HI + 1e-9, 2.5):
        yy = y_for_phase(ctx, lam, fly)
        if yy is None: continue
        d = core3(ctx, yy, lam)
        if d: pts.append((lam, d['r_p']))
    def g2(l):
        yy = y_for_phase(ctx, l, fly)
        if yy is None: return np.nan
        dd = core3(ctx, yy, l)
        return np.nan if dd is None else dd['r_p'] - R_PL
    cross = []
    for (l1, r1), (l2, r2) in zip(pts[:-1], pts[1:]):
        if (r1 - R_PL)*(r2 - R_PL) < 0:
            mid = np.linspace(l1, l2, 9)[1:-1]
            if any(not np.isfinite(g2(m)) for m in mid): continue
            cross.append(brentq(g2, l1, l2, xtol=1e-4))
    if tol > 0.0:
        cand = [(0.0, l) for l in cross] + \
               [(abs(rp - R_PL), l) for l, rp in pts if abs(rp - R_PL) <= tol and rp > r_moon]
        if not cand: return []
        cand.sort()
        out, seen = [], []
        for miss, l in cand[:6]:
            if any(abs(l - s) < 0.05 for s in seen): continue
            seen.append(l)
            yy = y_for_phase(ctx, l, fly)
            d = core3(ctx, yy, l)
            if d is not None and d['r_p'] > r_moon:
                out.append((yy, l, d))
        return out
    out = []
    for lam_s in cross:
        y_s = y_for_phase(ctx, lam_s, fly)
        d = core3(ctx, y_s, lam_s)
        if d is None: continue
        out.append((y_s, lam_s, d))
    return out


def loi1(vinf):
    a8 = (muM*(8*3600.0)**2/(4*math.pi**2))**(1/3)
    return math.sqrt(vinf**2 + 2*muM/R_PL) - math.sqrt(muM*(2/R_PL - 1/a8))

def perilune_hat(d, ctx):
    (_, _, _, _,_,_, _,_,_, _,_,_, Mx,My,Mz, vMx,vMy,vMz) = ctx
    rho = np.array(d['S']) - np.array([Mx,My,Mz])
    u = np.array(d['vsc']) - np.array([vMx,vMy,vMz])
    rhoh = unit(rho)
    tm = unit(u - float(np.dot(u, rhoh))*rhoh)
    return unit(math.cos(d['f2'])*rhoh - math.sin(d['f2'])*tm)

# ================= v18.2: 环月轨道面倾角(白道/月赤道) =================
# 路径A: IAU pck00010 线性极轴模型 (ICRF)
#   RA  = 269.9949 - 0.0031*T
#   DEC =  66.5392 + 0.0130*T    (T: 自 J2000 的儒略世纪数)
# 与 DE421/DE440 物理天平动偏差 <0.01°。de405/de440s 无 [3,310] 天平动段 → 路径A唯一选项。

def moon_pole(jd):
    T = (jd - 2451545.0)/36525.0          # 69s TT-UTC 在 T 中是 2e-8 世纪, 忽略
    ra  = math.radians(269.9949 - 0.0031*T)
    dec = math.radians(66.5392 + 0.0130*T)
    return np.array([math.cos(dec)*math.cos(ra),
                     math.cos(dec)*math.sin(ra),
                     math.sin(dec)])

def lunar_orbit_normal(d, ctx):
    """环月轨道面法向 ĥ (ICRF) = normalize(ρ × u), 与 perilune_hat 同源中间量"""
    (_, _, _, _,_,_, _,_,_, _,_,_, Mx,My,Mz, vMx,vMy,vMz) = ctx
    rho = np.array(d['S']) - np.array([Mx,My,Mz])
    u   = np.array(d['vsc']) - np.array([vMx,vMy,vMz])
    return unit(np.cross(rho, u))

def inc_white(d, ctx):
    """相对白道面(瞬时月球轨道面, 由 M_E × v_M 决定)的环月轨道倾角"""
    (_, _, _, _,_,_, _,_,_, _,_,_, Mx,My,Mz, vMx,vMy,vMz) = ctx
    k = unit(np.cross(np.array([Mx,My,Mz]), np.array([vMx,vMy,vMz])))
    h = lunar_orbit_normal(d, ctx)
    return math.degrees(math.acos(max(-1.0, min(1.0, float(np.dot(h, k))))))

def inc_meq(d, ctx, jd):
    """相对月赤道面的环月轨道倾角 (路径A)"""
    h = lunar_orbit_normal(d, ctx)
    return math.degrees(math.acos(max(-1.0, min(1.0, float(np.dot(h, moon_pole(jd)))))))
# ======================================================================

def verify3(fly, y, d, ctx):
    (_, _, _, _,_,_, _,_,_, nx,ny,nz, Mx,My,Mz, vMx,vMy,vMz) = ctx
    P = np.array(d['P']); S = np.array(d['S']); n_hat = np.array([nx,ny,nz])
    eng = y*y/2 - muE/r_park
    a = -muE/(2*eng); p = (r_park*y)**2/muE; e = math.sqrt(1 - p/a)
    sh = S/np.linalg.norm(S)
    ok_sense = float(np.dot(np.cross(P, S), n_hat)) > 0
    f = math.acos(max(-1.0, min(1.0, float(np.dot(P, sh)))))
    r_f = p/(1 + e*math.cos(f))
    ok_r = abs(r_f/np.linalg.norm(S) - 1.0) < 1e-4
    E = 2*math.atan(math.sqrt((1-e)/(1+e))*math.tan(f/2))
    t_chk = (E - e*math.sin(E))*math.sqrt(a**3/muE)
    ok_t = abs(t_chk - d['t_e']) < 0.5
    rho = S - np.array([Mx,My,Mz]); u = np.array(d['vsc']) - np.array([vMx,vMy,vMz])
    hm = float(np.linalg.norm(np.cross(rho, u)))
    eng_m = float(u@u)/2 - muM/r_s
    a_m = -muM/(2*eng_m); p_m = hm*hm/muM; e_m = math.sqrt(1 - p_m/a_m)
    r_p_t = p_m/(1 + e_m)
    ok_rp = abs(r_p_t/d['r_p'] - 1.0) < 1e-5
    ok_fly = abs((d['t_e'] + d['t_hyp'])/3600.0 - fly*24.0) < 0.02
    return (ok_r and ok_t and ok_rp and ok_fly and ok_sense), r_p_t

def solve_phi_s(n_hat, P, r_site):
    tdir = np.cross(n_hat, P)
    best = None
    for phid in np.arange(math.degrees(PHI_LO), math.degrees(PHI_HI) + 1e-9, 1.0):
        ph = math.radians(phid)
        r_b = r_park*(P*math.cos(ph) - tdir*math.sin(ph))
        arc = math.degrees(math.acos(max(-1.0, min(1.0,
              float(np.dot(r_site, r_b/np.linalg.norm(r_b)))))))*111.195
        if ARC_UP_LO <= arc <= ARC_UP_HI:
            sc = abs(ph - PHI_S)
            if best is None or sc < best[0]: best = (sc, ph, arc)
    return (None, None) if best is None else (best[1], best[2])

def _build(t0, t_arr, branch, n_hat, az, fly, y_s, lam_s, d, ctx, tE, phi_s=None, t_imp=None):
    phi_s = PHI_S if phi_s is None else phi_s
    t_imp = T_IMP if t_imp is None else t_imp
    t_b = t0 + T_UP/86400.0
    ok, r_p_t = verify3(fly, y_s, d, ctx)
    P = np.array(d['P'])
    tdir = np.cross(n_hat, P)
    r_b = r_park*(P*math.cos(phi_s) - tdir*math.sin(phi_s))
    th_b = math.radians(gmst_deg(t_b))
    lat_b = math.degrees(math.asin(max(-1.0, min(1.0, float(r_b[2]/r_park)))))
    lon_b = math.degrees(math.atan2(float(r_b[1]), float(r_b[0])) - th_b)
    lon_b = (lon_b + 180.0) % 360.0 - 180.0
    r_site = site_hat(t0)
    arc_up = math.degrees(math.acos(max(-1.0, min(1.0, float(np.dot(r_site, r_b/np.linalg.norm(r_b)))))))*111.195
    if not (ARC_UP_LO <= arc_up <= ARC_UP_HI): return None
    ph = perilune_hat(d, ctx)
    S = np.array(d['S'])
    (_,_,_, _,_,_, _,_,_, _,_,_, Mx,My,Mz, vMx,vMy,vMz) = ctx
    eta = math.degrees(math.acos(max(-1.0, min(1.0, float(np.dot(ph, -unit(S - np.array([Mx,My,Mz]))))))))
    coast_ign = phi_s/N_ORB - 0.5*BURN
    # v18.2: 环月轨道面与两个参考面的倾角 (LOI 共面假设下 = 双曲线面)
    h_hat = lunar_orbit_normal(d, ctx)
    k_w = unit(np.cross(np.array([Mx,My,Mz]), np.array([vMx,vMy,vMz])))
    i_white = math.degrees(math.acos(max(-1.0, min(1.0, float(np.dot(h_hat, k_w))))))
    i_meq   = math.degrees(math.acos(max(-1.0, min(1.0, float(np.dot(h_hat, moon_pole(tE)))))))
    return dict(t0=t0, t_b=t_b, tc=COAST, tL=t0 + t_imp/86400.0,
                t_li=t0 + (t_imp - 0.5*BURN)/86400.0, coast=coast_ign,
                fly=fly, y=y_s, lam=lam_s, d=d, az=az, inc=INC_TGT,
                arc_up=arc_up, lat_b=lat_b, lon_b=lon_b, branch=branch,
                tE=tE, hM=d['hM'], eta=eta,
                inc_white=i_white, inc_meq=i_meq, h_hat=h_hat,
                dv_tli=(y_s - v_circ)*1e3, dv_loi=loi1(d['vinf'])*1e3,
                ok=ok, r_p_t=r_p_t)

def solve_event(t0, t_arr, branch=None, tol=0.0, tE_warm=None):
    tL = t0 + T_IMP/86400.0
    fly = t_arr - tL
    if not (2.5 <= fly <= 5.5): return None
    r_site = site_hat(t0)
    branches = (branch,) if branch else (+1, -1)
    for br in branches:
        n_hat = plane_from_site(r_site, INC_TGT, br)
        if n_hat is None: continue
        az = site_azimuth(n_hat, r_site)
        if not (AZ_LO <= az <= AZ_HI): continue
        cands = [tL + fr*fly for fr in TE_FR if abs(float(np.dot(_moon(tL + fr*fly), n_hat))) < 0.97*r_s]
        if tE_warm is not None:
            cands = [tE_warm] + [c for c in cands if abs(c - tE_warm)*86400.0 > 300.0]
        for tE in cands:
            for it in range(6):
                M_E = _moon(tE); v_M = _moonv(tE)
                ctx = make_ctx(n_hat, M_E, v_M)
                if ctx is None: break
                ev = None
                for y_s, lam_s, d in solve2_3d(ctx, fly, tol):
                    ev = _build(t0, t_arr, br, n_hat, az, fly, y_s, lam_s, d, ctx, tE)
                    if ev is not None: break
                if ev is None: break
                tE_new = tL + ev['d']['t_e']/86400.0
                conv = abs(tE_new - tE)*86400.0 < 5.0
                tE = tE_new
                if conv: return ev
    return None

# ================= v18.3: 正向预测器 — 倾角由几何输出, h_M≡0 =================
def plane_optimal(r_site, M_E):
    """转移面 = 发射点 × 月球 张成的平面, 法向强制顺行"""
    n = np.cross(r_site, M_E)
    if np.linalg.norm(n) < 1e-12: return None
    n = unit(n)
    if n[2] < 0: n = -n
    return n

def solve_event_fwd(t0, t_arr, tol=0.0, tE_warm=None, phi_search=False):  # v18.5: +phi_search
    tL = t0 + T_IMP/86400.0
    fly = t_arr - tL
    if not (2.5 <= fly <= 5.5):
        return None
    r_site = site_hat(t0)
    cands = [tL + fr*fly for fr in TE_FR]          # fwd 模式 h_M≡0, 月球-平面预筛门恒过
    if tE_warm is not None:
        cands = [tE_warm] + [c for c in cands if abs(c - tE_warm)*86400.0 > 300.0]
    for tE in cands:
        for it in range(6):
            M_E = _moon(tE); v_M = _moonv(tE)
            n_hat = plane_optimal(r_site, M_E)     # 不动点内每轮重建
            if n_hat is None:
                break
            az = site_azimuth(n_hat, r_site)
            if not (AZ_LO <= az <= AZ_HI):
                break
            ctx = make_ctx(n_hat, M_E, v_M)
            if ctx is None:
                break
            t_imp_eff = T_IMP
            ev = None
            for y_s, lam_s, d in solve2_3d(ctx, fly, tol):
                br = +1 if az > 90.0 else -1       # 分支连续化: 由射向判(仅标签)
                ev = _build(t0, t_arr, br, n_hat, az, fly, y_s, lam_s, d, ctx, tE)
                if ev is None and phi_search:      # v18.5: 上升弧门兜底 — φ 搜索
                    phi_s, arc_t = solve_phi_s(n_hat, np.array(d['P']), r_site)
                    if phi_s is not None:
                        t_imp_eff = T_UP + phi_s/N_ORB
                        ev = _build(t0, t_arr, br, n_hat, az, fly, y_s, lam_s, d, ctx, tE, phi_s, t_imp_eff)
                if ev is not None:
                    break
            if ev is None:
                break
            tL_e = t0 + t_imp_eff/86400.0
            tE_new = tL_e + ev['d']['t_e']/86400.0
            conv = abs(tE_new - tE)*86400.0 < 5.0
            tE = tE_new
            if conv:
                ev['inc'] = math.degrees(math.acos(max(-1.0, min(1.0, float(n_hat[2])))))
                return ev                          # 倾角=输出, 覆写后 print_event 直显
    return None

# ==============================================================================

def diag(t0, t_arr):
    tL = t0 + T_IMP/86400.0
    fly = t_arr - tL
    print("[门1] 时序: fly=%.3fh %s" % (fly*24, "OK" if 2.5 <= fly <= 5.5 else "FAIL"))
    r_site = site_hat(t0)
    for br, name in ((+1, "东南(+1)"), (-1, "东北(-1)")):
        print("-" * 100)
        n_hat = plane_from_site(r_site, INC_TGT, br)
        if n_hat is None:
            print("[%s] 平面无解(|lat|>i)" % name); continue
        inc_eff = math.degrees(math.acos(min(1.0, abs(float(n_hat[2])))))
        az = site_azimuth(n_hat, r_site)
        print("[门2] %s 平面: 射向=%.2f° (实测 %.1f°) 有效倾角=%.3f° | 射向门 %s" % (
            name, az, AZ_MEAS, inc_eff, "OK" if AZ_LO <= az <= AZ_HI else "FAIL"))
        prof = [(tL + fr*fly, abs(float(np.dot(_moon(tL + fr*fly), n_hat)))) for fr in np.arange(0.70, 1.0001, 0.005)]
        tE_min, h_min = min(prof, key=lambda x: x[1])
        n_pass = sum(1 for _, h in prof if h < 0.97*r_s)
        print("[门3] h_M: min|h_M|=%.0f km @ tL+%.3f·fly | 硬门 %s | 引导网格过门 %d/%d" % (
            h_min, (tE_min - tL)/fly, "OK" if h_min < r_s else "FAIL(面与SOI不相交)", n_pass, len(prof)))
        if h_min >= r_s: continue
        best = None
        for fr in TE_FR:
            tE = tL + fr*fly
            if abs(float(np.dot(_moon(tE), n_hat))) >= 0.97*r_s: continue
            ctx = make_ctx(n_hat, _moon(tE), _moonv(tE))
            if ctx is None: continue
            for lam in np.arange(LAM_LO, LAM_HI + 1e-9, 2.5):
                yy = y_for_phase(ctx, lam, fly)
                if yy is None: continue
                d = core3(ctx, yy, lam)
                if not d: continue
                miss = abs(d['r_p'] - R_PL)
                if best is None or miss < best[0]: best = (miss, fr, tE, lam, yy, d)
        if best is None:
            print("[门4] 双钉: 全域无 tof 根或无 r_p 样本 → y 能量域/远地点门全灭"); continue
        miss, fr, tE, lam_a, y_a, d_a = best
        print("[门4] 双钉: min|r_p−2137.4|=%+.0f km @ tE=tL+%.3f·fly λ=%.1f° y=%.5f" % (
            d_a['r_p'] - R_PL, fr, lam_a, y_a))
        print("    分解: t_e=%.1fh t_hyp=%.1fh (和=%.2fh vs fly=%.2fh) vinf=%.3f C3=%.3f hM=%+.0f" % (
            d_a['t_e']/3600, d_a['t_hyp']/3600, (d_a['t_e']+d_a['t_hyp'])/3600, fly*24,
            d_a['vinf'], d_a['C3'], d_a['hM']))
        print("    → 近月点门: 粗样本偏差 %+.0f km (2.5°λ网格分辨率内; solve_event 精化后→0, 参见 ce5 对账)"
              % (d_a['r_p'] - R_PL))
        ctx = make_ctx(n_hat, _moon(tE), _moonv(tE))
        ev = _build(t0, t_arr, br, n_hat, az, fly, y_a, lam_a, d_a, ctx, tE)
        if ev is None:
            P = np.array(d_a['P']); tdir = np.cross(n_hat, P)
            r_b = r_park*(P*math.cos(PHI_S) - tdir*math.sin(PHI_S))
            arc = math.degrees(math.acos(max(-1.0, min(1.0, float(np.dot(r_site, r_b/np.linalg.norm(r_b)))))))*111.195
            print("[门5] 发射几何: arc_up=%.0f km → 门[500,4000] %s" % (
                arc, "OK" if ARC_UP_LO <= arc <= ARC_UP_HI else "FAIL"))
        else:
            print("[门5] 发射几何: arc_up=%.0f km OK → 最近可达样本(严格时序解不存在, 此为 r_p-最近构造)" % ev['arc_up'])
            print_event(ev)

def print_event(e):
    d = e['d']
    print("点火[%s] 入轨[%s] 滑%5.1fmin TLI点火[%s] 脉冲[%s] | fly=%6.2fh TLIΔv=%4.0f LOI1=%3.0f" % (
        to_bj(e['t0']), to_bj(e['t_b']), e['coast']/60, to_bj(e['t_li']), to_bj(e['tL']),
        e['fly']*24, e['dv_tli'], e['dv_loi']))
    print("  C3=%5.2f λ=%5.1f° η=%5.1f° 射向=%5.1f°(%s支) 倾角=%5.2f° 上升弧=%4.0fkm "
          "入轨星下点(%5.2f°N,%6.2f°E) hM=%+6.0fkm r_p高=%.0fkm 自检%s" % (
        d['C3'], e['lam'], e['eta'], e['az'], "东南" if e['branch'] > 0 else "东北",
        e['inc'], e['arc_up'], e['lat_b'], e['lon_b'], e['hM'], d['r_p'] - r_moon,
        "PASS" if e['ok'] else "FAIL"))
    print("  环月轨道: 白道倾=%5.1f° 月赤倾=%5.1f° (LOI共面假设下=双曲线面)" % (          # v18.2
        e['inc_white'], e['inc_meq']))

def merge_windows(evs, gap_h=1.0):
    evs = sorted(evs, key=lambda e: e['t0'])
    segs = []
    for e in evs:
        if segs and (e['t0'] - segs[-1]['t1'])*24.0 <= gap_h:
            segs[-1]['t1'] = e['t0']; segs[-1]['n'] += 1
        else:
            segs.append(dict(t0=e['t0'], t1=e['t0'], n=1))
    return segs

def scan(t0_lo, t0_hi, t_step_h, fly_lo=3.0, fly_hi=4.95, fly_step=0.05, quiet=True):
    t0s = np.arange(t0_lo, t0_hi + 1e-9, t_step_h/24.0)
    flys = np.arange(fly_lo, fly_hi + 1e-9, fly_step)
    evs = []
    for i, t0 in enumerate(t0s):
        if not quiet:
            print("  点火 %s ..." % to_bj(t0), end="\r", flush=True)
        for fly in flys:
            e = solve_event(t0, t0 + fly)
            if e is not None: evs.append(e)
    if not quiet: print()
    return evs

def has_sol(t0, fly, tol=0.0):
    return solve_event(t0, t0 + fly, tol=tol) is not None

def refine_edge(t_in, t_out, fly, tol=0.0, tol_min=1.0):
    lo, hi = (t_in, t_out) if t_in < t_out else (t_out, t_in)
    f_lo = has_sol(lo, fly, tol)
    while (hi - lo)*86400.0 > tol_min*60.0:
        m = 0.5*(lo + hi)
        if has_sol(m, fly, tol) == f_lo: lo = m
        else: hi = m
    return lo if f_lo else hi


Y_WIN_LO = 10.78

def _y_fast(ctx, lam, fly):
    T = fly * 86400.0
    def tof(yv):
        d = core3(ctx, yv, lam)
        return None if d is None else d['t_e'] + d['t_hyp']
    y1 = Y_WIN_LO
    f1 = tof(y1)
    while f1 is None and y1 < Y_HI - 0.0005:
        y1 = min(y1 + 0.005, Y_HI - 0.0005)
        f1 = tof(y1)
    if f1 is None: return None
    y2 = Y_HI - 0.0005
    f2 = tof(y2)
    if f2 is None or f2 >= f1: return None
    if not (f2 <= T <= f1): return None
    y3 = y2
    for _ in range(30):
        y3 = y2 + (T - f2)*(y1 - y2)/(f1 - f2)
        y3 = min(max(y3, y1), Y_HI - 0.0005)
        f3 = tof(y3)
        if f3 is None: return None
        if abs(f3 - T) < 2.0: return y3
        if (f1 - T)*(f3 - T) <= 0.0:
            y2, f2 = y3, f3
        else:
            y1, f1 = y3, f3
    return y3 if abs(f3 - T) < 10.0 else None

def _probe(t0, fly, tol_km, warm, phi_force=None):
    tL = t0 + T_IMP/86400.0
    if not (2.5 <= fly <= 5.5): return (9e9, None)
    r_site = site_hat(t0)
    F_best, ev_got = 9e9, None
    for br in (+1, -1):
        n_hat = plane_from_site(r_site, INC_TGT, br)
        if n_hat is None: continue
        az = site_azimuth(n_hat, r_site)
        if not (AZ_LO <= az <= AZ_HI): continue
        tE_w, lam_w = warm.get(br, (None, None))
        tE = tE_w if tE_w is not None else tL + 0.84*fly
        Fb = None
        phi_locked = None
        for _it in range(6):
            M_E = _moon(tE); v_M = _moonv(tE)
            ctx = make_ctx(n_hat, M_E, v_M)
            if ctx is None: break
            def scan(lams):
                out = []
                for lam in lams:
                    yy = _y_fast(ctx, lam, fly)
                    if yy is None: continue
                    d = core3(ctx, yy, lam)
                    if d and d['r_p'] > RP_FLOOR:
                        out.append((lam, d['r_p'], yy, d))
                return out
            pts = scan(np.arange(lam_w - 1.2, lam_w + 1.2 + 1e-9, 0.02)) if lam_w is not None else []
            thr = max(tol_km, 200.0)
            if (not pts) or min(abs(p[1] - R_PL) for p in pts) > thr:
                cpts = scan(np.arange(LAM_LO, LAM_HI + 1e-9, 2.5))
                if cpts:
                    l0 = min(cpts, key=lambda p: abs(p[1] - R_PL))[0]
                    pts = pts + cpts + scan(np.arange(max(LAM_LO, l0 - 2.0), min(LAM_HI, l0 + 2.0) + 1e-9, 0.02))
            if not pts: break
            lam_b, rp_b, yy_b, d_b = min(pts, key=lambda p: abs(p[1] - R_PL))
            Fb = abs(rp_b - R_PL)
            warm[br] = [tE, lam_b]
            if phi_force is not None:
                phi_s = phi_force
            elif phi_locked is not None:
                phi_s = phi_locked
            else:
                phi_s, arc_t = solve_phi_s(n_hat, np.array(d_b['P']), r_site)
                if phi_s is None: break
                phi_locked = phi_s
            coast_ign = phi_s/N_ORB - 0.5*BURN
            if not (COAST_IGN_LO <= coast_ign <= COAST_IGN_HI): break
            t_imp_e = T_UP + phi_s/N_ORB
            tL_e = t0 + t_imp_e/86400.0
            tE_new = tL_e + d_b['t_e']/86400.0
            if abs(tE_new - tE)*86400.0 >= 5.0:
                tE = tE_new
                continue
            if Fb > tol_km: break
            ev = None
            for dl in (0.0, 0.3, -0.3, 0.6, -0.6, 1.0, -1.0):
                lam_c = lam_b + dl
                if not (LAM_LO <= lam_c <= LAM_HI): continue
                if dl == 0.0:
                    yy, d = yy_b, d_b
                else:
                    yy = _y_fast(ctx, lam_c, fly)
                    if yy is None: continue
                    d = core3(ctx, yy, lam_c)
                    if d is None: continue
                    if d['r_p'] - R_PL > tol_km or d['r_p'] < RP_FLOOR: continue
                ev = _build(t0, tL_e + fly, br, n_hat, az, fly, yy, lam_c, d, ctx, tE, phi_s, t_imp_e)
                if ev is not None: break
            if ev is not None:
                ev_got = ev
                break
        if ev_got is not None: return (0.0, ev_got)
        if Fb is not None: F_best = min(F_best, Fb)
    return (F_best, None)

HMF_FR = np.arange(0.50, 0.9701, 0.02)
HMF_GATE = 1.08
DFLY_COARSE_H = 0.5

def _probe_gate(t0, fh):
    r_site = site_hat(t0)
    ns = []
    for br in (+1, -1):
        n_hat = plane_from_site(r_site, INC_TGT, br)
        if n_hat is not None and AZ_LO <= site_azimuth(n_hat, r_site) <= AZ_HI:
            ns.append(n_hat)
    if not ns: return False
    tL = t0 + T_IMP/86400.0
    flyd = fh/24.0
    for fr in HMF_FR:
        rM = _moon(tL + fr*flyd)
        for n_hat in ns:
            if abs(float(np.dot(rM, n_hat))) < HMF_GATE * r_s: return True
    return False

def traj_row(e):
    d = e['d']
    return dict(点火=to_bj(e['t0']), TLI点火=to_bj(e['t_li']), 等效脉冲=to_bj(e['tL']),
                fly_h=round(e['fly']*24.0, 2), 滑行min=round(e['coast']/60.0, 1),
                射向=round(e['az'], 1), 分支=("东南" if e['branch'] > 0 else "东北"),
                入轨纬度=round(e['lat_b'], 2), 入轨经度=round(e['lon_b'], 2),
                上升弧km=round(e['arc_up'], 0), TLIΔv=round(e['dv_tli'], 1),
                C3=round(d['C3'], 3), LOI1=round(e['dv_loi'], 1), vinf=round(d['vinf'], 3),
                到达=to_bj(e['tL'] + e['fly']), 近月高km=round(d['r_p'] - r_moon, 1),
                白道倾=round(e['inc_white'], 1), 月赤倾=round(e['inc_meq'], 1),
                自检=("PASS" if e['ok'] else "FAIL"))

def print_traj_row(r):
    print("  │ 点火%s TLI%s | fly=%6.2fh 滑%5.1fmin 射向%5.1f°(%s) 入轨(%5.2f°N,%6.2f°E) "
          "Δv=%6.1f C3=%6.2f LOI=%5.1f → 到达%s (近月高%.0fkm) 自检%s" % (
        r['点火'], r['TLI点火'], r['fly_h'], r['滑行min'], r['射向'], r['分支'],
        r['入轨纬度'], r['入轨经度'], r['TLIΔv'], r['C3'], r['LOI1'], r['到达'],
        r['近月高km'], r['自检']))

def write_csv(rows, path):
    if not rows: return
    import csv as _csv
    with open(path, "w", newline="", encoding="utf-8-sig") as fp:
        w = _csv.DictWriter(fp, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)

def run_win(date_str, tol_km=50.0, fly_lo=72.0, fly_hi=119.0, dfly_h=0.2, collect=False, n_traj=5, csv_path=None, warm0=None):
    y_, m_, d_ = map(int, date_str.split("-"))
    d0 = (datetime.date(y_, m_, d_) - datetime.date(1858, 11, 17)).days + 2400000.5 - 8.0/24
    if not collect:
        print("[v18] 联合窗口 %s (北京时) | tol=±%.0f km | fly∈[%.1f,%.1f]h 步%.0f min" % (
            date_str, tol_km, fly_lo, fly_hi, dfly_h*60))
    warm = dict(warm0[2]) if warm0 else {}
    seed_t0 = warm0[0] if warm0 else None
    slope, prev_c, prev_f = 0.0, None, None
    recs = []
    def hit_at(t0, fh):
        _, ev = _probe(t0, fh/24.0, tol_km, warm)
        return ev
    def best_on_grid(fh, use_seed=True):
        best = None
        t0s = list(d0 + np.arange(0.5, 24.0, 0.5)/24.0)
        if use_seed and seed_t0 is not None:
            t0s = [seed_t0] + t0s
        for t0c in t0s:
            if not _probe_gate(t0c, fh): continue
            _, ev = _probe(t0c, fh/24.0, tol_km, warm)
            if ev is None: continue
            sc = (abs(ev['coast'] - COAST), float(t0c))
            if best is None or sc < best[0]: best = (sc, float(t0c), ev)
            if sc[0] < 30.0: break
        return best
    def scan_fly(fhs, step_h, use_seed, out):
        nonlocal slope, prev_c, prev_f
        got = 0
        for fh in fhs:
            hit = None
            t0g = None
            if prev_c is not None and (fh - prev_f) <= 1.5*step_h:
                t0g = prev_c + slope*(fh - prev_f)/24.0
                for k in (0, -2, 2, -5, 5, -10, 10):
                    t0c = t0g + k/1440.0
                    if not _probe_gate(t0c, fh): continue
                    _, ev = _probe(t0c, fh/24.0, tol_km, warm)
                    if ev is not None: hit = (float(t0c), ev); break
            if hit is None:
                g = best_on_grid(fh, use_seed)
                if g is not None:
                    if t0g is not None and abs(g[1] - t0g) > 40.0/1440.0:
                        out.append((fh, None, None))
                        prev_c, prev_f = None, None
                    hit = (g[1], g[2])
            if hit is None:
                out.append((fh, None, None)); prev_c, prev_f = None, None
                continue
            t0c = hit[0]
            if prev_c is not None and fh > prev_f:
                slope = max(-1.5, min(0.5, (t0c - prev_c)/((fh - prev_f)/24.0)))
            prev_c, prev_f = t0c, fh
            out.append((fh, t0c, hit[1])); got += 1
        return got
    fhs_all = list(np.arange(fly_lo, fly_hi + 1e-9, dfly_h))
    if fhs_all[-1] < fly_hi - 1e-9: fhs_all.append(fly_hi)
    coarse_step = max(dfly_h, DFLY_COARSE_H)
    fhs_c = list(np.arange(fly_lo, fly_hi + 1e-9, coarse_step))
    if fhs_c[-1] < fly_hi - 1e-9: fhs_c.append(fly_hi)
    recs_a = []
    nA = scan_fly(fhs_c, coarse_step, True, recs_a)
    if nA == 0:
        scan_fly(fhs_all, dfly_h, True, recs)
    else:
        hits_c = sorted(r[0] for r in recs_a if r[1] is not None)
        clusters = []
        for fh in hits_c:
            if clusters and fh - clusters[-1][1] <= 2.0*coarse_step + 1e-9:
                clusters[-1][1] = fh
            else:
                clusters.append([fh, fh])
        for a, b in clusters:
            lo = max(fly_lo, a - 4.0*coarse_step); hi = min(fly_hi, b + 4.0*coarse_step)
            scan_fly([f for f in fhs_all if lo - 1e-9 <= f <= hi + 1e-9], dfly_h, False, recs)
    fd = {}
    for r in recs: fd[round(r[0], 9)] = r
    recs = [fd[k] for k in sorted(fd)]
    segs, cur = [], []
    for r in recs:
        gap = bool(cur) and (r[0] - cur[-1][0]) > 1.5*dfly_h + 1e-9
        if r[1] is None or gap:
            if cur: segs.append(cur); cur = []
        if r[1] is not None: cur.append(r)
    if cur: segs.append(cur)
    out_segs = []
    if not collect:
        print("解带 %d 段" % len(segs)); print("-" * 110)
    total = 0.0
    for si, seg in enumerate(segs):
        (f0, c0, _), (f1, c1, _) = seg[0], seg[-1]
        def edge(fh, c_in, sgn):
            a, b, t = c_in, None, c_in
            step = 0.5/1440.0
            for _ in range(30):
                t2 = t + sgn*step
                if hit_at(t2, fh) is not None:
                    a, t = t2, t2
                    step = min(step*2.0, 8.0/1440.0)
                else:
                    b = t2; break
            if b is None: b = t + sgn*step
            for _ in range(8):
                mm = 0.5*(a + b)
                if hit_at(mm, fh) is not None: a = mm
                else: b = mm
            return a
        eL, fL = min(((edge(f0, c0, -1), f0), (edge(f1, c1, -1), f1)), key=lambda p: p[0])
        eR, fR = max(((edge(f0, c0, +1), f0), (edge(f1, c1, +1), f1)), key=lambda p: p[0])
        ta0 = eL + T_IMP/86400.0 + fL/24.0
        ta1 = eR + T_IMP/86400.0 + fR/24.0
        w = (eR - eL)*1440.0
        total += w
        sl = (c1 - c0)/max(1e-9, (f1 - f0)/24.0)
        def traj_at(t0q, fh_hint=None):
            fh0 = fh_hint if fh_hint is not None else min(seg, key=lambda r: abs(r[1] - t0q))[0]
            for dh in (0.0, 0.1, -0.1, 0.25, -0.25, 0.5, -0.5, 1.0, -1.0, 1.5, -1.5, 2.0, -2.0):
                ev = hit_at(t0q, fh0 + dh)
                if ev is not None: return ev
            for back in (20.0, 40.0, 60.0):
                ev = hit_at(t0q - back/1440.0, fh0)
                if ev is not None: return ev
            return None
        rows = []
        t0_list = np.linspace(eL, eR, n_traj) if n_traj > 1 else [0.5*(eL + eR)]
        for i, t0q in enumerate(t0_list):
            fh_hint = None
            if n_traj > 1:
                if i == 0: fh_hint = fL
                elif i == len(t0_list) - 1: fh_hint = fR
            ev = traj_at(float(t0q), fh_hint)
            if ev is not None: rows.append(traj_row(ev))
        out_segs.append(dict(day=date_str, si=si+1, t0L=to_bj(eL), t0R=to_bj(eR),
                             width_min=w, fL=f0, fR=f1, slope=sl, rows=rows))
        if not collect:
            print("段%d: 窗口 %s ~ %s (宽 %5.1f min) | fly %6.2f~%6.2fh | "
                  "到达 %s ~ %s (漂 %4.1f min) | 带斜率 %+.2f" % (
                si+1, to_bj(eL), to_bj(eR), w, f0, f1,
                to_bj(ta0), to_bj(ta1), (ta1 - ta0)*1440.0, sl))
            for t0e, fe in ((eL, fL), ((eL + eR)/2, 0.5*(f0 + f1)), (eR, fR)):
                ev = hit_at(t0e, fe)
                if ev is not None: print_event(ev)
            for r in rows: print_traj_row(r)
            print("-" * 110)
    if not collect:
        print("联合窗口合计 %d 段, 总宽 %.1f min" % (len(segs), total))
    if csv_path:
        write_csv([r for s in out_segs for r in s['rows']], csv_path)
        if not collect:
            print("弹道族 CSV → %s" % csv_path)
    return out_segs

# 修正: h = 北京时小时; CE-5族解带@fly=112.46 ≈ [04:08,04:22] 五锚 + 镜像族01:0x三锚 + 散点
PHI_ANCHORS_H = (4.2, 4.15, 4.3, 4.1, 4.25, 1.4, 1.2, 1.7, 8.0, 12.0, 20.3, 5.0)

def _phi_task(args):
    phid, date_str, tol_km, fly_h, t0_step_min = args
    y_, m_, d_ = map(int, date_str.split("-"))
    d0 = (datetime.date(y_, m_, d_) - datetime.date(1858, 11, 17)).days + 2400000.5 - 8.0/24
    warm = {}
    phi = math.radians(phid)
    def hit(t0):
        if not _probe_gate(t0, fly_h): return None
        _, ev = _probe(t0, fly_h/24.0, tol_km, warm, phi)
        return ev
    a_hits = [t for t in (d0 + h/24.0 for h in PHI_ANCHORS_H) if hit(t) is not None]
    if not a_hits: return (phid, [])
    hits = set(a_hits)
    for t0a in a_hits:
        h0 = (t0a - d0)*24.0
        for h in np.arange(h0 - 50.0, h0 + 50.0 + 1e-9, t0_step_min/60.0):
            if 0.5 <= h <= 24.0 - 1e-9:
                t0 = float(d0 + h/24.0)
                if hit(t0) is not None: hits.add(t0)
    hs = sorted(hits); clusters = []
    for t in hs:
        if clusters and (t - clusters[-1][-1])*1440.0 <= 40.0: clusters[-1].append(t)
        else: clusters.append([t])
    return (phid, clusters)

def run_phi_mp(date_str, tol_km, fly_h, phi_step, t0_step_min, workers):
    from concurrent.futures import ProcessPoolExecutor, as_completed
    import multiprocessing
    phids = list(np.arange(math.degrees(PHI_LO), math.degrees(PHI_HI) + 1e-9, phi_step))
    print("[v18] 滑行扫描(并行) %s | fly=%.2fh | tol=±%.0fkm | φ步%.0f° t0步%.1fmin | 进程=%d" % (
        date_str, fly_h, tol_km, phi_step, t0_step_min, workers))
    tasks = [(float(p), date_str, tol_km, fly_h, t0_step_min) for p in phids]
    res = {}
    with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context("spawn")) as ex:
        futs = {ex.submit(_phi_task, t): t[0] for t in tasks}
        done = 0
        for fu in as_completed(futs):
            phid, cl = fu.result()
            res[phid] = cl
            done += 1
            print("  φ=%5.1f° 完成 (%d/%d)%s" % (phid, done, len(tasks), " 可行" if cl else ""), flush=True)
    print("%7s %7s  %-9s ~ %-9s %6s" % ("滑min", "φ°", "窗起", "窗止", "宽min"))
    n_hit = 0
    for phid in phids:
        for c in res.get(float(phid), []):
            n_hit += 1
            print("%7.1f %7.1f  %s ~ %s %6.0f" % (
                (math.radians(phid)/N_ORB - 0.5*BURN)/60.0, phid,
                to_bj(c[0]), to_bj(c[-1]), (c[-1] - c[0])*1440.0))
    print("可行 φ 簇: %d | φ 档: %d / %d" % (
        n_hit, sum(1 for cl in res.values() if cl), len(phids)))

def run_phi(date_str, tol_km=50.0, fly_h=112.46, phi_step=3.0, t0_step_min=5.0, workers=0):
    nw = int(workers) if workers and int(workers) > 1 else 1
    if nw > 1:
        run_phi_mp(date_str, tol_km, fly_h, phi_step, t0_step_min, nw)
        return
    y_, m_, d_ = map(int, date_str.split("-"))
    d0 = (datetime.date(y_, m_, d_) - datetime.date(1858, 11, 17)).days + 2400000.5 - 8.0/24
    warm = {}
    print("[v18] 滑行扫描 %s | fly=%.2fh | tol=±%.0fkm | φ步%.0f° t0步%.1fmin (锚点门控+簇细化)" % (
        date_str, fly_h, tol_km, phi_step, t0_step_min))
    print("%7s %7s  %-9s ~ %-9s %6s" % ("滑min", "φ°", "窗起", "窗止", "宽min"))
    n_hit = 0
    prev_anchors = []
    for phid in np.arange(math.degrees(PHI_LO), math.degrees(PHI_HI) + 1e-9, phi_step):
        phi = math.radians(phid)
        def hit(t0, phi=phi):
            if not _probe_gate(t0, fly_h): return None
            _, ev = _probe(t0, fly_h/24.0, tol_km, warm, phi)
            return ev
        cands = list(prev_anchors) + [d0 + h/24.0 for h in PHI_ANCHORS_H]
        a_hits = [t0a for t0a in cands[:12] if hit(t0a) is not None]
        if not a_hits:
            print("%7.1f %7.1f  -- (锚点快筛全败, 判不可行)" % (
                (phi/N_ORB - 0.5*BURN)/60.0, phid), flush=True)
            continue
        hits = set(a_hits)
        for t0a in a_hits:
            h0 = (t0a - d0)*24.0
            for h in np.arange(h0 - 50.0, h0 + 50.0 + 1e-9, t0_step_min/60.0):
                if 0.5 <= h <= 24.0 - 1e-9:
                    t0 = float(d0 + h/24.0)
                    if hit(t0) is not None: hits.add(t0)
        hs = sorted(hits)
        clusters = []
        for t0h in hs:
            if clusters and (t0h - clusters[-1][-1])*1440.0 <= 40.0:
                clusters[-1].append(t0h)
            else:
                clusters.append([t0h])
        prev_anchors = [c[0] for c in clusters] + [c[-1] for c in clusters]
        n_hit += 1
        for c in clusters:
            lo, hi = c[0], c[-1]
            print("%7.1f %7.1f  %s ~ %s %6.0f" % (
                (phi/N_ORB - 0.5*BURN)/60.0, phid, to_bj(lo), to_bj(hi), (hi - lo)*1440.0), flush=True)
    print("可行 φ 档: %d / %d" % (n_hit, int(math.degrees(PHI_HI - PHI_LO)/phi_step) + 1))

PROBE_TOL = 400.0

def day_probe(day_str, tol_km=PROBE_TOL, fly_lo=74.0, fly_hi=118.0, strict=False):
    y_, m_, d_ = map(int, day_str.split("-"))
    d0 = (datetime.date(y_, m_, d_) - datetime.date(1858, 11, 17)).days + 2400000.5 - 8.0/24
    warm = {}
    fhs = sorted(set(np.arange(fly_lo, fly_hi + 1e-9, 6.0).tolist() + [float(fly_hi)]))
    for fh in fhs:
        for h in np.arange(0.5, 24.0, 0.5):
            t0 = d0 + h/24.0
            if not _probe_gate(t0, fh): continue
            F, ev = _probe(t0, fh/24.0, tol_km, warm)
            if ev is not None or (not strict and F < min(tol_km, 200.0)):
                return (t0, fh, warm)
    return None

def run_auto(date_lo, date_hi, tol_km=50.0, fly_lo=72.0, fly_hi=119.0, dfly_h=0.2, n_traj=5, workers=0, csv_path=None):
    from concurrent.futures import ProcessPoolExecutor, as_completed
    import multiprocessing
    a_ = [int(x) for x in date_lo.split("-")]; b_ = [int(x) for x in date_hi.split("-")]
    dd0, dd1 = datetime.date(*a_), datetime.date(*b_)
    days = [(dd0 + datetime.timedelta(days=i)).strftime("%Y-%m-%d") for i in range((dd1 - dd0).days + 1)]
    nw = workers if workers > 0 else 1
    mp_ctx = multiprocessing.get_context("spawn") if nw > 1 else None
    print("[v18] 一键扫描 %s ~ %s (%d天) | tol=±%.0fkm fly=[%.0f,%.0f]h 步%.0fmin | "
          "滑行[%.0f,%.0f]min | 进程=%d" % (
        date_lo, date_hi, len(days), tol_km, fly_lo, fly_hi, dfly_h*60,
        COAST_IGN_LO/60, COAST_IGN_HI/60, nw))
    print("=" * 110)
    pf_day = functools.partial(day_probe, tol_km=tol_km, fly_lo=fly_lo, fly_hi=fly_hi)
    if mp_ctx is not None:
        with ProcessPoolExecutor(max_workers=nw, mp_context=mp_ctx) as ex:
            probe = dict(zip(days, ex.map(pf_day, days)))
    else:
        probe = {ds: pf_day(ds) for ds in days}
    hit = [d for d in days if probe[d]]
    print("粗筛: %d/%d 天有解 | 无解日: %s" % (
        len(hit), len(days), ", ".join(d for d in days if not probe[d]) or "无"))
    print("=" * 110)
    results = {}
    def _dump_day(ds, segs):
        results[ds] = segs
        rows = [dict(日期=ds, **r) for s in segs for r in s['rows']]
        if rows and csv_path:
            root, ext = os.path.splitext(csv_path)
            write_csv(rows, "%s_%s%s" % (root, ds, ext or ".csv"))
            print(">>> [%s] 已落盘 %s_%s%s (%d 条)" % (ds, root, ds, ext or ".csv", len(rows)))
    if hit:
        if mp_ctx is not None:
            with ProcessPoolExecutor(max_workers=nw, mp_context=mp_ctx) as ex:
                futs = {ex.submit(run_win, ds, tol_km, fly_lo, fly_hi, dfly_h, True, n_traj, None, probe[ds]): ds
                        for ds in hit}
                for fu in as_completed(futs):
                    ds = futs[fu]
                    try: _dump_day(ds, fu.result())
                    except Exception as exc:
                        print("[%s] 异常 %r" % (ds, exc)); results[ds] = []
        else:
            for ds in hit:
                try: _dump_day(ds, run_win(ds, tol_km, fly_lo, fly_hi, dfly_h, True, n_traj, None, warm0=probe[ds]))
                except Exception as exc:
                    print("[%s] 异常 %r" % (ds, exc)); results[ds] = []
    all_rows = []
    print("%-11s %3s %9s %6s %-36s %-17s %s" % ("日期", "段", "总宽min", "弹道", "最宽段(北京时)", "最早到达", "最小LOI"))
    for ds in sorted(results):
        segs = [s for s in results[ds] if s['rows']]
        if not segs:
            print("%-11s %3d %9.1f %6d (粗筛过但严格走廊无解)" % (
                ds, len(results[ds]), sum(s['width_min'] for s in results[ds]), 0))
            continue
        bw = max(segs, key=lambda s: s['width_min'])
        earl = min(r['到达'] for s in segs for r in s['rows'])
        mloi = min(r['LOI1'] for s in segs for r in s['rows'])
        print("%-11s %3d %9.1f %6d %s ~ %s (%5.1f min) %-17s %6.1f" % (
            ds, len(segs), sum(s['width_min'] for s in segs), sum(len(s['rows']) for s in segs),
            bw['t0L'], bw['t0R'], bw['width_min'], earl, mloi))
        for s in segs:
            for r in s['rows']:
                all_rows.append(dict(日期=ds, **r))
    if csv_path and all_rows:
        write_csv(all_rows, csv_path)
    print("\n弹道族 %d 条 → %s (utf-8-sig, Excel 可直接打开)" % (len(all_rows), csv_path))
    print("=" * 110)
    return results

def _day0_task(args):
    t0s, fly, tol = args
    return [(float(t0), solve_event(t0, t0 + fly, tol=tol) is not None) for t0 in t0s]

def run_day0(date_str, step=30.0, workers=0, tol_km=0.0):
    y, m, dd = map(int, date_str.split("-"))
    days = (datetime.date(y, m, dd) - datetime.date(1858, 11, 17)).days + 2400000.5 - 8.0/24
    FLY_NOM = 112.46/24.0
    t0s = list(np.arange(days, days + 1.0 - 1e-9, step/60.0/24.0))
    nw = int(workers) if workers and int(workers) > 1 else 1
    if nw > 1:
        from concurrent.futures import ProcessPoolExecutor
        import multiprocessing
        chunks = np.array_split(np.array(t0s), nw)
        with ProcessPoolExecutor(max_workers=nw, mp_context=multiprocessing.get_context("spawn")) as ex:
            parts = list(ex.map(_day0_task, [(list(c), FLY_NOM, tol_km) for c in chunks]))
        st = sorted([p for part in parts for p in part], key=lambda x: x[0])
    else:
        st = _day0_task((t0s, FLY_NOM, tol_km))
    win = []
    for i, (t0, ok) in enumerate(st):
        if not ok: continue
        a = t0 if i == 0 or not st[i-1][1] else None
        b = t0 if i == len(st)-1 or not st[i+1][1] else None
        if a is not None and i > 0: a = refine_edge(t0, st[i-1][0], FLY_NOM, tol_km)
        if b is not None and i < len(st)-1: b = refine_edge(t0, st[i+1][0], FLY_NOM, tol_km)
        win.append((a if a is not None else t0, b if b is not None else t0))
    merged = []
    for w in win:
        if merged and (w[0] - merged[-1][1])*24.0 < 1.0: merged[-1] = (merged[-1][0], w[1])
        else: merged.append(w)
    print("标称弧 fly=%.2fh tol=±%.0fkm | %d 个解, %d 个窗口段" % (
        FLY_NOM*24, tol_km, sum(1 for _, ok in st if ok), len(merged)))
    for w in merged:
        print("发射窗口(北京时): %s ~ %s" % (to_bj(w[0]), to_bj(w[1])))
    if tol_km <= 0.0 and merged and max((w[1]-w[0])*1440.0 for w in merged) < 2.0:
        print("(严格窗: 固定φ+r_p精确钉, 天然分钟级; 走廊窗在第4参传tolkm, 如 500)")
    print("-" * 100)
    for t0, ok in st:
        if ok:
            e = solve_event(t0, t0 + FLY_NOM, tol=tol_km)
            if e is not None: print_event(e)

def _span_task(args):
    day_str, tol_km, fly_lo, fly_hi = args
    p = day_probe(day_str, tol_km=tol_km, fly_lo=fly_lo, fly_hi=fly_hi, strict=True)
    if p is None: return (day_str, None, None, None)
    t0s, fh, warm = p
    def hit(t0):
        _, ev = _probe(t0, fh/24.0, tol_km, dict(warm))
        return ev is not None
    def edge(sgn):
        a, t, step, walked = t0s, t0s, 2.0/1440.0, 0.0
        for _ in range(24):
            t2 = t + sgn*step
            if hit(t2): a, t, walked = t2, t2, walked + step
            else: break
            if walked > 2.0/24.0: break
        b = t + sgn*step
        for _ in range(10):
            m = 0.5*(a + b)
            if hit(m): a = m
            else: b = m
        return a
    return (day_str, edge(-1), edge(+1), fh)

def run_span(d1s, d2s, tol_km=500.0, workers=0, fly_lo=74.0, fly_hi=118.0):
    from concurrent.futures import ProcessPoolExecutor
    import multiprocessing
    y1, m1, d1 = map(int, d1s.split("-")); y2, m2, d2 = map(int, d2s.split("-"))
    dd0, dd1 = datetime.date(y1, m1, d1), datetime.date(y2, m2, d2)
    days = [(dd0 + datetime.timedelta(days=i)).strftime("%Y-%m-%d") for i in range((dd1 - dd0).days + 1)]
    nw = int(workers) if workers and int(workers) > 1 else 1
    print("[v18.1] span(走廊速览) %s ~ %s | tol=±%.0fkm | 进程=%d (单fly截面, 全界用win/auto)" % (d1s, d2s, tol_km, nw))
    if nw > 1:
        with ProcessPoolExecutor(max_workers=nw, mp_context=multiprocessing.get_context("spawn")) as ex:
            res = list(ex.map(_span_task, [(ds, tol_km, fly_lo, fly_hi) for ds in days]))
    else:
        res = [_span_task((ds, tol_km, fly_lo, fly_hi)) for ds in days]
    n = 0
    print("%-11s %-10s ~ %-10s %8s %7s  %s" % ("日期", "窗起", "窗止", "宽min", "fly*h", "族"))
    for ds, lo, hi, fh in res:
        if lo is None:
            print("%-11s --" % ds); continue
        n += 1
        hbj = ((lo - 2400000.5 + 8.0/24.0) % 1.0) * 24.0
        fam = "东北镜像" if hbj < 3.0 else "东南任务"
        print("%-11s %s ~ %s %8.1f %7.1f  %s" % (ds, to_bj(lo), to_bj(hi), (hi - lo)*1440.0, fh, fam))
    print("%d/%d 天有走廊窗" % (n, len(days)))


BND_DAY = 0.20/24.0
DFS_DAY = 30.0/86400.0

def _day_find_fly(t0, fg, tol_km, tw=None):
    cand = np.arange(fg - BND_DAY, fg + BND_DAY + 1e-12, DFS_DAY)
    cand = cand[np.argsort(np.abs(cand - fg))]
    for fly in cand:
        e = solve_event(t0, t0 + fly, tol=tol_km, tE_warm=tw)
        if e is not None: return fly, e
    return None

def _day_trace(t0, fg, tw, direction, dt_min, tol_km):
    DT = dt_min/1440.0
    pts = []
    while True:
        t0n = t0 + direction*DT
        r = _day_find_fly(t0n, fg, tol_km, tw)
        if r is None: break
        fn, ev = r
        pts.append((t0n, fn)); t0, fg, tw = t0n, fn, ev['tE']
    return pts

def _day_task(args):
    t0s, fgs, tw, direction, dt_min, tol_km = args
    return _day_trace(t0s, fgs, tw, direction, dt_min, tol_km)

def run_day(date_str, dt_min=2.0, seed_h=4.5, tol_km=50.0, workers=0):
    y, m, dd = map(int, date_str.split("-"))
    days = (datetime.date(y, m, dd) - datetime.date(1858, 11, 17)).days + 2400000.5 - 8.0/24
    T0_SEED = days + seed_h/24.0
    FLY_SEED = 112.46/24.0
    seed = _day_find_fly(T0_SEED, FLY_SEED, tol_km)
    if seed is None:
        print(">>> 种子点无解! 请先用 day0 确认该日的解时刻, 再以第3参传入(北京时小时)")
        return
    nw = int(workers) if workers and int(workers) > 1 else 1
    if nw > 1:
        from concurrent.futures import ProcessPoolExecutor
        import multiprocessing
        with ProcessPoolExecutor(max_workers=2, mp_context=multiprocessing.get_context("spawn")) as ex:
            f_up = ex.submit(_day_task, (T0_SEED, seed[0], seed[1]['tE'], +1, dt_min, tol_km))
            f_dn = ex.submit(_day_task, (T0_SEED, seed[0], seed[1]['tE'], -1, dt_min, tol_km))
            up, dn = f_up.result(), f_dn.result()
    else:
        up = _day_trace(T0_SEED, seed[0], seed[1]['tE'], +1, dt_min, tol_km)
        dn = _day_trace(T0_SEED, seed[0], seed[1]['tE'], -1, dt_min, tol_km)
    pts = dn[::-1] + [(T0_SEED, seed[0])] + up
    t0_lo, t0_hi = pts[0][0], pts[-1][0]
    print("走廊 tol=±%.0f km" % tol_km)
    print("解曲线: %d 点 | 窗口 %s ~ %s (北京时, 宽 %.0f min)" % (
        len(pts), to_bj(t0_lo), to_bj(t0_hi), (t0_hi - t0_lo)*1440.0))
    print("fly 范围 %.3f~%.3fh | 到达历元随 t0 平移 %+.1f min (模型钉 r_p+tof, 到达不钉, 属本性行为)" % (
        pts[0][1]*24, pts[-1][1]*24,
        ((pts[-1][0]+pts[-1][1]) - (pts[0][0]+pts[0][1]))*1440.0))
    print("-" * 100)
    step_k = max(1, len(pts)//8)
    for k in range(0, len(pts), step_k):
        t0, fly = pts[k]
        e = solve_event(t0, t0 + fly, tol=tol_km)
        if e is not None: print_event(e)

# ================= v18.5: 问题3接通 — fwdspan 月扫每日最优倾角 i*(date) =================
def _fwdspan_task(args):
    day_str, fly_h, step_min, tol_km = args
    y_, m_, d_ = map(int, day_str.split("-"))
    d0 = (datetime.date(y_, m_, d_) - datetime.date(1858, 11, 17)).days + 2400000.5 - 8.0/24
    fly = fly_h/24.0
    warm = {"tE": None}
    def hit(t0):
        e = solve_event_fwd(t0, t0 + fly, tol=tol_km, tE_warm=warm["tE"], phi_search=True)
        if e is not None:
            warm["tE"] = e['tE']                   # 跨 t0 温启动: tE 平滑链
        return e
    hits = []
    for t0 in np.arange(d0 + 0.5/24.0, d0 + 1.0 - 1e-9, step_min/1440.0):
        e = hit(float(t0))
        if e is not None:
            hits.append((float(t0), e))
    if not hits:
        return (day_str, [])
    def refine(t_in, t_out):                       # 窗口边缘二分 (day0 同款)
        lo, hi = (t_in, t_out) if t_in < t_out else (t_out, t_in)
        f_lo = hit(lo) is not None
        while (hi - lo)*86400.0 > 60.0:
            m = 0.5*(lo + hi)
            if (hit(m) is not None) == f_lo:
                lo = m
            else:
                hi = m
        return lo if f_lo else hi
    segs, grp = [], [hits[0]]
    for t0, e in hits[1:]:
        if (t0 - grp[-1][0])*1440.0 <= step_min*1.4 + 1e-9:
            grp.append((t0, e))
        else:
            segs.append(grp); grp = [(t0, e)]
    segs.append(grp)
    out = []
    for grp in segs:
        a = refine(grp[0][0],  grp[0][0]  - 2.0*step_min/1440.0)
        b = refine(grp[-1][0], grp[-1][0] + 2.0*step_min/1440.0)
        evL, evR = hit(a), hit(b)
        if evL is None: evL = grp[0][1]
        if evR is None: evR = grp[-1][1]
        out.append(dict(a=a, b=b, evL=evL, evR=evR,
                        inc_lo=min(e['inc'] for _, e in grp),
                        inc_hi=max(e['inc'] for _, e in grp),
                        az_lo=min(e['az'] for _, e in grp),
                        az_hi=max(e['az'] for _, e in grp)))
    return (day_str, out)

def run_fwdspan(d1s, d2s, fly_h=112.46, step_min=20.0, tol_km=50.0, workers=0, csv_path=None):
    from concurrent.futures import ProcessPoolExecutor
    import multiprocessing
    y1, m1, d1 = map(int, d1s.split("-")); y2, m2, d2 = map(int, d2s.split("-"))
    dd0, dd1 = datetime.date(y1, m1, d1), datetime.date(y2, m2, d2)
    days = [(dd0 + datetime.timedelta(days=i)).strftime("%Y-%m-%d") for i in range((dd1 - dd0).days + 1)]
    nw = int(workers) if workers and int(workers) > 1 else 1
    print("[v18.5] fwdspan(每日最优倾角) %s ~ %s (%d天) | fly=%.2fh | t0步%.0fmin tol=±%.0fkm | 进程=%d"
          % (d1s, d2s, len(days), fly_h, step_min, tol_km, nw))
    print("=" * 110)
    tasks = [(ds, float(fly_h), float(step_min), float(tol_km)) for ds in days]
    if nw > 1:
        with ProcessPoolExecutor(max_workers=nw, mp_context=multiprocessing.get_context("spawn")) as ex:
            res = list(ex.map(_fwdspan_task, tasks))
    else:
        res = [_fwdspan_task(t) for t in tasks]
    print("%-11s %-23s %6s  %-14s %-14s %6s %6s" % ("日期", "窗口(北京时)", "宽min", "i*范围", "射向范围", "LOI", "族"))
    rows_csv = []
    n = 0
    for ds, segs in res:
        if not segs:
            print("%-11s --" % ds); continue
        n += 1
        for s in segs:
            print("%-11s %s ~ %s %6.1f  %5.2f~%5.2f  %5.1f~%5.1f  %6.1f %6s" % (
                ds, to_bj(s['a']), to_bj(s['b']), (s['b'] - s['a'])*1440.0,
                s['inc_lo'], s['inc_hi'], s['az_lo'], s['az_hi'],
                s['evL']['dv_loi'], "东南" if s['evL']['branch'] > 0 else "东北"))
            rows_csv.append(dict(日期=ds, 窗起=to_bj(s['a']), 窗止=to_bj(s['b']),
                                 宽min=round((s['b'] - s['a'])*1440.0, 1),
                                 istar_lo=round(s['inc_lo'], 3), istar_hi=round(s['inc_hi'], 3),
                                 az_lo=round(s['az_lo'], 2), az_hi=round(s['az_hi'], 2),
                                 fly_h=round(fly_h, 2),
                                 C3=round(s['evL']['d']['C3'], 3),
                                 LOI1=round(s['evL']['dv_loi'], 1),
                                 到达=to_bj(s['evL']['tL'] + s['evL']['fly']),
                                 分支=("东南" if s['evL']['branch'] > 0 else "东北")))
    curve = [(ds, 0.5*(s['inc_lo'] + s['inc_hi'])) for ds, segs in res for s in segs]
    if len(curve) >= 2:
        iv = [v for _, v in curve]
        k = int(np.argmax(iv))
        print("i* 月度趋势: %.2f° ~ %.2f° (极差 %.2f°, 峰值 %s)"
              % (min(iv), max(iv), max(iv) - min(iv), curve[k][0]))
    print("%d/%d 天存在 fwd 解" % (n, len(days)))
    if csv_path and rows_csv:
        write_csv(rows_csv, csv_path)
        print("i*(date) 表 CSV → %s (utf-8-sig)" % csv_path)
    print("=" * 110)
    return res


if __name__ == "__main__":
    print("自检: 月距 %.0f km [%s] | 近月点靶 %.1fkm | GMST(JD0)=%.1f° | "
          "T_UP=%.1fs COAST=%.1fs TLI@+%.1fs | T_IMP=%.1fs PHI_S=%.2f° | "
          "滑行界[%.0f,%.0f]min φ∈[%.1f,%.1f]° | 引导网格 fr=[%.2f,%.2f]步%.2f | Y_WIN_LO=%.3f" % (
        np.linalg.norm(_moon(JD0)), KER, R_PL, gmst_deg(JD0),
        T_UP, COAST, T_LI, T_IMP, math.degrees(PHI_S),
        COAST_IGN_LO/60, COAST_IGN_HI/60, math.degrees(PHI_LO), math.degrees(PHI_HI),
        TE_FR[0], TE_FR[-1], TE_FR[1]-TE_FR[0], Y_WIN_LO))
    mode = sys.argv[1] if len(sys.argv) > 1 else "ce5"
    _NEED = {"ce5": 0, "ce5fwd": 0, "diag": 0, "pole": 0, "day0": 1, "day": 1, "span": 2, "auto": 2, "win": 1, "phi": 1, "fwdspan": 2}   # v18.5
    if _NEED.get(mode) is not None and len(sys.argv) - 2 < _NEED[mode]:
        mode = "__usage__"
    if mode == "ce5":
        t0 = JD0 + 3.0 + (12*3600 + 6*60 + 30.8)/86400.0
        t_arr = t0 + 112.46/24.0
        print("=" * 100)
        print("CE-5 对账: 点火 %s UTC | 到达 %s UTC | tE 引导锚: 真解应在 tL+0.846·fly" % (to_utc(t0), to_utc(t_arr)))
        e = solve_event(t0, t_arr)
        if e is not None:
            print_event(e)
            print("  模型-实测: Δy=%+.1f m/s | ΔC3=%+.3f | Δ射向=%+.2f° | 近月点偏差=%+.0f km | "
                  "上升弧=%.0f km (实测量级 1500-2500)" % (
                (e['y'] - Y_MEAS)*1e3, e['d']['C3'] - C3_MEAS, e['az'] - AZ_MEAS,
                e['d']['r_p'] - R_PL, e['arc_up']))
        else:
            print(">>> 严格解无 → 自动进 diag 逐门定位:")
            diag(t0, t_arr)
    elif mode == "ce5fwd":                                                                   # v18.3
        t0 = JD0 + 3.0 + (12*3600 + 6*60 + 30.8)/86400.0
        t_arr = t0 + 112.46/24.0
        print("=" * 100)
        print("CE-5 正向预测: n̂*=normalize(r̂×M_E) → 倾角/射向均为输出, h_M≡0(构造)")
        e = solve_event_fwd(t0, t_arr)
        if e is not None:
            print_event(e)
            print("  预测-实测: i*=%.3f° (实测21.299) | az*=%.2f° (实测98.4) | hM=%+.1f km" % (
                e['inc'], e['az'], e['d']['hM']))
        else:
            print(">>> 前向无解 — 该历元几何不支持(射向门/走廊), 这也是预测结果")
    elif mode == "diag":
        t0 = JD0 + 3.0 + (12*3600 + 6*60 + 30.8)/86400.0
        t_arr = t0 + 112.46/24.0
        diag(t0, t_arr)
    elif mode == "pole":                                                                     # v18.3
        print("=" * 100)
        print("月北极自检 (IAU pck00010 线性模型) | 当前核: %s" % KER)
        for jd in (2451545.0, JD0):
            p = moon_pole(jd)
            ra = math.degrees(math.atan2(float(p[1]), float(p[0]))) % 360.0
            dc = math.degrees(math.asin(max(-1.0, min(1.0, float(p[2])))))
            print("  %s  RA=%9.4f°  DEC=%8.4f°  pole=(%+.6f, %+.6f, %+.6f)" % (
                to_utc(jd), ra, dc, p[0], p[1], p[2]))
        T20 = (JD0 - 2451545.0)/36525.0                       # 常数一致性检查(仅防粘贴损坏, 非独立验证)
        p_exp = moon_pole(2451545.0)                          # 独立代码路径: J2000 处率项为 0
        p20 = moon_pole(JD0)
        err = math.degrees(math.acos(max(-1.0, min(1.0, float(np.dot(p20, p_exp))))))
        print("  期望(JD0, pck00010手算): RA=%.4f° DEC=%.4f° | 常数一致性: %.5f°" % (
            269.9949 - 0.0031*T20, 66.5392 + 0.0130*T20, err))
        try:                                                  # 核含天平动段(de421/de440全量)则独立交叉验证
            seg = kernel[3, 310]
            def pole_ker(jd):
                a, _ = seg.compute_and_differentiate(jd + TT_UTC/86400.0)
                ph_, th_ = float(a[0]), float(a[1])   # ψ 不影响极轴方向, 只用 (φ,θ)
                return np.array([math.sin(th_)*math.sin(ph_),
                                 math.sin(th_)*math.cos(ph_), math.cos(th_)])
            for jd in (2451545.0, JD0):
                e2 = math.degrees(math.acos(max(-1.0, min(1.0,
                     float(np.dot(moon_pole(jd), pole_ker(jd)))))))
                print("  交叉验证 @%s: 路径A vs 星历天平动 = %.4f° %s" % (
                    to_utc(jd), e2, "OK" if e2 < 0.1 else "FAIL(调(φ,θ)顺序或y符号)"))
        except Exception as exc:
            print("  当前核无 [3,310] 天平动段(%s) → 路径A为唯一选项(本文件即此模式)" % type(exc).__name__)
    elif mode == "day0":
        run_day0(sys.argv[2],
                 float(sys.argv[3]) if len(sys.argv) > 3 else 30.0,
                 int(sys.argv[4]) if len(sys.argv) > 4 else 0,
                 float(sys.argv[5]) if len(sys.argv) > 5 else 0.0)
    elif mode == "day":
        run_day(sys.argv[2],
                float(sys.argv[3]) if len(sys.argv) > 3 else 2.0,
                float(sys.argv[4]) if len(sys.argv) > 4 else 4.5,
                float(sys.argv[5]) if len(sys.argv) > 5 else RP_TOL_DEF,
                int(sys.argv[6]) if len(sys.argv) > 6 else 0)
    elif mode == "span":
        run_span(sys.argv[2], sys.argv[3],
                 float(sys.argv[4]) if len(sys.argv) > 4 else 500.0,
                 int(sys.argv[5]) if len(sys.argv) > 5 else 0)
    elif mode == "auto":
        run_auto(sys.argv[2], sys.argv[3],
                 float(sys.argv[4]) if len(sys.argv) > 4 else 50.0,
                 float(sys.argv[5]) if len(sys.argv) > 5 else 72.0,
                 float(sys.argv[6]) if len(sys.argv) > 6 else 119.0,
                 float(sys.argv[7]) if len(sys.argv) > 7 else 0.2,
                 n_traj=5,
                 workers=int(sys.argv[8]) if len(sys.argv) > 8 else 0,
                 csv_path=(sys.argv[9] if len(sys.argv) > 9 else None))
    elif mode == "win":
        tol_km = float(sys.argv[3]) if len(sys.argv) > 3 else 50.0
        run_win(sys.argv[2], tol_km,
                float(sys.argv[4]) if len(sys.argv) > 4 else 72.0,
                float(sys.argv[5]) if len(sys.argv) > 5 else 119.0,
                float(sys.argv[6]) if len(sys.argv) > 6 else 0.2,
                csv_path=(sys.argv[7] if len(sys.argv) > 7 else None))
    elif mode == "phi":
        run_phi(sys.argv[2],
                float(sys.argv[3]) if len(sys.argv) > 3 else 50.0,
                float(sys.argv[4]) if len(sys.argv) > 4 else 112.46,
                float(sys.argv[5]) if len(sys.argv) > 5 else 3.0,
                float(sys.argv[6]) if len(sys.argv) > 6 else 5.0,
                int(sys.argv[7]) if len(sys.argv) > 7 else 0)
    elif mode == "fwdspan":   # v18.5: 月扫每日最优倾角 i*(date)
        run_fwdspan(sys.argv[2], sys.argv[3],
                float(sys.argv[4]) if len(sys.argv) > 4 else 112.46,
                float(sys.argv[5]) if len(sys.argv) > 5 else 20.0,
                float(sys.argv[6]) if len(sys.argv) > 6 else 50.0,
                int(sys.argv[7]) if len(sys.argv) > 7 else 0,
                csv_path=(sys.argv[8] if len(sys.argv) > 8 else None)) 
    else:
        print("用法: ce5 | ce5fwd | diag | pole | day0 YYYY-MM-DD [时步min] [进程] | "
              "day YYYY-MM-DD [t0步min] [种子BJ时] [走廊km] [进程] | "
              "span 起日 止日 [t步h] [fly步天] [进程] | "
              "auto 起日 止日 [tolkm] [fly_lo] [fly_hi] [fly步h] [进程] [csv] | "
              "win 日期 [tolkm] [fly_lo] [fly_hi] [fly步h] [csv] | "
              "phi 日期 [tolkm] [flyh] [φ步°] [t0步min] [进程] | "
              "fwdspan 起日 止日 [flyh] [t0步min] [tolkm] [进程] [csv]")
