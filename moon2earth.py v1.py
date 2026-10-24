# -*- coding: utf-8 -*-
# moon2earth.py — 月地返回窗口计算器 v1.0 [2026-10-22]
#
# v0.1 调试实录(三处根因, 供日后复盘):
#   ① _moon 曾对 jplephem 输出再 /1e3 → 月距 358km(千倍缩尺), 一切几何失效
#   ② Stumpff C/S 正负支写反 → 双曲/椭圆传播全错
#   ③ 括号扩张/二分方向反 + 地心椭圆弧 r(χ) 周期振荡使数值穿越失效
#     → 地心段改全解析(要素直取+开普勒飞行时间), 数值穿越只留月心段
#   求解结构: h* 先验(γ/r_p 联立解析根)定 θ 初值 + Gauss-Seidel 交替
#   secant(θ→r_p弹道, Δv→γ_EI) —— 返回走廊的物理是"渐近线对消月球轨道
#   角动量"(392,088 → ~71,600 km²/s), 可达 θ 窗 ~±10°, 粗网格必漏, 须先验引导
#
# diyue5/6(地→月) 的镜像: 停泊轨道 TEI 脉冲 → 月心双曲弧 → SOI 出口 →
# 地心弧 → EI 120km。patched conic 反着走。
#
# 几何与条件(镜像表):
#   LWC:  (λ,y) 二维 → r_p(月)=2137.4;   外扫 (点火 t0, fly)
#   返回: (θ,Δv) 二维 → γ_EI=γ_TGT 且 r_p弹道=RP_TGT;  外扫 (TEI t_b, Ω)
#   spot: 两公开历元(t_b, t_EI) → 三条件(t_EI 闭合, γ, r_p) 反演 (Ω,θ,Δv)
#         —— 与 ce6_close.py 的倾角反演同一方法论
#
# 物理口径:
#   - TEI 近月点面内顺行脉冲 → 逃逸渐近线在停泊轨道面内; 轨道面(对月球赤道
#     86°, RAAN Ω)唯一决定可达的地球方向集, Ω 是走廊主轴
#   - EI 120km 处 γ=−5.8°(半弹道跳跃) 对应弹道近地点在地面下 ~50km —— 跳段
#     不建模, 弹道弧穿到深近地点后由气动抬出, 本程序只算到 EI 几何
#   - 月球北极 IAU 常数(18.6y 章动忽略); LE 月赤道系 x 轴 = t_b 时月→地方向
#     (真实地球指向随天平动摆动 ±8°, v0.1 冻结于 t_b, 误差吸收进 Ω)
#   - 月心段 μM 二体, 出 SOI 后地心段 μE 二体(月引力截断, 同 diyue 口径)
#
# 锚点(公开, 多源一致):
#   CE-5 轨返组合体第一次月地转移入射 2020-12-12 09:54 BJT(近月点 ~200km)
#   CE-5 返回器 EI 2020-12-17 01:33 BJT, ~120km, ≈11.2km/s(第二宇宙速度)
#   着陆四子王旗(约 111.7°E, 41.4°N), 半弹道跳跃式再入

import os, sys, math, datetime, functools
import numpy as np
from jplephem.spk import SPK

# ---------------- 常量 ----------------
try:
    _bsp = SPK.open("de405.bsp"); _EPH = "de405.bsp"
except Exception:
    _bsp = SPK.open("de421.bsp"); _EPH = "de421.bsp"

muE = 398600.4418          # 地球 GM [km^3/s^2]
muM = 4902.8001            # 月球 GM
r_earth = 6378.137
r_moon = 1737.4
r_soi = 66200.0            # 月 SOI(与 diyue 同口径)
r_park = r_moon + 200.0    # 200km 圆停泊轨道
r_EI = r_earth + 120.0     # 再入界面
INC_ORB = 86.0             # 停泊轨道倾角(对月球赤道)
GAM_TGT = -5.8             # EI 航迹角目标 [deg]
RP_TGT_ALT = 50.0          # 弹道近地点高度(地面上 km) —— γ_TGT 的几何对应; 真空弧不碰地, 大气在近地点区接住(即跳跃)
SITE = (111.7, 41.4)       # 四子王旗(经,纬)
T_LUN = 2.0*math.pi*math.sqrt(r_park**3/muM)
_d = math.radians
_GAM_OVR = None   # γ_EI 覆盖(γ 扫描用)
def set_gam(x):
    global _GAM_OVR
    _GAM_OVR = x
_INC_OVR = None   # 倾角覆盖(spot i 扫描用); None=用 INC_ORB
def set_inc(x):
    global _INC_OVR
    _INC_OVR = x

# IAU 月球北极(ICRF)
_pole = np.array([math.cos(_d(65.6411))*math.cos(_d(266.8577)),
                  math.cos(_d(65.6411))*math.sin(_d(266.8577)),
                  math.sin(_d(65.6411))])

# ---------------- 星历 ----------------
@functools.lru_cache(maxsize=300000)
def _moon(jd):
    return _bsp[3, 301].compute(jd)   # jplephem 已是 km, 勿再 /1e3 (v0.1 曾错除, |r_M| 变 358km)

def _moon_v(jd):
    h = 0.5/86400.0
    return (_moon(jd+h) - _moon(jd-h))   # km/s (±0.5s 中心差分)

def to_bj(jd):
    d = datetime.datetime(1858, 11, 17) + datetime.timedelta(days=jd - 2400000.5 + 8.0/24)
    return d.strftime("%m-%d %H:%M:%S")

def gmst(jd):
    return (280.46061837 + 360.98564736629*(jd-2451545.0)
            + 0.000387933*((jd-2451545.0)/36525.0)**2) % 360.0

def subpoint(r, jd):
    r = np.asarray(r, float)
    lon = math.degrees(math.atan2(r[1], r[0])) - gmst(jd)
    lat = math.degrees(math.asin(max(-1.0, min(1.0, r[2]/float(np.linalg.norm(r))))))
    return ((lon+180.0) % 360.0 - 180.0, lat)

def _jd(date_str, hh_mm):
    y_, m_, d_ = map(int, date_str.split("-"))
    hh, mm = map(float, hh_mm.split(":"))
    return (datetime.date(y_, m_, d_) - datetime.date(1858, 11, 17)).days \
        + 2400000.5 - 8.0/24 + (hh + mm/60.0)/24.0

# ---------------- 万有变量传播核心 ----------------
def _stumpff(z):
    if z > 1e-6:
        s = math.sqrt(min(z, 2500.0)); return (1.0-math.cos(s))/s**2, (s-math.sin(s))/s**3
    if z < -1e-6:
        s = math.sqrt(-z); return (math.cosh(s)-1.0)/s**2, (math.sinh(s)-s)/s**3
    return (0.5 - z/24.0 + z*z/720.0, 1.0/6.0 - z/120.0 + z*z/5040.0)

def _chi_state(r0v, v0v, chi, mu):
    """给定万有异常 χ → (r⃗,v⃗,r,dt)"""
    r0 = float(np.linalg.norm(r0v)); v0 = float(np.linalg.norm(v0v))
    rdv = float(np.dot(r0v, v0v)); sqmu = math.sqrt(mu)
    alpha = 2.0/r0 - v0*v0/mu
    psi = chi*chi*alpha
    C, S = _stumpff(psi)
    r = chi*chi*C + rdv/sqmu*chi*(1.0-psi*S) + r0*(1.0-psi*S)
    dt = chi**3*S/sqmu + rdv/mu*chi*chi*C + r0/sqmu*chi*(1.0-psi*S)
    f = 1.0 - chi*chi/r0*C
    g = dt - chi**3*S/sqmu
    rv = f*r0v + g*v0v
    rr = float(np.linalg.norm(rv))
    fdot = sqmu/(rr*r0)*chi*(psi*S-1.0)
    gdot = 1.0 - chi*chi/rr*C
    vv = fdot*r0v + gdot*v0v
    return rv, vv, rr, dt

def _time_to_radius(r0v, v0v, r_tgt, mu):
    """|r⃗(χ)| = r_tgt 首穿(括号二分, 单调段内)。返回 (dt, r⃗, v⃗) 或 None。
    月心段: 近月点出发上升穿 SOI(单调升); 地心段: SOI 出口下降穿 EI
    (单调段止于近地点, r_p<r_tgt 已在上游守卫)。"""
    r0 = float(np.linalg.norm(r0v))
    rising = r_tgt > r0
    chi_lo, chi_hi = 1e-4, 2.0
    r_hi = float(np.linalg.norm(_chi_state(r0v, v0v, chi_hi, mu)[0]))
    for _ in range(40):
        if (r_hi > r_tgt) != rising:   # 未穿越 → 扩张
            chi_hi *= 2.0; chi_lo = chi_hi/2.0
        else:
            break
        r_hi = float(np.linalg.norm(_chi_state(r0v, v0v, chi_hi, mu)[0]))
    else:
        return None
    for _ in range(70):
        mid = 0.5*(chi_lo + chi_hi)
        r_mid = float(np.linalg.norm(_chi_state(r0v, v0v, mid, mu)[0]))
        if (r_mid > r_tgt) != rising: chi_lo = mid   # 未穿越 → 交点在更大 χ
        else: chi_hi = mid
    chi = 0.5*(chi_lo + chi_hi)
    rv, vv, rr, dtt = _chi_state(r0v, v0v, chi, mu)
    if dtt <= 0.0 or abs(rr - r_tgt) > 1.0: return None
    return dtt, rv, vv

# ---------------- 轨道要素直取(解析) ----------------
def _arc_geom(r0v, v0v, mu):
    """(r⃗,v⃗) → (r_p, h, eps, e⃗, h⃗)。r_p 解析: h²/(μ(1+e))"""
    h = np.cross(r0v, v0v)
    h2 = float(np.dot(h, h))
    eps = float(np.dot(v0v, v0v))/2.0 - mu/float(np.linalg.norm(r0v))
    evec = np.cross(v0v, h)/mu - r0v/float(np.linalg.norm(r0v))
    e = float(np.linalg.norm(evec))
    r_p = h2/(mu*(1.0+e))
    return r_p, math.sqrt(h2), eps, evec, h

def _gamma_at(r0v, v0v, mu, r_t):
    """固定弧上 r=r_t 处的航迹角(下降段取负)。返回 None 表示弧不穿该半径"""
    r_p, h, eps, evec, hvec = _arc_geom(r0v, v0v, mu)
    if r_p >= r_t: return None, None, None
    v2 = 2.0*(eps + mu/r_t)
    vr2 = v2 - h*h/(r_t*r_t)
    if vr2 <= 0.0: return None, None, None
    gamma = -math.degrees(math.asin(math.sqrt(vr2)/math.sqrt(v2)))
    return gamma, r_p, h

def _r_at_radius(r0v, v0v, mu, r_t):
    """弧上 r=r_t 下降段的位置向量(解析 θ_EI, pre-perigee 分支)"""
    r_p, h, eps, evec, hvec = _arc_geom(r0v, v0v, mu)
    e = float(np.linalg.norm(evec))
    if e < 1e-9: return None
    p = h*h/mu
    ct = (p/r_t - 1.0)/e
    if abs(ct) > 1.0: return None
    st = -math.sqrt(1.0-ct*ct)          # 下降段: θ_EI ∈ (−π,0)
    ehat = evec/e
    pdir = np.cross(hvec, evec)/e       # 单位, ⊥ê
    return p/(1.0+e*ct)*(ct*ehat + st*pdir)

# ---------------- 月心停泊轨道与 TEI ----------------
def _le_basis(jd_t):
    xw = -_moon(jd_t); xw = xw/float(np.linalg.norm(xw))
    xw = xw - float(np.dot(xw, _pole))*_pole
    xw = xw/float(np.linalg.norm(xw))
    return xw, np.cross(_pole, xw), _pole

def _orb_state(jd_b, theta_deg, raan_deg, basis):
    xw, yw, zw = basis
    i, om = _d(_INC_OVR if _INC_OVR is not None else INC_ORB), _d(raan_deg)
    nhat = np.array([math.cos(om), math.sin(om), 0.0])
    hhat = np.array([math.sin(i)*math.sin(om), -math.sin(i)*math.cos(om), math.cos(i)])
    mhat = np.cross(hhat, nhat)
    u = _d(theta_deg)
    v_c = math.sqrt(muM/r_park)
    r_le = math.cos(u)*nhat + math.sin(u)*mhat
    v_le = (-math.sin(u)*nhat + math.cos(u)*mhat)*v_c
    E = np.vstack([xw, yw, zw])
    return E.T @ (r_park*r_le), E.T @ v_le   # v0.2 修正: 基向量为行时正变换是 E.T, 原 E@ 是逆变换(面被放到 113°)

def _plane_icrf(jd_ref, raan_deg):
    """t_ref 时刻 LE 系 (i=INC_ORB, Ω) 的轨道面法向 → ICRF 单位向量。
    多日窗口扫描用: 真实停泊轨道的面在惯性系里不跟着月球转(节点进动
    ~0.x°/天 忽略, 注明), LE 系里的 Ω 则以 ~13°/天 漂移 —— 逐时刻重建
    LE 基再取同 Ω 是错的, 必须先在 t_ref 定面、此后冻结惯性法向。"""
    xw, yw, zw = _le_basis(jd_ref)
    E = np.vstack([xw, yw, zw])
    i, om = _d(_INC_OVR if _INC_OVR is not None else INC_ORB), _d(raan_deg)
    h_le = np.array([math.sin(i)*math.sin(om), -math.sin(i)*math.cos(om), math.cos(i)])
    hv = E.T @ h_le   # v0.2 修正: 同 _orb_state, 正变换为 E.T@
    return hv/float(np.linalg.norm(hv))

def _orb_state_h(jd_b, theta_deg, hhat):
    """固定惯性面(ĥ: ICRF 单位法向)上的圆停泊轨道状态(Moon 系)。
    θ 从升交点(相对月球赤道)起量, 与 spot 模式同约定: n̂ = normalize(p̂_M × ĥ)。"""
    hhat = np.asarray(hhat, float)
    hhat = hhat/float(np.linalg.norm(hhat))
    nhat = np.cross(_pole, hhat)
    nhat = nhat/float(np.linalg.norm(nhat))
    mhat = np.cross(hhat, nhat)
    u = _d(theta_deg)
    v_c = math.sqrt(muM/r_park)
    r = r_park*(math.cos(u)*nhat + math.sin(u)*mhat)
    v = v_c*(-math.sin(u)*nhat + math.cos(u)*mhat)
    return r, v

# ---------------- 返回弧 ----------------
def _earth_geom(rE0, vE0):
    """地心弧全解析: (r⃗,v⃗)@SOI出口 → dict(r_p, γ_EI, r⃗_EI, v⃗_EI, dt_EI, ε, a, e)。
    要素直取 + 开普勒飞行时间(椭圆 E−e·sinE / 双曲 e·sinhH−H)。
    [v0.1.2] 替换 χ 数值穿越 —— 地心弧是椭圆时 r(χ) 周期振荡, 括号扩张会跳过
    近地点窄窗(EI 交叉点距近地点仅 ~200km), 数值法不可靠; 解析法无此问题。"""
    r0v = np.asarray(rE0, float); v0v = np.asarray(vE0, float)
    r0 = float(np.linalg.norm(r0v)); v0 = float(np.linalg.norm(v0v))
    hv = np.cross(r0v, v0v); h = float(np.linalg.norm(hv))
    if h < 1.0: return None
    ev = np.cross(v0v, hv)/muE - r0v/r0
    e = float(np.linalg.norm(ev))
    eps = v0*v0/2.0 - muE/r0
    a = -muE/(2.0*eps)
    if abs(e) < 1e-12 or abs(a) < 1.0: return None
    r_p = a*(1.0-e)
    if r_p >= r_EI - 1.0: return None                # 弧不深入大气层
    # EI 真近点角(下降支: sinθ<0)
    cth = (h*h/(muE*r_EI) - 1.0)/e
    if abs(cth) > 1.0: return None
    sth = -math.sqrt(1.0-cth*cth)
    eph = ev/e
    hhat = hv/h
    qh = np.cross(hhat, eph)
    r_ei_v = r_EI*(cth*eph + sth*qh)
    vr = (muE/h)*e*sth                                # 径向速度(μ/h·e·sinθ)
    vt = h/r_EI
    gam = math.degrees(math.atan2(vr, vt))
    # 飞行时间: 当前 eccentric 异常 → EI
    if eps < 0.0:   # 椭圆
        n = math.sqrt(muE/a**3)
        cE0 = (1.0-r0/a)/e
        cE0 = max(-1.0, min(1.0, cE0))
        sE0 = float(np.dot(r0v, v0v))/(e*math.sqrt(muE*a))   # sinE = r·v/(e√(μa))
        E0 = math.atan2(sE0, cE0)
        EE = math.atan2(sth, cth)                     # EI 处 (sin<0 → EE∈(−π,0))
        dt = (EE - e*math.sin(EE) - (E0 - e*math.sin(E0)))/n
        if dt < 0.0: dt += 2.0*math.pi/n              # 前向绕到 EI
    else:           # 双曲
        na = math.sqrt(muE/(-a)**3)
        H0 = math.asinh(max(-1e12, min(1e12, float(np.dot(r0v, v0v))/(e*math.sqrt(muE*(-a))))))
        HE = -math.acosh(max(1.0, (1.0 - r_EI/a)/e))  # a<0 → (1+r/|a|)/e, 下降支 H<0
        dt = (e*math.sinh(HE) - HE - (e*math.sinh(H0) - H0))/na
        if dt < 0.0: return None
    # EI 状态矢量: r⃗ = r(cosθ ê_p + sinθ q̂); v⃗ = v_r r̂ + v_t θ̂, θ̂ = −sinθ ê_p + cosθ q̂
    rhat = cth*eph + sth*qh
    that = -sth*eph + cth*qh
    v_ei_v = vr*rhat + vt*that
    return dict(r_p=r_p, gamma=gam, r_ei=r_ei_v, v_ei=v_ei_v, dt=dt, eps=eps, a=a, e=e)

def _earth_arc(jd_b, theta_deg, raan_deg, basis, dvmag, orb=None):
    """(θ,Δv) → 返回弧几何 dict / None。
    月心段 χ 括号二分到 SOI 出口; 地心段全解析(_earth_geom)。
    orb: θ→(r⃗,v⃗) 状态函数(固定惯性面); 缺省用 LE 基(basis)构造。"""
    r_b, v_b = orb(theta_deg) if orb is not None else _orb_state(jd_b, theta_deg, raan_deg, basis)
    vhat = v_b/float(np.linalg.norm(v_b))
    v1 = v_b + vhat*dvmag
    # 月心段: 必须逃逸(能量>0), χ 括号二分到 r=SOI
    v2 = float(np.dot(v1, v1))
    if v2/2.0 - muM/r_park <= 0.0: return None      # 未逃逸
    out = _time_to_radius(r_b, v1, r_soi, muM)
    if out is None: return None
    dt_m, r_ex, v_ex = out
    t_exit = jd_b + dt_m/86400.0
    # 地心段
    rE0 = _moon(t_exit) + r_ex
    vE0 = _moon_v(t_exit) + v_ex
    g = _earth_geom(rE0, vE0)
    if g is None: return None
    return dict(dt_m=dt_m, dt_ei=g["dt"], t_exit=t_exit, r_p=g["r_p"], gamma=g["gamma"],
                r_ei=g["r_ei"], v_ei=g["v_ei"], v_ei_mag=float(np.linalg.norm(g["v_ei"])),
                dvmag=dvmag, jd_ei=jd_b + (dt_m+g["dt"])/86400.0)

def _h_target(gam_deg, r_ei, rp_tgt):
    """γ_EI 与弹道 r_p 联立 → 所需地心角动量 h*(解析根)。
    h = r_EI·v·cosγ, ε = h²/(2 r_EI² cos²γ) − μ/r_EI, r_p = a(1−e), e = √(1−h²/(μa))"""
    g = math.radians(gam_deg); cg2 = math.cos(g)**2
    def rp_of_h(h):
        eps = h*h/(2.0*r_ei*r_ei*cg2) - muE/r_ei
        if eps >= 0.0: return 1e9   # 双曲线无界? h 大到 ε>0 → r_p 仍 = h²/(μ(1+e)) 有限, 但此处抛物近似失效
        a = -muE/(2.0*eps)
        e = math.sqrt(max(0.0, 1.0 - h*h/(muE*a)))
        return a*(1.0-e)
    lo, hi = 100.0, 150000.0
    for _ in range(80):
        mid = 0.5*(lo+hi)
        if rp_of_h(mid) < rp_tgt: lo = mid   # h 越大近地点越浅
        else: hi = mid
    return 0.5*(lo+hi)

def _orb_fn(jd_b, raan_deg, hhat):
    """返回 θ → (r⃗,v⃗) 的状态函数: hhat 给定 → 固定惯性面(窗口扫描用);
    否则 LE 基冻结于 t_b(单点反演用)。"""
    if hhat is not None:
        return lambda th: _orb_state_h(jd_b, th, hhat)
    basis = _le_basis(jd_b)
    return lambda th: _orb_state(jd_b, th, raan_deg, basis)

def _solve_pair(jd_b, raan_deg, orb, th0, dv0, sweep=8):
    """Gauss-Seidel 交替 secant: θ → r_p=RP_TGT; Δv → γ=GAM_TGT。
    物理敏感性: θ 主导 h(角动量) → 同时控 r_p 与 γ; Δv 主导能量 → 细调 γ。
    返回收敛弧 dict / None。"""
    rp_tgt = r_earth + RP_TGT_ALT
    th, dv = float(th0) % 360.0, float(dv0)
    def arc(th_, dv_):
        return _earth_arc(jd_b, th_, raan_deg, None, dv_, orb=orb)
    a = arc(th, dv)
    for _ in range(sweep):
        if a is None: return None
        # --- θ → r_p (secant, 步长自适应) ---
        for _ in range(8):
            f = a["r_p"] - rp_tgt
            if abs(f) < 1.0: break
            dth = 1.0 if abs(f) > 3000.0 else (0.3 if abs(f) > 300.0 else 0.05)
            a2 = arc(th + dth, dv)
            if a2 is None: dth = -dth; a2 = arc(th + dth, dv)
            if a2 is None: return None
            sl = (a2["r_p"] - a["r_p"])/dth
            if abs(sl) < 1.0: return None
            th += max(-6.0, min(6.0, -f/sl))
            a = arc(th, dv)
            if a is None: return None
        # --- Δv → γ (secant) ---
        for _ in range(8):
            f = a["gamma"] - (_GAM_OVR if _GAM_OVR is not None else GAM_TGT)
            if abs(f) < 0.01: break
            ddv = 0.005
            a2 = arc(th, dv + ddv)
            if a2 is None: ddv = -ddv; a2 = arc(th, dv + ddv)
            if a2 is None: return None
            sl = (a2["gamma"] - a["gamma"])/ddv
            if abs(sl) < 1e-6: return None
            dv += max(-0.05, min(0.05, -f/sl))
            dv = min(max(dv, 0.62), 1.80)
            a = arc(th, dv)
            if a is None: return None
        if abs(a["r_p"] - rp_tgt) < 1.0 and abs(a["gamma"] - (_GAM_OVR if _GAM_OVR is not None else GAM_TGT)) < 0.01:
            break
    if abs(a["r_p"] - rp_tgt) > 5.0 or abs(a["gamma"] - (_GAM_OVR if _GAM_OVR is not None else GAM_TGT)) > 0.05: return None
    a["theta"], a["raan"], a["jd_b"] = th % 360.0, raan_deg, jd_b
    a["dvmag"] = dv
    return a

def solve_2d(jd_b, raan_deg, gam_tol=0.2, th_lo=None, dv_guess=0.90, hhat=None):
    """h* 先验定 θ 初值(角动量扫描), Gauss-Seidel 精化解。
    hhat 给定时面在惯性系冻结(多日窗口扫描); 缺省 LE 基冻结于 t_b(单点)。
    [v0.1.1 重写] 原 θ粗扫(10°) 跨不过返回走廊窄窗(Δh/Δθ ~ 6000 km²/s/°,
    h 需 ~71591/392000 → θ 窗 ~±10°), 且内/外层敏感性分派错误(见 _solve_pair)。
    [v0.2] h 扫描无括号时换 Δv 候选重试(一周扫描中所需 Δv 随几何漂移)。"""
    orb = _orb_fn(jd_b, raan_deg, hhat)
    h_star = _h_target(_GAM_OVR if _GAM_OVR is not None else GAM_TGT, r_EI, r_earth + RP_TGT_ALT)
    th_list = list(np.arange(0.0, 360.0, 5.0))
    for dv0 in (dv_guess, 0.80, 1.00, 1.10):
        # θ 粗扫: 只算 SOI 出口地心角动量 h(θ) (1 次月心段 + 叉积, 廉价)
        hs = []
        for th in th_list:
            r_b, v_b = orb(th)
            vhat = v_b/float(np.linalg.norm(v_b))
            v1 = v_b + vhat*dv0
            if float(np.dot(v1, v1))/2.0 - muM/r_park <= 0.0: hs.append(None); continue
            out = _time_to_radius(r_b, v1, r_soi, muM)
            if out is None: hs.append(None); continue
            dt_m, r_ex, v_ex = out
            t_exit = jd_b + dt_m/86400.0
            rE0 = _moon(t_exit) + r_ex; vE0 = _moon_v(t_exit) + v_ex
            hs.append(float(np.linalg.norm(np.cross(rE0, vE0))))
        # 括号: h(θ) 穿越 h*(每圈至多两次)
        cands = []
        n = len(hs)
        for k in range(n):
            if hs[k] is None: continue
            for j in range(k+1, min(k+3, n)):
                if hs[j] is None: continue
                if (hs[k] - h_star)*(hs[j] - h_star) <= 0.0:
                    th0 = th_list[k] + (th_list[j]-th_list[k])*(h_star-hs[k])/((hs[j]-hs[k]) or 1e-9)
                    c = _solve_pair(jd_b, raan_deg, orb, th0, dv0)
                    if c is not None: cands.append(c)
                    break
        if cands: break
    if not cands: return None
    cands.sort(key=lambda c: (c["dvmag"], abs(c["gamma"]-GAM_TGT)))   # 优选低 Δv 支(省推进剂), 残差兜底
    a = cands[0]
    lon, lat = subpoint(a["r_ei"], a["jd_ei"])
    a["ei_lon"], a["ei_lat"] = lon, lat
    p1, p2 = _d(lat), _d(SITE[1])
    dl = _d(lon - SITE[0])
    a["downrange"] = math.acos(max(-1.0, min(1.0,
        math.sin(p1)*math.sin(p2)+math.cos(p1)*math.cos(p2)*math.cos(dl))))*r_earth
    a["tof_h"] = (a["jd_ei"] - jd_b)*24.0
    return a

# ---------------- EI 航迹方向分析 ----------------
def _track(a):
    """EI 处方向判断(无需大气): 航向角 vs 站点方位角 → 横轨偏差;
    真空溅落点(弹道弧穿 r=6378.137 下降支) → 沿轨距离; 水漂需补程。
    物理口径: 真空弹道会穿过 50km 深的近地点, 大气在溅落点之前接住它,
    跳跃沿同一航迹向前延长射程 —— 方向与航迹线由真空段唯一决定。"""
    rv = np.asarray(a["r_ei"], float); vv = np.asarray(a["v_ei"], float)
    jd_ei = a["jd_ei"]
    r0 = float(np.linalg.norm(rv))
    hv = np.cross(rv, vv); h = float(np.linalg.norm(hv))
    ev = np.cross(vv, hv)/muE - rv/r0; e = float(np.linalg.norm(ev))
    # EI 星下点 + 当地东北天
    lon0, lat0 = subpoint(rv, jd_ei)
    p0, lam0 = _d(lat0), _d(lon0)
    east = np.array([-math.sin(lam0), math.cos(lam0), 0.0])
    north = np.array([-math.sin(p0)*math.cos(lam0), -math.sin(p0)*math.sin(lam0), math.cos(p0)])
    az_v = math.degrees(math.atan2(float(np.dot(vv, east)), float(np.dot(vv, north)))) % 360.0
    # 站点球面几何
    lon1, lat1 = SITE
    p1, lam1 = _d(lat1), _d(lon1)
    dl = lam1 - lam0
    c = math.acos(max(-1.0, min(1.0, math.sin(p0)*math.sin(p1)+math.cos(p0)*math.cos(p1)*math.cos(dl))))
    az_site = math.degrees(math.atan2(math.sin(dl)*math.cos(p1),
                            math.cos(p0)*math.sin(p1)-math.sin(p0)*math.cos(p1)*math.cos(dl))) % 360.0
    daz = (az_site - az_v + 180.0) % 360.0 - 180.0
    raz = math.radians(daz)
    xt_ang = math.asin(max(-1.0, min(1.0, math.sin(c)*math.sin(raz))))
    xtrack = xt_ang*r_earth
    cos_along = math.cos(c)/math.cos(xt_ang) if abs(math.cos(xt_ang)) > 1e-12 else 0.0
    along_ang = math.acos(max(-1.0, min(1.0, cos_along)))
    along = along_ang*r_earth if math.cos(raz) >= 0.0 else -along_ang*r_earth  # 站点在航迹前方为正
    # 真空溅落点(解析, 同 _earth_geom 教训: 短椭圆弧数值穿越不可靠)
    vac = None
    cth_s = (h*h/(muE*r_earth) - 1.0)/e
    if abs(cth_s) <= 1.0:
        sth_s = -math.sqrt(1.0-cth_s*cth_s)
        r_s_v = r_earth*(cth_s*eph + sth_s*qh)
        eps = v0*v0/2.0 - muE/r0
        a_o = -muE/(2.0*eps)
        if eps < 0.0 and e > 1e-12:
            n = math.sqrt(muE/a_o**3)
            def _M(rv_, vv_):
                r_ = float(np.linalg.norm(rv_))
                cE = max(-1.0, min(1.0, (1.0-r_/a_o)/e))
                sE = float(np.dot(rv_, vv_))/(e*math.sqrt(muE*a_o))
                return math.atan2(sE, cE) - e*math.atan2(sE, cE)/abs(math.atan2(sE, cE))*0.0 - e*sE, math.atan2(sE, cE)
            M0, E0 = _M(rv, vv); Ms, Es = _M(r_s_v, None)
            dt_s = (Ms - M0)/n
            if dt_s < 0.0: dt_s += 2.0*math.pi/n
            lon_s, lat_s = subpoint(r_s_v, jd_ei + dt_s/86400.0)
            ps, lms = _d(lat_s), _d(lon_s)
            gv = math.acos(max(-1.0, min(1.0, math.sin(p0)*math.sin(ps)+math.cos(p0)*math.cos(ps)*math.cos(lms-lam0))))*r_earth
            vac = dict(lon=lon_s, lat=lat_s, gdist=gv, dt=dt_s)
    return dict(az=az_v, az_site=az_site, daz=daz, xtrack=xtrack, along=along,
                c=c*r_earth, vac=vac)

def _print_track(t):
    print("  航向 %.1f° vs 站点方位 %.1f° → 偏 %.1f° | 横轨偏差 %.0f km | 站点沿轨 %.0f km"
          % (t["az"], t["az_site"], t["daz"], t["xtrack"], t["along"]))
    if t["vac"] is not None:
        need = t["c"] - t["vac"]["gdist"]
        print("  真空溅落 %.2f°E %.2f°N (EI 后 %.0f s, 沿轨 %.0f km) | 到四子王旗还差 %.0f km → 水漂需补"
              % (t["vac"]["lon"], t["vac"]["lat"], t["vac"]["dt"], t["vac"]["gdist"], need))
    else:
        print("  真空溅落: 未求得")

# ---------------- 模式 ----------------
def _print_sol(a, tof_t):
    print("spot 解: Ω=%.2f° θ=%.2f° Δv=%.1f m/s | tof=%.3fh (目标 %.3fh, 差 %+.1f min)"
          % (a["raan"], a["theta"], a["dvmag"]*1000, a["tof_h"], tof_t, (a["tof_h"]-tof_t)*60))
    print("  EI %s BJT | γ=%.2f° | v_EI=%.3f km/s | 弹道近地点高度 %.0f km"
          % (to_bj(a["jd_ei"]), a["gamma"], a["v_ei_mag"], a["r_p"] - r_earth))
    print("  EI 星下点 %.2f°E %.2f°N | 到四子王旗下程 %.0f km | 月心段 %.2fh + 地心段 %.2fh"
          % (a["ei_lon"], a["ei_lat"], a["downrange"], a["dt_m"]/3600.0, a["dt_ei"]/3600.0))
    v_esc = math.sqrt(2.0*muM/r_park)
    print("  自检: Δv=%.0f m/s vs 逃逸亏差 %.0f m/s (v_esc=%.0f m/s) | γ 残差 %+.2f° | r_p 残差 %+.1f km"
          % (a["dvmag"]*1000, (v_esc-math.sqrt(muM/r_park))*1000, v_esc*1000,
             a["gamma"]-GAM_TGT, (a["r_p"]-r_earth-RP_TGT_ALT)))
    _print_track(_track(a))

def spot(date_b, t_b, date_e, t_ei):
    """[锚点反演] 两公开历元 → 反演 (Ω, θ, Δv)"""
    jd_b, jd_e = _jd(date_b, t_b), _jd(date_e, t_ei)
    tof_t = (jd_e - jd_b)*24.0
    print("[返回] TEI %s %s BJT → EI %s %s BJT | tof=%.2fh | i=%.0f°(月赤道) | r_park=%.1fkm"
          % (date_b, t_b, date_e, t_ei, tof_t, INC_ORB, r_park))
    print("  星历 %s | γ目标=%.1f° | 弹道近地点高度 %.0f km" % (_EPH, GAM_TGT, RP_TGT_ALT))
    print("-"*112)
    oms = list(np.arange(0.0, 360.0, 10.0))
    fs = []
    for om in oms:
        a = solve_2d(jd_b, om)
        fs.append(None if a is None else a["tof_h"] - tof_t)
    brs = []
    for k in range(len(oms)):
        if fs[k] is None: continue
        for j in range(k+1, min(k+4, len(oms))):
            if fs[j] is None: continue
            if (fs[k] > 0) != (fs[j] > 0):
                if not brs or abs(oms[k] - brs[-1][1]) > 1e-6:
                    brs.append((oms[k], oms[j]))
                break
    if not brs:
        print("spot: 10° 粗网无 tof 变号 —— 86° 面内该历元对无解(或走廊外)"); return
    def _bisect_om(lo, hi):
        f_lo = None; a = None
        for _ in range(45):
            mid = 0.5*(lo+hi)
            a = solve_2d(jd_b, mid)
            if a is None: hi = mid; continue
            f = a["tof_h"] - tof_t
            if f_lo is None:
                a0 = solve_2d(jd_b, lo)
                if a0 is None: lo = mid; continue
                f_lo = a0["tof_h"] - tof_t
            if (f > 0) == (f_lo > 0): lo, f_lo = mid, f
            else: hi = mid
            if hi-lo < 1e-3: break
        return solve_2d(jd_b, 0.5*(lo+hi))
    sols = [s for s in (_bisect_om(lo, hi) for lo, hi in brs) if s is not None]
    uniq = []
    for s in sols:   # 括号重叠会产出重复解, 按 (Ω,θ) 去重
        if not any(abs(s["raan"]-u["raan"]) < 0.5 and abs(s["theta"]-u["theta"]) < 0.5 for u in uniq):
            uniq.append(s)
    sols = uniq
    if not sols:
        print("spot: 收敛失败"); return
    for a in sols:
        _print_sol(a, tof_t)
    if len(sols) > 1:
        print("共 %d 个解分支(镜像族, 对应停泊轨道面朝向两解)" % len(sols))

def scan(date_b, t_b, om_lo=0.0, om_hi=360.0, dom=10.0):
    """[走廊] 固定 TEI, 扫 RAAN: 哪些面朝向能以 γ=GAM_TGT 返回"""
    jd_b = _jd(date_b, t_b)
    print("[走廊扫描] TEI %s %s BJT | i=%.0f°(月赤道) | γ=%.1f° | Ω 步 %.0f°"
          % (date_b, t_b, INC_ORB, GAM_TGT, dom))
    print("%7s %7s %8s %8s %7s %9s  %s" % ("Ω", "θ", "Δv m/s", "tof h", "γ", "下程 km", "EI 星下点"))
    hits = 0
    n = 0
    for om in np.arange(om_lo, om_hi+1e-9, dom):
        n += 1
        a = solve_2d(jd_b, float(om))
        if a is None:
            print("%7.1f     --" % om); continue
        hits += 1
        print("%7.1f %7.2f %8.1f %8.2f %7.2f %9.0f  %.1f°E %.1f°N"
              % (a["raan"], a["theta"], a["dvmag"]*1000, a["tof_h"],
                 a["gamma"], a["downrange"], a["ei_lon"], a["ei_lat"]))
    print("命中 %d/%d" % (hits, n))

def win(date_lo, t_lo, days, raan_deg, step_h=2.0, jd_anchor=None):
    """[窗口搜索] 固定 86° 面(Ω=raan_deg 于锚定历元定面, 惯性冻结), 扫 TEI 时刻
    共 days 天: 每个时刻解 (θ,Δv), 收敛即"该时刻能回家"。输出窗口分组。
    jd_anchor: Ω 标签的锚定历元(缺省=扫描起点)。要复现 spot 的解, 锚定历元
    必须取 spot 的 t_b —— 月球 ~13.2°/天 公转, LE 系 Ω 标签随之漂移。"""
    jd_ref = _jd(date_lo, t_lo)
    hhat = _plane_icrf(jd_anchor if jd_anchor is not None else jd_ref, raan_deg)
    rp_tgt = r_earth + RP_TGT_ALT
    print("[窗口] TEI %s %s BJT 起 +%g 天, 步长 %gh | 面 i=%.0f°(月赤道) Ω=%.2f°@%s(惯性冻结)"
          % (date_lo, t_lo, days, step_h, INC_ORB, raan_deg,
             to_bj(jd_anchor) if jd_anchor is not None else (date_lo+" "+t_lo)))
    print("  γ=%.1f° | 弹道近地点高度 %.0f km | 落区匹配: 下程 %.0f~%.0f km"
          % (GAM_TGT, RP_TGT_ALT, DR_LO, DR_HI))
    print("-"*112)
    hits = []
    n = int(round(days*24.0/step_h))
    for k in range(n+1):
        jd_b = jd_ref + k*step_h/24.0
        a = solve_2d(jd_b, raan_deg, hhat=hhat)
        if a is None or not (TOF_LO <= a["tof_h"] <= TOF_HI): continue
        hits.append((jd_b, a))
        print("%s  θ=%6.2f° Δv=%5.1f tof=%6.2fh γ=%6.2f 下程%6.0fkm EI(%7.2f°E,%6.2f°N)%s"
              % (to_bj(jd_b), a["theta"], a["dvmag"]*1000, a["tof_h"], a["gamma"],
                 a["downrange"], a["ei_lon"], a["ei_lat"],
                 "  ✓" if DR_LO <= a["downrange"] <= DR_HI else ""))
    if not hits:
        print("本周无返回窗口(该面朝向下)"); return
    # 分组: 相邻命中间隔 ≤ 2 步长 → 同一窗口
    wins = []
    cur = [hits[0]]
    for h_ in hits[1:]:
        if (h_[0] - cur[-1][0])*24.0 <= 2.0*step_h + 1e-9: cur.append(h_)
        else: wins.append(cur); cur = [h_]
    wins.append(cur)
    print("-"*112)
    for w in wins:
        dvs = [x[1]["dvmag"]*1000 for x in w]
        tofs = [x[1]["tof_h"] for x in w]
        drs = [x[1]["downrange"] for x in w]
        n_ok = sum(1 for x in w if DR_LO <= x[1]["downrange"] <= DR_HI)
        print("窗口: %s → %s BJT | 长 %.1fh | %d 采样 | Δv %.0f~%.0f m/s | tof %.0f~%.0f h | 下程 %.0f~%.0f km | 落区匹配 %d"
              % (to_bj(w[0][0]), to_bj(w[-1][0]), (w[-1][0]-w[0][0])*24.0, len(w),
                 min(dvs), max(dvs), min(tofs), max(tofs), min(drs), max(drs), n_ok))

def design_point(jd_b, inc):
    """单设计点: 给定倾角 i 与点火时刻 t_b → 扫 Ω 找横轨归零(航迹过站点)。
    (θ,Δv) 由内层 2×2 解 (γ,r_p); Ω 是 LOI 时自主选的面朝向;
    到达时刻 t_EI 由几何确定(输出)。返回解行列表。"""
    set_inc(inc)
    oms = list(np.arange(0.0, 360.0, 10.0))
    xs, arcs = [], []
    for om in oms:
        a = solve_2d(jd_b, om)
        if a is None: xs.append(None); arcs.append(None); continue
        a["tof_h"] = (a["jd_ei"] - jd_b)*24.0
        if not (TOF_LO <= a["tof_h"] <= TOF_HI): xs.append(None); arcs.append(None); continue
        t = _track(a)
        a["xtrack"], a["daz"], a["along"] = t["xtrack"], t["daz"], t["along"]
        xs.append(a["xtrack"]); arcs.append(a)
    rows = []
    def _row(om):
        a = solve_2d(jd_b, om)
        if a is None: return None
        a["tof_h"] = (a["jd_ei"] - jd_b)*24.0
        t = _track(a)
        a["inc"], a["jd_b"], a["xtrack"], a["daz"] = inc, jd_b, t["xtrack"], t["daz"]
        return a
    for k in range(len(oms)):
        if xs[k] is None: continue
        for j in range(k+1, min(k+3, len(oms))):
            if xs[j] is None: continue
            if (xs[k] > 0) != (xs[j] > 0):
                lo, hi = oms[k], oms[j]
                f_lo = xs[k]
                for _ in range(14):
                    mid = 0.5*(lo+hi)
                    r = _row(mid)
                    if r is None: hi = mid; continue
                    if (r["xtrack"] > 0) == (f_lo > 0): lo, f_lo = mid, r["xtrack"]
                    else: hi = mid
                    if abs(r["xtrack"]) < 20.0: break
                r = _row(0.5*(lo+hi))
                if r is not None and abs(r["xtrack"]) < 80.0:
                    if not any(abs(r["raan"]-u["raan"]) < 0.5 for u in rows):
                        rows.append(r)
                break
    set_inc(None)
    return rows

def design(date_lo, t_lo, days, dtb_h, incs, outfile=None):
    """[设计/窗口模式] (i × 点火日期) 网格 → 航迹过站点的 (Ω, θ, Δv, 到达时刻) 表。
    outfile 给定时: 追加写 + 断点续跑(跳过已完成的 (i,点火时刻)) + 末尾窗口分组汇总。"""
    jd0 = _jd(date_lo, t_lo)
    n = int(round(days*24.0/dtb_h))
    print("[窗口] 点火 %s %s BJT 起 +%g 天 步 %gh | i=%s | 落站 %s | γ=%.1f° 近地点 %.0fkm"
          % (date_lo, t_lo, days, dtb_h, incs, SITE, _GAM_OVR if _GAM_OVR is not None else GAM_TGT, RP_TGT_ALT), flush=True)
    print("%5s %16s %7s %7s %8s %16s %7s %9s %8s" %
          ("i", "点火 BJT", "Ω", "θ", "Δv m/s", "EI BJT", "tof h", "横轨 km", "下程km"), flush=True)
    done = set()
    if outfile and os.path.exists(outfile):
        for line in open(outfile):
            t = line.split()
            if len(t) >= 11 and t[0].replace('.','',1).isdigit():
                try: done.add((float(t[0]), to_bj(_jd(t[1], t[2]))))
                except Exception: pass
        if done: print("[续跑] 跳过已完成 %d 点" % len(done), flush=True)
    all_rows = []
    for inc in incs:
        for k in range(n):
            jd_b = jd0 + k*dtb_h/24.0
            if (inc, to_bj(jd_b)) in done: continue
            rows = design_point(jd_b, inc)
            for r in rows:
                print("%5.1f %16s %7.2f %7.2f %8.1f %16s %7.2f %9.0f %8.0f"
                      % (inc, to_bj(jd_b), r["raan"], r["theta"], r["dvmag"]*1000,
                         to_bj(r["jd_ei"]), r["tof_h"], r["xtrack"], r["downrange"]), flush=True)
                r["jd_b"] = jd_b; all_rows.append(r)
            if outfile:
                with open(outfile, "a") as f:
                    for r in rows:
                        f.write("%5.1f %16s %7.2f %7.2f %8.1f %16s %7.2f %9.0f %8.0f\n"
                                % (inc, to_bj(jd_b), r["raan"], r["theta"], r["dvmag"]*1000,
                                   to_bj(r["jd_ei"]), r["tof_h"], r["xtrack"], r["downrange"]))
    if outfile and all_rows:
        all_rows.sort(key=lambda r: r["jd_b"])
        print("-"*104, flush=True)
        wins, cur = [], [all_rows[0]]
        for r in all_rows[1:]:
            if (r["jd_b"]-cur[-1]["jd_b"])*24.0 <= 2.0*dtb_h + 1e-9: cur.append(r)
            else: wins.append(cur); cur = [r]
        wins.append(cur)
        for w in wins:
            ok = [r for r in w if DR_LO <= r["downrange"] <= DR_HI and abs(r["xtrack"]) < 50.0]
            print("返回窗口: %s → %s 点火 | 长 %.1fh | 方案 %d | 可执行(横轨<50,下程带内) %d | 到达 %s ~ %s"
                  % (to_bj(w[0]["jd_b"]), to_bj(w[-1]["jd_b"]), (w[-1]["jd_b"]-w[0]["jd_b"])*24.0,
                     len(w), len(ok), to_bj(min(r["jd_ei"] for r in w)), to_bj(max(r["jd_ei"] for r in w))), flush=True)

DR_LO, DR_HI = 4000.0, 8000.0   # 半弹道跳跃下程量级(落区匹配判据, v0.2 经验带)
TOF_LO, TOF_HI = 48.0, 144.0    # 地心弧 2~6 天(滤掉多圈绕地解, 任务上无意义)

USAGE = """用法:
  python moon2earth.py spot 2020-12-12 09:54 2020-12-17 01:33   # CE-5 两历元反演 (Ω,θ,Δv)
  python moon2earth.py scan 2020-12-12 09:54 [Ω起 Ω止 步长]      # 固定 TEI, 扫 RAAN 走廊
  python moon2earth.py win  2020-12-09 00:00 7 [Ω] [步长h]       # 固定 86° 面, 搜一周返回窗口
  # Ω 缺省用 spot 反演值 7.44 (CE-5 分支 A); win 是多日扫描, 惯性冻结面"""

if __name__ == "__main__":
    # 位置式默认模式: python moon2earth.py 2020-07-24 86 8 [步长h] [经,纬]
    _p = sys.argv[1].split("-") if len(sys.argv) >= 4 else []
    if len(_p) == 3:
        try:
            int(_p[0]); int(_p[1]); int(_p[2])
            inc = float(sys.argv[2]); days = float(sys.argv[3])
            step = float(sys.argv[4]) if len(sys.argv) >= 5 else 3.0
            if len(sys.argv) >= 6:
                SITE = tuple(float(x) for x in sys.argv[5].split(","))
            _out = "out/win_%s_i%g_%gd.txt" % (sys.argv[1], inc, days)
            os.makedirs("out", exist_ok=True)
            design(sys.argv[1], "00:00", days, step, [inc], outfile=_out)
            sys.exit(0)
        except ValueError:
            pass
    if len(sys.argv) < 2:
        print(USAGE); sys.exit(0)
    cmd = sys.argv[1]
    if cmd == "spot" and len(sys.argv) >= 6:
        spot(sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5])
    elif cmd == "scan" and len(sys.argv) >= 4:
        args = [float(x) for x in sys.argv[4:7]] if len(sys.argv) >= 7 else (0.0, 360.0, 10.0)
        scan(sys.argv[2], sys.argv[3], *args)
    elif cmd == "design" and len(sys.argv) >= 4:
        incs = [float(x) for x in sys.argv[3].split(",")]
        dtb = float(sys.argv[4]) if len(sys.argv) >= 5 else 6.0
        days = float(sys.argv[5]) if len(sys.argv) >= 6 else 7.0
        design(sys.argv[2].split("+")[0], sys.argv[2].split("+")[1] if "+" in sys.argv[2] else "00:00",
               days, dtb, incs)
    elif cmd == "track" and len(sys.argv) >= 4:
        jd_b = _jd(sys.argv[2], sys.argv[3])
        raan = float(sys.argv[4]) if len(sys.argv) >= 5 else 90.80
        hhat = _plane_icrf(jd_b, raan)
        a = solve_2d(jd_b, raan, hhat=hhat)
        if a is None:
            print("track: 该历元/面朝向无解"); sys.exit(0)
        a["tof_h"] = (a["jd_ei"]-jd_b)*24.0
        print("[方向分析] TEI %s %s BJT | Ω=%.2f°@该历元 | θ=%.2f° Δv=%.1f m/s | tof=%.2fh"
              % (sys.argv[2], sys.argv[3], raan, a["theta"], a["dvmag"]*1000, a["tof_h"]))
        _print_track(_track(a))
    elif cmd == "win" and len(sys.argv) >= 5:
        raan = float(sys.argv[5]) if len(sys.argv) >= 6 else 90.80
        step = float(sys.argv[6]) if len(sys.argv) >= 7 else 2.0
        jd_anchor = None
        if len(sys.argv) >= 9:   # win 起 时 天 [Ω] [步] [锚日期 锚时]
            jd_anchor = _jd(sys.argv[7], sys.argv[8])
        win(sys.argv[2], sys.argv[3], float(sys.argv[4]), raan, step, jd_anchor)
    else:
        print(USAGE)
