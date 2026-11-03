#!/usr/bin/env python3
"""
SINGLE CYLINDER ENGINE SIMULATOR  -  Made by MD Rehan
--------------------------------
A real-time 4-stroke engine simulator inspired by AngeTheGreat's Engine Simulator,
reduced to ONE cylinder, ONE intake valve and ONE exhaust valve.

Physics (all solved live, every ~50 microseconds of simulated time):
  * slider-crank kinematics (bore, stroke, rod length, compression ratio)
  * single-zone ideal-gas cylinder model (mass + energy balance, first law)
  * compressible orifice flow through throttle, intake valve and exhaust valve
  * intake plenum (manifold) with its own pressure
  * Wiebe heat-release combustion, spark advance, air/fuel ratio effect
  * wall heat loss, friction, flywheel inertia, starter motor, dyno, brake load

Requires:  Python 3.8+  and  pygame      ->   pip install pygame
Run:       python single_cyl_engine_sim.py
"""
import math
import sys

import pygame

# =====================================================================
#  PHYSICAL CONSTANTS & ENGINE DEFINITION  (edit these to build your own engine)
# =====================================================================
P_AMB = 101325.0            # ambient pressure [Pa]
T_AMB = 300.0               # ambient temperature [K]
R_GAS = 287.0               # specific gas constant [J/kg/K]
GAMMA = 1.30                # in-cylinder ratio of specific heats
CV = R_GAS / (GAMMA - 1.0)
CP = CV + R_GAS
GF = 1.4                    # gamma used for orifice-flow equations

# --- geometry ---------------------------------------------------------
BORE = 0.086                # [m]
STROKE = 0.086              # [m]
ROD = 0.143                 # connecting-rod length [m]
CR = 10.0                   # compression ratio
CRANK_R = STROKE / 2.0
A_PIST = math.pi / 4.0 * BORE * BORE
V_DISP = A_PIST * STROKE
V_CLR = V_DISP / (CR - 1.0)

# --- valve train (crank angle: 0 deg = firing TDC, 720 deg = full cycle) ---
INT_D, INT_LIFT, INT_CENTER, INT_DUR = 0.034, 0.0090, 465.0, 240.0
EXH_D, EXH_LIFT, EXH_CENTER, EXH_DUR = 0.029, 0.0085, 255.0, 240.0
CD_VALVE = 0.65
INT_CAP = math.pi * INT_D * INT_D / 4.0
EXH_CAP = math.pi * EXH_D * EXH_D / 4.0
IVC_DEG = (INT_CENTER + INT_DUR / 2.0) % 720.0

# --- intake system -----------------------------------------------------
V_PLENUM = 0.003            # intake manifold volume [m^3]
CDA_THR = 8.0e-4            # throttle effective flow area at WOT [m^2]
P_EXH = P_AMB * 1.02        # exhaust back-pressure [Pa]
T_EXH_BACK = 900.0          # temperature of gas sucked back from exhaust [K]

# --- combustion ----------------------------------------------------------
LHV = 44.0e6                # fuel lower heating value [J/kg]
COMB_EFF = 0.80
IGN_DELAY = 8.0             # flame-development delay [deg]
BURN_DUR = 55.0             # 0-100 % burn duration [deg]
WIEBE_A = 5.0
FUEL_DENSITY = 745.0        # [kg/m^3]
HA_LOSS = 5.0               # wall heat transfer [W/K]
T_WALL = 450.0

# --- mechanical -------------------------------------------------------
INERTIA = 0.12              # crank + flywheel [kg m^2]
FRIC0, FRIC1 = 1.5, 0.012   # friction torque = FRIC0 + FRIC1 * omega  [N m]
STARTER_T, STARTER_W = 30.0, 85.0   # starter torque / cut-out speed [N m, rad/s]
LOAD_K = 0.06               # brake load: N m per rad/s at 100 %
REV_LIMIT = 7500.0          # rpm  (fuel + spark cut)
DYNO_KP, DYNO_KI = 1.5, 15.0

RPM2RAD = 2.0 * math.pi / 60.0
RAD2RPM = 60.0 / (2.0 * math.pi)
RAD2DEG = 180.0 / math.pi
FOUR_PI = 4.0 * math.pi

# --- orifice flow constants ----------------------------------------------
CHOKE_R = (2.0 / (GF + 1.0)) ** (GF / (GF - 1.0))
CHOKE_K = (math.sqrt(GF) * (2.0 / (GF + 1.0)) ** ((GF + 1.0) / (2.0 * (GF - 1.0)))
           / math.sqrt(R_GAS))
SUB_K = 2.0 * GF / (GF - 1.0)


def orifice(pu, tu, pd, a):
    """Mass flow [kg/s] from upstream (pu,tu) to downstream pd through effective area a."""
    r = pd / pu
    if r <= CHOKE_R:
        return a * pu * CHOKE_K / math.sqrt(tu)
    v = SUB_K * (r ** (2.0 / GF) - r ** ((GF + 1.0) / GF))
    if v <= 0.0:
        return 0.0
    return a * pu / math.sqrt(R_GAS * tu) * math.sqrt(v)


def flow(p1, t1, p2, t2, a):
    """Signed mass flow from side 1 to side 2."""
    if p1 >= p2:
        return orifice(p1, t1, p2, a)
    return -orifice(p2, t2, p1, a)


# =====================================================================
#  ENGINE MODEL
# =====================================================================
class Engine:
    def __init__(self):
        self.reset()

    def reset(self):
        # controls
        self.throttle = 0.08
        self.ign = False
        self.starter = False
        self.dyno = False
        self.dyno_rpm = 2000.0
        self.load = 0.0
        self.spark = 25.0           # deg BTDC
        self.afr = 13.8
        # crank state
        self.phi = math.radians(100.0)
        self.w = 0.0                # rad/s
        self._geom()
        # gas state
        self.T = T_AMB
        self.m = P_AMB * self.V / (R_GAS * T_AMB)
        self.mp = P_AMB * V_PLENUM / (R_GAS * T_AMB)
        self.pman = P_AMB
        # combustion
        self.burning = False
        self.bd = 0.0
        self.xb_prev = 0.0
        self.qtot = 0.0
        self.fuel_total = 0.0
        self.fuel_lph = 0.0
        self.misfire = False
        # measurements
        self.tq_f = 0.0
        self.tex = 600.0
        self.ve = 0.0
        self.dint = 0.0
        self.limiter = False
        self.p = P_AMB
        self.li = self.le = 0.0
        self.mdi = self.mde = 0.0
        # scope buffers (one slot per 2 deg of the 720 deg cycle)
        self.sp = [P_AMB] * 360
        self.sv = [self.V] * 360
        self.sli = [0.0] * 360
        self.sle = [0.0] * 360
        self.sfl = [0.0] * 360
        self.last_idx = -1

    def _geom(self):
        self.sn = math.sin(self.phi)
        self.cs = math.cos(self.phi)
        self.sq = math.sqrt(ROD * ROD - CRANK_R * CRANK_R * self.sn * self.sn)
        x = CRANK_R * self.cs + self.sq
        self.V = V_CLR + A_PIST * (CRANK_R + ROD - x)

    @property
    def rpm(self):
        return self.w * RAD2RPM

    def advance(self, t_avail, max_steps):
        """Advance simulation by up to t_avail seconds (at most max_steps steps)."""
        sin, cos, sqrt, exp = math.sin, math.cos, math.sqrt, math.exp
        phi, w, m, T, V = self.phi, self.w, self.m, self.T, self.V
        sn, cs, sq = self.sn, self.cs, self.sq
        mp, pman = self.mp, self.pman
        burning, bd, xb_prev, qtot = self.burning, self.bd, self.xb_prev, self.qtot
        tq_f, tex, dint = self.tq_f, self.tex, self.dint
        thr, ign, starter, dyno = self.throttle, self.ign, self.starter, self.dyno
        wt = self.dyno_rpm * RPM2RAD
        load = self.load
        spark_deg = 720.0 - self.spark
        afr = self.afr
        afr_eff = max(0.0, 1.0 - ((afr - 13.2) / 5.5) ** 2)
        ath = CDA_THR * (0.004 + 0.996 * thr)
        sp, sv, sli, sle, sfl = self.sp, self.sv, self.sli, self.sle, self.sfl
        last_idx = self.last_idx
        fuel_total, fuel_lph, ve = self.fuel_total, self.fuel_lph, self.ve
        misfire, limiter = self.misfire, self.limiter
        rev_w = REV_LIMIT * RPM2RAD
        t_done = 0.0
        n = 0
        li = le = mdi = mde = p = 0.0

        while t_done < t_avail and n < max_steps:
            dt = 50e-6
            lim = 0.025 / (w if w > 1.0 else 1.0)
            if lim < dt:
                dt = lim
            if t_done + dt > t_avail:
                dt = t_avail - t_done
            if dt < 1e-7:
                break
            n += 1
            t_done += dt

            p = m * R_GAS * T / V
            deg = phi * RAD2DEG
            dphi = w * dt * RAD2DEG

            # ---- valve lifts
            d = (deg - INT_CENTER + 360.0) % 720.0 - 360.0
            if -INT_DUR / 2.0 < d < INT_DUR / 2.0:
                li = INT_LIFT * 0.5 * (1.0 + cos(d * 6.283185307179586 / INT_DUR))
            else:
                li = 0.0
            d = (deg - EXH_CENTER + 360.0) % 720.0 - 360.0
            if -EXH_DUR / 2.0 < d < EXH_DUR / 2.0:
                le = EXH_LIFT * 0.5 * (1.0 + cos(d * 6.283185307179586 / EXH_DUR))
            else:
                le = 0.0

            # ---- flows
            mdi = mde = 0.0
            if li > 0.0:
                a = math.pi * INT_D * li
                if a > INT_CAP:
                    a = INT_CAP
                mdi = flow(pman, T_AMB, p, T, a * CD_VALVE)
            if le > 0.0:
                a = math.pi * EXH_D * le
                if a > EXH_CAP:
                    a = EXH_CAP
                mde = flow(P_EXH, T_EXH_BACK, p, T, a * CD_VALVE)
            mthr = flow(P_AMB, T_AMB, pman, T_AMB, ath)

            dmi = mdi * dt
            dme = mde * dt
            lim_m = 0.3 * m
            if dmi < -lim_m:
                dmi = -lim_m
            if dme < -lim_m:
                dme = -lim_m

            # ---- spark / fuel
            sd = (spark_deg - deg) % 720.0
            if sd < dphi:
                if w > rev_w:
                    limiter = True
                else:
                    limiter = False
                if ign and not limiter:
                    fuel_m = m / (afr + 1.0)
                    qtot = fuel_m * LHV * COMB_EFF * afr_eff
                    burning = True
                    bd = 0.0
                    xb_prev = 0.0
                    fuel_total += fuel_m
                    fuel_lph = fuel_m / FUEL_DENSITY * (w * RAD2RPM / 2.0) * 60.0 * 1000.0
                    misfire = afr_eff < 0.05
                else:
                    fuel_lph = 0.0

            dq = 0.0
            if burning:
                bd += dphi
                u = (bd - IGN_DELAY) / BURN_DUR
                if u > 0.0:
                    xb = 1.0 - exp(-WIEBE_A * u * u * u) if u < 2.0 else 1.0
                    dq = qtot * (xb - xb_prev)
                    xb_prev = xb
                    if u > 1.6:
                        burning = False

            # ---- crank torque balance
            dvdth = A_PIST * CRANK_R * sn * (1.0 + CRANK_R * cs / sq)
            tg = (p - P_AMB) * dvdth
            tf = FRIC0 + FRIC1 * w
            ts = STARTER_T * (1.0 - w / STARTER_W) if (starter and w < STARTER_W) else 0.0
            tl = load * LOAD_K * w
            td = 0.0
            if dyno:
                e = w - wt
                dint += e * dt
                if dint > 20.0:
                    dint = 20.0
                elif dint < -20.0:
                    dint = -20.0
                td = DYNO_KP * e + DYNO_KI * dint
                if td > 300.0:
                    td = 300.0
                elif td < -150.0:
                    td = -150.0
            w += (tg - tf - tl - td + ts) / INERTIA * dt
            if w < 0.0:
                w = 0.0

            # averaged brake torque (over ~2 revolutions)
            meas = td if dyno else (tg - tf)
            alpha = dt * w * 0.0795775
            if alpha < dt * 2.0:
                alpha = dt * 2.0
            if alpha > 1.0:
                alpha = 1.0
            tq_f += (meas - tq_f) * alpha

            # ---- advance crank, gas energy balance
            phi_n = phi + w * dt
            if phi_n >= FOUR_PI:
                phi_n -= FOUR_PI
            sn_n = sin(phi_n)
            cs_n = cos(phi_n)
            sq_n = sqrt(ROD * ROD - CRANK_R * CRANK_R * sn_n * sn_n)
            v_n = V_CLR + A_PIST * (CRANK_R + ROD - (CRANK_R * cs_n + sq_n))
            dV = v_n - V

            tsi = T_AMB if dmi >= 0.0 else T
            tse = T_EXH_BACK if dme >= 0.0 else T
            hl = HA_LOSS * (T - T_WALL) * dt
            T += (-p * dV + dq - hl
                  + dmi * (CP * tsi - CV * T)
                  + dme * (CP * tse - CV * T)) / (m * CV)
            if T < 150.0:
                T = 150.0
            elif T > 4500.0:
                T = 4500.0
            m += dmi + dme
            if m < 1e-6:
                m = 1e-6

            # ---- intake plenum
            mp += mthr * dt - dmi
            pman = mp * R_GAS * T_AMB / V_PLENUM
            if pman < 6000.0:
                pman = 6000.0
                mp = pman * V_PLENUM / (R_GAS * T_AMB)
            elif pman > P_AMB:
                pman = P_AMB
                mp = pman * V_PLENUM / (R_GAS * T_AMB)

            # ---- measurements
            if mde < 0.0:
                a2 = dt * 150.0
                tex += (T - tex) * (a2 if a2 < 1.0 else 1.0)
            if sd2_cross(deg, dphi):
                ve = m / (P_AMB / (R_GAS * T_AMB) * V_DISP)
            idx = int(deg * 0.5)
            if idx != last_idx and 0 <= idx < 360:
                sp[idx] = p
                sv[idx] = V
                sli[idx] = li
                sle[idx] = le
                sfl[idx] = -mde
                last_idx = idx

            phi, sn, cs, sq, V = phi_n, sn_n, cs_n, sq_n, v_n

        # write back
        self.phi, self.w, self.m, self.T, self.V = phi, w, m, T, V
        self.sn, self.cs, self.sq = sn, cs, sq
        self.mp, self.pman = mp, pman
        self.burning, self.bd, self.xb_prev, self.qtot = burning, bd, xb_prev, qtot
        self.tq_f, self.tex, self.dint = tq_f, tex, dint
        self.last_idx = last_idx
        self.fuel_total, self.fuel_lph, self.ve = fuel_total, fuel_lph, ve
        self.misfire, self.limiter = misfire, limiter
        self.p, self.li, self.le, self.mdi, self.mde = p, li, le, mdi, mde
        return t_done


def sd2_cross(deg, dphi):
    """True when the crank passes intake-valve-closing during this step."""
    return ((IVC_DEG - deg) % 720.0) < dphi


# =====================================================================
#  USER INTERFACE
# =====================================================================
W, H = 1280, 720
BG = (14, 15, 17)
LINE = (190, 190, 190)
WHITE = (240, 240, 240)
DIM = (120, 124, 130)
RED = (230, 60, 60)
ORANGE = (240, 140, 30)
CYAN = (90, 200, 230)
GREEN = (90, 220, 120)
YELLOW = (250, 220, 80)

LEFT_W = 300
RIGHT_X = 860
CENTER = pygame.Rect(LEFT_W, 0, RIGHT_X - LEFT_W, 400)


def lerp(a, b, t):
    return a + (b - a) * t


def lerp_col(c1, c2, t):
    t = max(0.0, min(1.0, t))
    return (int(lerp(c1[0], c2[0], t)), int(lerp(c1[1], c2[1], t)), int(lerp(c1[2], c2[2], t)))


GAS_STOPS = [(0.0, (30, 70, 140)), (0.30, (140, 60, 150)),
             (0.65, (240, 110, 30)), (1.0, (255, 235, 130))]


def gas_color(temp):
    f = max(0.0, min(1.0, (temp - 300.0) / 2400.0))
    for i in range(len(GAS_STOPS) - 1):
        f0, c0 = GAS_STOPS[i]
        f1, c1 = GAS_STOPS[i + 1]
        if f <= f1:
            return lerp_col(c0, c1, (f - f0) / (f1 - f0))
    return GAS_STOPS[-1][1]


class Fonts:
    def __init__(self):
        names = "consolas,lucidaconsole,couriernew,dejavusansmono,monospace"
        self.title = pygame.font.SysFont(names, 30, bold=True)
        self.big = pygame.font.SysFont(names, 22, bold=True)
        self.mid = pygame.font.SysFont(names, 16, bold=True)
        self.small = pygame.font.SysFont(names, 13, bold=True)
        self.tiny = pygame.font.SysFont(names, 11, bold=True)


def text(surf, font, s, pos, color=WHITE, center=False, right=False):
    img = font.render(s, True, color)
    r = img.get_rect()
    if center:
        r.center = pos
    elif right:
        r.topright = pos
    else:
        r.topleft = pos
    surf.blit(img, r)


def panel(surf, rect):
    pygame.draw.rect(surf, BG, rect)
    pygame.draw.rect(surf, LINE, rect, 1)


def draw_gauge(surf, fonts, cx, cy, r, value, vmin, vmax, readout,
               orange=None, red=None, ticks=8):
    rect = pygame.Rect(cx - r, cy - r, 2 * r, 2 * r)

    def ang(f):
        return math.radians(225.0 - 270.0 * f)

    pygame.draw.arc(surf, (60, 62, 66), rect, ang(1.0), ang(0.0), 2)
    if orange is not None:
        pygame.draw.arc(surf, ORANGE, rect, ang(red if red else 1.0), ang(orange), 4)
    if red is not None:
        pygame.draw.arc(surf, RED, rect, ang(1.0), ang(red), 4)
    for i in range(ticks + 1):
        a = ang(i / ticks)
        c, s = math.cos(a), math.sin(a)
        pygame.draw.line(surf, WHITE, (cx + c * r * 0.82, cy - s * r * 0.82),
                         (cx + c * r * 0.97, cy - s * r * 0.97), 2)
    for i in range(ticks * 2 + 1):
        a = ang(i / (ticks * 2))
        c, s = math.cos(a), math.sin(a)
        pygame.draw.line(surf, DIM, (cx + c * r * 0.90, cy - s * r * 0.90),
                         (cx + c * r * 0.97, cy - s * r * 0.97), 1)
    f = (value - vmin) / (vmax - vmin) if vmax != vmin else 0.0
    f = max(0.0, min(1.0, f))
    a = ang(f)
    pygame.draw.line(surf, RED, (cx - math.cos(a) * r * 0.12, cy + math.sin(a) * r * 0.12),
                     (cx + math.cos(a) * r * 0.80, cy - math.sin(a) * r * 0.80), 4)
    pygame.draw.circle(surf, WHITE, (cx, cy), 4)
    text(surf, fonts.small, readout, (cx, cy + int(r * 0.62)), WHITE, center=True)


def draw_engine(surf, fonts, eng):
    panel(surf, CENTER)
    s = 1.15
    cx = CENTER.centerx
    cy = CENTER.y + 338
    rr = CRANK_R * 1000.0
    half_b = BORE * 500.0 * s
    x_mm = (CRANK_R * eng.cs + eng.sq) * 1000.0
    pin = (cx, cy - x_mm * s)
    cpin = (cx + rr * eng.sn * s, cy - rr * eng.cs * s)
    crown_y = pin[1] - 25 * s
    head_y = cy - (CRANK_R + ROD) * 1000.0 * s - 25 * s - (V_CLR / A_PIST) * 1000.0 * s
    bdc_crown = cy - (ROD - CRANK_R) * 1000.0 * s - 25 * s
    liner_bottom = bdc_crown + 75 * s

    # ---- gas
    gc = gas_color(eng.T)
    pygame.draw.rect(surf, gc, (cx - half_b, head_y, 2 * half_b, max(1, crown_y - head_y)))
    # ---- cylinder walls & head
    pygame.draw.rect(surf, (70, 72, 78), (cx - half_b - 14, head_y, 14, liner_bottom - head_y))
    pygame.draw.rect(surf, (70, 72, 78), (cx + half_b, head_y, 14, liner_bottom - head_y))
    pygame.draw.rect(surf, (70, 72, 78), (cx - half_b - 14, head_y - 42, 2 * half_b + 28, 42))

    # ---- ports (coloured by flow)
    fin = min(1.0, max(0.0, eng.mdi) / 0.03)
    fout = min(1.0, max(0.0, -eng.mde) / 0.10)
    ix, ex = cx - 24, cx + 24
    pygame.draw.line(surf, lerp_col((30, 32, 36), CYAN, fin), (ix, head_y - 22), (ix - 50, head_y - 70), 16)
    pygame.draw.line(surf, lerp_col((30, 32, 36), ORANGE, fout), (ex, head_y - 22), (ex + 50, head_y - 70), 16)
    text(surf, fonts.small, "INTAKE", (ix - 80, head_y - 98), CYAN)
    text(surf, fonts.small, "EXHAUST", (ex + 20, head_y - 98), ORANGE)

    # ---- valves
    for vx, d, lift, col in ((ix, INT_D, eng.li, CYAN), (ex, EXH_D, eng.le, ORANGE)):
        lp = lift * 1000.0 * s * 1.5
        pygame.draw.line(surf, (170, 170, 175), (vx, head_y - 38), (vx, head_y + lp), 4)
        vw = d * 1000.0 * s
        pygame.draw.rect(surf, col if lift > 0.0002 else (150, 150, 155),
                         (vx - vw / 2, head_y + lp - 2, vw, 6))

    # ---- spark plug and spark
    pygame.draw.line(surf, (230, 230, 230), (cx, head_y - 42), (cx, head_y - 4), 6)
    if eng.burning and eng.bd < 12.0:
        for k in range(6):
            a = k * math.pi / 3.0 + eng.bd
            pygame.draw.line(surf, YELLOW, (cx, head_y + 3),
                             (cx + math.cos(a) * 14, head_y + 3 + math.sin(a) * 14), 2)

    # ---- piston, rod, crank
    pygame.draw.rect(surf, (225, 225, 228), (cx - half_b + 1, crown_y, 2 * half_b - 2, 62 * s))
    for k in range(3):
        yy = crown_y + 8 + k * 7
        pygame.draw.line(surf, (110, 112, 118), (cx - half_b + 1, yy), (cx + half_b - 1, yy), 2)
    pygame.draw.line(surf, (235, 235, 235), pin, cpin, 11)
    pygame.draw.circle(surf, (40, 40, 44), (int(pin[0]), int(pin[1])), 6)
    pygame.draw.circle(surf, (70, 72, 78), (cx, cy), 62)
    cw = (cx - rr * eng.sn * s * 0.9, cy + rr * eng.cs * s * 0.9)
    pygame.draw.circle(surf, (150, 152, 158), (int(cw[0]), int(cw[1])), 36)
    pygame.draw.circle(surf, (205, 205, 210), (int(cpin[0]), int(cpin[1])), 13)
    pygame.draw.circle(surf, (40, 40, 44), (int(cpin[0]), int(cpin[1])), 6)
    pygame.draw.circle(surf, (40, 40, 44), (cx, cy), 8)
    pygame.draw.rect(surf, LINE, CENTER, 1)

    # ---- label
    deg = (eng.phi * RAD2DEG) % 720.0
    stroke = ("POWER", "EXHAUST", "INTAKE", "COMPRESSION")[int(deg // 180) % 4]
    text(surf, fonts.big, "MADE BY MD REHAN", (CENTER.right - 12, CENTER.y + 8), YELLOW, right=True)
    text(surf, fonts.mid, "CRANK %5.1f DEG  [%s]" % (deg, stroke), (CENTER.x + 10, CENTER.y + 8), WHITE)
    text(surf, fonts.small, "CYL P %6.2f BAR   T %5.0f K" % (eng.p / 1e5, eng.T),
         (CENTER.x + 10, CENTER.y + 30), DIM)


def plot_frame(surf, fonts, rect, title):
    panel(surf, rect)
    text(surf, fonts.small, title, (rect.x + 8, rect.y + 4), WHITE)


def draw_pressure_plot(surf, fonts, rect, eng, pscale):
    plot_frame(surf, fonts, rect, "CYLINDER PRESSURE / VALVE LIFT vs CRANK ANGLE")
    inner = pygame.Rect(rect.x + 36, rect.y + 24, rect.w - 46, rect.h - 40)
    pygame.draw.rect(surf, (30, 32, 36), inner, 1)
    names = ("POWER", "EXHAUST", "INTAKE", "COMPR.")
    for i in range(5):
        x = inner.x + inner.w * i / 4.0
        pygame.draw.line(surf, (50, 52, 58), (x, inner.y), (x, inner.bottom), 1)
        text(surf, fonts.tiny, str(i * 180), (x, inner.bottom + 3), DIM, center=True)
        if i < 4:
            text(surf, fonts.tiny, names[i], (x + inner.w / 8.0, inner.y + 8), (70, 74, 82), center=True)
    for bar in (0, 20, 40, 60, 80):
        y = inner.bottom - inner.h * (bar * 1e5) / pscale
        if y > inner.y:
            pygame.draw.line(surf, (36, 38, 44), (inner.x, y), (inner.right, y), 1)
            text(surf, fonts.tiny, str(bar), (inner.x - 4, y - 6), DIM, right=True)
    n = 360
    ptsp, ptsi, ptse = [], [], []
    for i in range(n):
        x = inner.x + inner.w * i / (n - 1)
        ptsp.append((x, inner.bottom - inner.h * min(1.0, eng.sp[i] / pscale)))
        ptsi.append((x, inner.bottom - inner.h * 0.30 * eng.sli[i] / INT_LIFT))
        ptse.append((x, inner.bottom - inner.h * 0.30 * eng.sle[i] / EXH_LIFT))
    pygame.draw.lines(surf, CYAN, False, ptsi, 1)
    pygame.draw.lines(surf, ORANGE, False, ptse, 1)
    pygame.draw.lines(surf, RED, False, ptsp, 2)
    cur = int(((eng.phi * RAD2DEG) % 720.0) * 0.5) % 360
    x = inner.x + inner.w * cur / (n - 1)
    pygame.draw.line(surf, WHITE, (x, inner.y), (x, inner.bottom), 1)
    text(surf, fonts.tiny, "BAR", (rect.x + 6, rect.y + 24), DIM)


def draw_pv_plot(surf, fonts, rect, eng, pscale):
    plot_frame(surf, fonts, rect, "P-V DIAGRAM")
    inner = pygame.Rect(rect.x + 34, rect.y + 24, rect.w - 44, rect.h - 34)
    pygame.draw.rect(surf, (30, 32, 36), inner, 1)
    vmax = V_CLR + V_DISP
    pts = []
    for i in range(360):
        x = inner.x + inner.w * eng.sv[i] / vmax
        y = inner.bottom - inner.h * min(1.0, eng.sp[i] / pscale)
        pts.append((x, y))
    pts.append(pts[0])
    pygame.draw.lines(surf, GREEN, False, pts, 2)
    x = inner.x + inner.w * eng.V / vmax
    y = inner.bottom - inner.h * min(1.0, eng.p / pscale)
    pygame.draw.circle(surf, WHITE, (int(x), int(y)), 4)
    text(surf, fonts.tiny, "VOLUME ->", (inner.right - 4, inner.bottom - 14), DIM, right=True)


def draw_flow_plot(surf, fonts, rect, eng):
    plot_frame(surf, fonts, rect, "TOTAL EXHAUST FLOW")
    inner = pygame.Rect(rect.x + 8, rect.y + 24, rect.w - 16, rect.h - 34)
    pygame.draw.rect(surf, (30, 32, 36), inner, 1)
    fmax = max(0.02, max(eng.sfl) * 1.1)
    zero_y = inner.bottom - inner.h * 0.15
    pygame.draw.line(surf, (50, 52, 58), (inner.x, zero_y), (inner.right, zero_y), 1)
    pts = []
    for i in range(360):
        x = inner.x + inner.w * i / 359.0
        pts.append((x, zero_y - (inner.h * 0.8) * eng.sfl[i] / fmax))
    pygame.draw.lines(surf, ORANGE, False, pts, 2)
    cur = int(((eng.phi * RAD2DEG) % 720.0) * 0.5) % 360
    x = inner.x + inner.w * cur / 359.0
    pygame.draw.line(surf, WHITE, (x, inner.y), (x, inner.bottom), 1)
    text(surf, fonts.tiny, "%.0f g/s peak" % (fmax * 1000 / 1.1), (inner.right - 4, inner.y + 3), DIM, right=True)


def lamp(surf, fonts, x, y, label, on, col=GREEN):
    pygame.draw.rect(surf, col if on else (45, 30, 30), (x, y + 2, 12, 12))
    text(surf, fonts.small, label, (x + 18, y), WHITE if on else DIM)


def draw_left(surf, fonts, eng, fps, speed_ratio, tscale):
    panel(surf, pygame.Rect(0, 0, LEFT_W, H))
    text(surf, fonts.title, "ENGINE", (14, 8), WHITE)
    text(surf, fonts.title, "SIMULATOR", (14, 38), WHITE)
    text(surf, fonts.small, "SINGLE CYLINDER  4-STROKE", (14, 76), CYAN)
    text(surf, fonts.small, "1 INTAKE VALVE / 1 EXHAUST VALVE", (14, 92), DIM)
    text(surf, fonts.small, "%.0f CC  %d x %d MM  CR %.1f:1" %
         (V_DISP * 1e6, BORE * 1000, STROKE * 1000, CR), (14, 108), DIM)
    pygame.draw.line(surf, LINE, (0, 130), (LEFT_W, 130), 1)

    lamp(surf, fonts, 14, 140, "IGNITION", eng.ign)
    lamp(surf, fonts, 150, 140, "STARTER", eng.starter, YELLOW)
    lamp(surf, fonts, 14, 160, "DYNO HOLD", eng.dyno, CYAN)
    lamp(surf, fonts, 150, 160, "LIMITER", eng.limiter and eng.rpm > 1000, RED)
    pygame.draw.line(surf, LINE, (0, 186), (LEFT_W, 186), 1)

    rows = [
        ("THROTTLE", "%5.1f %%" % (eng.throttle * 100)),
        ("SPARK ADV", "%5.1f DEG BTDC" % eng.spark),
        ("AIR/FUEL", "%5.2f : 1" % eng.afr),
        ("BRAKE LOAD", "%5.0f %%" % (eng.load * 100)),
        ("DYNO RPM", "%5.0f" % eng.dyno_rpm),
        ("TIME SCALE", "%5.2fx" % tscale),
        ("FPS", "%5.1f" % fps),
        ("SIM SPEED", "%5.2f x RT" % speed_ratio),
    ]
    y = 194
    for k, v in rows:
        text(surf, fonts.small, k, (14, y), DIM)
        text(surf, fonts.small, v, (LEFT_W - 14, y), WHITE, right=True)
        y += 19
    # throttle bar
    pygame.draw.rect(surf, (40, 42, 48), (14, y + 2, LEFT_W - 28, 10))
    pygame.draw.rect(surf, ORANGE, (14, y + 2, (LEFT_W - 28) * eng.throttle, 10))
    pygame.draw.line(surf, LINE, (0, y + 22), (LEFT_W, y + 22), 1)

    y += 32
    text(surf, fonts.mid, "CONTROLS", (14, y), CYAN)
    y += 22
    help_lines = [
        "A        ignition on/off",
        "S (hold) starter motor",
        "UP/DOWN  throttle (or wheel)",
        "D        dyno hold on/off",
        "LEFT/RT  dyno target rpm",
        "K / L    brake load -/+",
        "F / V    spark advance +/-",
        "M / N    air-fuel ratio +/-",
        "Z / X    slow-mo / faster",
        "SPACE    reset engine",
        "ESC      quit",
    ]
    for line in help_lines:
        text(surf, fonts.tiny, line, (14, y), WHITE)
        y += 15
    text(surf, fonts.tiny, "TO START: A, hold S, give ~10-15% throttle", (14, y + 6), YELLOW)
    pygame.draw.line(surf, LINE, (0, H - 52), (LEFT_W, H - 52), 1)
    text(surf, fonts.small, "MADE BY", (LEFT_W // 2, H - 40), DIM, center=True)
    text(surf, fonts.big, "MD REHAN", (LEFT_W // 2, H - 20), WHITE, center=True)


def draw_right(surf, fonts, eng, disp, tscale):
    cw, rh = 140, 240
    cells = [
        ("ENGINE SPEED", disp["rpm"], 0, 8000, "%.0f RPM" % disp["rpm"], 0.80, 0.92, 8),
        ("MANIFOLD", disp["map"], 0, 110, "%.0f KPA" % disp["map"], None, None, 11),
        ("TORQUE", disp["tq"], 0, 50, "%.1f NM" % disp["tq"], None, None, 10),
        ("POWER", disp["pw"], 0, 30, "%.1f KW" % disp["pw"], None, None, 6),
        ("IN. AFR", eng.afr, 8, 20, "%.1f" % eng.afr, None, None, 6),
        ("VOLUMETRIC EFF.", disp["ve"], 0, 120, "%.0f %%" % disp["ve"], None, None, 6),
        ("CYL PRESSURE", disp["pc"], 0, 80, "%.1f BAR" % disp["pc"], 0.7, 0.88, 8),
        ("EXHAUST TEMP", disp["ex"], 0, 1100, "%.0f C" % disp["ex"], 0.75, 0.9, 11),
    ]
    for i, (title, val, vmin, vmax, ro, og, rd, tk) in enumerate(cells):
        col, row = i % 3, i // 3
        rect = pygame.Rect(RIGHT_X + col * cw, row * rh, cw, rh)
        panel(surf, rect)
        text(surf, fonts.small, title, (rect.centerx, rect.y + 16), WHITE, center=True)
        draw_gauge(surf, fonts, rect.centerx, rect.y + 120, 52, val, vmin, vmax, ro, og, rd, tk)
    # fuel cell
    rect = pygame.Rect(RIGHT_X + 2 * cw, 2 * rh, cw, rh)
    panel(surf, rect)
    text(surf, fonts.small, "FUEL", (rect.centerx, rect.y + 16), WHITE, center=True)
    lit = eng.fuel_total / FUEL_DENSITY * 1000.0
    text(surf, fonts.big, "%.4f L" % lit, (rect.centerx, rect.y + 70), WHITE, center=True)
    text(surf, fonts.small, "%.1f ML" % (lit * 1000), (rect.centerx, rect.y + 100), DIM, center=True)
    text(surf, fonts.small, "%.2f L/H" % disp["lph"], (rect.centerx, rect.y + 130), CYAN, center=True)
    bsfc = (disp["lph"] * FUEL_DENSITY * 1000.0 / 1000.0 / disp["pw"]) if disp["pw"] > 0.3 else 0.0
    text(surf, fonts.small, "BSFC %.0f G/KWH" % bsfc if bsfc else "BSFC --", (rect.centerx, rect.y + 160),
         DIM, center=True)
    if eng.misfire:
        text(surf, fonts.small, "MISFIRE (AFR)", (rect.centerx, rect.y + 200), RED, center=True)


def main():
    pygame.init()
    flags = pygame.SCALED | pygame.RESIZABLE
    try:
        screen = pygame.display.set_mode((W, H), flags)
    except pygame.error:
        screen = pygame.display.set_mode((W, H))
    pygame.display.set_caption("Single Cylinder Engine Simulator - Made by MD Rehan")
    clock = pygame.time.Clock()
    fonts = Fonts()
    eng = Engine()

    scales = [0.005, 0.02, 0.05, 0.1, 0.25, 0.5, 1.0]
    si = len(scales) - 1
    disp = dict(rpm=0.0, map=101.0, tq=0.0, pw=0.0, ve=0.0, pc=1.0, ex=300.0, lph=0.0)
    pscale = 2.0e6
    speed_ratio = 1.0
    running = True

    while running:
        dtf = min(clock.tick(60) / 1000.0, 0.05)
        for ev in pygame.event.get():
            if ev.type == pygame.QUIT:
                running = False
            elif ev.type == pygame.KEYDOWN:
                k = ev.key
                if k == pygame.K_ESCAPE:
                    running = False
                elif k == pygame.K_a:
                    eng.ign = not eng.ign
                elif k == pygame.K_d:
                    eng.dyno = not eng.dyno
                    if eng.dyno:
                        eng.dint = eng.tq_f / DYNO_KI
                elif k == pygame.K_SPACE:
                    eng.reset()
                elif k == pygame.K_z:
                    si = max(0, si - 1)
                elif k == pygame.K_x:
                    si = min(len(scales) - 1, si + 1)
                elif k == pygame.K_k:
                    eng.load = max(0.0, eng.load - 0.1)
                elif k == pygame.K_l:
                    eng.load = min(1.0, eng.load + 0.1)
                elif k == pygame.K_f:
                    eng.spark = min(50.0, eng.spark + 1.0)
                elif k == pygame.K_v:
                    eng.spark = max(-5.0, eng.spark - 1.0)
                elif k == pygame.K_m:
                    eng.afr = min(20.0, eng.afr + 0.1)
                elif k == pygame.K_n:
                    eng.afr = max(8.0, eng.afr - 0.1)
            elif ev.type == pygame.MOUSEWHEEL:
                eng.throttle = max(0.0, min(1.0, eng.throttle + ev.y * 0.02))

        keys = pygame.key.get_pressed()
        eng.starter = bool(keys[pygame.K_s])
        if keys[pygame.K_UP]:
            eng.throttle = min(1.0, eng.throttle + 0.5 * dtf)
        if keys[pygame.K_DOWN]:
            eng.throttle = max(0.0, eng.throttle - 0.5 * dtf)
        if keys[pygame.K_RIGHT]:
            eng.dyno_rpm = min(8000.0, eng.dyno_rpm + 1500.0 * dtf)
        if keys[pygame.K_LEFT]:
            eng.dyno_rpm = max(0.0, eng.dyno_rpm - 1500.0 * dtf)

        tscale = scales[si]
        want = dtf * tscale
        done = eng.advance(want, 260)
        speed_ratio += ((done / dtf if dtf > 0 else 0.0) - speed_ratio) * 0.1

        # smoothed readouts
        k = min(1.0, dtf * 10.0)
        pw = eng.tq_f * eng.w / 1000.0
        targets = dict(rpm=eng.rpm, map=eng.pman / 1000.0, tq=max(0.0, eng.tq_f),
                       pw=max(0.0, pw), ve=eng.ve * 100.0, pc=eng.p / 1e5,
                       ex=eng.tex - 273.15, lph=eng.fuel_lph)
        for key, v in targets.items():
            disp[key] += (v - disp[key]) * (1.0 if key == "pc" else k)
        pm = max(eng.sp) * 1.15
        pm = max(pm, 5.0e5)
        pscale += (pm - pscale) * (0.5 if pm > pscale else 0.05)

        # ---- draw
        screen.fill(BG)
        draw_left(screen, fonts, eng, clock.get_fps(), speed_ratio, tscale)
        draw_engine(screen, fonts, eng)
        draw_pressure_plot(screen, fonts, pygame.Rect(LEFT_W, 400, RIGHT_X - LEFT_W, 160), eng, pscale)
        draw_pv_plot(screen, fonts, pygame.Rect(LEFT_W, 560, 280, 160), eng, pscale)
        draw_flow_plot(screen, fonts, pygame.Rect(LEFT_W + 280, 560, RIGHT_X - LEFT_W - 280, 160), eng)
        draw_right(screen, fonts, eng, disp, tscale)
        pygame.display.flip()

    pygame.quit()
    sys.exit(0)


if __name__ == "__main__":
    main()
