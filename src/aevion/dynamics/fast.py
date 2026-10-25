"""Fused Numba kernel for the multirotor step (CPU fast path).

This re-implements :meth:`Multirotor._deriv` + the integrator + contact/collision/battery in a
single compiled, multi-threaded loop over vehicles. The generic NumPy/torch implementation in
``multirotor.py`` is the reference; ``tests/test_fast_parity.py`` asserts the two agree to
round-off. The kernel is used automatically for the NumPy backend when Numba is installed.
"""

from __future__ import annotations

import math

import numpy as np

from ..core.constants import G0, LAPSE, P0, R_AIR, R_EARTH, R_VAPOUR, RHO0, T0

try:
    from numba import njit, prange

    HAVE_NUMBA = True
except Exception:  # pragma: no cover
    HAVE_NUMBA = False

if HAVE_NUMBA:
    from ..sensors.raycast import _terrain_height

    @njit(cache=True)
    def _density_ratio(z, atm_on, base, dT, p_sl, humidity):
        if atm_on == 0:
            return 1.0
        h = z + base
        t_std = T0 - LAPSE * h
        p = P0 * (t_std / T0) ** (G0 / (R_AIR * LAPSE))
        if p_sl != P0:
            p = p * (p_sl / P0)
        t = t_std + dT
        if humidity <= 0.0:
            return p / (R_AIR * t) / RHO0
        tc = t - 273.15  # moist air: vapour pressure from the Magnus formula, capped below the total pressure
        e = min(humidity * 610.94 * math.exp(17.625 * tc / (tc + 243.04)), 0.99 * p)
        return ((p - e) / (R_AIR * t) + e / (R_VAPOUR * t)) / RHO0

    @njit(cache=True)
    def _gravity(z, kind, g0, g_lat, alt_ref):
        if kind == 0:
            return g0
        h = z + alt_ref
        if kind == 1:
            return g0 * (R_EARTH / (R_EARTH + h)) ** 2
        return g_lat - 3.086e-6 * h

    @njit(cache=True)
    def _ground(x, y, hm, has_hm, ox, oy, res):
        if has_hm:
            return _terrain_height(hm, ox, oy, res, x, y)
        return 0.0

    @njit(cache=True)
    def _rot(q, R):
        w, x, y, z = q[0], q[1], q[2], q[3]
        R[0, 0] = 1 - 2 * (y * y + z * z)
        R[0, 1] = 2 * (x * y - w * z)
        R[0, 2] = 2 * (x * z + w * y)
        R[1, 0] = 2 * (x * y + w * z)
        R[1, 1] = 1 - 2 * (x * x + z * z)
        R[1, 2] = 2 * (y * z - w * x)
        R[2, 0] = 2 * (x * z - w * y)
        R[2, 1] = 2 * (y * z + w * x)
        R[2, 2] = 1 - 2 * (x * x + y * y)

    @njit(cache=True)
    def _f(y, dy, cmd, wnd, i, M, P, aux, want_aux, R):
        """Time derivative of the state vector y = [p(3), v(3), q(4), w(3), rotor(M)]."""
        (mass, J, Jinv, kT, kQ, tau_up, tau_dn, health, dlin, dquad, rdrag, adamp, com, rpos, spin, sc, hm, ext) = P
        Jr, Rp, inflow, ge_flag = sc[0], sc[1], sc[2], sc[3]
        gkind, g0, g_lat, alt_ref = int(sc[4]), sc[5], sc[6], sc[7]
        atm_on, atm_base, atm_dT = int(sc[8]), sc[9], sc[10]
        has_hm, hm_ox, hm_oy, hm_res = int(sc[11]), sc[12], sc[13], sc[14]
        bat_eff, p_av = sc[15], sc[16]
        p_sl, humidity = sc[17], sc[18]

        qn = y[6:10]
        _rot(qn, R)  # R is per-vehicle scratch handed in by the caller: no allocation inside the parallel loop
        z = y[2]
        g = _gravity(z, gkind, g0, g_lat, alt_ref)
        rho = _density_ratio(z, atm_on, atm_base, atm_dT, p_sl, humidity)
        ox, oy, oz = y[10], y[11], y[12]

        vr0, vr1, vr2 = y[3] - wnd[0], y[4] - wnd[1], y[5] - wnd[2]
        vb0 = R[0, 0] * vr0 + R[1, 0] * vr1 + R[2, 0] * vr2
        vb1 = R[0, 1] * vr0 + R[1, 1] * vr1 + R[2, 1] * vr2
        vb2 = R[0, 2] * vr0 + R[1, 2] * vr1 + R[2, 2] * vr2

        f_ge = 1.0
        if ge_flag > 0.5:
            agl = z - _ground(y[0], y[1], hm, has_hm, hm_ox, hm_oy, hm_res)
            ratio = Rp / (4.0 * max(agl, 0.5 * Rp))
            f_ge = min(max(1.0 / (1.0 - ratio * ratio + 1e-6), 1.0), 1.5)

        t_scale = ext[i, 6]  # thrust multiplier from proximity effects (wall, ceiling, ...)
        Tsum = 0.0
        tx = 0.0
        ty = 0.0
        tz = 0.0
        wsum = 0.0
        hz = 0.0
        pm = 0.0
        for m in range(M):
            w = y[13 + m]
            f_in = 1.0
            if inflow != 0.0:
                f_in = min(max(1.0 - inflow * vb2 / (max(w, 50.0) * Rp), 0.0), 1.5)
            eff = rho * health[i, m]
            Tm = kT[i, m] * eff * w * w * f_in * f_ge * t_scale
            Qm = kQ[i, m] * eff * w * w
            if want_aux:
                aux[3 + m] = Tm
            tau = tau_up[i, m] if cmd[m] > w else tau_dn[i, m]
            wd = (cmd[m] - w) / tau
            dy[13 + m] = wd
            Tsum += Tm
            tx += (rpos[m, 1] - com[i, 1]) * Tm
            ty -= (rpos[m, 0] - com[i, 0]) * Tm
            tz -= spin[m] * (Qm + Jr * wd)
            wsum += w
            hz += spin[m] * Jr * w
            pm += Qm * w

        speed = math.sqrt(vb0 * vb0 + vb1 * vb1 + vb2 * vb2)
        Fb0 = -(dlin[i, 0] + dquad[i, 0] * speed) * vb0 * rho - rdrag[i] * wsum * vb0
        Fb1 = -(dlin[i, 1] + dquad[i, 1] * speed) * vb1 * rho - rdrag[i] * wsum * vb1
        Fb2 = Tsum - (dlin[i, 2] + dquad[i, 2] * speed) * vb2 * rho
        m_i = mass[i]
        fb0, fb1, fb2 = Fb0 / m_i, Fb1 / m_i, Fb2 / m_i
        a0 = R[0, 0] * fb0 + R[0, 1] * fb1 + R[0, 2] * fb2
        a1 = R[1, 0] * fb0 + R[1, 1] * fb1 + R[1, 2] * fb2
        a2 = R[2, 0] * fb0 + R[2, 1] * fb1 + R[2, 2] * fb2 - g
        a0 += ext[i, 0] / m_i  # external force (world frame), e.g. the tension of a tether
        a1 += ext[i, 1] / m_i
        a2 += ext[i, 2] / m_i
        dy[0], dy[1], dy[2] = y[3], y[4], y[5]
        dy[3], dy[4], dy[5] = a0, a1, a2

        # attitude kinematics q_dot = 1/2 q ⊗ [0, w]
        qw, qx, qy, qz = y[6], y[7], y[8], y[9]
        dy[6] = 0.5 * (-qx * ox - qy * oy - qz * oz)
        dy[7] = 0.5 * (qw * ox + qy * oz - qz * oy)
        dy[8] = 0.5 * (qw * oy - qx * oz + qz * ox)
        dy[9] = 0.5 * (qw * oz + qx * oy - qy * ox)

        # rotational dynamics
        t0 = tx + oy * (-hz) - adamp[i, 0] * ox + ext[i, 3]  # + external torque (body frame)
        t1 = ty + ox * hz - adamp[i, 1] * oy + ext[i, 4]
        t2 = tz - adamp[i, 2] * oz + ext[i, 5]
        Jw0 = J[i, 0, 0] * ox + J[i, 0, 1] * oy + J[i, 0, 2] * oz
        Jw1 = J[i, 1, 0] * ox + J[i, 1, 1] * oy + J[i, 1, 2] * oz
        Jw2 = J[i, 2, 0] * ox + J[i, 2, 1] * oy + J[i, 2, 2] * oz
        r0 = t0 - (oy * Jw2 - oz * Jw1)
        r1 = t1 - (oz * Jw0 - ox * Jw2)
        r2 = t2 - (ox * Jw1 - oy * Jw0)
        dy[10] = Jinv[i, 0, 0] * r0 + Jinv[i, 0, 1] * r1 + Jinv[i, 0, 2] * r2
        dy[11] = Jinv[i, 1, 0] * r0 + Jinv[i, 1, 1] * r1 + Jinv[i, 1, 2] * r2
        dy[12] = Jinv[i, 2, 0] * r0 + Jinv[i, 2, 1] * r1 + Jinv[i, 2, 2] * r2

        if want_aux:
            aux[0], aux[1], aux[2] = fb0, fb1, fb2
            aux[3 + M] = pm / bat_eff + p_av
            aux[4 + M], aux[5 + M], aux[6 + M] = a0, a1, a2

    @njit(cache=True)
    def _norm_q(y):
        n = math.sqrt(y[6] * y[6] + y[7] * y[7] + y[8] * y[8] + y[9] * y[9])
        n = max(n, 1e-12)
        y[6] /= n
        y[7] /= n
        y[8] /= n
        y[9] /= n

    @njit(parallel=True, cache=True)
    def step_kernel(
        pos,
        vel,
        quat,
        omega,
        rotor,
        soc,
        energy,
        cmd_in,
        wind,
        w_hi_scale,
        mass,
        J,
        Jinv,
        kT,
        kQ,
        tau_up,
        tau_dn,
        health,
        dlin,
        dquad,
        rdrag,
        adamp,
        com,
        rpos,
        spin,
        hm,
        ext,
        sc,
        w_min,
        w_max,
        dt,
        method,
        cap_wh,
        restitution,
        mu,
        ang_damp_contact,
        ground_offset,
        radius,
        boxes,
        spheres,
        cyls,
        has_obst,
        o_pos,
        o_vel,
        o_quat,
        o_omega,
        o_rotor,
        o_soc,
        o_energy,
        o_acc,
        o_fb,
        o_thrust,
        o_power,
        o_contact,
        o_collided,
        o_impact,
    ):
        N = pos.shape[0]
        M = rotor.shape[1]
        NY = 13 + M
        P = (mass, J, Jinv, kT, kQ, tau_up, tau_dn, health, dlin, dquad, rdrag, adamp, com, rpos, spin, sc, hm, ext)
        has_hm = int(sc[11])
        hm_ox, hm_oy, hm_res = sc[12], sc[13], sc[14]
        g0 = sc[5]
        # All scratch memory is allocated here, once, as one row per vehicle: allocating inside a prange body proved to be a source
        # of rare, load-dependent NaNs (a race in numba's parfor allocation handling), so the parallel loop only reads and writes rows.
        Y = np.zeros((N, NY))
        K1 = np.zeros((N, NY))
        K2 = np.zeros((N, NY))
        K3 = np.zeros((N, NY))
        K4 = np.zeros((N, NY))
        YT = np.zeros((N, NY))
        AUX = np.zeros((N, 7 + M))
        CMD = np.zeros((N, M))
        RM = np.zeros((N, 3, 3))
        for i in prange(N):  # type: ignore[attr-defined]
            y = Y[i]
            k1 = K1[i]
            k2 = K2[i]
            k3 = K3[i]
            k4 = K4[i]
            yt = YT[i]
            aux = AUX[i]
            cmd = CMD[i]
            Rm = RM[i]
            wnd = wind[i]
            for k in range(3):
                y[k] = pos[i, k]
                y[3 + k] = vel[i, k]
                y[10 + k] = omega[i, k]
            for k in range(4):
                y[6 + k] = quat[i, k]
            for m in range(M):
                y[13 + m] = rotor[i, m]
                c = cmd_in[i, m]
                hi = w_max[i, m] * w_hi_scale[i]
                c = min(max(c, w_min), hi)
                if health[i, m] <= 0.0:
                    c = 0.0
                cmd[m] = c

            _f(y, k1, cmd, wnd, i, M, P, aux, True, Rm)
            if method == 3:  # rk4
                for k in range(NY):
                    yt[k] = y[k] + 0.5 * dt * k1[k]
                _norm_q(yt)
                _f(yt, k2, cmd, wnd, i, M, P, aux, False, Rm)
                for k in range(NY):
                    yt[k] = y[k] + 0.5 * dt * k2[k]
                _norm_q(yt)
                _f(yt, k3, cmd, wnd, i, M, P, aux, False, Rm)
                for k in range(NY):
                    yt[k] = y[k] + dt * k3[k]
                _norm_q(yt)
                _f(yt, k4, cmd, wnd, i, M, P, aux, False, Rm)
                for k in range(NY):
                    y[k] = y[k] + (dt / 6.0) * (k1[k] + 2 * k2[k] + 2 * k3[k] + k4[k])
            elif method == 2:  # midpoint
                for k in range(NY):
                    yt[k] = y[k] + 0.5 * dt * k1[k]
                _norm_q(yt)
                _f(yt, k2, cmd, wnd, i, M, P, aux, False, Rm)
                for k in range(NY):
                    y[k] = y[k] + dt * k2[k]
            elif method == 1:  # semi-implicit
                for k in range(3):
                    y[3 + k] = y[3 + k] + dt * k1[3 + k]
                    y[k] = y[k] + dt * y[3 + k]
                for k in range(6, NY):
                    y[k] = y[k] + dt * k1[k]
            else:  # euler
                for k in range(NY):
                    y[k] = y[k] + dt * k1[k]
            _norm_q(y)

            # ---- ground contact
            gz = _ground(y[0], y[1], hm, has_hm, hm_ox, hm_oy, hm_res) + ground_offset
            below = y[2] <= gz
            impact = 0.0
            if below:
                if y[5] < 0.0:
                    impact = -y[5]
                    y[5] = -restitution * y[5]
                y[2] = gz
                sxy = math.sqrt(y[3] * y[3] + y[4] * y[4] + 1e-18)
                fr = max(1.0 - mu * g0 * dt / sxy, 0.0)
                y[3] *= fr
                y[4] *= fr
                d = math.exp(-ang_damp_contact * dt)
                y[10] *= d
                y[11] *= d
                y[12] *= d

            # ---- obstacle collisions (sphere vs boxes / spheres / vertical cylinders)
            collided = False
            if has_obst:
                px, py, pz = 0.0, 0.0, 0.0
                for b in range(boxes.shape[0]):
                    cx = min(max(y[0], boxes[b, 0]), boxes[b, 3])
                    cy = min(max(y[1], boxes[b, 1]), boxes[b, 4])
                    cz = min(max(y[2], boxes[b, 2]), boxes[b, 5])
                    dx, dy_, dz = y[0] - cx, y[1] - cy, y[2] - cz
                    dist = math.sqrt(dx * dx + dy_ * dy_ + dz * dz + 1e-18)
                    inside = (
                        y[0] > boxes[b, 0]
                        and y[0] < boxes[b, 3]
                        and y[1] > boxes[b, 1]
                        and y[1] < boxes[b, 4]
                        and y[2] > boxes[b, 2]
                        and y[2] < boxes[b, 5]
                    )
                    if inside:
                        best = 1e30
                        nx, ny_, nz = 0.0, 0.0, 0.0
                        for ax in range(3):
                            lo = y[ax] - boxes[b, ax]
                            hi2 = boxes[b, 3 + ax] - y[ax]
                            if lo <= best + 1e-12:
                                best = lo
                                nx, ny_, nz = 0.0, 0.0, 0.0
                                if ax == 0:
                                    nx = -1.0
                                elif ax == 1:
                                    ny_ = -1.0
                                else:
                                    nz = -1.0
                            if hi2 < best - 1e-12:
                                best = hi2
                                nx, ny_, nz = 0.0, 0.0, 0.0
                                if ax == 0:
                                    nx = 1.0
                                elif ax == 1:
                                    ny_ = 1.0
                                else:
                                    nz = 1.0
                        dep = best + radius
                        px += nx * dep
                        py += ny_ * dep
                        pz += nz * dep
                    elif dist < radius:
                        dep = radius - dist
                        px += dx / dist * dep
                        py += dy_ / dist * dep
                        pz += dz / dist * dep
                for s in range(spheres.shape[0]):
                    dx, dy_, dz = y[0] - spheres[s, 0], y[1] - spheres[s, 1], y[2] - spheres[s, 2]
                    dist = math.sqrt(dx * dx + dy_ * dy_ + dz * dz + 1e-18)
                    dep = spheres[s, 3] + radius - dist
                    if dep > 0.0:
                        px += dx / dist * dep
                        py += dy_ / dist * dep
                        pz += dz / dist * dep
                for c in range(cyls.shape[0]):
                    zc = min(max(y[2], cyls[c, 2]), cyls[c, 3])
                    dx, dy_, dz = y[0] - cyls[c, 0], y[1] - cyls[c, 1], y[2] - zc
                    dist = math.sqrt(dx * dx + dy_ * dy_ + dz * dz + 1e-18)
                    dep = cyls[c, 4] + radius - dist
                    if dep > 0.0:
                        px += dx / dist * dep
                        py += dy_ / dist * dep
                        pz += dz / dist * dep
                mag = math.sqrt(px * px + py * py + pz * pz + 1e-18)
                if mag > 1e-6:
                    collided = True
                    nx, ny_, nz = px / mag, py / mag, pz / mag
                    y[0] += nx * mag
                    y[1] += ny_ * mag
                    y[2] += nz * mag
                    vn = y[3] * nx + y[4] * ny_ + y[5] * nz
                    if vn < 0.0:
                        y[3] -= (1.0 + restitution) * vn * nx
                        y[4] -= (1.0 + restitution) * vn * ny_
                        y[5] -= (1.0 + restitution) * vn * nz

            # ---- write back
            for k in range(3):
                o_pos[i, k] = y[k]
                o_vel[i, k] = y[3 + k]
                o_omega[i, k] = y[10 + k]
                o_fb[i, k] = aux[k]
                o_acc[i, k] = aux[4 + M + k]
            for k in range(4):
                o_quat[i, k] = y[6 + k]
            for m in range(M):
                o_rotor[i, m] = y[13 + m]
                o_thrust[i, m] = aux[3 + m]
            pw = aux[3 + M]
            o_power[i] = pw
            e_step = pw * dt
            o_energy[i] = energy[i] + e_step
            o_soc[i] = max(soc[i] - e_step / (cap_wh * 3600.0), 0.0)
            o_contact[i] = y[2] <= _ground(y[0], y[1], hm, has_hm, hm_ox, hm_oy, hm_res) + ground_offset + 1e-3
            o_collided[i] = collided
            o_impact[i] = impact


METHOD_IDS = {"euler": 0, "semi_implicit": 1, "midpoint": 2, "rk4": 3}
