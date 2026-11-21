"""Procedural macarena simulators for the VLMA paper.

Every number in the paper comes from the files this script writes to figures/data/.
Nothing here is a hardware measurement: the dancer is procedural, the camera is a
renderer, the events are emulated, the pose is tracked on rendered frames and the
signal strength comes from a path-loss model.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import ndimage

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
EPISODE = DATA / "episode_000"
SEED = 20261121

FPS = 30
BPM = 103.0
BEAT = 60.0 / BPM
CYCLES = 2
COUNTS = 16
DURATION = CYCLES * COUNTS * BEAT
NF = int(round(DURATION * FPS))
W, H = 192, 144
F_PX = 180.0
CAM = np.array([0.0, 1.1, 3.4])
CX, CY = W / 2, H / 2
WALL_Z = -1.3

L_UP, L_FORE = 0.30, 0.27
SH_HALF = 0.19
SH_Y = 1.45

# ---------------------------------------------------------------- choreography
# body frame: x = dancer's left, y = up, z = forward (towards the camera)
DOWN = np.array([0.0, -1.0, 0.0])


def unit(v):
    v = np.asarray(v, float)
    return v / np.linalg.norm(v)


def mirror(v):
    return np.array([-v[0], v[1], v[2]])


# per arm: (elbow direction, hand target or None, pronation 0=palm down 1=palm up)
def arm_rest(side):
    return (unit([0.12 * side, -1, 0.05]), None, 0.0)


R_SH = np.array([-SH_HALF, SH_Y, 0.0])
L_SH = np.array([SH_HALF, SH_Y, 0.0])

MOVES = [
    ("R arm forward, palm down", "R", (unit([0, 0, 1]), None, 0.0)),
    ("L arm forward, palm down", "L", (unit([0, 0, 1]), None, 0.0)),
    ("R palm up", "R", (unit([0, 0, 1]), None, 1.0)),
    ("L palm up", "L", (unit([0, 0, 1]), None, 1.0)),
    ("R hand to L shoulder", "R", (unit([0.25, -0.35, 0.9]), np.array([0.17, 1.42, 0.10]), 0.0)),
    ("L hand to R shoulder", "L", (unit([-0.25, -0.30, 0.92]), np.array([-0.17, 1.40, 0.13]), 0.0)),
    ("R hand behind head", "R", (unit([-0.75, 0.55, 0.15]), np.array([-0.03, 1.68, -0.12]), 0.0)),
    ("L hand behind head", "L", (unit([0.75, 0.55, 0.15]), np.array([0.03, 1.70, -0.12]), 0.0)),
    ("R hand to L front hip", "R", (unit([0.05, -0.8, 0.55]), np.array([0.13, 0.98, 0.15]), 0.0)),
    ("L hand to R front hip", "L", (unit([-0.05, -0.8, 0.55]), np.array([-0.13, 0.98, 0.16]), 0.0)),
    ("R hand to R back pocket", "R", (unit([-0.35, -0.85, -0.4]), np.array([-0.12, 0.92, -0.15]), 0.0)),
    ("L hand to L back pocket", "L", (unit([0.35, -0.85, -0.4]), np.array([0.12, 0.92, -0.15]), 0.0)),
    ("Hip sway", None, None),
    ("Hip sway", None, None),
    ("Hip sway", None, None),
    ("Jump, quarter turn", None, None),
]


def state_table():
    """Cumulative arm configuration after each count (the FSM of Appendix C)."""
    R = arm_rest(-1)
    L = arm_rest(+1)
    states = []
    for _, side, cfg in MOVES:
        if side == "R":
            R = cfg
        elif side == "L":
            L = cfg
        states.append((R, L))
    return states


def arm_points(sh, cfg):
    e_dir, target, pron = cfg
    elbow = sh + L_UP * e_dir
    if target is None:
        w_dir = e_dir
    else:
        w_dir = unit(target - elbow)
    hand = elbow + L_FORE * w_dir
    return elbow, hand, pron


def slerp(a, b, s):
    a, b = unit(a), unit(b)
    d = np.clip(np.dot(a, b), -1, 1)
    om = np.arccos(d)
    if om < 1e-6:
        return a
    return (np.sin((1 - s) * om) * a + np.sin(s * om) * b) / np.sin(om)


def smoothstep(x):
    x = np.clip(x, 0, 1)
    return x * x * (3 - 2 * x)


JOINTS = ["head", "neck", "r_sh", "l_sh", "r_el", "l_el", "r_wr", "l_wr",
          "pelvis", "r_hip", "l_hip", "r_kn", "l_kn", "r_an", "l_an"]
JI = {n: i for i, n in enumerate(JOINTS)}
SEGMENTS = [  # (a, b, radius m, intensity)
    ("neck", "pelvis", 0.11, 0.70),
    ("r_sh", "l_sh", 0.05, 0.70),
    ("r_hip", "l_hip", 0.07, 0.50),
    ("r_sh", "r_el", 0.045, 0.66),
    ("r_el", "r_wr", 0.040, 0.66),
    ("l_sh", "l_el", 0.045, 0.66),
    ("l_el", "l_wr", 0.040, 0.66),
    ("r_hip", "r_kn", 0.07, 0.48),
    ("r_kn", "r_an", 0.055, 0.48),
    ("l_hip", "l_kn", 0.07, 0.48),
    ("l_kn", "l_an", 0.055, 0.48),
]
DISCS = [("head", 0.11, 0.86), ("r_wr", 0.062, 0.97), ("l_wr", 0.062, 0.97)]


def choreography():
    """Returns world joints (NF,15,3), heading (NF,), count index (NF,), pronation (NF,2)."""
    states = state_table()
    t = np.arange(NF) / FPS
    beat_pos = t / BEAT
    joints = np.zeros((NF, len(JOINTS), 3))
    heading = np.zeros(NF)
    count = np.zeros(NF, int)
    pron = np.zeros((NF, 2))
    rest = (arm_rest(-1), arm_rest(+1))
    for i, b in enumerate(beat_pos):
        k_abs = int(np.floor(b))
        cyc, k = divmod(k_abs, COUNTS)
        ph = b - k_abs
        s = smoothstep(ph / 0.45)
        prev = states[k - 1] if k > 0 else (states[COUNTS - 1] if cyc > 0 else rest)
        cur = states[k]
        if k == 0 and cyc > 0:
            prev = states[COUNTS - 1]
        arms = []
        for a in range(2):
            sh = R_SH if a == 0 else L_SH
            e0, h0, p0 = arm_points(sh, prev[a])
            e1, h1, p1 = arm_points(sh, cur[a])
            # interpolate directions so limb lengths stay fixed
            ed = slerp(e0 - sh, e1 - sh, s)
            el = sh + L_UP * ed
            hd = slerp(h0 - e0, h1 - e1, s)
            hand = el + L_FORE * hd
            arms.append((el, hand))
            pron[i, a] = (1 - s) * p0 + s * p1
        # body: bounce on each beat, sway on 13-15, jump and quarter turn on 16
        bob = -0.025 * np.sin(np.pi * ph) ** 2
        sway = 0.0
        if k in (12, 13, 14):
            sway = 0.08 * np.sin(np.pi * (k - 12 + ph))
        jump = 0.0
        yaw_extra = 0.0
        if k == 15:
            jump = 0.09 * np.sin(np.pi * np.clip((ph - 0.1) / 0.6, 0, 1))
            yaw_extra = -0.5 * np.pi * smoothstep((ph - 0.1) / 0.6)
        heading[i] = -0.5 * np.pi * cyc + yaw_extra
        body = {
            "head": np.array([0, 1.64, 0.0]),
            "neck": np.array([0, 1.50, 0.0]),
            "r_sh": R_SH, "l_sh": L_SH,
            "r_el": arms[0][0], "l_el": arms[1][0],
            "r_wr": arms[0][1], "l_wr": arms[1][1],
            "pelvis": np.array([0, 0.98, 0.0]),
            "r_hip": np.array([-0.10, 0.95, 0.0]), "l_hip": np.array([0.10, 0.95, 0.0]),
            "r_kn": np.array([-0.11, 0.52, 0.03]), "l_kn": np.array([0.11, 0.52, 0.03]),
            "r_an": np.array([-0.12, 0.08, 0.0]), "l_an": np.array([0.12, 0.08, 0.0]),
        }
        c, sn = np.cos(heading[i]), np.sin(heading[i])
        Ry = np.array([[c, 0, sn], [0, 1, 0], [-sn, 0, c]])
        for name, p in body.items():
            q = p.copy()
            lift = jump + bob * (0 if name.endswith("an") else 1)
            if name in ("r_kn", "l_kn"):
                lift = jump + 0.5 * bob
            q = Ry @ q
            q[0] += sway * (0.3 if name.endswith("an") else 1.0)
            q[1] += lift
            joints[i, JI[name]] = q
        count[i] = k_abs
    return joints, heading, count, pron


# ---------------------------------------------------------------- renderer
def project(P):
    zc = CAM[2] - P[..., 2]
    u = CX + F_PX * (P[..., 0] - CAM[0]) / zc
    v = CY - F_PX * (P[..., 1] - CAM[1]) / zc
    return np.stack([u, v], -1), zc


def background(rng):
    yy, xx = np.mgrid[0:H, 0:W].astype(float)
    zc = CAM[2] - WALL_Z
    floor_v = CY - F_PX * (0.0 - CAM[1]) / zc
    wall_y = CAM[1] - (yy - CY) * zc / F_PX
    log_h = 0.24
    frac = (wall_y / log_h) % 1.0
    img = 0.22 + 0.14 * np.sin(np.pi * frac) ** 0.7
    grain = ndimage.gaussian_filter(rng.normal(0, 1, (H, W)), (0.6, 4.0))
    img += 0.05 * grain / grain.std()
    seam = frac < 0.07
    img[seam] = 0.12
    floor = yy > floor_v
    img[floor] = 0.40 + 0.03 * np.sin(xx[floor] * 0.9)
    img *= 1.0 - 0.25 * (yy / H)
    depth = np.full((H, W), zc)
    depth[floor] = np.clip(F_PX * CAM[1] / np.maximum(yy[floor] - CY, 1e-3), 1.0, zc)
    return np.clip(img, 0.02, 1.0), depth


YY, XX = np.mgrid[0:H, 0:W].astype(np.float32)


def seg_alpha(a, b, r):
    ab = b - a
    L2 = max(float(ab @ ab), 1e-9)
    t = np.clip(((XX - a[0]) * ab[0] + (YY - a[1]) * ab[1]) / L2, 0, 1)
    dx = XX - (a[0] + t * ab[0])
    dy = YY - (a[1] + t * ab[1])
    d = np.sqrt(dx * dx + dy * dy)
    return np.clip(r + 0.5 - d, 0, 1)


def render(joints, bg, bg_depth):
    """Renders (NF,H,W) linear radiance, depth maps and a hand-visibility mask."""
    frames = np.empty((NF, H, W), np.float32)
    depth = np.empty((NF, H, W), np.float32)
    uv, zc = project(joints)
    visible = np.ones((NF, 2), bool)
    for i in range(NF):
        img = bg.copy()
        dep = bg_depth.copy()
        prims = []
        for a, b, r, c in SEGMENTS:
            ia, ib = JI[a], JI[b]
            z = 0.5 * (zc[i, ia] + zc[i, ib])
            prims.append((z, "s", ia, ib, F_PX * r / z, c))
        for n, r, c in DISCS:
            j = JI[n]
            prims.append((zc[i, j] - 0.02, "d", j, j, F_PX * r / zc[i, j], c))
        prims.sort(key=lambda p: -p[0])
        hand_alpha = {}
        for z, kind, ia, ib, rp, c in prims:
            al = seg_alpha(uv[i, ia], uv[i, ib], rp)
            shade = c * (0.92 + 0.08 * np.clip((YY - uv[i, ia][1]) / 40.0, -1, 1))
            img = img * (1 - al) + shade * al
            dep = np.where(al > 0.5, z, dep)
            if kind == "d" and ia in (JI["r_wr"], JI["l_wr"]):
                hand_alpha[ia] = al
            else:
                for j, ha in hand_alpha.items():
                    hand_alpha[j] = ha * (1 - al)
        for k, j in enumerate((JI["r_wr"], JI["l_wr"])):
            visible[i, k] = hand_alpha[j].sum() > 6.0
        frames[i] = img
        depth[i] = dep
    return frames, depth, uv, visible


# ---------------------------------------------------------------- sensor and veil
def steam_field(rng, n=6):
    k = rng.normal(0, 1, (n, 2)) * np.array([0.05, 0.08])
    ph = rng.uniform(0, 2 * np.pi, n)
    speed = rng.uniform(15, 30, n)

    def field(i):
        t = i / FPS
        acc = np.zeros((H, W), np.float32)
        for m in range(n):
            acc += np.sin(k[m, 0] * XX + k[m, 1] * (YY + speed[m] * t) + ph[m])
        return 0.5 + 0.5 * acc / n * 2.0

    return field


def apply_veil(frames, depth, tau_dancer, rng, airlight=0.80):
    """I' = T I + (1 - T) A with T = exp(-beta d), heterogeneous beta (Eq. 7 to 9)."""
    if tau_dancer <= 0:
        return frames.copy(), np.ones_like(frames)
    beta0 = tau_dancer / (CAM[2] - 0.0)
    field = steam_field(rng)
    out = np.empty_like(frames)
    T = np.empty_like(frames)
    for i in range(frames.shape[0]):
        beta = beta0 * np.clip(0.4 + 1.2 * field(i), 0.2, 2.0)
        T[i] = np.exp(-beta * depth[i])
        out[i] = T[i] * frames[i] + (1 - T[i]) * airlight
    return out, T


def sensor(frames, rng, read=0.008, gain=0.0008):
    noisy = frames + np.sqrt(read ** 2 + gain * np.clip(frames, 0, 1)) * rng.normal(0, 1, frames.shape)
    return (np.clip(np.round(np.clip(noisy, 0, 1) * 255), 0, 255) / 255.0).astype(np.float32)


def snow(frames, rng, n=140, gain=0.35):
    out = frames * gain
    x0 = rng.uniform(0, W, n)
    y0 = rng.uniform(-H, H, n)
    vy = rng.uniform(25, 55, n)
    vx = rng.normal(0, 4, n)
    r = rng.uniform(0.7, 1.4, n)
    for i in range(out.shape[0]):
        t = i / FPS
        ys = (y0 + vy * t) % (2 * H) - H / 2
        xs = (x0 + vx * t) % W
        for m in range(n):
            if -3 < ys[m] < H + 3:
                d2 = (XX - xs[m]) ** 2 + (YY - ys[m]) ** 2
                al = np.clip(r[m] + 0.5 - np.sqrt(d2), 0, 1)
                out[i] = out[i] * (1 - al) + 0.55 * al
    return out


# ---------------------------------------------------------------- event emulator
def emulate_events(frames, C=0.2, sigma_C=0.03, t_ref=1e-3, ba_rate=0.1, eps=0.02,
                   rng=None, keep=False):
    """Log-intensity threshold model with per-pixel threshold mismatch and a refractory
    period (Eq. 1 to 4). Timestamps are interpolated linearly between frames."""
    rng = rng or np.random.default_rng(SEED)
    n, h, w = frames.shape
    dt = 1.0 / FPS
    c_on = np.clip(C * (1 + sigma_C / C * rng.normal(0, 1, (h, w))), 0.01, None)
    c_off = np.clip(C * (1 + sigma_C / C * rng.normal(0, 1, (h, w))), 0.01, None)
    L_ref = np.log(frames[0] + eps)
    last_t = np.full((h, w), -np.inf)
    count_on = np.zeros(n, np.int64)
    count_off = np.zeros(n, np.int64)
    ev_frames = np.zeros((n, h, w), np.int16)
    out = []
    for k in range(1, n):
        L = np.log(frames[k] + eps)
        d = L - L_ref
        t0 = (k - 1) * dt
        for pol, cmap in ((1, c_on), (-1, c_off)):
            mag = d * pol
            nc = np.floor(np.where(mag > 0, mag / cmap, 0)).astype(np.int32)
            if nc.max() == 0:
                continue
            ys, xs = np.nonzero(nc)
            ncs = nc[ys, xs]
            mags = mag[ys, xs]
            cs = cmap[ys, xs]
            emitted = np.zeros(len(ys), np.int32)
            for j in range(1, ncs.max() + 1):
                act = ncs >= j
                tj = t0 + dt * (j * cs[act]) / mags[act]
                ok = tj - last_t[ys[act], xs[act]] >= t_ref
                idx = np.nonzero(act)[0][ok]
                last_t[ys[idx], xs[idx]] = tj[ok]
                emitted[idx] += 1
                if keep and len(idx):
                    out.append(np.stack([tj[ok], xs[idx], ys[idx], np.full(len(idx), pol)], 1))
            L_ref[ys, xs] += pol * ncs * cs
            ev_frames[k, ys, xs] += emitted.astype(np.int16)
            if pol > 0:
                count_on[k] += emitted.sum()
            else:
                count_off[k] += emitted.sum()
        nb = rng.poisson(ba_rate * dt * h * w)
        if nb:
            by = rng.integers(0, h, nb)
            bx = rng.integers(0, w, nb)
            bp = rng.choice([-1, 1], nb)
            np.add.at(ev_frames[k], (by, bx), 1)
            count_on[k] += (bp > 0).sum()
            count_off[k] += (bp < 0).sum()
            if keep:
                out.append(np.stack([t0 + dt * rng.uniform(0, 1, nb), bx, by, bp], 1))
    ev = np.concatenate(out) if keep and out else None
    if ev is not None:
        ev = ev[np.argsort(ev[:, 0], kind="stable")]
    return count_on, count_off, ev_frames, ev


# ---------------------------------------------------------------- trackers
def track_hands_frames(frames, gt_uv, kappa=5.0, gate=14.0, gate_lost=40.0):
    """Blob detector plus nearest-neighbour association. Hand-scale DoG peaks above
    kappa robust standard deviations are candidates; the head (largest blob) is masked.
    Initialised from the renderer at frame 0 only."""
    Rs = ndimage.gaussian_filter(frames, (0, 1.2, 1.2)) - ndimage.gaussian_filter(frames, (0, 2.6, 2.6))
    Rl = ndimage.gaussian_filter(frames, (0, 3.5, 3.5)) - ndimage.gaussian_filter(frames, (0, 7.0, 7.0))
    sig = 1.4826 * np.median(np.abs(Rs - np.median(Rs, axis=(1, 2), keepdims=True)), axis=(1, 2))
    n = frames.shape[0]
    est = np.zeros((n, 2, 2))
    found = np.zeros((n, 2), bool)
    p = gt_uv[0, [JI["r_wr"], JI["l_wr"]]].astype(float).copy()
    lost = np.zeros(2, int)
    for i in range(n):
        R = Rs[i] / max(sig[i], 1e-6)
        hy, hx = np.unravel_index(np.argmax(Rl[i]), Rl[i].shape)
        R = np.where((XX - hx) ** 2 + (YY - hy) ** 2 < 81.0, 0, R)
        mx = ndimage.maximum_filter(R, 5)
        ys, xs = np.nonzero((R == mx) & (R >= kappa))
        cand = np.stack([xs, ys], 1).astype(float)
        used = np.zeros(len(cand), bool)
        for a in np.argsort(lost, kind="stable"):
            if len(cand):
                d = np.linalg.norm(cand - p[a], axis=1)
                d[used] = np.inf
                k = int(np.argmin(d))
                if d[k] < (gate if lost[a] < 5 else gate_lost):
                    p[a] = cand[k]
                    used[k] = True
                    found[i, a] = True
                    lost[a] = 0
                    continue
            lost[a] += 1
        est[i] = p
    return est, found


def event_motion_detection(ev_frames, gt_uv, radius=5, min_events=3, speed=1.0):
    """Fraction of frames in which a moving hand (speed > 1 px/frame) has at least
    min_events events within radius px, and the same test on still frames (false alarms)."""
    hands = gt_uv[:, [JI["r_wr"], JI["l_wr"]]]
    sp = np.linalg.norm(np.diff(hands, axis=0, prepend=hands[:1]), axis=-1)
    cnt = np.zeros(sp.shape)
    for a in range(2):
        for i in range(len(hands)):
            x, y = int(hands[i, a, 0]), int(hands[i, a, 1])
            cnt[i, a] = ev_frames[i, max(0, y - radius):y + radius + 1, max(0, x - radius):x + radius + 1].sum()
    mv = sp > speed
    hit = cnt >= min_events
    return float(hit[mv].mean()), float(hit[~mv].mean())


# ---------------------------------------------------------------- retargeting
HUMAN_LIM = np.array([[-1.2, 3.1], [-0.6, 2.6], [-1.6, 1.6], [0.0, 2.6]])
ROBOT_LIM = np.array([[-0.8, 2.4], [-0.3, 1.6], [-1.2, 1.2], [0.0, 2.0]])
ROBOT_UP, ROBOT_FORE = 0.22, 0.24
Q_NAMES = ["shoulder pitch", "shoulder roll", "shoulder yaw", "elbow"]


def rodrigues(v, k, th):
    c, s = np.cos(th)[..., None], np.sin(th)[..., None]
    return v * c + np.cross(k, v) * s + k * np.sum(k * v, -1, keepdims=True) * (1 - c)


def arm_fk(q, side, l1, l2):
    """q (...,4) = pitch, roll, yaw, elbow. Returns elbow and hand relative to shoulder."""
    p, r, y, e = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    zero = np.zeros_like(p)
    u = np.stack([zero, -np.cos(p), np.sin(p)], -1)
    ax = np.stack([np.ones_like(p), zero, zero], -1)
    rr = side * r
    cz, sz = np.cos(rr)[..., None], np.sin(rr)[..., None]

    def rz(v):
        return np.stack([v[..., 0] * cz[..., 0] - v[..., 1] * sz[..., 0],
                         v[..., 0] * sz[..., 0] + v[..., 1] * cz[..., 0], v[..., 2]], -1)

    u = rz(u)
    ax = rz(ax)
    ax = rodrigues(ax, u, side * y)
    f = rodrigues(u, ax, -e)
    elbow = l1 * u
    hand = elbow + l2 * f
    return elbow, hand


def ik_track(el_t, hd_t, side, lim, l1, l2, w_el=0.5, lam=0.05, iters=8):
    """Batched damped Gauss-Newton IK with box limits and a temporal term (Eq. 10).
    el_t, hd_t: (R,N,3) targets relative to the shoulder."""
    Rn, N, _ = hd_t.shape
    q = np.tile(np.array([0.1, 0.1, 0.0, 0.3]), (Rn, 1))
    out = np.zeros((Rn, N, 4))
    sw = np.sqrt(np.array([w_el] * 3 + [1.0] * 3 + [lam] * 4))
    h = 1e-4
    for i in range(N):
        q_prev = q.copy()
        tgt = np.concatenate([el_t[:, i], hd_t[:, i]], -1)

        def resid(qq):
            e, hh = arm_fk(qq, side, l1, l2)
            return np.concatenate([e, hh, qq - q_prev], -1) * sw - np.concatenate(
                [tgt, q_prev], -1) * sw * np.r_[np.ones(6), np.zeros(4)]

        for _ in range(iters):
            r0 = resid(q)
            J = np.empty((Rn, 10, 4))
            for j in range(4):
                dq = q.copy()
                dq[:, j] += h
                J[:, :, j] = (resid(dq) - r0) / h
            JT = np.transpose(J, (0, 2, 1))
            A = JT @ J + 1e-3 * np.eye(4)
            g = (JT @ r0[..., None])[..., 0]
            step = np.linalg.solve(A, g[..., None])[..., 0]
            q = np.clip(q - step, lim[:, 0], lim[:, 1])
        out[:, i] = q
    return out


def pd_track(q_ref, kp=60.0, kd=2 * np.sqrt(60.0) * 0.9, inertia=1.0, tau_max=40.0, sub=16):
    """Joint-space PD tracking of a 30 Hz reference with torque saturation (Eq. 11)."""
    q = q_ref[..., 0, :].copy()
    qd = np.zeros_like(q)
    dt = 1.0 / FPS / sub
    out = np.zeros_like(q_ref)
    N = q_ref.shape[-2]
    for i in range(N):
        a = q_ref[..., i, :]
        b = q_ref[..., min(i + 1, N - 1), :]
        for s in range(sub):
            ref = a + (b - a) * s / sub
            tau = np.clip(kp * (ref - q) - kd * qd, -tau_max, tau_max)
            qd = qd + dt * tau / inertia
            q = q + dt * qd
        out[..., i, :] = q
    return out


def dtw(a, b, band=15):
    """Sakoe-Chiba banded DTW with Euclidean local cost (Eq. 12 to 13)."""
    n, m = len(a), len(b)
    D = np.full((n + 1, m + 1), np.inf)
    D[0, 0] = 0.0
    for i in range(1, n + 1):
        j0, j1 = max(1, i - band), min(m, i + band)
        c = np.linalg.norm(a[i - 1] - b[j0 - 1:j1], axis=1)
        for jj, j in enumerate(range(j0, j1 + 1)):
            D[i, j] = c[jj] + min(D[i - 1, j - 1], D[i - 1, j], D[i, j - 1])
    return D[n, m]


SIGMA_S = 0.25


def similarity(a, b):
    return float(np.exp(-dtw(a, b) / (len(a) * SIGMA_S)))


# ---------------------------------------------------------------- RF model
FREQ = 2.437e9
LAMBDA = 299792458.0 / FREQ
P_TX = 20.0
N_EXP = 2.2
SIGMA_SH = 2.0
TAU_SH = 2.0
A_BODY = 8.0
WALLS = [  # (x0,y0,x1,y1, loss dB, kind)
    (0, 0, 12, 0, 10.0, "exterior"), (0, 8, 12, 8, 10.0, "exterior"),
    (0, 0, 0, 8, 10.0, "exterior"), (12, 0, 12, 8, 10.0, "exterior"),
    (7, 0, 7, 3.0, 6.0, "interior"), (7, 3.9, 7, 8, 6.0, "interior"),
    (7, 4, 8.2, 4, 6.0, "interior"), (9.0, 4, 12, 4, 6.0, "interior"),
    (10, 0, 10, 1.2, 14.0, "sauna"), (10, 1.9, 10, 4, 14.0, "sauna"),
]
ROUTER = np.array([1.0, 7.0])
MACS = {"MacBook 1": np.array([6.0, 1.5]), "MacBook 2": np.array([9.0, 6.5]),
        "MacBook 3": np.array([8.5, 1.0])}
DANCE_SPOT = np.array([3.5, 4.25])
ROOMS = {"Living room": (0, 0, 7, 8), "Kitchen": (7, 4, 12, 8), "Dressing room": (7, 0, 10, 4),
         "Sauna (hot room)": (10, 0, 12, 4), "Deck and hot tub": (12, 0, 15, 4)}
HOT_TUB = np.array([13.5, 2.0])


def seg_intersect(p, q, a, b):
    d1, d2 = q - p, b - a
    den = d1[..., 0] * d2[1] - d1[..., 1] * d2[0]
    with np.errstate(divide="ignore", invalid="ignore"):
        t = ((a[0] - p[..., 0]) * d2[1] - (a[1] - p[..., 1]) * d2[0]) / den
        s = ((a[0] - p[..., 0]) * d1[..., 1] - (a[1] - p[..., 1]) * d1[..., 0]) / den
    return (np.abs(den) > 1e-12) & (t > 0) & (t < 1) & (s >= 0) & (s <= 1)


def wall_loss(p, q):
    loss = np.zeros(np.broadcast(p[..., 0], q[..., 0]).shape)
    for x0, y0, x1, y1, L, _ in WALLS:
        loss = loss + L * seg_intersect(p, q, np.array([x0, y0]), np.array([x1, y1]))
    return loss


def pl0():
    return 20 * np.log10(4 * np.pi * 1.0 / LAMBDA)


def strip_circle_fraction(h, w, r):
    """Area fraction of a disc of radius r covered by a strip of width w centred at offset h."""
    def F(x):
        x = np.clip(x, -r, r)
        return (x * np.sqrt(r * r - x * x) + r * r * np.arcsin(x / r)) / (np.pi * r * r)
    return np.clip(F(h + w / 2) - F(h - w / 2), 0, 1)


def body_loss(tx, rx, pos, heading):
    link = rx - tx
    d = np.linalg.norm(link)
    u = link / d
    rel = pos - tx
    along = rel @ u if rel.ndim == 1 else rel @ u
    perp = np.abs(rel[..., 0] * u[1] - rel[..., 1] * u[0])
    inside = (along > 0.05) & (along < d - 0.05)
    d1 = np.clip(along, 0.05, d - 0.05)
    r1 = np.sqrt(LAMBDA * d1 * (d - d1) / d)
    facing = np.array([np.sin(heading), np.cos(heading)]).T if np.ndim(heading) else np.array([np.sin(heading), np.cos(heading)])
    cosang = np.abs(facing @ u) if np.ndim(heading) else abs(float(facing @ u))
    w_eff = 0.26 + (0.46 - 0.26) * np.sqrt(np.clip(1 - cosang ** 2, 0, 1))
    frac = strip_circle_fraction(perp, w_eff, r1)
    return A_BODY * frac * inside, frac * inside, r1


def rssi_link(tx, rx, pos, heading, rng, fs=10.0):
    d = np.linalg.norm(rx - tx)
    base = P_TX - (pl0() + 10 * N_EXP * np.log10(d)) - float(wall_loss(tx, rx))
    n = len(pos)
    a = np.exp(-1 / (fs * TAU_SH))
    x = np.zeros(n)
    x[0] = rng.normal(0, SIGMA_SH)
    for i in range(1, n):
        x[i] = a * x[i - 1] + np.sqrt(1 - a * a) * rng.normal(0, SIGMA_SH)
    bl, frac, _ = body_loss(tx, rx, pos, heading)
    return base, base + x - bl, bl, frac


# ---------------------------------------------------------------- main
def main():
    t_start = time.time()
    DATA.mkdir(exist_ok=True)
    EPISODE.mkdir(exist_ok=True)
    rng = np.random.default_rng(SEED)
    joints, heading, count, pron = choreography()
    bg, bg_depth = background(np.random.default_rng(SEED + 1))
    clean, depth, uv, visible = render(joints, bg, bg_depth)
    hand_uv = uv[:, [JI["r_wr"], JI["l_wr"]]]
    np.savez_compressed(DATA / "dancer.npz", joints=joints, heading=heading, count=count, pron=pron,
                        uv=uv, visible=visible, fps=FPS, bpm=BPM)
    np.savez_compressed(DATA / "frames_preview.npz", clean=clean[::3], depth=depth[0])
    moves = pd.DataFrame({"count": np.arange(1, 17), "move": [m[0] for m in MOVES],
                          "arm": [m[1] or "body" for m in MOVES]})
    moves.to_csv(DATA / "macarena_fsm.csv", index=False)

    # act one: clean clip, main event stream
    cam = sensor(clean, np.random.default_rng(SEED + 2))
    on, off, evf, ev = emulate_events(cam, C=0.2, rng=np.random.default_rng(SEED + 3), keep=True)
    np.savez_compressed(DATA / "events_main.npz", t=ev[:, 0].astype(np.float32), x=ev[:, 1].astype(np.uint16),
                        y=ev[:, 2].astype(np.uint16), p=ev[:, 3].astype(np.int8))
    pd.DataFrame({"frame": np.arange(NF), "t_s": np.arange(NF) / FPS, "count": count + 1,
                  "on": on, "off": off}).to_csv(DATA / "event_counts_main.csv", index=False)

    # event frames at several thresholds (one window)
    thr = [0.10, 0.15, 0.20, 0.30, 0.40, 0.50]
    win_frames = np.arange(int(4.35 * BEAT * FPS), int(4.35 * BEAT * FPS) + 3)
    snaps = {}
    rows = []
    for C in thr:
        o, f_, evf_c, _ = emulate_events(cam, C=C, rng=np.random.default_rng(SEED + 3))
        pos = np.zeros((H, W))
        neg = np.zeros((H, W))
        snaps[f"C{int(C*100):02d}"] = evf_c[win_frames].sum(0)
        rows.append({"C": C, "events_total": int(o.sum() + f_.sum()), "on": int(o.sum()), "off": int(f_.sum())})
    # polarity-resolved snapshot via keep=True for the window only
    pol_snaps = {}
    for C in thr:
        _, _, _, evk = emulate_events(cam[: win_frames[-1] + 1], C=C, rng=np.random.default_rng(SEED + 3), keep=True)
        t0, t1 = (win_frames[0] - 1) / FPS, win_frames[-1] / FPS
        m = (evk[:, 0] >= t0) & (evk[:, 0] < t1)
        pol_snaps[f"C{int(C*100):02d}"] = evk[m]
    np.savez_compressed(DATA / "threshold_snapshots.npz", **pol_snaps, window=win_frames)
    pd.DataFrame(rows).to_csv(DATA / "threshold_totals.csv", index=False)

    # retargeting on 3D pose (ground truth plus Gaussian noise as a stand-in for the estimator)
    noise = np.random.default_rng(SEED + 4).normal(0, 0.015, joints.shape)
    est3d = joints + noise
    np.savez_compressed(EPISODE / "pose.npz", joints_est=est3d.astype(np.float32), names=np.array(JOINTS),
                        uv_gt=uv.astype(np.float32))

    def body_frame(P):
        c, s = np.cos(-heading), np.sin(-heading)
        out = P.copy()
        out[..., 0] = c[:, None] * P[..., 0] + s[:, None] * P[..., 2]
        out[..., 2] = -s[:, None] * P[..., 0] + c[:, None] * P[..., 2]
        return out

    def retarget_all(P, lim, l1, l2, scale):
        Pb = body_frame(P)
        qs = []
        for side, sh, el, wr in ((-1, "r_sh", "r_el", "r_wr"), (1, "l_sh", "l_el", "l_wr")):
            s = Pb[..., JI[sh], :]
            el_t = (Pb[..., JI[el], :] - s) * scale
            hd_t = (Pb[..., JI[wr], :] - s) * scale
            if el_t.ndim == 2:
                el_t, hd_t = el_t[None], hd_t[None]
            qs.append(ik_track(el_t, hd_t, side, lim, l1, l2))
        return np.concatenate(qs, -1)

    q_human = retarget_all(joints, HUMAN_LIM, L_UP, L_FORE, 1.0)[0]
    scale = (ROBOT_UP + ROBOT_FORE) / (L_UP + L_FORE)
    q_robot = retarget_all(est3d, ROBOT_LIM, ROBOT_UP, ROBOT_FORE, scale)[0]
    q_pd = pd_track(q_robot[None])[0]
    lim8 = np.concatenate([ROBOT_LIM, ROBOT_LIM])
    human_clip = (q_human < lim8[:, 0] - 1e-6) | (q_human > lim8[:, 1] + 1e-6)
    np.savez_compressed(DATA / "retarget.npz", q_human=q_human, q_robot=q_robot, q_pd=q_pd,
                        robot_lim=ROBOT_LIM, human_lim=HUMAN_LIM)
    np.savez_compressed(EPISODE / "humanoid.npz", q_ref=q_robot.astype(np.float32), q_pd=q_pd.astype(np.float32),
                        joint_names=np.array([f"{s} {n}" for s in ("right", "left") for n in Q_NAMES]))
    hand_err_robot = []
    rob_e, rob_h = arm_fk(q_robot[:, :4], -1, ROBOT_UP, ROBOT_FORE)
    ret = {
        "clip_frac_any": float(human_clip.any(1).mean()),
        "clip_frac_joint": human_clip.mean(0).tolist(),
        "pd_rmse": float(np.sqrt(np.mean((q_pd - q_robot) ** 2))),
        "pd_max": float(np.max(np.abs(q_pd - q_robot))),
    }

    # act two: steam veil sweep
    taus = np.round(np.arange(0, 3.01, 0.25), 2)
    seeds = range(6)
    ref_seq = q_pd
    sweep = []
    gt_hand = hand_uv
    vis = visible
    pck_r = 8.0
    tracks_preview = {}
    run_meta = []
    hd_targets = []
    for tau in taus:
        for sd in seeds:
            r = np.random.default_rng(SEED + 1000 + int(tau * 100) * 10 + sd)
            veiled, T = apply_veil(clean, depth, tau, r)
            obs = sensor(veiled, r)
            est, found = track_hands_frames(obs, uv)
            err = np.linalg.norm(est - gt_hand, axis=-1)
            pck = float((err < pck_r).mean())
            # lift tracked hands to 3D with the renderer's depth (depth oracle); other joints from the estimate
            P = est3d.copy()
            for a, j in enumerate((JI["r_wr"], JI["l_wr"])):
                zc = CAM[2] - joints[:, j, 2]
                P[:, j, 0] = CAM[0] + (est[:, a, 0] - CX) * zc / F_PX
                P[:, j, 1] = CAM[1] - (est[:, a, 1] - CY) * zc / F_PX
            hd_targets.append(P)
            Tm = float(np.mean([T[i][int(np.clip(gt_hand[i, 0, 1], 0, H - 1)), int(np.clip(gt_hand[i, 0, 0], 0, W - 1))]
                                for i in range(0, NF, 10)]))
            run_meta.append({"tau": float(tau), "seed": sd, "pck": pck, "found": float(found.mean()),
                             "median_err_px": float(np.median(err)), "T_hand": Tm})
            if sd == 0:
                tracks_preview[f"tau{int(tau*100):03d}"] = est
                if tau in (0.0, 0.75, 1.5, 2.25, 3.0) or tau == 1.0 or tau == 2.0:
                    np.save(DATA / f"veil_frame_tau{int(tau*100):03d}.npy", obs[int(6.4 * BEAT * FPS)])
    P_all = np.stack(hd_targets)
    q_sweep_rob = retarget_all(P_all, ROBOT_LIM, ROBOT_UP, ROBOT_FORE, scale)
    q_sweep = pd_track(q_sweep_rob)
    for k, m in enumerate(run_meta):
        a = q_sweep[k][::2]
        b = ref_seq[::2]
        m["similarity"] = similarity(a, b)
        m["joint_rmse"] = float(np.sqrt(np.mean((q_sweep[k] - ref_seq) ** 2)))
        sweep.append(m)
    pd.DataFrame(sweep).to_csv(DATA / "veil_sweep.csv", index=False)
    np.savez_compressed(DATA / "veil_tracks.npz", gt=gt_hand, visible=vis, **tracks_preview)
    np.savez_compressed(DATA / "veil_humanoid.npz", taus=taus,
                        q=np.stack([q_sweep[k] for k, m in enumerate(run_meta) if m["seed"] == 0]))

    # event-based tracker heatmap: severity x contrast threshold
    heat = []
    for tau in taus:
        r = np.random.default_rng(SEED + 5000 + int(tau * 100))
        veiled, _ = apply_veil(clean, depth, tau, r)
        obs = sensor(veiled, r)
        for C in thr:
            o, f_, evf_c, _ = emulate_events(obs, C=C, rng=np.random.default_rng(SEED + 7))
            det, fa = event_motion_detection(evf_c, uv)
            heat.append({"tau": float(tau), "C": C, "success": det, "still_hits": fa,
                         "events_per_s": float((o.sum() + f_.sum()) / DURATION)})
    pd.DataFrame(heat).to_csv(DATA / "event_tracker_heatmap.csv", index=False)

    # thermal domains
    dom_rows = []
    for dom in ("dry", "steam", "frozen"):
        for sd in range(4):
            r = np.random.default_rng(SEED + 9000 + sd + {"dry": 0, "steam": 100, "frozen": 200}[dom])
            if dom == "dry":
                img = clean
            elif dom == "steam":
                img, _ = apply_veil(clean, depth, 1.5, r)
            else:
                img = snow(clean, r)
            obs = sensor(img, r)
            o, f_, evf_c, _ = emulate_events(obs, C=0.2, rng=r)
            est, _ = track_hands_frames(obs, uv)
            err = np.linalg.norm(est - gt_hand, axis=-1).mean(1)
            for i in range(1, NF, 2):
                dom_rows.append({"domain": dom, "seed": sd, "frame": i,
                                 "event_rate_keps": (o[i] + f_[i]) * FPS / 1e3, "hand_err_px": err[i]})
    pd.DataFrame(dom_rows).to_csv(DATA / "thermal_domains.csv", index=False)

    # RF: dancer walks in, dances on the router-MacBook 1 link, walks out
    fs = 10.0
    t_walk = 3.0
    T_total = DURATION + 2 * t_walk
    tt = np.arange(0, T_total, 1 / fs)
    start = np.array([0.8, 1.2])
    exitp = np.array([6.2, 6.8])
    pos = np.zeros((len(tt), 2))
    hdg = np.zeros(len(tt))
    for i, t in enumerate(tt):
        if t < t_walk:
            s = smoothstep(t / t_walk)
            pos[i] = start + s * (DANCE_SPOT - start)
            hdg[i] = np.arctan2(*(DANCE_SPOT - start))
        elif t < t_walk + DURATION:
            k = int(min((t - t_walk) * FPS, NF - 1))
            pos[i] = DANCE_SPOT + np.array([joints[k, JI["pelvis"], 0], -joints[k, JI["pelvis"], 2]])
            hdg[i] = heading[k] + np.pi
        else:
            s = smoothstep((t - t_walk - DURATION) / t_walk)
            pos[i] = DANCE_SPOT + s * (exitp - DANCE_SPOT)
            hdg[i] = np.arctan2(*(exitp - DANCE_SPOT))
    rf = {"t_s": tt, "x_m": pos[:, 0], "y_m": pos[:, 1]}
    rf_meta = {}
    for name, mac in MACS.items():
        base, rssi, bl, frac = rssi_link(ROUTER, mac, pos, hdg, np.random.default_rng(SEED + 300 + len(rf)))
        key = name.replace(" ", "")
        rf[f"{key}_rssi_dbm"] = rssi
        rf[f"{key}_body_db"] = bl
        rf[f"{key}_fresnel_frac"] = frac
        rf_meta[key] = {"base_dbm": float(base), "max_body_db": float(bl.max()),
                        "d_m": float(np.linalg.norm(mac - ROUTER)), "walls_db": float(wall_loss(ROUTER, mac))}
    rf_df = pd.DataFrame(rf)
    rf_df.to_csv(DATA / "rssi_sim.csv", index=False)
    logger = rf_df.iloc[:: int(fs)].copy()
    for name in MACS:
        key = name.replace(" ", "")
        logger[f"{key}_rssi_dbm"] = np.round(logger[f"{key}_rssi_dbm"])
    logger.to_csv(EPISODE / "rssi_simulated_1hz.csv", index=False)

    gx, gy = np.meshgrid(np.arange(0.05, 15, 0.1), np.arange(0.05, 8, 0.1))
    G = np.stack([gx, gy], -1)
    d = np.maximum(np.linalg.norm(G - ROUTER, axis=-1), 0.3)
    field = P_TX - (pl0() + 10 * N_EXP * np.log10(d)) - wall_loss(ROUTER[None, None], G)
    np.savez_compressed(DATA / "rf_field.npz", gx=gx, gy=gy, field=field)
    _, _, r1_mid = body_loss(ROUTER, MACS["MacBook 1"], DANCE_SPOT, 0.0)
    rf_meta["fresnel_r1_mid_m"] = float(r1_mid)
    rf_meta["lambda_m"] = float(LAMBDA)
    rf_meta["pl0_db"] = float(pl0())
    rf_meta["t_walk_s"] = t_walk

    # episode metadata
    ev_npz = EPISODE / "events.npz"
    np.savez_compressed(ev_npz, t=ev[:, 0].astype(np.float32), x=ev[:, 1].astype(np.uint16),
                        y=ev[:, 2].astype(np.uint16), p=ev[:, 3].astype(np.int8))
    meta = {
        "episode": "episode_000",
        "subject": "procedural dancer (no human was recorded)",
        "streams": {
            "events.npz": "SIMULATED: log-intensity threshold emulation of rendered frames",
            "pose.npz": "SIMULATED: ground-truth renderer pose plus Gaussian noise (stand-in for an estimator)",
            "humanoid.npz": "SIMULATED: kinematic retargeting plus PD tracking",
            "rssi_simulated_1hz.csv": "SIMULATED: log-distance path loss, walls, shadowing, body occlusion",
        },
        "real_streams": [],
        "fps": FPS, "bpm": BPM, "duration_s": DURATION, "resolution": [W, H], "seed": SEED,
    }
    (EPISODE / "meta.json").write_text(json.dumps(meta, indent=2))
    sizes = {
        "frames_bytes": int(NF * H * W),
        "events_bytes": int(len(ev) * (4 + 2 + 2 + 1)),
        "pose_bytes": int(est3d.astype(np.float32).nbytes),
        "humanoid_bytes": int(q_robot.astype(np.float32).nbytes + q_pd.astype(np.float32).nbytes),
        "rssi_bytes": int(len(logger) * 3 * 4),
        "n_events": int(len(ev)),
    }
    summary = {
        "n_frames": NF, "duration_s": DURATION, "fps": FPS, "bpm": BPM, "width": W, "height": H,
        "seed": SEED, "on_total": int(on.sum()), "off_total": int(off.sum()),
        "hand_visible_frac": float(visible.mean()),
        "retarget": ret, "rf": rf_meta, "sizes": sizes,
        "taus": taus.tolist(), "thresholds": thr, "pck_radius_px": pck_r, "event_radius_px": 5, "event_min_events": 3,
        "sim_wall_time_s": time.time() - t_start,
    }
    (DATA / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps({k: summary[k] for k in ("n_frames", "on_total", "off_total", "sim_wall_time_s")}))


if __name__ == "__main__":
    main()
