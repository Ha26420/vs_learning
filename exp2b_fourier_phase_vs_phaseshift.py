# -*- coding: utf-8 -*-
"""
fourier_phase_vs_phaseshift.py
==============================
第 4 章（傅里叶/相位）→ demo1（相移条纹 → 相位）的桥。

一句话：**条纹图的相位藏在它的频谱里**。把一张条纹图做 2D FFT，会看到 3 个亮点
（直流 + 两个一级侧带）；把其中一个侧带抠出来、移到中心、再 IFFT，得到的复数场的
`angle` 就是相位。这就是**傅里叶变换轮廓术（FTP）**，单帧就能出相位；
而 demo1 的相移法是"时域解"：拍 4 帧、每帧相移 90°，用 atan2 直接解。

所以这一步不是学新东西：
    第4章 傅里叶变换 → 频谱里找侧带 → 反变换得相位（FTP，1 帧）
    demo1 相移法     → 4 帧做差 → atan2 得相位（4 帧，更准）
两者算的是同一个 φ，只是"频域解"和"时域解"的区别。

运行：
    python fourier_phase_vs_phaseshift.py
输出：
    fourier_01_spectrum_and_phase.png   # 条纹图 + 对数频谱(标出侧带) + 两种相位之差
    fourier_02_height_profile.png       # 沿一行的高度：真值 / 相移 / FTP
    + 终端打印：掩膜半径 5~32 px 时的相位误差与高度误差对比表（看半径多大时会吃到直流而崩掉）
"""
import sys
from pathlib import Path

import numpy as np

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

OUT = Path(__file__).resolve().parent
rng = np.random.default_rng(0)

# ----------------------------- 与 demo1 一致的参数 -----------------------------
NY, NX = 360, 480
F0 = 1.0 / 16.0                 # 载频：16 px 一个条纹 → 全宽 30 个条纹
K_H = 2 * np.pi * 0.9           # 相位-高度灵敏度 (rad/mm)，demo1 同款
NOISE = 1.5                     # 灰度噪声


def make_data():
    """生成 4 帧相移条纹（和 demo1 的 simulate() 同款：A/B 不均匀 + 噪声 + 真值面形）。"""
    y, x = np.mgrid[0:NY, 0:NX].astype(float)
    h_true = 2.0 * np.exp(-(((x - 0.52 * NX) ** 2 + (y - 0.5 * NY) ** 2) / (2 * 110.0 ** 2)))
    phi = 2 * np.pi * F0 * x + K_H * h_true          # 载频 + 高度调制
    A = 90 + 45 * np.exp(-(((x - 0.18 * NX) ** 2 + (y - 0.85 * NY) ** 2) / (2 * 180.0 ** 2)))
    B = 55 * np.exp(-(((x - 0.5 * NX) ** 2 + (y - 0.5 * NY) ** 2) / (2 * (0.75 * NX) ** 2)))
    B *= 0.8 + 0.2 * np.sin(np.pi * x / NX)
    I = np.empty((4, NY, NX))
    for k in range(4):
        I[k] = A + B * np.cos(phi + 2 * np.pi * k / 4) + rng.normal(0, NOISE, (NY, NX))
    return I, h_true, phi


def phase_from_four_step(I):
    """demo1 的做法：4 帧相移 → atan2(I3-I1, I0-I2)。"""
    return np.arctan2(I[3] - I[1], I[0] - I[2])


def wrap(a):
    """把相位折回 (-π, π]。"""
    return np.mod(a + np.pi, 2 * np.pi) - np.pi


def unwrap_rowcol(phi_w):
    """和 demo1 一样的行列展开：先逐行 np.unwrap，再用第 0 列把行间 2π 偏移补上。"""
    u = np.empty_like(phi_w)
    for i in range(phi_w.shape[0]):
        u[i] = np.unwrap(phi_w[i])
    for j in range(1, u.shape[0]):
        u[j] -= 2 * np.pi * np.round((u[j, 0] - u[j - 1, 0]) / (2 * np.pi))
    return u


def ftp_phase(fringe, radius=40, side=+1, show_peak=False):
    """傅里叶变换轮廓术：FFT → 取一级侧带 → 移到中心 → IFFT → angle。

    side=+1 取 +f0 侧带，-1 取 -f0 侧带（相位符号相反）。
    radius 是圆形掩膜半径：太小丢高频（平滑掉细节），太大把直流带进来（误差爆炸）。
    注意：把侧带搬到零频时，**载频被自动去掉了**，所以返回的是 ψ = K·h，
    不是 φ = 2πf0x + K·h —— 这就是 FTP 和相移法"差一个载频"的原因。
    """
    F = np.fft.fftshift(np.fft.fft2(fringe))
    H, W = fringe.shape
    cy, cx = H // 2, W // 2
    shift_x = int(round(W * F0))                 # 载频在频域的位置：W*f0 = 30
    peak = (cy, cx + side * shift_x)

    # 以峰值点为中心做圆形掩膜，再移到中心
    yy, xx = np.mgrid[0:H, 0:W]
    mask = (yy - peak[0]) ** 2 + (xx - peak[1]) ** 2 <= radius ** 2
    Fm = F * mask
    Fm = np.roll(Fm, (cy - peak[0], cx - peak[1]), axis=(0, 1))   # 侧带搬到中心
    c = np.fft.ifft2(np.fft.ifftshift(Fm))                        # 复解析信号
    phase = np.angle(c)                                           # 相位（含载频）
    if show_peak:
        return phase, F, peak, mask
    return phase


def height_from_phase(phi_unwrap):
    """相位 → 高度（减掉载频，除以灵敏度），和 demo1 的线性相位-高度模型一致。"""
    y, x = np.mgrid[0:NY, 0:NX].astype(float)
    return (phi_unwrap - 2 * np.pi * F0 * x) / K_H


def height_from_ftp_phase(psi_unwrap):
    """FTP 的相位已经是 ψ=K·h（载频被侧带搬移去掉了），所以直接除以 K。"""
    return psi_unwrap / K_H


def rms(a):
    return float(np.sqrt(np.mean(np.asarray(a) ** 2)))


def main():
    I, h_true, phi_true = make_data()
    phi_w_ps = phase_from_four_step(I)             # 相移法（4 帧）
    phi_u_ps = unwrap_rowcol(phi_w_ps)
    h_ps = height_from_phase(phi_u_ps)

    print("=" * 66)
    print("[data] 合成条纹 360x480，载频 16 px/周期，面形峰值 2 mm（和 demo1 同款）")
    print(f"[4step] 4 帧相移 → 包裹相位 RMS 误差 = {np.rad2deg(rms(wrap(phi_w_ps - phi_true))):.3f} °"
          f"；重建高度 RMS = {rms(h_ps - h_true) * 1000:.1f} μm")

    # 单帧 FTP：用第 0 帧，看频谱
    print("-" * 66)
    print("[FTP] 单帧傅里叶变换轮廓术：对同一张 I0 做 FFT → 抠侧带 → IFFT")
    phase_dummy, F, peak, mask = ftp_phase(I[0], radius=10, side=+1, show_peak=True)
    mag = np.log1p(np.abs(F))
    print(f"[FTP] 频谱峰值位置（fftshift 后）= (row={peak[0]}, col={peak[1]})，"
          f"即载频在 ±{int(round(NX*F0))} 处 → 频谱里能看到'直流 + 两个一级侧带'三个亮点")

    # 掩膜半径扫描：这是 FTP 最关键的参数
    print("-" * 66)
    print("[FTP] 掩膜半径对结果的影响（同一张图）：")
    print(f"  {'半径':>6} {'包裹相位误差RMS':>16} {'重建高度RMS':>14}")
    rows = []
    for r in (5, 10, 20, 25, 28, 32):
        p = ftp_phase(I[0], radius=r, side=+1)
        e_deg = np.rad2deg(rms(wrap(p - K_H * h_true)))     # FTP 输出 ψ=K·h，拿它和 K·h 比
        h = height_from_ftp_phase(unwrap_rowcol(p))
        rows.append((r, e_deg, rms(h - h_true) * 1000))
        print(f"  {r:>6} {e_deg:>14.3f} ° {rows[-1][2]:>12.1f} μm")
    best = min([t for t in rows if t[0] < 30], key=lambda t: t[2])   # 半径 ≥ 载频间隔(30) 会吃到直流，排除
    print(f"[FTP] 最好的半径 {best[0]} px：相位误差 {best[1]:.2f} °，高度 RMS {best[2]:.1f} μm")
    print(f"[FTP] 注意半径 ≥ 30 px 那一行：掩膜把直流（背景光强）也圈进来了 → 误差从几度跳到 97° 以上。")
    print("      这就是为什么条纹载频必须足够高：侧带要和直流分开，掩膜半径 < 载频间隔。")

    best_r = best[0]
    phase_ftp = ftp_phase(I[0], radius=best_r, side=+1)   # 用最优半径重新算，供出图和对比

    # 结论对照：同一张图、两种解法
    print("-" * 66)
    print("[compare] 同一套条纹：")
    print(f"  相移法（4 帧） : 高度 RMS {rms(h_ps - h_true)*1000:6.1f} μm，需要 4 次曝光")
    print(f"  FTP（1 帧）    : 高度 RMS {best[2]:6.1f} μm，只需要 1 次曝光")
    print("  → 相移更准；FTP 单帧、适合动态/快速，但精度受频谱泄漏、载频选择、掩膜带宽限制。")

    # ---------------- 出图 ----------------
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False

    fig, axs = plt.subplots(1, 3, figsize=(16, 4.6))
    axs[0].imshow(I[0], cmap="gray")
    axs[0].set_title("单帧条纹 I0（只给这一张）")
    axs[1].imshow(mag, cmap="viridis")
    axs[1].plot(peak[1], peak[0], "r+", ms=14, label="选中的 +1 级侧带")
    axs[1].plot(NX // 2, NY // 2, "w+", ms=12, label="直流（背景光强）")
    axs[1].set_title("对数频谱：直流 + 两个一级侧带")
    axs[1].legend(fontsize=7, loc="upper right")
    diff = np.rad2deg(wrap(phase_ftp - K_H * h_true))
    im = axs[2].imshow(diff, cmap="RdBu_r", vmin=-10, vmax=10)
    axs[2].set_title("FTP 相位 − K·h（去载频后，单位：度）")
    fig.colorbar(im, ax=axs[2], fraction=0.046)
    plt.tight_layout()
    p1 = OUT / "fourier_01_spectrum_and_phase.png"
    plt.savefig(p1, dpi=150, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4.2))
    row = NY // 2
    ax.plot(h_true[row] * 1e3, "k-", lw=2, label="真值")
    ax.plot(h_ps[row] * 1e3, "C0--", label="4 步相移（4 帧）")
    ax.plot(height_from_ftp_phase(unwrap_rowcol(phase_ftp))[row] * 1e3, "C1:", label=f"FTP（1 帧，半径 {best_r}）")
    ax.set_xlabel("x / pixel")
    ax.set_ylabel("高度 / μm")
    ax.set_title(f"沿 y={row} 行的高度：相移 vs 傅里叶单帧")
    ax.legend()
    plt.tight_layout()
    p2 = OUT / "fourier_02_height_profile.png"
    plt.savefig(p2, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\n[out] 已保存 {p1.name}、{p2.name}")


if __name__ == "__main__":
    main()
