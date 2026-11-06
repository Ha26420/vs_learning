# -*- coding: utf-8 -*-
"""
exp7b_matchtemplate_to_dic.py
=============================
把 exp7 的「模板匹配」一步步变成 demo2 的「DIC 子区相关」——四个台阶，代码是同一套。

为什么这算"同一步"：
    exp7 里那一行  cv2.matchTemplate(img, template, cv2.TM_CCOEFF_NORMED)
    做的就是"在整张图里滑模板、算零均值归一化相关（ZNCC）、取最大值"。
    DIC 只是把它改成：
        ① 在"变形图"里找同一个子区（模板来自参考图）
        ② 对图像上很多个点各做一次（得到整场位移）
        ③ 峰值不取整像素，而是在相关峰上做抛物面拟合（亚像素）
        ④ 对位移场求梯度（得到应变）
    所以 demo2 的 dic_demo.py 里没有魔法，只是把 matchTemplate 的思想手写 + 加了网格和亚像素。

运行：
    python exp7b_matchtemplate_to_dic.py
    python exp7b_matchtemplate_to_dic.py --ref 参考图.png --def 变形图.png   # 换成你自己的图

预期输出（合成数据，真值已知）：
    [1] exp7 式 matchTemplate 的整像素位移 ≈ (1, -1) px
    [2] 手写 ZNCC 与 matchTemplate 同一点的相关值一致（差 <1e-4）→ 证明 TM_CCOEFF_NORMED 就是 ZNCC
    [3] 单点亚像素：位移误差 < 0.05 px
    [4] 全场网格：位移 RMS ≈ 0.01 px，应变 ex ≈ 2000 με
"""
import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

OUT = Path(__file__).resolve().parent
rng = np.random.default_rng(0)


# ---------------------------------------------------------------- 数据
def make_speckle(h=600, w=600):
    """合成散斑：随机噪声 + 高斯模糊 + 对比度拉伸（散斑粒径约 3~5 px）。"""
    img = rng.random((h, w)).astype(np.float32)
    img = cv2.GaussianBlur(img, (0, 0), 2.0)
    img = (img - img.min()) / (img.max() - img.min()) * 255
    return img.astype(np.uint8)


def deform(ref, u0=0.70, v0=-1.20, ex=2000e-6, ey=-800e-6):
    """按已知位移场变形：def(x,y) = ref(x-u, y-v)。返回 def 和图上的真值场。"""
    h, w = ref.shape
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    u = u0 + ex * (xx - w / 2)      # 刚体 + 均匀正应变
    v = v0 + ey * (yy - h / 2)
    m = np.dstack([(xx - u), (yy - v)]).astype(np.float32)
    return cv2.remap(ref, m, None, cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE), u, v


# ---------------------------------------------------------------- 三个核心函数
def zncc_manual(a, b):
    """手写 ZNCC：和 cv2.TM_CCOEFF_NORMED 同一个公式，用来证明两者等价。"""
    a = a.astype(np.float64) - a.mean()
    b = b.astype(np.float64) - b.mean()
    den = np.sqrt((a * a).sum() * (b * b).sum())
    return float((a * b).sum() / den) if den > 0 else 0.0


def parabolic_peak(res, x, y):
    """在相关峰 (x,y) 上做三点抛物面拟合，得到亚像素偏移（exp7 没有这一步）。"""
    def offset(v_m, v_0, v_p):
        den = (v_m - 2 * v_0 + v_p)
        return 0.0 if abs(den) < 1e-12 else 0.5 * (v_m - v_p) / den

    dx = offset(res[y, x - 1], res[y, x], res[y, x + 1])
    dy = offset(res[y - 1, x], res[y, x], res[y + 1, x])
    return dx, dy


def match_one(ref, defimg, cx, cy, half=20):
    """exp7 的精髓：拿参考图里 (cx,cy) 附近的一块当模板，在变形图里找它。
    返回 (整像素位移, 亚像素位移, 相关峰值, 相关图 res, 模板左上角)。"""
    x0, y0 = cx - half, cy - half
    tmpl = ref[y0:y0 + 2 * half + 1, x0:x0 + 2 * half + 1]
    res = cv2.matchTemplate(defimg, tmpl, cv2.TM_CCOEFF_NORMED)   # ← exp7 的那一行
    _, max_val, _, max_loc = cv2.minMaxLoc(res)
    int_u = max_loc[0] - x0
    int_v = max_loc[1] - y0
    dx, dy = parabolic_peak(res, max_loc[0], max_loc[1])
    return (int_u, int_v), (int_u + dx, int_v + dy), max_val, res, (x0, y0)


# ---------------------------------------------------------------- 主流程
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", default="", help="参考图路径（不给就用合成散斑）")
    ap.add_argument("--def", dest="defimg", default="", help="变形图路径")
    ap.add_argument("--half", type=int, default=20, help="子区半宽（21x21→41x41）")
    args = ap.parse_args()

    if args.ref and args.defimg:
        ref = cv2.imread(args.ref, cv2.IMREAD_GRAYSCALE)
        defimg = cv2.imread(args.defimg, cv2.IMREAD_GRAYSCALE)
        if ref is None or defimg is None:
            raise SystemExit("读不到图片，检查 --ref/--def 路径")
        truth_known = False
        print("[data] 使用你自己的两张图（没有真值，只能看位移场是否合理）")
    else:
        ref = make_speckle()
        defimg, _, _ = deform(ref)
        truth_known = True
        print("[data] 合成散斑 600x600，真值：刚体 u=+0.70 px, v=-1.20 px；应变 ex=+2000 με, ey=-800 με")

    h, w = ref.shape
    cx, cy = w // 2, h // 2

    # ---- [1] exp7 式：单点整像素匹配
    (iu, iv), (su, sv), score, res, (x0, y0) = match_one(ref, defimg, cx, cy, args.half)
    dx, dy = su - iu, sv - iv          # 亚像素在整像素基础上的偏移
    print(f"\n[1] exp7 式单点匹配（子区 {2*args.half+1}px）：整像素位移 = ({iu}, {iv}) px，相关峰值 = {score:.4f}")

    # ---- [2] 证明 TM_CCOEFF_NORMED 就是 ZNCC
    tmpl = ref[y0:y0 + 2 * args.half + 1, x0:x0 + 2 * args.half + 1]
    patch = defimg[y0 + iv:y0 + iv + tmpl.shape[0], x0 + iu:x0 + iu + tmpl.shape[1]]
    z = zncc_manual(tmpl, patch)
    print(f"[2] 手写 ZNCC 与 matchTemplate 的峰值：{z:.6f} vs {score:.6f}  → 差 {abs(z - score):.2e}"
          f"（{'同一个东西' if abs(z - score) < 1e-4 else '检查一下'}）")

    # ---- [3] 亚像素（exp7 没有、DIC 必须有）
    print(f"[3] 亚像素（抛物面拟合）：位移 = ({su:+.3f}, {sv:+.3f}) px，相关峰值 {score:.4f}")
    if truth_known:
        tu = 0.70 + 2000e-6 * (cx - w / 2)
        tv = -1.20 - 800e-6 * (cy - h / 2)
        print(f"    真值 = ({tu:+.3f}, {tv:+.3f}) px  → 误差 ({su-tu:+.3f}, {sv-tv:+.3f}) px")

    # ---- [4] 全场：把 [1]+[3] 在每个网格点做一遍 = DIC
    step = 20
    pts = [(x, y) for y in range(args.half + 10, h - args.half - 10, step)
           for x in range(args.half + 10, w - args.half - 10, step)]
    U, V, GX, GY = [], [], [], []
    for (x, y) in pts:
        _, (mu, mv), _, _, _ = match_one(ref, defimg, x, y, args.half)
        U.append(mu); V.append(mv); GX.append(x); GY.append(y)
    U = np.array(U); V = np.array(V); GX = np.array(GX); GY = np.array(GY)
    print(f"\n[4] 全场 DIC：{len(pts)} 个测点（网格步长 {step} px）")
    if truth_known:
        tu = 0.70 + 2000e-6 * (GX - w / 2)
        tv = -1.20 - 800e-6 * (GY - h / 2)
        rms = float(np.sqrt(np.mean((U - tu) ** 2 + (V - tv) ** 2)))
        print(f"    位移 RMS 误差 = {rms:.4f} px  （demo2 的手写版是 0.0128 px）")
        print(f"    位移均值 = ({U.mean():+.2f}, {V.mean():+.2f}) px；真值均值 = ({tu.mean():+.2f}, {tv.mean():+.2f})")

    # 应变：对位移场做平面拟合取斜率（和 demo2 一样）
    def slope(xs, ys, zs):
        A = np.c_[xs, ys, np.ones_like(xs)]
        coef, *_ = np.linalg.lstsq(A, zs, rcond=None)
        return coef[0], coef[1]
    du_dx, du_dy = slope(GX, GY, U)
    dv_dx, dv_dy = slope(GX, GY, V)
    print(f"    平面拟合应变：ex = {du_dx*1e6:+.0f} με（真值 +2000），ey = {dv_dy*1e6:+.0f} με（真值 -800）")

    # ---- 出图
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False

    fig, axs = plt.subplots(1, 3, figsize=(15, 4.2))
    # 左：参考图 + 子区框
    axs[0].imshow(ref, cmap="gray")
    axs[0].add_patch(plt.Rectangle((x0, y0), tmpl.shape[1], tmpl.shape[0], fill=False, ec="r", lw=1.5))
    axs[0].plot(cx, cy, "r+", ms=10)
    axs[0].set_title(f"参考图 + 子区（{tmpl.shape[1]}px）")
    # 中：matchTemplate 输出的相似度图
    peak_x, peak_y = x0 + iu, y0 + iv
    im = axs[1].imshow(res, cmap="viridis")
    axs[1].plot(peak_x, peak_y, "r+", ms=12)
    axs[1].set_title("ZNCC 相似度图（exp7 的匹配结果）")
    fig.colorbar(im, ax=axs[1], fraction=0.046)
    # 右：过峰值的那一行相关值 —— 三个点拟合抛物线就是亚像素
    xs = np.arange(max(0, peak_x - 6), min(res.shape[1], peak_x + 7))
    axs[2].plot(xs - peak_x, res[peak_y, xs], "o-", label="ZNCC")
    axs[2].axvline(dx, color="g", ls="--", label=f"抛物面峰 {dx:+.2f} px")
    axs[2].set_title("沿 x 的相关峰（三点拟合 → 亚像素）")
    axs[2].set_xlabel("相对整像素峰的偏移 (px)")
    axs[2].legend(fontsize=8)
    plt.tight_layout()
    p1 = OUT / "exp7b_01_zncc_peak.png"
    plt.savefig(p1, dpi=150, bbox_inches="tight")
    plt.close(fig)

    fig, axs = plt.subplots(1, 2, figsize=(11, 4.2))
    sc = axs[0].scatter(GX, GY, c=U, s=18, cmap="coolwarm")
    axs[0].invert_yaxis()
    axs[0].set_title("全场 u 位移（px）")
    fig.colorbar(sc, ax=axs[0], fraction=0.046)
    sc = axs[1].scatter(GX, GY, c=V, s=18, cmap="coolwarm")
    axs[1].invert_yaxis()
    axs[1].set_title("全场 v 位移（px）")
    fig.colorbar(sc, ax=axs[1], fraction=0.046)
    plt.tight_layout()
    p2 = OUT / "exp7b_02_dic_field.png"
    plt.savefig(p2, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\n[out] 已保存 {p1.name}、{p2.name}")
    print("[下一步] 打开 demo2_dic/dic_demo.py 对比：它的 zncc 函数、抛物面拟合、网格循环就是刚才这三步的工程版。")


if __name__ == "__main__":
    main()
