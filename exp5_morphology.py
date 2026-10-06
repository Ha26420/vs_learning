import cv2
import os
import numpy as np

# ---------- 用 Pillow 画中文标题（没装也能跑，会自动退回英文） ----------
try:
    from PIL import Image, ImageDraw, ImageFont
    HAVE_PIL = True
except Exception:
    HAVE_PIL = False

def load_font(size):
    """加载中文字体，找不到就用默认字体。"""
    candidates =[
            r"C:\Windows\Fonts\msyh.ttc",          # 微软雅黑(Windows)
            r"C:\Windows\Fonts\simhei.ttf",        # 黑体(Windows)
            r"C:\Windows\Fonts\simsun.ttc",        # 宋体(Windows)
            r"/System/Library/Fonts/PingFang.ttc",  # macOS
            r"/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",  # Linux
    ]
    for p in candidates:
        if os.path.exists(p):
            try:
             return ImageFont.truetype(p, size)
            except Exception:
                pass
    return ImageFont.load_default()
    
def add_label(image, text_cn, text_en):
    """在图像顶部加一条白色标题栏：有 Pillow 显示中文，否则用 OpenCV 显示英文。"""
    if image.ndim == 2:                 # 灰度/二值图只有 1 个通道，先转成 3 通道 BGR
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    h, w = image.shape[:2]
    bar_h = max(36, int(h * 0.06))     # 标题栏高度，随图高自适应
    bar = np.full((bar_h, w, 3), 255, dtype=np.uint8)   # 纯白底

    if HAVE_PIL:
        pil = Image.fromarray(cv2.cvtColor(bar, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(pil)
        font = load_font(int(bar_h * 0.62))
        draw.text((12, bar_h // 2), text_cn, font=font, fill=(0, 0, 0), anchor="lm")
        bar = cv2.cvtColor(np.asarray(pil), cv2.COLOR_RGB2BGR)
    else:
        cv2.putText(bar, text_en, (12, int(bar_h * 0.72)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2, cv2.LINE_AA)

    return np.vstack([bar, image])     # 标题栏叠在图像上方

# ---------- 1. 读图、灰度、二值反相 ----------
img = cv2.imread("test_3.jpg")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# 反相：让「线条」变成白色前景（形态学操作的是白色区域)
_, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))  # 3×3 方形结构元

erode = cv2.erode(bw, kernel)                             # 腐蚀：白区瘦一圈
dilate = cv2.dilate(bw, kernel)                           # 膨胀：白区胖一圈
opening = cv2.morphologyEx(bw, cv2.MORPH_OPEN, kernel)    # 开：去孤立白噪点
closing = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, kernel)   # 闭：填白区里的小黑洞、接断线

# ---------- 3. 每张图加标题（前 5 格为结果图，最后 1 格为原图参考） ----------
cells = [
    add_label(bw,      "二值反相 binary inv", "binary inv"),
    add_label(erode,   "腐蚀 erode",          "erode"),
    add_label(dilate,  "膨胀 dilate",         "dilate"),
    add_label(opening, "开运算 opening",      "opening"),
    add_label(closing, "闭运算 closing",      "closing"),
    add_label(img,     "原图 original(参考)", "original"),
]

# ---------- 4. 统一每格尺寸（按原图宽高比，避免拼接时大小不一） ----------
h0, w0 = bw.shape[:2]
label_h = max(36, int(h0 * 0.06))
cell_w = 420
cell_h = int(cell_w * h0 / w0) + label_h
cells = [cv2.resize(c, (cell_w, cell_h)) for c in cells]

# ---------- 5. 拼成 2 行 3 列 ----------
row1 = np.hstack(cells[0:3])     # 二值反相 | 腐蚀 | 膨胀
row2 = np.hstack(cells[3:6])     # 开运算   | 闭运算 | 原图
grid = np.vstack([row1, row2])

# ---------- 6. 保存并显示 ----------
cv2.imwrite("morphology_compare.jpg", grid)
cv2.namedWindow("morphology compare", cv2.WINDOW_NORMAL)
cv2.resizeWindow("morphology compare",
                 min(cell_w * 3, 1600),
                 min(cell_h * 2, 1000))
cv2.imshow("morphology compare", grid)
cv2.waitKey(0)
cv2.destroyAllWindows()

#cv2.namedWindow("binary inv", cv2.WINDOW_NORMAL)
#cv2.resizeWindow("binary inv", 300, 400)

#cv2.namedWindow("erode", cv2.WINDOW_NORMAL)
#cv2.resizeWindow("erode", 300, 400)

#cv2.namedWindow("dilate", cv2.WINDOW_NORMAL)
#cv2.resizeWindow("dilate", 300, 400)

#cv2.namedWindow("opening", cv2.WINDOW_NORMAL)
#cv2.resizeWindow("opening", 300, 400)

#cv2.namedWindow("closing", cv2.WINDOW_NORMAL)
#cv2.resizeWindow("closing", 300, 400)

#cv2.imshow("binary inv", bw)
#cv2.imshow("erode", erode)
#cv2.imshow("dilate", dilate)
#cv2.imshow("opening", opening)
#cv2.imshow("closing", closing)
#cv2.waitKey(0)
#cv2.destroyAllWindows()
