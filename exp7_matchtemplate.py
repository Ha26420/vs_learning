import cv2

img = cv2.imread("test_5.jpg")    # 大图（场景）

# ★ 关键：直接从场景图里抠模板（切片 = 先 y 后 x！）
x, y, w, h = 620, 420, 160, 120        # ← 换成目标在大图里的真实位置和大小
template = img[y:y+h, x:x+w].copy() # 小图（模板）

h, w = template.shape[:2] # 模板的高、宽

# 归一化相关系数匹配：结果 res 是一张「相似度图」，值越接近 1 越像
res = cv2.matchTemplate(img, template, cv2.TM_CCOEFF_NORMED)

min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res) # 找到最大值位置

# ★ 工程习惯：分数不够就报「未找到」，不硬画框
if max_val < 0.8:
    print(f"相似度 {max_val:.2f}，低于 0.8，判定：未找到可信匹配")
else:
    top_left = max_loc # 左上角坐标
    # 在大图上画框
    bottom_right = (top_left[0] + w, top_left[1] + h)
    cv2.rectangle(img, top_left, bottom_right, (0, 0, 255), 3)
    print("最高相似度 =", round(max_val, 3)) # 1.0 表示完美匹配


cv2.namedWindow("template", cv2.WINDOW_NORMAL)
cv2.resizeWindow("template", 120, 120)

cv2.namedWindow("result", cv2.WINDOW_NORMAL)
cv2.resizeWindow("result", 600, 800)

cv2.namedWindow("similarity", cv2.WINDOW_NORMAL)
cv2.resizeWindow("similarity", 600, 800)

cv2.imshow("template", template)
cv2.imshow("result", img)
cv2.imshow("similarity", res)
cv2.waitKey(0)
cv2.destroyAllWindows()