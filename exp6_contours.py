import cv2

img = cv2.imread("test_3.jpg")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# ① 二值化：INV 让黑色物体变成白色前景
_, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

# ② 开运算去噪（实验 5 直接复用）
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))  # 5×5 方形结构元
bw = cv2.morphologyEx(bw, cv2.MORPH_OPEN, kernel)

# ③ 找轮廓（RETR_EXTERNAL：只要最外层轮廓）
contours, _ = cv2.findContours(bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
print("找到轮廓数：", len(contours))

# ④ 逐个量面积、过滤噪声、框出来并标注
for i, c in enumerate(contours):
    area = cv2.contourArea(c)
    if area < 500:
        continue
    x, y, w, h = cv2.boundingRect(c)
    cv2.rectangle(img, (x,y), (x + w, y + h), (0, 0, 255), 2)
    cv2.putText(img, f"#{i} area={int(area)}", (x, y - 8), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    cv2.imshow("binary", bw)
    cv2.imshow("result", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    