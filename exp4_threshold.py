import cv2
import matplotlib.pyplot as plt

img = cv2.imread("test_3.jpg")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

_, thresh_manual = cv2.threshold(gray, 30, 255, cv2.THRESH_BINARY)


ret, thresh_otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
print("Otsu 自动找到的阈值 =", ret)

thresh_adap = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 33, 2)

cv2.namedWindow("manual 30", cv2.WINDOW_NORMAL)
cv2.resizeWindow("manual 30", 600, 800)

cv2.namedWindow("otsu", cv2.WINDOW_NORMAL)
cv2.resizeWindow("otsu", 600, 800)

cv2.namedWindow("adaptive", cv2.WINDOW_NORMAL)
cv2.resizeWindow("adaptive", 600, 800)

cv2.imshow("manual 30", thresh_manual)
cv2.imshow("otsu", thresh_otsu)
cv2.imshow("adaptive", thresh_adap)
cv2.waitKey(0)
cv2.destroyAllWindows()

plt.hist(gray.ravel(), 256, [0, 256])
plt.axvline(ret, color='r', linestyle='--', label=f'Otsu = {int(ret)}')
plt.legend()
plt.show()
