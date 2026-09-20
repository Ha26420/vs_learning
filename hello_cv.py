import cv2

img = cv2.imread("test.jpg")
print("图片尺寸(高,宽,通道):", img.shape)
cv2.imshow("my first image", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
