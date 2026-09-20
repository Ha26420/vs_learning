import numpy as np


a = np.zeros((3,3))
print(a)


b = np.array([[1, 2, 3], 
              [4, 5, 6]])
print("形状：",b.shape)
print("第0行第一列:",b[0,1])
print("切片:",b[0:2, 1:3])
print("平均值:",b.mean())
