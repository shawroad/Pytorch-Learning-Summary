from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

img = Image.open("../img/python.jpg")
print(img)   # <PIL.JpegImagePlugin.JpegImageFile image mode=RGB size=478x260 at 0x28751B79080>
print(img.format)  # 图片的格式  JPEG
print(img.size)  # 注意，这省略的通道数   (478, 260)
print(img.mode)  # L为灰度图。RGB为彩色图 RGBA为加了透明通道  RGB
# img.show()  # 显示图片  在你默认显示图片的应用上进行显示

img_arr = np.array(img)
plt.imshow(img_arr)
plt.axis('off')  # 关闭坐标轴
plt.show()


# # 转换为灰度图
# gray = Image.open("../img/python.jpg").convert('L')
# gray.show()


# 读取不出图片会抛出IO异常
# try:
#     img = Image.open('../img/haha.jpg')
# except IOError:
#     print("图片读取异常")


# pillow读进来的图片不是矩阵，我们将图片转换为矩阵
img = Image.open('../img/python.jpg').convert('L')
arr = np.array(img)
print(arr.shape)   # (260, 478)
print(arr.dtype)   # uint8
print(arr)


# 存储图片
new_img = Image.fromarray(arr)   # 从array数组中还原图片
new_img.save('python_gray.jpg')


# 分离通道
img = Image.open('../img/python.jpg')
r, g, b = img.split()
img = Image.merge('RGB', (b, g, r))
img = img.copy()
img.show()


# ROI获取
img = Image.open('../img/python.jpg')
roi = img.crop((0, 0, 50, 50))  #(左上x，左上y，右下x，右下y)坐标
roi.show()



