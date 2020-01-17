import cv2
import numpy as np
import matplotlib.pyplot as plt
from week1homework import *

# 图片读取用cv2.imread
img_ori = cv2.imread('lenna.jpg',1)
print(img_ori.shape)

#cv2.imshow可以打印图片
# cv2.imshow('lenna_photo',img_ori)
# key = cv2.waitKey(0)

#plt也可以打印图片，但要注意是否为灰度图，以及通道顺序
# plt.figure(figsize=(2,2))
# plt.imshow(img_ori,cmap='gray')
# plt.show()

#subplot子图的使用，感兴趣可以了解一下
# plt.subplot(121)#1行2列，这是第1个
# plt.imshow(img_ori)
# plt.subplot(122)#1行2列，这是第2个，有兴趣可以改着玩
# plt.imshow(cv2.cvtColor(img_ori,cv2.COLOR_BGR2RGB))
# plt.show()


#image_crop使用
# img=image_crop(img_ori,200,300,100,200)
# my_show(img)

## 图像通道的分别处理
# B,G,R = cv2.split(img_ori)
# cv2.imshow('B',B)
# cv2.imshow('G',G)
# cv2.imshow('R',R)
# key = cv2.waitKey(0)

#用plt显示三通道
# plt.subplot(131)
# plt.imshow(B,cmap='gray')
# plt.subplot(132)
# plt.imshow(G,cmap='gray')
# plt.subplot(133)
# plt.imshow(R,cmap='gray')
# plt.show()

#测试cv里面用的是bgr
# B,G,R = cv2.split(img_ori)  #cv2里面的是bgr
# img_rgb = cv2.merge((R,G,B))
# my_show(img_rgb)


# 对不同通道进行函数关系映射
# def img_cooler(img, b_increase, r_decrease):
#     B, G, R = cv2.split(img)
#     b_lim = 255 - b_increase   #大于255就设置255
#     B[B > b_lim] = 255
#     B[B <= b_lim] = (b_increase + B[B <= b_lim]).astype(img.dtype)
#
#     r_lim = r_decrease  #小于0就设置为0
#     R[R < r_lim] = 0
#     R[R >= r_lim] = (R[R >= r_lim] - r_decrease).astype(img.dtype)
#     return cv2.merge((B, G, R))
#
# cooler_image = img_cooler(img_ori,30,10)
# my_show(cooler_image)

#color_shift函数使用
# my_show(color_shift(img_ori,-10,0,30))

# Look Up Table效率很高  gamma变换
# def adjust_gamma(img, gamma=1.0):
#     invGamma = 1.0/gamma
#     table = []
#     for i in range(256):
#         table.append(((i/255.0)**invGamma)*255)
#     table = np.array(table).astype('uint8')
#     return cv2.LUT(img,table)
#
# #直方图均衡
# img_dark = cv2.imread('dark.jpg',1)
# my_show(img_dark,size=(8,8))
# img_brighter = adjust_gamma(img_dark,2)
# my_show(img_brighter,size=(8,8))
#
# plt.subplot(121)
# plt.hist(img_dark.flatten(),256,[0,256],color='r')
# plt.subplot(122)
# plt.hist(img_brighter.flatten(),256,[0,256],color='b')
# plt.show()
#
# img_yuv = cv2.cvtColor(img_dark,cv2.COLOR_BGR2YUV) # BGR转YUV
# img_yuv[:,:,0] = cv2.equalizeHist(img_yuv[:,:,0]) # 对Y通道单独进行调整
# img_output = cv2.cvtColor(img_yuv,cv2.COLOR_YUV2BGR) # YUV转BGR
# my_show(img_output,size=(8,8))
#
# plt.subplot(131)
# plt.hist(img_dark.flatten(),256,[0,256],color='r')
# plt.subplot(132)
# plt.hist(img_brighter.flatten(),256,[0,256],color='b')
# plt.subplot(133)
# plt.hist(img_output.flatten(),256,[0,256],color='g')
# plt.show()

# rotation函数使用
# img_rotate = rotation(img_ori,60,0.7)
# my_show(img_rotate)

# Affine Transform 失去保角性，保持平行，需要三对点
# img=img_ori
# rows, cols, ch = img.shape
# pts1 = np.float32([[0, 0], [cols - 1, 0], [0, rows - 1]])
# pts2 = np.float32([[cols * 0.2, rows * 0.1], [cols * 0.9, rows * 0.2], [cols * 0.1, rows * 0.9]])
#
# M = cv2.getAffineTransform(pts1, pts2)
# dst = cv2.warpAffine(img, M, (cols, rows))
#
# cv2.imshow('affine lenna', dst)
# key = cv2.waitKey(0)

## perspective transform函数使用
pts1 = [[0,0],[0,500],[500,0],[500,500]]  # 源点创建
pts2 = [[5,19],[19,460],[460,9],[410,320]]  # 目标点创建
my_show(perspective_transform(img_ori,pts1,pts2))