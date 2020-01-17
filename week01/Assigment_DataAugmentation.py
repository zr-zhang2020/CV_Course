import cv2
import numpy as np
import matplotlib.pyplot as plt

def image_crop(img,x1,x2,y1,y2): # you code here
    img_crop = img[x1:x2,y1:y2]

    return img_crop

def color_shift(img,R_change=0,G_change=0,B_change=0): # you code here
        B, G, R = cv2.split(img)
        #处理B通道
        if B_change>=0:
            b_maxLimit= 255 - B_change #大于等于255就设置为255
            B[B > b_maxLimit] = 255
            B[B <= b_maxLimit] = (B_change + B[B <= b_maxLimit]).astype(img.dtype)
        else:
            b_minLimit= 0-B_change  #小于0就设置为0
            B[B < b_minLimit] = 0
            B[B >= b_minLimit] = (B[B >= b_minLimit] + B_change).astype(img.dtype)
        #处理G通道
        if G_change >= 0:
            g_maxLimit = 255 - G_change
            G[G > g_maxLimit] = 255
            G[G <= g_maxLimit] = (G_change + G[B <= g_maxLimit]).astype(img.dtype)
        else:
            g_minLimit = 0 - G_change  # 小于0就设置为0
            G[G < g_minLimit] = 0
            G[G >= g_minLimit] = (G[G >= g_minLimit] + G_change).astype(img.dtype)
        #处理R通道
        if R_change>=0:
            r_maxLimit= 255 - R_change
            R[R > r_maxLimit] = 255
            R[R<= r_maxLimit] = (R_change + R[R <= r_maxLimit]).astype(img.dtype)
        else:
            r_minLimit= 0-R_change  #小于0就设置为0
            R[R < r_minLimit] = 0
            R[R >= r_minLimit] = (R[R >= r_minLimit] + R_change).astype(img.dtype)

        return cv2.merge((B, G, R))


def rotation(img,angle,scale): # you code here
    M = cv2.getRotationMatrix2D((img.shape[0] / 2, img.shape[1] / 2), angle,scale)  # center, angle, scale
    img_rotate = cv2.warpAffine(img, M, (img.shape[1], img.shape[0]))
    return img_rotate

def perspective_transform(img,srcPts,desPts): # you code here

    rows, cols, ch = img.shape
    pts1 = np.float32([[0, 0], [0, 500], [500, 0], [500, 500]])  # 源点创建
    pts2 = np.float32([[5, 19], [19, 460], [460, 9], [410, 320]])  # 目标点创建

    M = cv2.getPerspectiveTransform(pts1, pts2)  # 计算得到单应性矩阵
    img_warp = cv2.warpPerspective(img, M, (rows, cols))  # 通过得到的矩阵对图片进行变换
    return img_warp

def my_show(img, size=(3,3)):
    plt.figure(figsize=size)
    plt.imshow(cv2.cvtColor(img,cv2.COLOR_BGR2RGB))
    plt.show()