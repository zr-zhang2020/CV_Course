# DataAugmentation
## Example

## 1.image_crop
```
      import cv2
      from week1homework import *
      
      img=image_crop(img_ori,200,300,100,200)
      my_show(img)

```

## 2.color_shift
```
      import cv2
      from week1homework import *
      
      my_show(color_shift(img_ori,-10,0,30))
```

## 3.rotation

```
      import cv2
      from week1homework import *
      
      img_rotate = rotation(img_ori,60,0.7)
      my_show(img_rotate)
 ```

## 4.perspective_transform
```
      import cv2
      from week1homework import *
      
      pts1 = [[0,0],[0,500],[500,0],[500,500]]  # 源点创建
      pts2 = [[5,19],[19,460],[460,9],[410,320]]  # 目标点创建
      my_show(perspective_transform(img_ori,pts1,pts2))
```
