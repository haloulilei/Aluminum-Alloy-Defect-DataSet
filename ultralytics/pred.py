# from ultralytics import YOLO
#
# if __name__=='__main__':
#     model = YOLO(r'H:\ultralytics10.24_1\runs\detect\train212\weights\best.pt')
#     model.val(data=r'H:\coco128\coco128\val\images\41.jpg')
from ultralytics import YOLO
from PIL import Image
import cv2

model = YOLO(r'H:\ultralytics10.24_1\runs\detect\train213\weights\best.pt')
# # 接受所有格式-image/dir/Path/URL/video/PIL/ndarray。0用于网络摄像头
# results = model.predict(source="0")
# results = model.predict(source="folder", show=True) # 展示预测结果

# from PIL
im1 = Image.open(r"H:\coco128\coco128\train\images\19.jpg")
results = model.predict(source=im1, save=True)  # 保存绘制的图像

# from ndarray
im2 = cv2.imread(r"H:\coco128\coco128\train\images\19.jpg")
results = model.predict(source=im2, save=True, save_txt=True)  # 将预测保存为标签

# from list of PIL/ndarray
results = model.predict(source=[im1, im2])
