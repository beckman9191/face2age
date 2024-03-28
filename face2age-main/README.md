模型在model文件夹里，有2个，我感觉没什么区别，用哪个都可以

centerface.py是调用模型

img_detect.py是生成标注的图片，存在result/detect

img_crop.py是截取部分画面（大头照），存在result/crop


crop版本应该能更好的辅助训练

detect版本用于成果展示，在每一个人对应的框上标注年龄

parameters:

confindency 检查模型的置信度

padding 向外扩展的大小（为了包含整个头部而不止是框中的部分）