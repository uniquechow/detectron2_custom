
from PIL import Image, ImageFont,ImageDraw
import cv2
import numpy as np
import torch

def is_ascii(str=''):
    # Is string composed of all ASCII (no UTF) characters?
    return len(str.encode().decode('ascii', 'ignore')) == len(str)

class Colors:
    # Ultralytics color palette https://ultralytics.com/
    def __init__(self):
        # hex = matplotlib.colors.TABLEAU_COLORS.values()
        #hex = ('FF3838', 'FF9D97', 'FF701F', 'FFB21D', 'CFD231', '48F90A', '92CC17', '3DDB86', '1A9334', '00D4BB',
        #     '2C99A8', '00C2FF', '344593', '6473FF', '0018EC', '8438FF', '520085', 'CB38FF', 'FF95C8', 'FF37C7')
        hex = ('00FFFF', '00FF00', 'FF0000')  # 用于画图颜色
        # hex = ('0000FF', '006400', 'FF0000')
        self.palette = [self.hex2rgb('#' + c) for c in hex]
        self.n = len(self.palette)

    def __call__(self, i, bgr=False):
        c = self.palette[int(i) % self.n]
        return (c[2], c[1], c[0]) if bgr else c

    @staticmethod
    def hex2rgb(h):  # rgb order (PIL)
        return tuple(int(h[1 + i:1 + i + 2], 16) for i in (0, 2, 4))

def plot_one_box(box, im, color=(128, 128, 128), txt_color=(255, 255, 255), label=None, line_width=3, use_pil=False, label_s =None):
    # Plots one xyxy box on image im with label
    """一般会用在detect.py中在nms之后变量每一个预测框，再将每个预测框画在原图上
       使用opencv在原图im上画一个bounding box
       :params x: 预测得到的bounding box  [x1 y1 x2 y2]
       :params im: 原图 要将bounding box画在这个图上  array
       :params color: bounding box线的颜色
       :params labels: 标签上的框框信息  类别 + score
       :params line_thickness: bounding box的线宽
       """
    # 检查内存是否连续
    assert im.data.contiguous, 'Image not contiguous. Apply np.ascontiguousarray(im) to plot_on_box() input image.'
    lw = line_width or max(int(min(im.size) / 200), 2)  # line width

    if use_pil or not is_ascii(label):  # use PIL
        im = Image.fromarray(im)
        draw = ImageDraw.Draw(im)
        draw.rectangle(box, width=lw + 1, outline=color)  # plot
        if label:
            font = ImageFont.truetype("Arial.ttf", size=max(round(max(im.size) / 40), 12))
            txt_width, txt_height = font.getsize(label)
            draw.rectangle([box[0], box[1] - txt_height + 4, box[0] + txt_width, box[1]], fill=color)
            draw.text((box[0], box[1] - txt_height + 1), label, fill=txt_color, font=font)
        return np.asarray(im)
    else:  # use OpenCV, 不支持中文 no-ascii
        # c1 = (x1, y1)=矩形框左上角  c2 = (x2, y2) = 矩形框右下角
        c1, c2 = (int(box[0]), int(box[1])), (int(box[2]), int(box[3]))
        # cv2.rectangle: 在im上画出框框   c1: start_point(x1, y1)  c2: end_point(x2, y2)
        # 注意: 这里的c1+c2可以是左上角+右下角  也可以是左下角+右上角都可以
        cv2.rectangle(im, c1, c2, color, thickness=lw, lineType=cv2.LINE_AA)
        # 如果label不为空还要在框框上面显示标签label + score
        if label:
            tf = max(lw - 1, 1)  # font thickness， label字体的线宽
            # cv2.getTextSize: 根据输入的label信息计算文本字符串的宽度和高度
            # 0: 文字字体类型  fontScale: 字体缩放系数  thickness: 字体笔画线宽
            # 返回retval 字体的宽高 (width, height), baseLine 相对于最底端文本的 y 坐标
            txt_width, txt_height = cv2.getTextSize(label, 0, fontScale=lw / 3, thickness=tf)[0]
            c2 = c1[0] + txt_width, c1[1] - txt_height - 3
            # 同上面一样是个画框的步骤  但是线宽thickness=-1表示整个矩形都填充color颜色
            if not label == 'orange':
                cv2.rectangle(im, c1, c2, color, -1, cv2.LINE_AA)  # filled
                # cv2.putText: 在图片上写文本 这里是在上面这个矩形框里写label + score文本
                # (c1[0], c1[1] - 2)文本左下角坐标  0: 文字样式  fontScale: 字体缩放系数
                # txt_color: 文字颜色  thickness: tf字体笔画线宽     lineType: 线样式
                # cv2.putText(img,‘OpenCV’,(50,200), font, 3,(0,255,255),5,cv2.LINE_AA)
                test_label = label
                cv2.putText(im, label, (c1[0], c1[1] - 2), 0, lw / 3, txt_color, thickness=tf, lineType=cv2.LINE_AA)
                labels = label_s
            # cv2.putText(im, label_s, (int(im.shape[0]/4), int(im.shape[1]/4)), 0, lw/3, txt_color, thickness=tf, lineType=cv2.LINE_AA)
        return im


# detect on image
# prediction , draw_instance error ,now 3.30
def on_image(image_path, predictor, names, line_thickness, ):
    colors = Colors()
    im = cv2.imread(image_path)

    outputs = predictor(im)['instances'].to('cpu')
    if len(outputs):
        outputs_dict = outputs._fields
        boxes, scores, classes = outputs_dict['pred_boxes'].tensor, outputs_dict['scores'].view(-1, 1), outputs_dict[
            'pred_classes'].view(-1, 1)
        predtensor = torch.cat((boxes, scores, classes), dim=1)

        for *xyxy, conf, cls in predtensor:
            c = int(cls)  # integer class
            label = (f'{names[c]} {conf:.2f}')
            im0 = plot_one_box(xyxy, im, label=label, color=colors(c, True), line_width=line_thickness)
            # plot_one_box(xyxy, im, label='hello')

        cv2.imshow('win1', im0)
        k = cv2.waitKey(0)
        if k == '27' or k == ord('q'):
            cv2.destroyAllWindows()


