# @Time : 2022/6/25 19:22
# @Author : yss
# @File : weak_label_generation.py
# @Software: PyCharm


from PIL import Image
import sys
import numpy as np
import torch.nn.functional as F
from collections import Counter
from torchvision import transforms as trn
import jieba
import random
import math
from scripts.resnet import resnet101
from scripts.resnet_utils import myResnet
import argparse
import torch.nn as nn
import cv2
import torch
from torch.autograd import Function
from torchvision import models, transforms
import json
import os

random.seed(2)
preprocess = trn.Compose([
    trn.ToTensor(),
    trn.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

])

# define the pathological words for Chinese segmentation tool
jieba.load_userdict(r"/devdata/Dataset_CT/data2077/all_medical_nouns.txt")

temp = {
    '颅底层面眦耳线层面': '枕大池	大枕大池 蝶鞍 上颌窦 颌窦 额窦 筛窦 蝶窦 前床突 眼球 眼眶 蝶骨  海绵窦 垂体窝 颞骨 颞弓 颞极 海马回 岩骨 枕骨 '
                 '颈内动脉 破裂孔 鼓室 外耳道 乳突 乳突气房 第四脑室 脑干 小脑 小脑半球 小脑溪 小脑蚓部 延髓 桥脑 脑桥 脑桥小脑角尺 小脑扁桃体'
                 '内枕骨粗隆 桥臂 颅骨 副鼻窦炎 鼻甲 鼻腔 鼻中隔',
    '鞍上池层面': '	额叶直回 额叶  纵裂池 外侧裂池 大脑中动脉 基底动脉 海马 颞叶 颞枕叶 鞍上池 环池 枕叶 四叠体池 中脑 中脑大脑脚'
             '侧脑室下角 脚间池 下丘 小脑蚓部',
    '第三脑室下部层面': '额叶 侧脑室前角 外侧裂池 岛叶 丘脑 侧脑室后角 侧脑室下角 四叠体池 枕叶 松果体 上丘 第三脑室 尾状核头 纵隔裂'
                '小脑蚓部 窦汇 枕叶 小脑幕 颞叶 外侧裂 脑中线 中线 额颞区',
    '第三脑室上部层面': '基底节区 基底节 大脑镰 胼胝体 胼胝体膝部 侧脑室前角 侧脑室后角 侧脑室下角 侧脑室三角 透明隔 内囊前肢 内囊后肢 中间帆池 '
                '视辐射 枕叶 颞枕叶 脉络丛 丘脑 穹窿部 穹窿柱 苍白球 壳核 颞叶 第三脑室 外侧裂池 尾状核头部 额叶 四叠体池'
                ' 松果体 岛叶 脑中线 中线 额颞区 ',
    '侧脑室体部层面': '侧脑室 纵裂池 额叶 透明隔 放射冠 颞叶 额颞区 侧脑室体部 上矢状窦 枕叶 颞枕叶 胼胝体 胼胝体压部 尾状核体部 胼胝体膝部 大脑镰 脑中线 中线',
    '侧脑室上部层面': '侧脑室 大脑镰 额叶 顶叶 顶部 顶颞部 脑沟 脑室 上矢状窦 枕叶 侧脑室体 胼胝体 顶枕沟 顶枕部  中央沟 脑中线 中线',
    '半卵圆中心层面': '	纵裂池 额叶 半卵圆中心 缘上回 角回 上矢状窦 楔前叶 顶叶 顶部 大脑镰 扣带回 脑沟 脑室 脑实质 脑中线 中线 额骨 额顶骨',
    '大脑皮质上部层面': '大脑镰 额上回 额中回 脑回 中央前沟 中央前回 中央沟 中央后回 中央后沟 上矢状窦 顶叶 顶部 中央旁小叶 放射冠 额叶 脑灰质 脑沟'}
temp_name = list(temp.keys())


def main():
    # Loading the dataset
    data = json.load(open('/home/ai/data/yss/data/long_annotations_clean/brain_ct_data_2048.json', 'r'))['annotations']
    root = '/home/ai/data/yss/data/png_process'
    net = resnet101(num_classes=2)
    # Loading the pretrained resnet101
    net.load_state_dict(torch.load("/home/ai/data/yss/data/dan_CQ500_resnet101.pth"), strict=False)
    net.cuda()
    net.eval()
    model = myResnet(net)
    model.cuda()
    model.eval()
    grad_cam = GradCam(model=net, feature_module=net.layer4, target_layer_names=["2"], use_cuda=True)
    slice_num = []
    count = 0
    frame_map = [[] for i in range(2076)]
    att_map = [[] for i in range(2076)]
    for info_index, info in enumerate(data):
        dir = os.path.join(root, str(info['image_id']))
        del_img_list = sorted(os.listdir(dir))[0:24]
        list_num = [i for i in range(len(del_img_list))]

        bbox_per_persion = []
        fc = np.zeros([24, 2048], dtype='float32')
        att = np.zeros([24, 14, 14, 2048], dtype='float32')
        for i_, i in enumerate(list_num):
            img_path = os.path.join(dir, del_img_list[i])
            print(i_, ':', img_path)
            I = np.array(Image.open(img_path).convert('RGB'))
            I = preprocess(I)
            with torch.no_grad():
                # Extract visual features by pretrained resnet101
                tmp_fc, tmp_att = model(I.cuda(), 14)
                name = str(info['image_id'])
                fc[i_] = tmp_fc.data.cpu().float().numpy()
                att[i_] = tmp_att.data.cpu().float().numpy()
            
            ########### Start: Weakly Labels for Frame Attention ###########
            # step1. Image-layer Matching: This process is early conducted by radioligists by matching 3 consecutive image slices to one standard anatomical layer. Resulting in the dataset package '/home/ai/data/yss/data/png_process'.
            # step2. Layer-term Matching: This process is also early conducted by radioligistsis, see the 'temp' dictionary in line 306.
            para_supervse = []
            for sen in info['caption'].split('。'):
                if not sen:
                    continue
                sen_temp = []
                # step3. Sentence-Layer Mapping: Use medical terms to bridge the anatomical layers and textual sentences
                for word in list(jieba.cut(sen.strip().replace(u'\n', ''), cut_all=False)):
                    for te in temp:
                        if word in temp[te].split():
                            sen_temp.append(temp_name.index(te))
                    if len(sen_temp):
                        break
                # step4. Layer Interval Determination: The 'sen_temp' list records the layer interval for current sentence, representing the relevant event segments.
                # step5. Image Interval Expansion: The layer interval is expanded threefold to determine the image interval for each sentence.
                sen_supervise = get_sen_supervise(sen_temp)
                para_supervse.append(sen_supervise)
            ########### End: Weakly Labels for Frame Attention ###########

            ########### Start: Weakly Labels for Spatial Attention ###########
            img = cv2.imread(img_path, 1)
            img = np.float32(img) / 255
            img = img[:, :, ::-1]
            input_img = preprocess_image(img)
            target_category = None
            grayscale_cam = grad_cam(input_img, target_category)

            grayscale_cam = cv2.resize(grayscale_cam, (img.shape[1], img.shape[0]))
            cam = show_cam_on_image(img, grayscale_cam)

            binarized_image = binarize(cam, percentage=0.15)
            img = binarized_image
            lower_red = np.array([160, 60, 60])
            upper_red = np.array([180, 255, 255])
            lower_red2 = np.array([0, 60, 60])
            upper_red2 = np.array([10, 255, 255])

            # change to hsv model
            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            mask_r = cv2.inRange(hsv, lower_red, upper_red)
            mask_r2 = cv2.inRange(hsv, lower_red2, upper_red2)
            mask = mask_r + mask_r2

            # generating bbox
            try:
                x, y, width, height = compute_bounding_box(mask)
                origin_img = cv2.resize(cv2.imread(img_path, 1), (512, 512))
                # draw bounding box
                bounded_binarized_image = draw_bounding_box(x=x, y=y, width=width, height=height, image=origin_img)
            except:
                x, y, width, height = [0, 0, 0, 0]

            att2_target = np.zeros((14, 14))
            x_left = math.floor(float(x) / 512 * 14)
            x_right = math.ceil(float(x + width) / 512 * 14)
            y_top = math.floor(float(y) / 512 * 14)
            y_bottom = math.ceil(float(y + height) / 512 * 14)
            att2_target[y_top: y_bottom, x_left: x_right] = 1
            bbox_per_persion.append(att2_target.tolist())
            ########### End: Weakly Labels for Spatial Attention ###########


        ########### Append the two types of weak labels into lists ###########
        frame_map[info['image_id']] = para_supervse
        att_map[info['image_id']] = bbox_per_persion

    ########### Save the two types of weak labels into json files ###########
    with open('/home/ai/data/yss/data/long_annotations_clean/frame_map.json', 'w') as f:
        json.dump(frame_map, f)

    with open('/home/ai/data/yss/data/long_annotations_clean/att_map.json', 'w') as f:
        json.dump(att_map, f)




class FeatureExtractor():
    """ Class for extracting activations and
    registering gradients from targetted intermediate layers """

    def __init__(self, model, target_layers):
        self.model = model
        self.target_layers = target_layers
        self.gradients = []

    def save_gradient(self, grad):
        self.gradients.append(grad)

    def __call__(self, x):
        outputs = []
        self.gradients = []
        for name, module in self.model._modules.items():
            x = module(x)
            if name in self.target_layers:
                x.register_hook(self.save_gradient)
                outputs += [x]
        return outputs, x


class ModelOutputs():
    """ Class for making a forward pass, and getting:
    1. The network output.
    2. Activations from intermeddiate targetted layers.
    3. Gradients from intermeddiate targetted layers. """

    def __init__(self, model, feature_module, target_layers):
        self.model = model
        self.feature_module = feature_module
        self.feature_extractor = FeatureExtractor(self.feature_module, target_layers)

    def get_gradients(self):
        return self.feature_extractor.gradients

    def __call__(self, x):
        target_activations = []
        for name, module in self.model._modules.items():
            if module == self.feature_module:
                target_activations, x = self.feature_extractor(x)
            elif "avgpool" in name.lower():
                x = module(x)
                x = x.view(x.size(0), -1)
            else:
                x = module(x)

        return target_activations, x


def preprocess_image(img):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    preprocessing = transforms.Compose([
        transforms.ToTensor(),
        normalize,
    ])
    return preprocessing(img.copy()).unsqueeze(0)


def show_cam_on_image(img, mask):
    heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255
    cam = heatmap + np.float32(img)
    cam = cam / np.max(cam)
    return np.uint8(255 * cam)


class GradCam:
    def __init__(self, model, feature_module, target_layer_names, use_cuda):
        self.model = model
        self.feature_module = feature_module
        self.model.eval()
        self.cuda = use_cuda
        if self.cuda:
            self.model = model.cuda()

        self.extractor = ModelOutputs(self.model, self.feature_module, target_layer_names)

    def forward(self, input_img):
        return self.model(input_img)

    def __call__(self, input_img, target_category=None):
        if self.cuda:
            input_img = input_img.cuda()

        features, output = self.extractor(input_img)

        if target_category == None:
            target_category = np.argmax(output.cpu().data.numpy())

        one_hot = np.zeros((1, output.size()[-1]), dtype=np.float32)
        one_hot[0][target_category] = 1
        one_hot = torch.from_numpy(one_hot).requires_grad_(True)
        if self.cuda:
            one_hot = one_hot.cuda()

        one_hot = torch.sum(one_hot * output)

        self.feature_module.zero_grad()
        self.model.zero_grad()
        one_hot.backward(retain_graph=True)

        grads_val = self.extractor.get_gradients()[-1].cpu().data.numpy()

        target = features[-1]
        target = target.cpu().data.numpy()[0, :]

        weights = np.mean(grads_val, axis=(2, 3))[0, :]
        cam = np.zeros(target.shape[1:], dtype=np.float32)

        for i, w in enumerate(weights):
            cam += w * target[i, :, :]

        cam = np.maximum(cam, 0)
        cam = cv2.resize(cam, input_img.shape[2:])
        cam = cam - np.min(cam)
        cam = cam / np.max(cam)
        return cam


def binarize(image, percentage=0.15):
    # Retrieve the maximum intensity
    maximum_intensity = image.max()

    # Define the threshold as a percentage of the maximum intensity
    threshold_value = percentage * maximum_intensity

    # Display some information if DEBUG is enabled.
    # print(f'Maximum intensity: {maximum_intensity} \nThreshold value: {threshold_value}')

    # Binarize the image at the threshold value
    _, binarized_image = cv2.threshold(src=image, thresh=threshold_value, maxval=255, type=cv2.THRESH_BINARY)

    return binarized_image


def draw_bounding_box(x, y, width, height, image, color=(255, 0, 0)):
    # Draw the bounding box!
    bounded_image = cv2.rectangle(img=image, \
                                  pt1=(x, y),  # Top left coordinate\
                                  pt2=(x + width, y + height),  # Bottom right coordinate\
                                  color=color,  # Color of bounding box\
                                  thickness=3  # thickness of bounding box line\
                                  )

    return bounded_image


def compute_bounding_box(image):
    # Finds the largest contiguous block in the image
    contours, hierarchy = cv2.findContours(image=image, \
                                           mode=cv2.RETR_TREE, \
                                           method=cv2.CHAIN_APPROX_SIMPLE \
                                           )

    # If there are multiple contiguous blocks, select the largest
    if len(contours) > 0:
        contour = max(contours, key=cv2.contourArea)
    else:
        contour = contours[0]

    # Determine the coordinates of the minimum bounding box around the contour
    x, y, width, height = cv2.boundingRect(contour)

    return x, y, width, height


def get_sen_supervise(sen_supervise):
    """
    :param sen_supervise: layer index
    :return: a list with 24 elements
    """
    super_info = [0 for i in range(24)]
    for i in sen_supervise:
        for j in range(3):
            super_info[i * 3 + j] = 1

    return super_info


if __name__ == '__main__':
    main()