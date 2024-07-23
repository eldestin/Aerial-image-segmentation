import numpy as np
import time
import random
import torch
import torchvision.transforms as transforms
import albumentations as albu

def reseed(seed):
    random.seed(seed)
    torch.manual_seed(seed)

CLASSES = ('Building', 'Road', 'Tree', 'LowVeg', 'Moving_Car',  'Static_Car', 'Human', 'Clutter')
Building = np.array([128, 0, 0])  # label 0
Road = np.array([128, 64, 128]) # label 1
Tree = np.array([0, 128, 0]) # label 2
LowVeg = np.array([128, 128, 0]) # label 3
Moving_Car = np.array([64, 0, 128]) # label 4
Static_Car = np.array([192, 0, 192]) # label 5
Human = np.array([64, 64, 0]) # label 6
Clutter = np.array([0, 0, 0]) # label 7
Boundary = np.array([255, 255, 255]) # label 255
num_classes = 8

def label2rgb(mask):
    h, w = mask.shape[0], mask.shape[1]
    mask_rgb = np.zeros(shape=(h, w, 3), dtype=np.uint8)
    mask_convert = mask[np.newaxis, :, :]
    mask_rgb[np.all(mask_convert == 0, axis=0)] = Building
    mask_rgb[np.all(mask_convert == 1, axis=0)] = Road
    mask_rgb[np.all(mask_convert == 2, axis=0)] = Tree
    mask_rgb[np.all(mask_convert == 3, axis=0)] = LowVeg
    mask_rgb[np.all(mask_convert == 4, axis=0)] = Moving_Car
    mask_rgb[np.all(mask_convert == 5, axis=0)] = Static_Car
    mask_rgb[np.all(mask_convert == 6, axis=0)] = Human
    mask_rgb[np.all(mask_convert == 7, axis=0)] = Clutter
    mask_rgb[np.all(mask_convert == 255, axis=0)] = Boundary
    return mask_rgb


def rgb2label(label):
    label_seg = np.zeros(label.shape[:2], dtype=np.uint8)
    label_seg[np.all(label == Building, axis=-1)] = 0
    label_seg[np.all(label == Road, axis=-1)] = 1
    label_seg[np.all(label == Tree, axis=-1)] = 2
    label_seg[np.all(label == LowVeg, axis=-1)] = 3
    label_seg[np.all(label == Moving_Car, axis=-1)] = 4
    label_seg[np.all(label == Static_Car, axis=-1)] = 5
    label_seg[np.all(label == Human, axis=-1)] = 6
    label_seg[np.all(label == Clutter, axis=-1)] = 7
    label_seg[np.all(label == Boundary, axis=-1)] = 255
    return label_seg

# video transform
im_mean = (124, 116, 104)

im_normalization = transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )

inv_im_trans = transforms.Normalize(
                mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225],
                std=[1/0.229, 1/0.224, 1/0.225])

# static transform
def get_training_transform():
    train_transform = [
        albu.HorizontalFlip(p=0.5),
        albu.VerticalFlip(p=0.5),
        albu.RandomBrightnessContrast(brightness_limit=0.25, contrast_limit=0.25, p=0.25),
        albu.Normalize()
    ]
    return albu.Compose(train_transform)


def train_aug(img, mask):
    # crop_aug = SmartCropV1(crop_size=768, max_ratio=0.75, ignore_index=255, nopad=False)
    # img, mask = crop_aug(img, mask)
    img, mask = np.array(img), np.array(mask)
    aug = get_training_transform()(image=img.copy(), mask=mask.copy())
    img, mask = aug['image'], aug['mask']
    return img, mask


def get_val_transform():
    val_transform = [
        albu.Normalize()
    ]
    return albu.Compose(val_transform)


def val_aug(img, mask):
    img, mask = np.array(img), np.array(mask)
    aug = get_val_transform()(image=img.copy(), mask=mask.copy())
    img, mask = aug['image'], aug['mask']
    return img, mask

class Evaluator(object):
    def __init__(self, num_class):
        self.num_class = num_class
        self.confusion_matrix = np.zeros((self.num_class,) * 2)
        self.eps = 1e-8

    def get_tp_fp_tn_fn(self):
        tp = np.diag(self.confusion_matrix)
        fp = self.confusion_matrix.sum(axis=0) - np.diag(self.confusion_matrix)
        fn = self.confusion_matrix.sum(axis=1) - np.diag(self.confusion_matrix)
        tn = np.diag(self.confusion_matrix).sum() - np.diag(self.confusion_matrix)
        return tp, fp, tn, fn

    def Precision(self):
        tp, fp, tn, fn = self.get_tp_fp_tn_fn()
        precision = tp / (tp + fp)
        return precision

    def Recall(self):
        tp, fp, tn, fn = self.get_tp_fp_tn_fn()
        recall = tp / (tp + fn)
        return recall

    def F1(self):
        tp, fp, tn, fn = self.get_tp_fp_tn_fn()
        Precision = tp / (tp + fp)
        Recall = tp / (tp + fn)
        F1 = (2.0 * Precision * Recall) / (Precision + Recall)
        return F1

    def OA(self):
        OA = np.diag(self.confusion_matrix).sum() / (self.confusion_matrix.sum() + self.eps)
        return OA

    def Intersection_over_Union(self):
        tp, fp, tn, fn = self.get_tp_fp_tn_fn()
        IoU = tp / (tp + fn + fp)
        return IoU

    def Dice(self):
        tp, fp, tn, fn = self.get_tp_fp_tn_fn()
        Dice = 2 * tp / ((tp + fp) + (tp + fn))
        return Dice

    def Pixel_Accuracy_Class(self):
        #         TP                                  TP+FP
        Acc = np.diag(self.confusion_matrix) / (self.confusion_matrix.sum(axis=0) + self.eps)
        return Acc

    def Frequency_Weighted_Intersection_over_Union(self):
        freq = np.sum(self.confusion_matrix, axis=1) / (np.sum(self.confusion_matrix) + self.eps)
        iou = self.Intersection_over_Union()
        FWIoU = (freq[freq > 0] * iou[freq > 0]).sum()
        return FWIoU

    def _generate_matrix(self, gt_image, pre_image):
        mask = (gt_image >= 0) & (gt_image < self.num_class)
        label = self.num_class * gt_image[mask].astype('int') + pre_image[mask]
        count = np.bincount(label, minlength=self.num_class ** 2)
        confusion_matrix = count.reshape(self.num_class, self.num_class)
        return confusion_matrix

    def add_batch(self, gt_image, pre_image):
        assert gt_image.shape == pre_image.shape, 'pre_image shape {}, gt_image shape {}'.format(pre_image.shape,
                                                                                                 gt_image.shape)
        self.confusion_matrix += self._generate_matrix(gt_image, pre_image)

    def reset(self):
        self.confusion_matrix = np.zeros((self.num_class,) * 2)
