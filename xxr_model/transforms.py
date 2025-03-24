import cv2
import numpy as np
from PIL import Image
from torchvision import transforms
import torchvision.transforms.functional as TF
import random

class FOVExtraction(object):
    
    def __call__(self, img, tol=7):
        img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR) # PIL to cv
        
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        mask = gray_img > tol
        img_blue = img[:, :, 0][np.ix_(mask.any(1), mask.any(0))]
        img_green = img[:, :, 1][np.ix_(mask.any(1), mask.any(0))]
        img_red = img[:, :, 2][np.ix_(mask.any(1), mask.any(0))]
        img = np.stack([img_blue, img_green, img_red], axis=-1)
            
        img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)) # cv to PIL
        
        return img


class CircleCrop(object):

    def __call__(self, img):
        img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR) # PIL to cv

        height, width, _ = img.shape

        x = int(width/2)
        y = int(height/2)
        r = np.amin((x, y))

        circle_img = np.zeros((height, width), np.uint8)
        cv2.circle(circle_img, (x, y), int(r), 1, thickness=-1)
        img = cv2.bitwise_and(img, img, mask=circle_img)

        img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)) # cv to PIL
        
        return img
    

class RandomHorizontalFlipDual(object):

    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, images):
        img_left = images[0]
        img_right = images[1]
        
        transformed_img_left = img_left
        transformed_img_right = img_right
        
        if random.random() < self.p:
            transformed_img_left = TF.hflip(img_right)
            transformed_img_right = TF.hflip(img_left)
        
        return transformed_img_left, transformed_img_right
    

class RandomVerticalFlipDual(object):

    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, images):
        img_left = images[0]
        img_right = images[1]
        
        if random.random() < self.p:
            img_left = TF.vflip(img_left)
            img_right = TF.vflip(img_right)
        
        return img_left, img_right
    

class RandomRotationDual(object):

    def __init__(self, degrees=30):
        self.low = -degrees
        self.high = degrees

    def __call__(self, images):
        img_left = images[0]
        img_right = images[1]
        
        degrees_left = random.randint(self.low, self.high)
        img_left = TF.rotate(img_left, degrees_left)

        degrees_right = random.randint(self.low, self.high)
        img_right = TF.rotate(img_right, degrees_right)
        
        return img_left, img_right
    

class ColorJitterDual(object):

    def __init__(self, brightness=0, contrast=0, saturation=0, hue=0):
        self.brightness_range = [max(0, 1 - brightness), 1 + brightness]
        self.contrast_range = [max(0, 1 - contrast), 1 + contrast]
        self.saturation_range = [max(0, 1 - saturation), 1 + saturation]
        self.hue_range = [-hue, hue]
        self.transform = transforms.ColorJitter()

    def __call__(self, images):
        img_left = images[0]
        img_right = images[1]
        
        _, brightness, contrast, saturation, hue = self.transform.get_params(brightness=self.brightness_range, contrast=self.contrast_range, saturation=self.saturation_range, hue=self.hue_range)

        img_left = TF.adjust_brightness(img_left, brightness)
        img_left = TF.adjust_contrast(img_left, contrast)
        img_left = TF.adjust_saturation(img_left, saturation)
        img_left = TF.adjust_hue(img_left, hue)
        
        img_right = TF.adjust_brightness(img_right, brightness)
        img_right = TF.adjust_contrast(img_right, contrast)
        img_right = TF.adjust_saturation(img_right, saturation)
        img_right = TF.adjust_hue(img_right, hue)
        
        return img_left, img_right
    

class ToTensorDual(object):

    def __call__(self, images):
        img_left = images[0]
        img_right = images[1]
        
        img_left = TF.to_tensor(img_left)
        img_right = TF.to_tensor(img_right)
        
        return img_left, img_right


class NormalizeDual(object):
    
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, images):
        img_left = images[0]
        img_right = images[1]
        
        img_left = TF.normalize(img_left, self.mean, self.std)
        img_right = TF.normalize(img_right, self.mean, self.std)
        
        return img_left, img_right
    

class RandomCropDual(object):
    
    def __init__(self, size):
        self.transform = transforms.RandomCrop(size=size)

    def __call__(self, images):
        img_left = images[0]
        img_right = images[1]
        
        img_left = self.transform(img_left)
        img_right = self.transform(img_right)
        
        return img_left, img_right
    

class CenterCropDual(object):
    
    def __init__(self, size):
        self.transform = transforms.CenterCrop(size=size)

    def __call__(self, images):
        img_left = images[0]
        img_right = images[1]
        
        img_left = self.transform(img_left)
        img_right = self.transform(img_right)
        
        return img_left, img_right
    
    
# 新增自定义变换类（放在文件开头，transform定义之前）
class CircularBorderCroppingDual:
    """双通道黑边裁剪"""
    def __init__(self, tolerance=6):
        self.tolerance = tolerance

    def __call__(self, images):
        
        left_img = images[0]
        right_img = images[1]
        def _single_crop(img):
            img_np = np.array(img)
            gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
            mask = gray > self.tolerance
            if np.any(mask):
                coords = np.argwhere(mask)
                x0, y0 = coords.min(axis=0)
                x1, y1 = coords.max(axis=0)
                return img.crop((y0, x0, y1+1, x1+1))
            return img
        
        return _single_crop(left_img), _single_crop(right_img)

class EnhanceContrastDual:
    """双通道对比度增强"""
    def __init__(self, clip_limit=2, tile_grid_size=(8,8)):
        self.clip_limit = clip_limit
        self.tile_grid_size = tile_grid_size

    def __call__(self, images):
        
        left_img = images[0]
        right_img = images[1]
        def _single_enhance(img):
            img_np = np.array(img)
            lab = cv2.cvtColor(img_np, cv2.COLOR_RGB2LAB)
            l, a, b = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=self.clip_limit, 
                                  tileGridSize=self.tile_grid_size)
            cl = clahe.apply(l)
            limg = cv2.merge((cl, a, b))
            enhanced = cv2.cvtColor(limg, cv2.COLOR_LAB2RGB)
            enhanced = cv2.medianBlur(enhanced, 3)
            return Image.fromarray(enhanced)
        
        return _single_enhance(left_img), _single_enhance(right_img)
