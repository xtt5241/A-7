import cv2
import numpy as np
import torch

# 在文件顶部添加
import warnings

def get_grad_cam(model, image_tensor, class_idx):
    # 添加模型检查
    if model is None:
        warnings.warn("Grad-CAM不可用：模型未加载")
        return np.zeros((456, 456))  # 返回空白热力图
    
    model.eval()
    target_layer = model.features[-1]  # 获取模型最后一层

    activation = None
    gradient = None

    def forward_hook(module, input, output):
        nonlocal activation
        activation = output

    def backward_hook(module, grad_in, grad_out):
        nonlocal gradient
        gradient = grad_out[0]

    target_layer.register_forward_hook(forward_hook)
    target_layer.register_backward_hook(backward_hook)

    output = model(image_tensor)
    model.zero_grad()
    class_score = output[0, class_idx]
    class_score.backward()

    # 计算 Grad-CAM
    weights = gradient.mean(dim=(2, 3), keepdim=True)
    cam = (weights * activation).sum(dim=1, keepdim=True)
    cam = cam.squeeze().detach().numpy()
    cam = np.maximum(cam, 0)  # ReLU 操作
    cam = cv2.resize(cam, (image_tensor.shape[2], image_tensor.shape[3]))

    return cam
