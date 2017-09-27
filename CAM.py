from __future__ import division
from __future__ import print_function

import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn

from torchvision import transforms
from torch.autograd import Variable

from PIL import Image

def overlay(img, heatmap, cmap='jet', alpha=0.5):

    if isinstance(img, np.ndarray):
        img = Image.fromarray(img)

    if isinstance(heatmap, np.ndarray):
        colorize = plt.get_cmap(cmap)
        # Normalize
        heatmap = heatmap - np.min(heatmap)
        heatmap = heatmap / np.max(heatmap)
        heatmap = colorize(heatmap, bytes=True)
        heatmap = Image.fromarray(heatmap[:, :, :3], mode='RGB')

    # Resize the heatmap to cover whole img
    heatmap = heatmap.resize((img.size[0], img.size[1]), resample=Image.BILINEAR)
    # Display final overlayed output
    result = Image.blend(img, heatmap, alpha)
    return result


def CAM(input_img, model, feature_layer_name, weight_layer_name, transform=None, USE_GPU=False):
    """Compute Class Activation Map (CAM) of input_img

    Args:
        input_img (PIL.Image.Image): image file feed into model
        model (torchvision.models): the DCNN model
        feature_layer_name (string): should be the name of model's last conv layer
        weight_layer_name (string): should be the name of model's classifier weight layer
        transform (torchvision.transforms): transformations perform on input_img
        USE_GPU (bool): GPU configuration

    Returns:
        The return image (PIL.Image.Image). With input_img blended with CAM.

    """
    if transform is not None:
        img = transform(input_img)

    model.eval()

    # hook the feature extractor
    feature_maps = []
    def hook(module, input, output):
        if USE_GPU:
            feature_maps.append(output.cpu().data.numpy())
        else:
            feature_maps.append(output.data.numpy())
    handle = model._modules.get(feature_layer_name).register_forward_hook(hook)

    params = model.state_dict()[weight_layer_name]
    if USE_GPU:
        weight_softmax = np.squeeze(params.cpu().numpy())
    else:
        weight_softmax = np.squeeze(params.numpy())

    # fake batch dim
    img = torch.unsqueeze(img, 0)

    if USE_GPU:
        img = Variable(img).cuda(async=True)
    else:
        img = Variable(img)

    # forward
    output = model(img)

    # remove the hook
    handle.remove()

    class_idx = torch.max(output, 1)[1].data.cpu().numpy()[0]

    # compute CAM
    bz, nc, h, w = feature_maps[0].shape
    features = feature_maps[0].reshape(bz*nc, h*w)
    cam = weight_softmax[class_idx].dot(features)
    cam = cam.reshape(h, w)
    cam = cam - np.min(cam)
    cam = cam / np.max(cam)
    cam_img = np.uint8(255 * cam)

    result = overlay(input_img, cam_img)

    return result
