from __future__ import print_function

import torchvision.models as models
import torchvision.transforms as transforms

from CAM import CAM
from PIL import Image

USE_GPU = False

def main():

    input_img = Image.open('demo/school_bus.png')
    input_img = input_img.convert('RGB')

    transformations = transforms.Compose([
        transforms.Scale((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    model = models.resnet50(pretrained=True)

    if USE_GPU:
        model = model.cuda()

    output_img = CAM(input_img, model, 'layer4', 'fc.weight', transformations, USE_GPU)

    output_img.save('demo/school_bus_CAM.png')

if __name__ == '__main__':
    main()
