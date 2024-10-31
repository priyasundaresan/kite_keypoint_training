import pickle
import cv2
import os
import torch
from torchvision import transforms
from torch.utils.data import DataLoader
from config import *
from src.prediction import Prediction
from datetime import datetime
from PIL import Image
import numpy as np
from src.model_clip import CLIPLingUNet
from src.dataset import LanguageKeypointsDataset, transform

os.environ["CUDA_VISIBLE_DEVICES"]="0"

# model
cfg = {'train': {'batchnorm': True, 'lang_fusion_type': 'mult'}}
model = CLIPLingUNet((IMG_HEIGHT, IMG_WIDTH, 3), 1, cfg, 'cuda:0', None)
dataset_dir = 'semantic_grasping_dset'
model.load_state_dict(torch.load('checkpoints/%s/best_model.pth'%dataset_dir))

# cuda
use_cuda = torch.cuda.is_available()
if use_cuda:
    torch.cuda.set_device(0)
    model = model.cuda()

prediction = Prediction(model, 1, IMG_HEIGHT, IMG_WIDTH, use_cuda)
transform = transforms.Compose([
    transforms.ToTensor()
])

if not os.path.exists('preds'):
    os.mkdir('preds')

workers=0
dataset_dir = '%s'%dataset_dir
dataset = LanguageKeypointsDataset(1, 'data/%s_test/images'%dataset_dir, 'data/%s_test/keypoints'%dataset_dir, 'data/%s_test/lang'%dataset_dir, IMG_HEIGHT, IMG_WIDTH, transform, multimodal=True, gauss_sigma=GAUSS_SIGMA)
data = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=workers)

for i, (img, gt_gauss, text, img_np) in enumerate(data):
    print(i)
    text = text[0]
    heatmap = model(img.cuda(), text)
    heatmap = heatmap.detach().cpu().numpy()
    img_np = img_np.squeeze().cpu().numpy()
    prediction.plot(img_np, heatmap, text, image_id=i)
