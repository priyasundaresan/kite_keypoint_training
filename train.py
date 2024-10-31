import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from src.model_clip import CLIPLingUNet
from src.dataset import LanguageKeypointsDataset, transform
from config import *

def forward(sample_batched, model):
    img, gt_gauss, text, _ = sample_batched
    img = img.cuda() if use_cuda else img
    text = text[0]
    pred_gauss = model.forward(img, text).double()
    loss = nn.BCELoss()(pred_gauss, gt_gauss)
    return loss

def train(train_data, test_data, model, epochs, checkpoint_path = ''):
    best_loss = float('inf')
    for epoch in range(epochs):

        train_loss = 0.0
        for i_batch, sample_batched in enumerate(train_data):
            optimizer.zero_grad()
            loss = forward(sample_batched, model)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            print('Epoch: [%d/%d], Batch: %5d, Loss: %.3f' % (epoch + 1, epochs, i_batch + 1, loss.item()), end='')
            print('\r', end='')
        print('train loss:', train_loss / i_batch)
        
        test_loss = 0.0
        for i_batch, sample_batched in enumerate(test_data):
            loss = forward(sample_batched, model)
            test_loss += loss.item()

        curr_loss = test_loss/i_batch
        print('test loss:', curr_loss)

        if curr_loss < best_loss:
            print('Saving with loss', curr_loss)
            torch.save(model.state_dict(), checkpoint_path + '/best_model.pth')
            best_loss = curr_loss


### Directory to save checkpoints
save_dir = os.path.join(output_dir, dataset_dir)
if not os.path.exists(output_dir):
    os.mkdir(output_dir)
if not os.path.exists(save_dir):
    os.mkdir(save_dir)

### Specify datasets for training / testing
train_dataset = LanguageKeypointsDataset(NUM_KEYPOINTS, 'data/%s/images'%dataset_dir, 'data/%s/keypoints'%dataset_dir, 'data/%s/lang'%dataset_dir, IMG_HEIGHT, IMG_WIDTH, transform, multimodal=False, gauss_sigma=GAUSS_SIGMA)
train_data = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=workers)

test_dataset = LanguageKeypointsDataset(NUM_KEYPOINTS, 'data/%s_test/images'%dataset_dir, 'data/%s/keypoints'%dataset_dir, 'data/%s/lang'%dataset_dir, IMG_HEIGHT, IMG_WIDTH, transform, multimodal=False, gauss_sigma=GAUSS_SIGMA)
test_data = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=workers)

### Use cuda
os.environ["CUDA_VISIBLE_DEVICES"]="0"
use_cuda = torch.cuda.is_available()
Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
if use_cuda:
    torch.cuda.set_device(0)

### Specify model
cfg = {'train': {'batchnorm': True, 'lang_fusion_type': 'mult'}}
model = CLIPLingUNet((IMG_HEIGHT, IMG_WIDTH, 3), NUM_KEYPOINTS, cfg, 'cuda:0', None)

### Train
optimizer = optim.Adam(model.parameters(), lr=1.0e-4, weight_decay=1.0e-4)
train(train_data, test_data, model, epochs=epochs, checkpoint_path=save_dir)
