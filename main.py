import glob
import random
import os
import math
import itertools
import sys

import pandas as pd
import numpy as np

from matplotlib import pyplot as plt
from PIL import Image
import argparse# Importing Libraries

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from torchvision.utils import save_image, make_grid
from torch.utils.data import DataLoader
from torch.autograd import Variable

from torchvision.models import vgg19


from ECV_Generator import *
from util import *

import torch

model = color_ecv()
model.load_state_dict(torch.load("pretrained_models/generator.pth", map_location=torch.device('cpu')))
model.eval()

class TestDataset(Dataset):
    def __init__(self, root, single_image = True):
        if single_image:
            self.files = [root]
        else:
            self.files = sorted(glob.glob(root + "/*.*"))
        
    def __getitem__(self, index):
       
        black_path = self.files[index % len(self.files)]
        img_black = np.asarray(Image.open(black_path))
        if(img_black.ndim==2):
            img_black = np.tile(img_black[:,:,None],3)
        (tens_l_orig, tens_l_rs) = preprocess_img(img_black, HW=(400, 400))

        return {"black": tens_l_rs.squeeze(0), 'orig': tens_l_orig.squeeze(0), 'path' : black_path}
    
    def __len__(self):
        return len(self.files)
    
def predict_outputs(model, dataset):
    #image = single_image
    batch_size = 1
    dataloader = DataLoader(
        dataset,
        batch_size = batch_size,
        shuffle = False,
        num_workers = 0,
    )

    cuda = torch.cuda.is_available()
    if cuda:
        model = model.to('cuda')

    Tensor = torch.cuda.FloatTensor if cuda else torch.Tensor
    outputs = {}
    for i, imgs in enumerate(dataloader):
        imgs_black = Variable(imgs["black"].type(Tensor))
        imgs_black_orig = Variable(imgs["orig"].type(Tensor))
        gen_ab = model(imgs_black)
        gen_ab.detach_
        gen_color = postprocess_tens_new(imgs_black_orig, gen_ab)[0].transpose(1,2,0)
        outputs[imgs["path"][0]] = gen_color
    return outputs

def print_images(outputs):
    for i in outputs.keys():
        print("----------- The Black and White Image -----------")
        plt.imshow(plt.imread(i))
        plt.show()
        print("----------- The Colourfull Image Generated -----------")
        plt.imshow(outputs[i])
        plt.show()

def save_outputs(outputs, folder_path, single_image=False):
    os.makedirs(folder_path,  exist_ok=True)
    for i in outputs.keys():
        if single_image:
            name = i.split('/')[-1]
        else:
            name = i.split('\\')[-1]
        image = Image.fromarray((outputs[i] * 255).astype(np.uint8)) 
        image.save(folder_path + '/' + name)

def save_colorized_images(outputs, output_dir='colorized_images'):
    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Iterate over the outputs dictionary
    for key, value in outputs.items():
        # Generate the output file path
        output_file = os.path.join(output_dir, f"{os.path.splitext(os.path.basename(key))[0]}_colorized.jpg")
        
        # Save the colorized image to the output file
        plt.imsave(output_file, value)
    
    print(f"Colorized images saved to {output_dir}")

# dataset = TestDataset(input("Enter Path "))

# output = predict_outputs(model,dataset)

# print(output)
# print_images(output)