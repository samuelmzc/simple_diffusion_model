import os
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import torch as torch
import torchvision as tvis
import torch.nn.functional as F
from model import *

SIZE = 64
BATCH_SIZE = 128

def load_preprocess_data():
    data_transformations = [
        tvis.transforms.Resize((SIZE, SIZE)),
        tvis.transforms.ToTensor(),
        tvis.transforms.Lambda(lambda img: (img * 2) - 1)
    ]

    data_tr = tvis.transforms.Compose(data_transformations)
    train_set = tvis.datasets.Flowers102(root = ".", split = "train", transform = data_tr, download=True)
    test_set = tvis.datasets.Flowers102(root = ".", split = "train", transform = data_tr, download=True)

    return torch.utils.data.ConcatDataset([train_set, test_set])


def show_tensor_image(image):
    reverse_transforms = tvis.transforms.Compose([
        tvis.transforms.Lambda(lambda img : (img + 1) / 2),
        tvis.transforms.Lambda(lambda img : img.permute(1, 2, 0)),
        tvis.transforms.Lambda(lambda img : img * 255.),
        tvis.transforms.Lambda(lambda img : img.numpy().astype(np.uint8)),
        tvis.transforms.ToPILImage()
    ])

    if len(image.shape) == 4: #if there is minibatch, pick first image
        image = image[0, :, :, :]
    plt.imshow(reverse_transforms(image))


data = load_preprocess_data()
data_loader = torch.utils.data.DataLoader(data, batch_size = BATCH_SIZE, shuffle = True)

T = 100
beta_t = variance_schedule(T)
c_alphas = cumm_alphas(beta_t)

images_shown = int(100)
stepsize = int(T/images_shown)

image = next(iter(data_loader))[0]

plt.title(f"Forward process for t = {0}")
show_tensor_image(image)
plt.savefig(f"pictures/forward_t_{0}.jpg")

for idx in range(0, T, stepsize):
    idx_t = torch.Tensor([idx]).type(torch.int64)
    #plt.subplot(1, images_shown + 1, (idx/stepsize) + 1)
    image_diff, _ = forward_diffusion_step(image, idx_t, c_alphas[idx])
    plt.clf()
    plt.title(f"Forward process for t = {idx}")
    show_tensor_image(image_diff)
    plt.savefig(f"pictures/forward_t_{idx}.jpg")
    if idx%10 == 0:
        print(f"Denoising diffusion process: {idx}/{T}")