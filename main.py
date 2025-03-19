import torch
from torchvision.utils import save_image
from PIL import Image
import os

import torch.optim as optim
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.models as models

# Load the dataset
dataset_path = 'e:/Tester/NeuralStyleTransfer/NeuralStyleImages/Data/Artworks'
style_image_path = os.path.join(dataset_path, '81842.jpg')  # Example style image
content_image_path = os.path.join(dataset_path, '933391.jpg')  # Example content image

# Image loading and preprocessing
def load_image(image_path, transform=None, max_size=400, shape=None):
    image = Image.open(image_path)
    if max_size:
        size = max(image.size)
        if size > max_size:
            size = max_size
    if shape:
        size = shape
    if transform:
        image = transform(image).unsqueeze(0)
    return image

# Define the image transformations
transform = transforms.Compose([
    transforms.Resize((400, 400)),
    transforms.ToTensor(),
    transforms.Lambda(lambda x: x.mul(255))
])

# Load the images
print("Searching for CUDA Drivers... ", 'Found' if torch.cuda.is_available() else 'Not Found')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
style_image = load_image(style_image_path, transform).to(device)
content_image = load_image(content_image_path, transform).to(device)

# Define the model
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1)
        self.conv5 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.pool(x)
        x = self.relu(self.conv2(x))
        x = self.pool(x)
        x = self.relu(self.conv3(x))
        x = self.pool(x)
        x = self.relu(self.conv4(x))
        x = self.pool(x)
        x = self.relu(self.conv5(x))
        return x

# Initialize the model
model = SimpleCNN().to(device).train()

# Define the optimizer and loss functions
optimizer = optim.Adam(model.parameters(), lr=0.003)
mse_loss = nn.MSELoss()

# Training loop
num_steps = 1000
style_weight = 1e6
content_weight = 0.5

for step in range(num_steps):
    optimizer.zero_grad()
    
    content_image.requires_grad_(True)
    target_features = model(content_image)
    content_features = model(content_image)
    style_features = model(style_image)

    style_loss = content_loss = 0

    for target_feature, content_feature, style_feature in zip(target_features, content_features, style_features):
        content_loss += mse_loss(target_feature, content_feature)

        c, h, w = target_feature.size()
        target_gram = target_feature.view(c, h * w).mm(target_feature.view(c, h * w).t())
        style_gram = style_feature.view(c, h * w).mm(style_feature.view(c, h * w).t())
        style_loss += mse_loss(target_gram, style_gram) / (c * h * w)

    total_loss = content_weight * content_loss + style_weight * style_loss

    total_loss.backward()
    optimizer.step()

    if step % 100 == 0:
        print(f"Step [{step}/{num_steps}], Content Loss: {content_loss.item()}, Style Loss: {style_loss.item()}")
        save_image(content_image, f"outputImages\\output_{step}.png")