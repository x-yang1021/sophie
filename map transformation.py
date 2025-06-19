from torchvision import models, transforms
from PIL import Image
import torch
import pickle

# Path to your PNG map
img_path = './img/North_shared_map.png'
save_path = './img/North_shared_map.pkl'

# Load pretrained VGG16 (remove classifier head)
vgg = models.vgg16(pretrained=True).features.eval()

# Transform image
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # match input size
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])
image = Image.open(img_path).convert('RGB')
img_tensor = transform(image).unsqueeze(0)  # shape: [1, 3, 224, 224]

# Extract features
with torch.no_grad():
    features = vgg(img_tensor).squeeze(0).numpy()

# Save to .pkl
with open(save_path, 'wb') as f:
    pickle.dump(features, f)
