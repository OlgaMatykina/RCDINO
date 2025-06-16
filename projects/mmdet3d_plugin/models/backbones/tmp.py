from dino import DINOFeaturesExtractor
from PIL import Image
from torchvision import transforms
import torch
from einops import rearrange

# Инициализация модели
model_type = "dino_vits16"
load_size = 256
stride = 16
facet = "token"
num_patches_h, num_patches_w = 16, 44  # зависит от входного изображения

extractor = DINOFeaturesExtractor(
    model=model_type,
    load_size=load_size,
    stride=stride,
    facet=facet,
    num_patches_h=num_patches_h,
    num_patches_w=num_patches_w
)

# Загрузка и предобработка изображения
image_path = "/home/docker_rctrans/HPR3/nuscenes/samples/CAM_FRONT/n008-2018-09-18-14-54-39-0400__CAM_FRONT__1537297347362404.jpg"  # Укажите путь к изображению
image = Image.open(image_path).convert("RGB")

# image.save("/home/docker_rctrans/RCTrans/origin.png")

transform = transforms.Compose([
    transforms.Resize((256, 704)),
    transforms.ToTensor(),
])
image_tensor = transform(image).unsqueeze(0)  # Добавляем размерность батча

image_tensor = image_tensor.to("cuda" if torch.cuda.is_available() else "cpu")

image_path = "/home/docker_rctrans/HPR3/nuscenes/samples/CAM_BACK/n008-2018-09-18-14-54-39-0400__CAM_BACK__1537297347387558.jpg"  # Укажите путь к изображению
image = Image.open(image_path).convert("RGB")

# image.save("/home/docker_rctrans/RCTrans/origin_back.png")

image_tensor2 = transform(image).unsqueeze(0)  # Добавляем размерность батча

image_tensor2 = image_tensor2.to("cuda" if torch.cuda.is_available() else "cpu")

image_tensor = torch.cat([image_tensor, image_tensor2], dim=0)

print('Image shape:', image_tensor.shape)

# Извлечение признаков
features = extractor.forward(image_tensor, facet="token")

# Вывод результата
print("Feature shape:", features.shape)

features = features.transpose(2,3)
features = rearrange(features, 'b n c (h w) -> b n c h w', h=num_patches_h)

print("Feature shape:", features.shape)

print(torch.equal(features[0], features[1]))

feature_map_2d = features.squeeze().cpu()  # Замените на ваши данные

import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

feature_map = feature_map_2d[0].mean(dim=0).reshape(16, 44).numpy()

# Строим heatmap
plt.figure(figsize=(6, 6))
plt.imshow(feature_map, cmap="viridis")
plt.savefig("/home/docker_rctrans/RCTrans/dinov2_features2front.png", dpi=300, bbox_inches='tight')

feature_map = feature_map_2d[1].mean(dim=0).reshape(16, 44).numpy()

# Строим heatmap
plt.figure(figsize=(6, 6))
plt.imshow(feature_map, cmap="viridis")
plt.savefig("/home/docker_rctrans/RCTrans/dinov2_features2back.png", dpi=300, bbox_inches='tight')


print(torch.equal(feature_map_2d[0], feature_map_2d[1]))