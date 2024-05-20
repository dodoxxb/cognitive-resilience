import torch
import torchvision.transforms as transforms
from torchvision import models
from torch.autograd import Variable
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from PIL import Image
import os

# 加载预训练的ResNet-50模型
resnet50 = models.resnet50(pretrained=True)
resnet50.eval()

# 图像预处理函数
def preprocess_image(image_path):
    image = Image.open(image_path).convert('RGB')
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    input_image = transform(image)
    input_image = input_image.unsqueeze(0)
    return input_image

# 提取图像特征
def extract_features(image_path):
    input_image = preprocess_image(image_path)
    input_var = Variable(input_image)
    features = resnet50(input_var)
    features = features.data.numpy().flatten()
    return features

# 获取数据集中的图像路径和对应的分组标签
image_paths = []
group_labels = []
for dataset in os.listdir("./image_dataset"):
    for model_name in os.listdir(os.path.join("./image_dataset", dataset)):
        if '.jpg' in model_name or '.png' in model_name:
            image_paths.append(os.path.join("./image_dataset", dataset, model_name))
            group_labels.append(dataset)

# 提取特征并构建特征矩阵
feature_matrix = np.array([extract_features(path) for path in image_paths])

# 使用t-SNE算法降维到二维空间
tsne = TSNE(n_components=2, random_state=0)
tsne_result = tsne.fit_transform(feature_matrix)

# 绘制t-SNE图，根据分组标签使用不同颜色的点
plt.figure(figsize=(10, 8))
for label in set(group_labels):
    indices = np.where(np.array(group_labels) == label)
    plt.scatter(tsne_result[indices, 0], tsne_result[indices, 1], label=f'{label}')

# plt.title('t-SNE Visualization with Group Information')
plt.legend()
plt.show()
