import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import random

"""该代码实现了一个基于 Transformer 的图像分类模型，专注于对 CIFAR-10 数据集进行分类。模型主要分为三部分：
Patch Embedding：将输入图像划分为小块（patches）并嵌入高维空间。
Transformer Encoder：通过自注意力机制和前馈网络提取全局特征。
分类头（MLP Head）：用于最终的分类预测。"""
"""该类是一个图像分类模型，继承自 PyTorch 的 nn.Module，其主要目的是对输入图像进行特征提取并分类。"""
class TransformerClassifier(nn.Module):
    def __init__(self, num_classes, img_size=32, patch_size=4, dim=256, depth=8, heads=8, mlp_dim=512):
        super(TransformerClassifier, self).__init__()
        self.patch_size = patch_size
        self.dim = dim

        self.embedding = nn.Conv2d(3, dim, kernel_size=patch_size, stride=patch_size)
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.position_embeddings = nn.Parameter(torch.randn((img_size // patch_size) ** 2 + 1, dim))

        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(dim, heads, mlp_dim, dropout=0.2, batch_first=True),
            num_layers=depth
        )

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, mlp_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(mlp_dim, num_classes)
        )
"""模型特点
Patch Embedding：
将图像转换为类似序列的结构，为 Transformer 提供输入。
多头注意力机制：
可以捕获图像中远距离区域的相关性，适合复杂模式的分类任务。
可学习的位置编码：
为每个 patch 添加唯一的位置标识，保留图像的空间结构。
分类标记：
汇总所有 patch 的信息，用于最终的分类任务。
模块化设计：
通过调整 dim、depth 和 heads 等超参数，可以灵活适配不同的数据集和任务需求。"""

def forward(self, x):
        b, c, h, w = x.size()
        x = self.embedding(x)  # Convert to patch embeddings
        x = x.flatten(2).transpose(1, 2)  # (batch, tokens, dim)

        cls_tokens = self.cls_token.expand(b, -1, -1)  # Add class token
        x = torch.cat((cls_tokens, x), dim=1)

        x += self.position_embeddings
        x = self.transformer(x)

        return self.mlp_head(x[:, 0])  # Output class prediction

"""代码实现了 CutMix 数据增强技术，CutMix 是一种通过混合两张图片及其对应的标签来增强数据多样性的方法。
它将一张图片的部分区域替换为另一张图片的对应区域，并调整两张图片的标签比例。"""
def cutmix(data, targets, alpha=1.0):
    indices = torch.randperm(data.size(0))
    shuffled_data = data[indices]
    shuffled_targets = targets[indices]

    lam = random.betavariate(alpha, alpha)
    bx1, by1, bx2, by2 = random_bbox(data.size(2), data.size(3), lam)
    data[:, :, bx1:bx2, by1:by2] = shuffled_data[:, :, bx1:bx2, by1:by2]

    return data, (targets, shuffled_targets, lam)
"""根据混合比例 lam，随机生成裁剪区域的坐标。
该裁剪区域的大小由 lam 确定，并以随机位置出现在图片中。
参数
height 和 width：图片的高和宽。
lam：图片混合的比例，控制裁剪区域的面积。"""
def random_bbox(height, width, lam):
    cut_ratio = (1.0 - lam) ** 0.5
    cut_h = int(height * cut_ratio)
    cut_w = int(width * cut_ratio)

    cy = random.randint(0, height)
    cx = random.randint(0, width)

    bx1 = max(0, cy - cut_h // 2)
    by1 = max(0, cx - cut_w // 2)
    bx2 = min(height, cy + cut_h // 2)
    by2 = min(width, cx + cut_w // 2)

    return bx1, by1, bx2, by2
"""实现了基于 Transformer 的图像分类任务，对 CIFAR-10 数据集进行训练和评估。核心功能包括：

数据预处理与加载。
构建和初始化模型。
使用 CutMix 数据增强技术进行训练。
模型评估并输出分类准确率。"""
if __name__ == "__main__":
    # Hyperparameters
    batch_size = 128
    learning_rate = 0.0001
    epochs = 10
    num_classes = 10
    img_size = 32
    patch_size = 4

    # 准备 CIFAR-10 dataset
    data_transform_train = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(img_size, padding=4),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    data_transform_test = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    train_dataset = datasets.CIFAR10(root='./data', train=True, transform=data_transform_train, download=True)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)

    test_dataset = datasets.CIFAR10(root='./data', train=False, transform=data_transform_test, download=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    # 初始化模型，损失函数
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TransformerClassifier(num_classes, img_size, patch_size).to(device)

    """损失函数：
CrossEntropyLoss 用于计算预测值与真实标签之间的差异，是分类任务中常用的损失函数。
优化器：
AdamW 是一种常用的优化器，具备较好的收敛性，并通过 weight_decay 防止过拟合。
学习率调度器：
OneCycleLR 动态调整学习率，使模型在训练初期快速学习，在后期更稳定地优化。"""
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
    scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=learning_rate, steps_per_epoch=len(train_loader), epochs=epochs)

    # 训练啦
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            # 应用数据增强
            if random.random() < 0.5:
                images, (labels_a, labels_b, lam) = cutmix(images, labels)
                outputs = model(images)
                loss = lam * criterion(outputs, labels_a) + (1 - lam) * criterion(outputs, labels_b)
            else:
                outputs = model(images)
                loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            running_loss += loss.item()

        print(f"Epoch [{epoch + 1}/{epochs}], Loss: {running_loss / len(train_loader):.4f}")

    # 开始评估模型性能
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f"Test Accuracy: {100 * correct / total:.2f}%")
"""Test Accuracy: 66.53%"""
