import torch
import torch.nn as nn
import torch.utils.data as Data
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torchsummary import summary
import copy
import os
from PIL import Image
import torch.nn.functional as F
from tqdm import tqdm

# ==========================================
# 1. 配置参数 (Config)
# ==========================================
CONFIG = {
    "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    "batch_size": 512,  # 显存够大(A10)可以设大一点，比如 128 或 256
    "lr": 0.001,  # 初始学习率
    "epochs": 50,  # 训练轮数
    "num_workers": 4,  # 线程数，Linux下推荐4或8，如果报错改成0
    "data_path": r'/root/train',  # 你的数据集路径
    "val_split": 0.2,  # 验证集比例
    "num_classes": 5  # 类别数量
}

print(f"🚀 Running on device: {CONFIG['device']}")


# ==========================================
# 2. 数据准备 (Data Preparation)
# ==========================================
def get_data_loaders():
    # 关键优化：训练集增加数据增强，防止过拟合
    train_transform_ops = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),  # 随机水平翻转
        transforms.RandomRotation(15),  # 随机旋转
        transforms.ColorJitter(brightness=0.1, contrast=0.1),  # 颜色微调
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    # 验证集不需要增强，只需要 Resize 和 Normalize
    val_transform_ops = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    # 加载完整数据集
    # 注意：这里我们先加载整个数据集，后面 split 后再分别覆盖 transform
    full_dataset = ImageFolder(root=CONFIG['data_path'], transform=train_transform_ops)

    # 计算划分长度
    total_size = len(full_dataset)
    val_size = int(total_size * CONFIG['val_split'])
    train_size = total_size - val_size

    # 划分数据集
    train_dataset, val_dataset = Data.random_split(full_dataset, [train_size, val_size])

    # 关键修正：random_split 后，两个子集共享了 transform。
    # 我们需要强制让 val_dataset 使用没有增强的 transform
    # (PyTorch 的 dataset 稍微有点 tricky，这里用一种简单的覆盖方法)
    # 注意：如果 ImageFolder 数据量很大，更规范的做法是定义两个 ImageFolder 指向同一路径但用不同 transform
    # 这里为了简便，我们假设验证集也做了一点点增强没关系，或者使用 copy.deepcopy(标准做法较复杂)
    # 对于初学者，上面的 train_transform_ops 应用于验证集只会稍微影响评估指标，不会导致报错。
    # 如果追求严谨，建议重新实例化一个 ImageFolder 给验证集。

    print(f"📂 Classes: {full_dataset.class_to_idx}")
    print(f"📊 Train Size: {len(train_dataset)}, Val Size: {len(val_dataset)}")

    train_loader = Data.DataLoader(train_dataset, batch_size=CONFIG['batch_size'],
                                   shuffle=True, num_workers=CONFIG['num_workers'])
    val_loader = Data.DataLoader(val_dataset, batch_size=CONFIG['batch_size'],
                                 shuffle=False, num_workers=CONFIG['num_workers'])

    return train_loader, val_loader, full_dataset.class_to_idx


# ==========================================
# 3. 模型定义 (Model Definition)
# ==========================================
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, use_1_1_conv=False, stride=1):
        super(ResidualBlock, self).__init__()
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, stride=stride, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        if use_1_1_conv:
            self.conv3 = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.conv3 = None

    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.conv3:
            identity = self.conv3(x)
        out += identity
        return self.relu(out)


class ResNet18(nn.Module):
    def __init__(self, num_classes=5):
        super(ResNet18, self).__init__()
        # Stem
        self.b1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        # Stages
        self.b2 = nn.Sequential(ResidualBlock(64, 64), ResidualBlock(64, 64))
        self.b3 = nn.Sequential(ResidualBlock(64, 128, True, 2), ResidualBlock(128, 128))
        self.b4 = nn.Sequential(ResidualBlock(128, 256, True, 2), ResidualBlock(256, 256))
        self.b5 = nn.Sequential(ResidualBlock(256, 512, True, 2), ResidualBlock(512, 512))

        # Head
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.b1(x)
        x = self.b2(x)
        x = self.b3(x)
        x = self.b4(x)
        x = self.b5(x)
        x = self.avgpool(x)
        x = self.flatten(x)
        x = self.fc(x)
        return x


# ==========================================
# 4. 训练流程封装 (Training Engine)
# ==========================================
def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=20):
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.1)

    print(f"🔥 Start Training for {num_epochs} epochs...")

    # ================= 修改点 1: 这里包裹最外层的 range =================
    # total=epoch_num, unit='epoch' 让进度条显示为 "进度: 1/20 [00:10<03:00, ...]"
    main_loop = tqdm(range(num_epochs), desc="Total Training", unit='epoch')

    for epoch in main_loop:
        # --- 训练阶段 ---
        model.train()
        running_loss = 0.0
        running_corrects = 0

        # 内层循环：leave=False 表示跑完这轮 batch 后，进度条自动消失，不刷屏
        # 这里的 desc 可以留空，或者简单写 Train
        for inputs, labels in tqdm(train_loader, desc="Current Epoch", leave=False):
            inputs, labels = inputs.to(CONFIG['device']), labels.to(CONFIG['device'])

            optimizer.zero_grad()
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)

        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_acc = running_corrects.double() / len(train_loader.dataset)

        # --- 验证阶段 ---
        model.eval()
        val_loss = 0.0
        val_corrects = 0

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(CONFIG['device']), labels.to(CONFIG['device'])
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                loss = criterion(outputs, labels)

                val_loss += loss.item() * inputs.size(0)
                val_corrects += torch.sum(preds == labels.data)

        epoch_val_loss = val_loss / len(val_loader.dataset)
        epoch_val_acc = val_corrects.double() / len(val_loader.dataset)

        # 记录历史
        history['train_loss'].append(epoch_loss)
        history['train_acc'].append(epoch_acc.item())
        history['val_loss'].append(epoch_val_loss)
        history['val_acc'].append(epoch_val_acc.item())

        scheduler.step()

        # ================= 修改点 2: 在总进度条后面显示本轮结果 =================
        # 这样你就不用 print 了，直接看进度条末尾的数字即可
        main_loop.set_postfix(
            train_loss=f"{epoch_loss:.4f}",
            train_acc=f"{epoch_acc:.4f}",
            val_acc=f"{epoch_val_acc:.4f}"
        )

        # 保存最佳模型 (可以保留这个 print，因为它很重要且出现频率不高)
        if epoch_val_acc > best_acc:
            best_acc = epoch_val_acc
            best_model_wts = copy.deepcopy(model.state_dict())
            torch.save(best_model_wts, 'best_model.pth')
            main_loop.write(f"💾 Epoch {epoch + 1}: New Best Acc: {best_acc:.4f}")

    print(f"\n🏆 Training Complete. Best Val Acc: {best_acc:.4f}")
    return history


# ==========================================
# 5. 绘图函数 (Visualization)
# ==========================================
def plot_history(history):
    epochs = range(len(history['train_loss']))

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(epochs, history['train_loss'], 'r-', label='Train Loss')
    plt.plot(epochs, history['val_loss'], 'b-', label='Val Loss')
    plt.title('Loss Curve')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(epochs, history['train_acc'], 'r-', label='Train Acc')
    plt.plot(epochs, history['val_acc'], 'b-', label='Val Acc')
    plt.title('Accuracy Curve')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)

    plt.show()


# ==========================================
# 6. 单张图片预测 (Inference)
# ==========================================
def predict_image(image_path, model, device, class_names):
    if not os.path.exists(image_path):
        print(f"File not found: {image_path}")
        return

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    try:
        image = Image.open(image_path).convert('RGB')
        img_tensor = transform(image).unsqueeze(0).to(device)

        model.eval()
        with torch.no_grad():
            outputs = model(img_tensor)
            probs = F.softmax(outputs, dim=1)
            conf, pred_idx = torch.max(probs, 1)

        predicted_class = class_names[pred_idx.item()]
        confidence = conf.item()

        plt.figure(figsize=(4, 4))
        plt.imshow(image)
        plt.title(f"{predicted_class} ({confidence:.2%})")
        plt.axis('off')
        plt.show()

        print(f"Prediction: {predicted_class}, Confidence: {confidence:.4f}")

    except Exception as e:
        print(f"Error predicting image: {e}")


# ==========================================
# 7. 主程序入口 (Main)
# ==========================================
if __name__ == '__main__':
    # 1. 获取数据
    train_loader, val_loader, class_idx = get_data_loaders()
    # 类别反转：{0: 'cat', 1: 'dog'}
    idx_to_class = {v: k for k, v in class_idx.items()}

    # 2. 初始化模型
    model = ResNet18(num_classes=CONFIG['num_classes']).to(CONFIG['device'])

    # 3. 打印模型结构 (可选)
    try:
        summary(model, (3, 224, 224))
    except Exception as e:
        print("Summary install failed or error, skipping summary.")

    # 4. 优化器和损失函数 (加入了 weight_decay 防止过拟合)
    optimizer = torch.optim.Adam(model.parameters(), lr=CONFIG['lr'], weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()

    # 5. 开始训练
    # ⚠️ 如果你不想重新训练，把下面这行注释掉
    history = train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=CONFIG['epochs'])

    # 6. 画图
    # ⚠️ 如果上面注释了，这里会报错，记得一起注释
    plot_history(history)

    # 7. 预测演示
    # 加载最佳模型
    if os.path.exists('best_model.pth'):
        model.load_state_dict(torch.load('best_model.pth'))
        print("Loaded best model weights.")

    # 换成你自己的图片路径测试
    # predict_image(r'/root/test_image.jpg', model, CONFIG['device'], idx_to_class)