
import os
import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader


if __name__ == '__main__':



    # 数据集文件夹路径
    dataset_dir = 'data'
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # 训练集图像预处理：缩放裁剪、图像增强、转 Tensor、归一化
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # 测试集图像预处理-RCTN：缩放、裁剪、转 Tensor、归一化
    test_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    train_path = os.path.join(dataset_dir, 'train')
    test_path = os.path.join(dataset_dir, 'val')



    # 载入训练集
    train_dataset = torchvision.datasets.ImageFolder(train_path, train_transform)

    # 载入测试集
    test_dataset = torchvision.datasets.ImageFolder(test_path, test_transform)

    BATCH_SIZE = 32

    # 训练集的数据加载器
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)

    # 测试集的数据加载器
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    # 各类别名称
    class_names = train_dataset.classes
    n_class = len(class_names)

    # 获得一个 batch 的数据和标注
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
    images, labels = next(iter(train_loader))

    # 构建模型
    model = torchvision.models.resnet18(pretrained=True)
    model.fc = nn.Linear(model.fc.in_features, n_class)  # 修改全连接层，使得输出与当前数据集类别数对应
    model = model.to(device)

    # 优化器和损失函数
    optimizer = optim.Adam(model.fc.parameters())
    criterion = nn.CrossEntropyLoss()

    # 训练轮次 Epoch
    EPOCHS = 20

    # 遍历每个 EPOCH
    for epoch in range(EPOCHS):
        model.train()
        print("training(%s/20)"%(epoch+1))

        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0

        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (preds == labels).sum()

        print('测试集上的准确率为 {:.4f} %'.format(100 * correct / total))

    torch.save(model, 'dog.pth')
