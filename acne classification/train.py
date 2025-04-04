# train.py

import argparse
import os

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

import numpy as np

# ---------- 1. 引入你定义的 Dataset 和 模型 ----------
from dataset import AcneDataset  # 这里替换成实际文件名/路径
from model.mutil_cnn import MultiBranchAcneModel, MultiBranchEfficientNetModel  # 同上

# 用于计算多分类的准确率、PR曲线等
from sklearn.metrics import accuracy_score, precision_recall_curve, average_precision_score
from sklearn.preprocessing import label_binarize

# 如果想要保存/绘制loss曲线和PR曲线，可用matplotlib
import matplotlib.pyplot as plt

from tqdm import tqdm
def validate(model, dataloader, device, criterion):
    """
    在测试/验证集上跑一遍 forward，计算平均loss和accuracy，并记录所有预测分数用于绘制PR曲线。

    :param model: 已训练好的模型
    :param dataloader: 测试数据的 DataLoader
    :param device: 'cpu' or 'cuda'
    :param criterion: 损失函数 (CrossEntropyLoss)
    :return:
       avg_loss: float, 测试集平均loss
       accuracy: float, 测试集准确率
       y_true_all: (N,) numpy数组, 测试集所有样本的真实标签
       y_score_all: (N, num_classes) numpy数组, 模型对每个样本输出的logits或概率
    """
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_labels = []
    all_scores = []

    with torch.no_grad():
        for (imgs, labels) in tqdm(dataloader):
            # imgs: [img_front, img_left, img_right]
            imgs = [img.to(device) for img in imgs]
            labels = labels.to(device)

            logits = model(imgs)  # [B, num_classes]

            # 计算 loss
            loss = criterion(logits, labels)
            total_loss += loss.item()

            # 保存真实标签
            all_labels.append(labels.cpu().numpy())

            # 保存原始logits (或softmax之后的prob也行，这里先保留logits方便后面做多种处理)
            all_scores.append(logits.cpu().numpy())

            # 取预测标签
            preds = torch.argmax(logits, dim=1)
            all_preds.append(preds.cpu().numpy())

    # 汇总
    avg_loss = total_loss / len(dataloader)

    all_preds = np.concatenate(all_preds, axis=0)  # shape=(N,)
    all_labels = np.concatenate(all_labels, axis=0)  # shape=(N,)
    all_scores = np.concatenate(all_scores, axis=0)  # shape=(N, num_classes)

    # 计算准确率
    accuracy = accuracy_score(all_labels, all_preds)

    model.train()  # 切回训练模式
    return avg_loss, accuracy, all_labels, all_scores


def main():
    # ---------- 2. 命令行参数解析 (可选) ----------
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_path', type=str, default='./data.xls',
                        help='训练集 XLS 文件路径')
    parser.add_argument('--test_path', type=str, default='./data.xls',
                        help='测试集 XLS 文件路径')
    parser.add_argument('--train_dir', type=str, default='/root/autodl-tmp/acne',
                        help='存放所有 jpg 图片的文件夹路径')
    parser.add_argument('--test_dir', type=str, default='/root/autodl-tmp/acne',
                        help='存放所有 jpg 图片的文件夹路径')
    parser.add_argument('--num_classes', type=int, default=4,
                        help='痤疮严重程度分类数，例如 4')
    parser.add_argument('--batch_size', type=int, default=8,
                        help='批大小')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='初始学习率')
    parser.add_argument('--num_epochs', type=int, default=100,
                        help='训练轮数')
    parser.add_argument('--save_path', type=str, default='acne_model.pth',
                        help='训练完成后保存模型的路径')
    args = parser.parse_args()

    # ---------- 3. 构建 Dataset 和 DataLoader ----------
    transform = transforms.Compose([
    transforms.Resize((224, 224)),      # 将图像缩放到 224×224
    transforms.ToTensor(),              # PIL -> Tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406],  # 可选: ImageNet 均值方差
                 std=[0.229, 0.224, 0.225]) 
    # 如果需要归一化，可再加 transforms.Normalize(mean, std)
])
    # 如果需要图像预处理/数据增强，可使用 torchvision.transforms.Compose([...])
    # 这里演示就写成 None

    train_dataset = AcneDataset(
        csv_path=args.train_path,
        img_dir=args.train_dir,
        transform=transform
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4
    )

    test_dataset = AcneDataset(
        csv_path=args.test_path,
        img_dir=args.test_dir,
        transform=transform,
        is_train=False  # 假设你在Dataset里用 is_train 区分训练/测试逻辑
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,  # 测试集一般不需要 shuffle
        num_workers=4
    )

    # ---------- 4. 构建模型、损失函数与优化器 ----------
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Using device:", device)

    # model = MultiBranchAcneModel(num_classes=args.num_classes)
    model = MultiBranchEfficientNetModel(num_classes=args.num_classes)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # ---------- 用于记录训练 & 测试loss, accuracy 的列表 ----------
    train_losses = []
    test_losses = []
    train_accuracies = []
    test_accuracies = []

    # ---------- 5. 训练循环 ----------
    for epoch in range(args.num_epochs):
        # -- 5.1 训练一个 epoch --
        model.train()
        running_loss = 0.0
        for batch_idx, (imgs, labels) in tqdm(enumerate(train_loader)):
            imgs = [img.to(device) for img in imgs]
            labels = labels.to(device)

            logits = model(imgs)
            loss = criterion(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        avg_train_loss = running_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        # -- 5.2 在训练完一个 epoch 后，用训练集自身做accuracy(可选) --
        # 为了省时间，也可以只用 test_set 做评估，这里示例多加一步
        train_loss_epoch, train_acc_epoch, y_true_train, y_score_train = validate(
            model, train_loader, device, criterion
        )
        train_accuracies.append(train_acc_epoch)

        # -- 5.3 测试集验证 --
        test_loss_epoch, test_acc_epoch, y_true_test, y_score_test = validate(
            model, test_loader, device, criterion
        )
        test_losses.append(test_loss_epoch)
        test_accuracies.append(test_acc_epoch)

        print(f"Epoch [{epoch + 1}/{args.num_epochs}] "
              f"Train Loss: {avg_train_loss:.4f}, "
              f"Train Acc: {train_acc_epoch:.4f}, "
              f"Test Loss: {test_loss_epoch:.4f}, "
              f"Test Acc: {test_acc_epoch:.4f}")

        # -- 5.4 生成并保存 PR 曲线 (可选，每个epoch都保存) --
        # 由于是多分类，我们可以采用 'micro' 方式把多类展开计算
        # y_true_test.shape = [N], y_score_test.shape = [N, num_classes]
        # 如果你的标签是 0~(num_classes-1)，无需再做减1
        # 如果你的标签是 1~4，请先减1 => [0~3]

        # 1) 需要先把 y_true_test 转为 one-hot 形式
        #    label_binarize: classes=[0,1,2,3,...]
        y_true_bin = label_binarize(y_true_test, classes=list(range(args.num_classes)))
        # shape = [N, num_classes]

        # 2) 计算 "micro" 整体的 precision, recall
        #    先把 logits 转化为概率
        y_prob_test = torch.softmax(torch.from_numpy(y_score_test), dim=1).numpy()
        # shape=[N, num_classes]
        precision, recall, thresholds = precision_recall_curve(
            y_true_bin.ravel(), y_prob_test.ravel()
            , pos_label=1
        )
        ap_score = average_precision_score(y_true_bin, y_prob_test, average='micro')

        # 3) 绘制并保存
        plt.figure()
        plt.step(recall, precision, where='post', label=f'AP={ap_score:.4f}')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title(f'Epoch {epoch + 1} PR Curve (micro-average)')
        plt.legend()
        plt.savefig(f'pr_curve_epoch{epoch + 1}.png')
        plt.close()

    # ---------- 6. 训练完成后，保存模型 ----------
    os.makedirs(os.path.dirname(args.save_path), exist_ok=True)  # 确保目录存在
    torch.save(model.state_dict(), args.save_path)
    print(f"模型已保存至: {args.save_path}")

    # ---------- 7. 保存并绘制训练 & 测试Loss/Acc 曲线 (可选) ----------
    # 你可以将数据写入文件，也可以直接用 matplotlib 绘图保存

    # 7.1 写入一个 .txt 或 .csv
    with open("training_log.csv", "w") as f:
        f.write("epoch,train_loss,train_acc,test_loss,test_acc\n")
        for i in range(args.num_epochs):
            f.write(f"{i + 1},{train_losses[i]:.4f},{train_accuracies[i]:.4f},"
                    f"{test_losses[i]:.4f},{test_accuracies[i]:.4f}\n")

    # 7.2 绘制 Loss 曲线
    epochs_range = range(1, args.num_epochs + 1)

    plt.figure()
    plt.plot(epochs_range, train_losses, label='Train Loss')
    plt.plot(epochs_range, test_losses, label='Test Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Train & Test Loss')
    plt.legend()
    plt.savefig('loss_curve.png')
    plt.close()

    # 7.3 绘制 Accuracy 曲线
    plt.figure()
    plt.plot(epochs_range, train_accuracies, label='Train Acc')
    plt.plot(epochs_range, test_accuracies, label='Test Acc')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Train & Test Accuracy')
    plt.legend()
    plt.savefig('acc_curve.png')
    plt.close()


if __name__ == '__main__':
    main()