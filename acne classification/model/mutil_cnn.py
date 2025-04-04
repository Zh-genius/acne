import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class MultiBranchAcneModel(nn.Module):
    def __init__(self, num_classes=4):
        """
        :param num_classes: 痤疮严重等级分类数 (例如 4)
        """
        super(MultiBranchAcneModel, self).__init__()

        # ---- Branch 1: 正脸 ----
        self.branch_front = models.resnet18(pretrained=True)
        # 把最后一层 fc 去掉，便于提取特征
        # resnet18 默认 fc 输出维度是 1000
        # 这里我们保留全局池化后的 512-d 特征
        num_feats = self.branch_front.fc.in_features
        self.branch_front.fc = nn.Identity()  # 直接置空

        # ---- Branch 2: 左侧 ----
        self.branch_left = models.resnet18(pretrained=True)
        self.branch_left.fc = nn.Identity()

        # ---- Branch 3: 右侧 ----
        self.branch_right = models.resnet18(pretrained=True)
        self.branch_right.fc = nn.Identity()

        # 上面 3 个分支提取出来的特征拼接后就是 512*3 = 1536 维
        # 你可以再加个中间层或直接输出
        self.classifier = nn.Sequential(
            nn.Linear(num_feats * 3, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )

    def forward(self, imgs):
        """
        :param imgs: 一个包含 3 个张量的 list [img_front, img_left, img_right]
                     每个张量形状 [B, C, H, W]
        :return: logits, shape=[B, num_classes]
        """
        img_front, img_left, img_right = imgs  # 解包

        # 分别通过三个分支
        feat_front = self.branch_front(img_front)  # shape=[B, 512]
        feat_left = self.branch_left(img_left)  # shape=[B, 512]
        feat_right = self.branch_right(img_right)  # shape=[B, 512]

        # 拼接特征
        feat_cat = torch.cat([feat_front, feat_left, feat_right], dim=1)
        # shape=[B, 1536]

        # 最后分类输出
        logits = self.classifier(feat_cat)  # shape=[B, 4]

        return logits
    


class MultiBranchEfficientNetModel(nn.Module):
    def __init__(self, num_classes=4):
        """
        :param num_classes: 痤疮严重等级分类数 (例如 4)
        """
        super(MultiBranchEfficientNetModel, self).__init__()

        # ---- Branch 1: 正脸 ----
        self.branch_front = models.efficientnet_b0(pretrained=True)
        # EfficientNet 的分类头通常是 self.branch_front.classifier
        # 替换它以获得特征向量
        # efficientnet_b0 的最后一个全连接层的输入特征维度通常是 1280
        num_feats = self.branch_front.classifier[1].in_features  
        # 移除最后的分类器层，使之成为特征提取器
        self.branch_front.classifier = nn.Identity()

        # ---- Branch 2: 左侧 ----
        self.branch_left = models.efficientnet_b0(pretrained=True)
        self.branch_left.classifier = nn.Identity()

        # ---- Branch 3: 右侧 ----
        self.branch_right = models.efficientnet_b0(pretrained=True)
        self.branch_right.classifier = nn.Identity()

        # 三个分支提取的特征拼接后维度
        fused_feat_dim = num_feats * 3

        # 分类器
        self.classifier = nn.Sequential(
            nn.Linear(fused_feat_dim, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )

    def forward(self, imgs):
        """
        :param imgs: 一个包含 3 个张量的 list [img_front, img_left, img_right]
                     每个张量形状 [B, C, H, W]
        :return: logits, shape=[B, num_classes]
        """
        img_front, img_left, img_right = imgs

        # 分别通过三个分支提取特征
        feat_front = self.branch_front(img_front)  # shape=[B, 1280]
        feat_left  = self.branch_left(img_left)    # shape=[B, 1280]
        feat_right = self.branch_right(img_right)  # shape=[B, 1280]

        # 拼接特征
        feat_cat = torch.cat([feat_front, feat_left, feat_right], dim=1)  # shape=[B, 1280*3]

        # 最后分类输出
        logits = self.classifier(feat_cat)  # shape=[B, num_classes]
        return logits