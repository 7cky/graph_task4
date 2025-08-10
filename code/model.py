import torch
import torch.nn as nn
import torch.nn.functional as F

class TransE(nn.Module):
    def __init__(self, num_entities, num_relations, embedding_dim=100, margin=1.0):
        super().__init__()
        # （保持原有初始化代码）
        self.num_entities = num_entities
        self.num_relations = num_relations
        self.embedding_dim = embedding_dim
        self.margin = margin  # 用于MarginRankingLoss的边际值
        self.entity_emb = nn.Embedding(num_entities, embedding_dim)
        self.rel_emb = nn.Embedding(num_relations, embedding_dim)
        self.entity_emb.weight.data.uniform_(-6 / (embedding_dim ** 0.5), 6 / (embedding_dim ** 0.5))
        self.rel_emb.weight.data.uniform_(-6 / (embedding_dim ** 0.5), 6 / (embedding_dim ** 0.5))
        self.entity_emb.weight.data = F.normalize(self.entity_emb.weight.data, p=2, dim=1)  # 初始化归一化

    def forward(self, h_idx, r_idx, t_idx):
        h = self.entity_emb(h_idx)
        r = self.rel_emb(r_idx)
        t = self.entity_emb(t_idx)
        
        # 关键修复：训练时对实体嵌入动态归一化
        h = F.normalize(h, p=2, dim=1)  # 确保h的模长为1
        t = F.normalize(t, p=2, dim=1)  # 确保t的模长为1
        
        # 计算TransE分数（距离越小越合理）
        score = torch.norm(h + r - t, p=2, dim=1)
        return score


class RotatE(nn.Module):
    """RotatE模型：将关系表示为复平面上的旋转操作"""
    def __init__(self, num_entities, num_relations, embedding_dim=100):
        super().__init__()
        self.num_entities = num_entities
        self.num_relations = num_relations
        self.embedding_dim = embedding_dim  # 实部+虚部的总维度（原论文中为2*k）

        # 实体嵌入（分为实部和虚部）和关系嵌入（角度）
        self.entity_emb = nn.Embedding(num_entities, embedding_dim)
        self.rel_emb = nn.Embedding(num_relations, embedding_dim // 2)  # 角度维度为embedding_dim/2

        # 初始化嵌入
        self.entity_emb.weight.data.uniform_(-0.05, 0.05)
        self.rel_emb.weight.data.uniform_(-0.05, 0.05)

    def forward(self, h_idx, r_idx, t_idx):
        """计算三元组(h, r, t)的得分"""
        h = self.entity_emb(h_idx)  # (batch_size, embedding_dim)
        r = self.rel_emb(r_idx)     # (batch_size, embedding_dim//2)
        t = self.entity_emb(t_idx)  # (batch_size, embedding_dim)

        # 将实体嵌入分为实部和虚部（各占一半维度）
        h_re, h_im = torch.chunk(h, 2, dim=1)  # 各为(batch_size, embedding_dim//2)
        t_re, t_im = torch.chunk(t, 2, dim=1)

        # 关系角度的余弦和正弦（旋转因子）
        r_cos = torch.cos(r)  # (batch_size, embedding_dim//2)
        r_sin = torch.sin(r)

        # 旋转操作：h_rot = h * r（复平面乘法）
        h_rot_re = h_re * r_cos - h_im * r_sin  # 实部
        h_rot_im = h_re * r_sin + h_im * r_cos  # 虚部

        # RotatE得分函数：||(h_rot_re, h_rot_im) - (t_re, t_im)||_2
        score = torch.norm(torch.cat([h_rot_re - t_re, h_rot_im - t_im], dim=1), p=2, dim=1)
        return score

class ConvE(nn.Module):
    """ConvE模型：通过卷积操作捕捉实体和关系的交互特征"""
    def __init__(self, num_entities, num_relations, embedding_dim=200, hidden_dim=9728):
        super().__init__()
        self.num_entities = num_entities
        self.num_relations = num_relations
        self.embedding_dim = embedding_dim
        
        # 计算嵌入reshape的维度（确保height * width = embedding_dim）
        self.height = int(embedding_dim ** 0.5)  # 开平方得到高度
        self.width = embedding_dim // self.height  # 宽度 = 嵌入维度 / 高度
        
        # 确保height * width = embedding_dim
        if self.height * self.width != embedding_dim:
            raise ValueError(f"embedding_dim必须是完全平方数，当前为{embedding_dim}")

        # 实体和关系嵌入层
        self.entity_emb = nn.Embedding(num_entities, embedding_dim)
        self.rel_emb = nn.Embedding(num_relations, embedding_dim)

        # 卷积层
        self.conv1 = nn.Conv2d(
            in_channels=1,        # 输入通道数（单通道）
            out_channels=32,      # 输出通道数（32个卷积核）
            kernel_size=(3, 3),   # 卷积核大小3x3
            padding=0             # 不填充
        )

        # 全连接层（维度需根据卷积输出计算）
        self.fc1 = nn.Linear(hidden_dim, embedding_dim)
        self.bn1 = nn.BatchNorm2d(1)    # 卷积前的批归一化
        self.bn2 = nn.BatchNorm2d(32)   # 卷积后的批归一化
        self.bn3 = nn.BatchNorm1d(embedding_dim)  # 全连接后的批归一化
        self.dropout = nn.Dropout(0.2)  # dropout层防止过拟合

        # 初始化嵌入和权重
        self.entity_emb.weight.data.uniform_(-0.05, 0.05)
        self.rel_emb.weight.data.uniform_(-0.05, 0.05)
        nn.init.xavier_uniform_(self.conv1.weight)
        nn.init.xavier_uniform_(self.fc1.weight)

    def forward(self, h_idx, r_idx, t_idx):
        """计算三元组(h, r, t)的得分"""
        h = self.entity_emb(h_idx)  # (batch_size, embedding_dim)
        r = self.rel_emb(r_idx)     # (batch_size, embedding_dim)
        t = self.entity_emb(t_idx)  # (batch_size, embedding_dim)

        # 1. 将实体和关系嵌入reshape为2D矩阵并拼接
        h_reshaped = h.view(-1, 1, self.height, self.width)  # 动态适应不同嵌入维度
        r_reshaped = r.view(-1, 1, self.height, self.width)
        x = torch.cat([h_reshaped, r_reshaped], dim=2)  # 在高度方向拼接

        # 2. 卷积特征提取
        x = self.bn1(x)
        x = self.conv1(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = x.view(x.shape[0], -1)  # 展平

        # 3. 全连接层映射到实体嵌入空间
        x = self.fc1(x)
        x = self.bn3(x)
        x = F.relu(x)

        # 4. ConvE得分函数：与目标实体嵌入的点积（取负）
        score = -torch.sum(x * t, dim=1)  # (batch_size,)
        return score



