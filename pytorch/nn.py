import torch
import torch.nn as nn
import torch.nn.functional as F

# 搭建MLP分类模型
class MLPClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes):
        super().__init__()
        # 三层全连接线性层
        self.linear1 = nn.Linear(input_dim, hidden_dim)    # 输入层→隐藏层1
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)  # 隐藏层1→隐藏层2
        self.linear3 = nn.Linear(hidden_dim, num_classes) # 隐藏层→输出层

    def forward(self, x):
        # 前向传播流程
        x = F.relu(self.linear1(x))  # 第一层+激活函数
        x = F.relu(self.linear2(x))  # 第二层+激活函数
        out = self.linear3(x)        # 输出层不加激活，后续用交叉熵损失自带softmax
        return out

# ---------------------- 测试代码 ----------------------
if __name__ == "__main__":
    # 超参数设置
    input_size = 10    # 输入特征维度（比如10维数据）
    hidden_size = 32   # 隐藏层神经元数量
    class_num = 3      # 分类类别数（3分类任务）

    # 初始化模型
    model = MLPClassifier(input_size, hidden_size, class_num)

    # 构造模拟输入：batch_size=4，每个样本10维特征
    sample_input = torch.randn(4, input_size)

    # 前向传播得到预测分数
    pred = model(sample_input)
    print("输出形状 [批次, 类别数]:", pred.shape)
    print("原始预测logits：\n", pred)