import torch
import torch.nn as nn
import torch.nn.functional as F


# 倒立振子ゲーム用の Q-network
# 4次元ベクトル（振子の状態を表現）を入力，2次元ベクトル（2種類の行動のQ値に表現）を出力とする
class CartPoleQNet(nn.Module):

    def __init__(self):
        super(CartPoleQNet, self).__init__()
        # 3層構造，中間層のユニット数は 16 とする（適当に決めたもので，これが良いかどうかは不明）
        self.fc1 = nn.Linear(in_features=4, out_features=16)
        self.fc2 = nn.Linear(in_features=16, out_features=16)
        self.fc3 = nn.Linear(in_features=16, out_features=2)

    def forward(self, x):
        h = F.relu(self.fc1(x))
        h = F.relu(self.fc2(h))
        y = self.fc3(h)
        return y


# Pendulum用の Q_pol ネットワーク
# 3次元の状態ベクトルを入力，1次元の行動ベクトルを出力とする
class PendulumQPolNet(nn.Module):

    def __init__(self):
        super(PendulumQPolNet, self).__init__()
        self.fc1 = nn.Linear(in_features=3, out_features=16)
        self.fc2 = nn.Linear(in_features=16, out_features=16)
        self.fc3 = nn.Linear(in_features=16, out_features=8)
        self.fc4 = nn.Linear(in_features=8, out_features=1)

    def forward(self, x):
        h = F.relu(self.fc1(x))
        h = F.relu(self.fc2(h))
        h = F.relu(self.fc3(h))
        y = 2 * torch.tanh(self.fc4(h))
        return y


# Pendulum用の Q_val ネットワーク
# 3次元の状態ベクトルと1次元の行動ベクトルのペアを入力とし，単一のスカラー値（Q値）を出力する
class PendulumQValNet(nn.Module):

    def __init__(self):
        super(PendulumQValNet, self).__init__()
        self.fc1 = nn.Linear(in_features=3, out_features=16)
        self.fc2 = nn.Linear(in_features=16+1, out_features=16)
        self.fc3 = nn.Linear(in_features=16, out_features=8)
        self.fc4 = nn.Linear(in_features=8, out_features=1)

    def forward(self, x, a):
        h = F.relu(self.fc1(x))
        h = torch.cat([h, a], dim=1)
        h = F.relu(self.fc2(h))
        h = F.relu(self.fc3(h))
        y = self.fc4(h)
        return y
