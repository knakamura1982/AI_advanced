import gym
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from networks import PendulumQPolNet


##### 実行設定: ここから #####

# 使用するデバイス
DEVICE = 'cpu'

# 何エピソード分，テスト実行するか
N_EPISODES = 10

# 1エピソードあたりの最大ステップ数
N_STEPS = 200

##### 実行設定: ここまで #####


# Q-network を用意
QP_net = PendulumQPolNet() # Q_pol
QP_net.load_state_dict(torch.load('Pendulum_QP.pth'))
QP_net = QP_net.to(DEVICE)
QP_net.eval()

# ゲーム環境を作成
env = gym.make('Pendulum-v0')

# ここからテスト実行．ゲームを N_EPISODES 回実行
for e in range(N_EPISODES):

    print('Episode {0}:'.format(e + 1))

    # ゲームを初期化し，初期状態を取得
    current_state = env.reset()

    # 1エピソード分を実行
    total_reward = 0
    steps_to_live = N_STEPS
    for t in range(N_STEPS):

        # 現在のゲーム画面をレンダリング
        env.render()

        # AI の行動を決める
        s = torch.tensor(np.asarray([current_state]), dtype=torch.float32).to(DEVICE)
        action = QP_net(s).detach().reshape(-1).numpy() # 現在の Q-network の下で行動を選択

        # 選択した行動を実行
        next_state, reward, done, info = env.step(action)
        total_reward += reward

        # 現在状態の更新
        current_state = next_state

        # 終了フラグが立った場合は次のエピソードに移行
        if done:
            steps_to_live = t + 1
            break

    print('  episode finished after {0} steps'.format(steps_to_live))
    print('  total reward = {0}'.format(total_reward))
    print('')
