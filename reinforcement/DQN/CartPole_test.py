import torch
import torch.nn as nn
import torch.optim as optim
import gymnasium as gym
import numpy as np
from networks import CartPoleQNet


##### 実行設定: ここから #####

# 使用するデバイス
DEVICE = 'cpu'

# 何エピソード分，テスト実行するか
N_EPISODES = 10

# 1エピソードあたりの最大ステップ数
N_STEPS = 200

##### 実行設定: ここまで #####


# Q-network を用意
Q_net = CartPoleQNet() # Q-network
Q_net.load_state_dict(torch.load('CartPole_Q.pth'))
Q_net = Q_net.to(DEVICE)
Q_net.eval()

# 倒立振子ゲームの環境を作成
# ちなみに，render_mode=None を指定するとゲーム画面が描画されなくなる
env = gym.make('CartPole-v1', render_mode='human')

# ここから学習．ゲームを N_EPISODES 回実行
eps = 1.0
for e in range(N_EPISODES):

    print('Episode {0}:'.format(e + 1))

    # ゲームを初期化し，初期状態を取得
    current_state, info = env.reset()

    # 1エピソード分を実行
    total_reward = 0
    steps_to_live = N_STEPS
    for t in range(N_STEPS):

        # AI の行動を決める
        s = torch.tensor(np.asarray([current_state]), dtype=torch.float32).to(DEVICE)
        action = int(torch.argmax(Q_net(s), dim=1)) # 現在の Q-network の下で Q 値最大の行動を選択

        # 選択した行動を実行
        next_state, reward, done, truncated, info = env.step(action)
        total_reward += reward

        # 現在状態の更新
        current_state = next_state

        # このプログラムでは終了フラグは敢えて無視（警告メッセージが出力されるが，気にしない）
        if done:
            if steps_to_live == N_STEPS:
                steps_to_live = t + 1
            #break # 終了フラグを無視したくない場合は，この break をアンコメントすればよい

    print('  episode finished after {0} steps'.format(steps_to_live))
    print('  total reward = {0}'.format(total_reward))
    print('')
