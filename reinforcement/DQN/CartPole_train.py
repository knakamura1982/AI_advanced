import os
import torch
import torch.nn as nn
import torch.optim as optim
import gymnasium as gym
import numpy as np
from utils import EPSController, ReplayMemory, save_checkpoint, load_checkpoint, save_replay_memory, load_replay_memory
from networks import CartPoleQNet


##### 中断／再開に関する設定: ここから #####

# 前回の試行の続きを行いたい場合は True にする -> 再開モードになる
RESTART_MODE = False

# 中断／再開の際に用いる一時ファイル
CHECKPOINT_REPLAY_MEMORY = 'CartPole_checkpoint_memory.pkl'
CHECKPOINT_EPISODE_ID = 'CartPole_checkpoint_epoch.pkl'
CHECKPOINT_MODEL = 'CartPole_checkpoint_model.pth'
CHECKPOINT_OPT = 'CartPole_checkpoint_opt.pth'
CHECKPOINT_EPS = 'CartPole_checkpoint_eps.pkl'

##### 中断／再開に関する設定: ここまで #####


##### 学習設定: ここから #####

# 使用するデバイス
DEVICE = 'cpu'

# 学習時のバッチサイズ
BATCH_SIZE = 128

# Replay memory に登録で可能なデータ数の最大値
REPLAY_MEMORY_CAPACITY = 10000

# Fixed target Q-network において target network のパラメータを何エピソードに 1 回の割合で更新するか
TARGET_UPDATE_FREQ = 10

# 何エピソード分，学習を実行するか
# 再開モードの場合も, このエピソード数の分だけ追加学習される（N_EPISODESは最終エピソード番号ではない）
N_EPISODES = 100

# 1エピソードあたりの最大ステップ数
N_STEPS = 200

# 割引報酬和を計算する際の割引率
DISCOUNT_FACTOR = 0.99

# ε-greedy に関する設定値（ utils.py の関連部分を参照のこと）
EPS_START = 0.9
EPS_END = 0.01
EPS_DECAY = 5000

##### 学習設定: ここまで #####


# Q-network を用意
Q_net = CartPoleQNet().to(DEVICE) # 学習対象の Q-network
T_net = CartPoleQNet().to(DEVICE) # Fixed target Q-network 用．学習対象の Q network とは別物として用意
Q_net.train() # Q-network は学習モードで運用
T_net.eval() # target network は学習対象ではないので，非学習モードで運用

# 最適化アルゴリズムの指定
optimizer = optim.Adam(Q_net.parameters())

# 損失関数
loss_func = nn.SmoothL1Loss()

# Experience replay 用の replay memory を用意
memory = ReplayMemory(capacity=REPLAY_MEMORY_CAPACITY)

# ε-greedy における ε を決定するクラスを用意
EPS = EPSController(EPS_START, EPS_END, EPS_DECAY)

# 再開モードの場合は，前回チェックポイントから情報をロードして学習再開
if RESTART_MODE:
    INIT_EPISODE_ID, LAST_EPISODE_ID, Q_net, optimizer, EPS = load_checkpoint(CHECKPOINT_EPISODE_ID, CHECKPOINT_MODEL, CHECKPOINT_OPT, CHECKPOINT_EPS, N_EPISODES, Q_net, optimizer)
    memory = load_replay_memory(CHECKPOINT_REPLAY_MEMORY)
    print('')
else:
    INIT_EPISODE_ID = 0
    LAST_EPISODE_ID = INIT_EPISODE_ID + N_EPISODES

# 倒立振子ゲームの環境を作成
# render_mode=None を指定するとゲーム画面が描画されなくなり，学習を高速化できる
env = gym.make('CartPole-v1', render_mode='human')

# ここから学習．ゲームを N_EPISODES 回実行
eps = 1.0
for e in range(INIT_EPISODE_ID, LAST_EPISODE_ID):

    print('Episode {0}:'.format(e + 1))
    print('  epsilon = {0}'.format(eps))

    # Fixed target Q-network における target network のパラメータを
    # 学習対象の Q network のパラメータに同期させる
    if e % TARGET_UPDATE_FREQ == 0:
        T_net.load_state_dict(Q_net.state_dict())

    # ゲームを初期化し，初期状態を取得
    current_state, info = env.reset()

    # 1エピソード分を実行
    n_data = 0
    sum_loss = 0
    total_reward = 0
    steps_to_live = N_STEPS
    for t in range(N_STEPS):

        # Replay memory に登録されているデータ数を確認する
        mem_size = len(memory)
        learn_flag = True
        if mem_size < BATCH_SIZE:
            learn_flag = False # 登録データ数が BATCH_SIZE に満たない間は学習せず，事例収集に専念する

        # ε-greedy で AI の行動を決める
        if learn_flag:
            eps = EPS.get()
        else:
            eps = 1.0 # 事例収集に専念するときは，ε=1 として 100% ランダム行動するようにする
        p = np.random.rand()
        if p < eps:
            action = env.action_space.sample() # ランダムに行動を選択
        else:
            Q_net.eval() # 行動選択のため，Q-network を一時的に非学習モードにする
            s = torch.tensor(np.asarray([current_state]), dtype=torch.float32).to(DEVICE)
            action = int(torch.argmax(Q_net(s), dim=1)) # 現在の Q-network の下で Q 値最大の行動を選択
            Q_net.train()

        # 選択した行動を実行
        next_state, reward, done, truncated, info = env.step(action)
        total_reward += reward

        # Replay memory にデータを登録
        data = [current_state, action, reward, next_state, done]
        memory.append(data)

        # 現在状態の更新
        current_state = next_state

        # Replay memory の登録データ数が BATCH_SIZE を超えている場合のみ学習処理を実行
        if learn_flag:

            for param in Q_net.parameters():
                param.grad = None

            # Replay memory からランダムにデータをサンプリングしてミニバッチを構成
            C, A, R, N, D = memory.sample(batch_size=BATCH_SIZE)
            C = C.to(DEVICE)
            A = A.to(DEVICE)
            R = R.to(DEVICE)
            N = N.to(DEVICE)
            D = D.to(DEVICE)

            # まず Q(s^t, a^t; θ)を求める．
            # このプログラムでは C が s^t，A が a^t に相当する．
            # Q_net(C) で全行動に関するQ値が得られるが，ここから実際に選択された行動に関するQ値のみを抜き出す
            Q = Q_net(C).gather(1, A)

            # 次に max_m Q(s^{t+1}, a_m; θ^-) を求める
            # このプログラムでは N が s^{t+1} に相当する．
            # D は終了フラグが立っている時は 0，そうでない時は 1 であり，これを乗じることにより最終時刻における損失関数の変化に対応している
            Qm = torch.max(T_net(N).detach(), dim=1, keepdim=True)[0] * D

            # 学習
            loss = loss_func(Q, R + DISCOUNT_FACTOR * Qm)
            loss.backward()
            optimizer.step()
            sum_loss += float(loss) * len(C)
            n_data += len(C)

        # 振子が倒れたら次のエピソードに移行
        if done:
            steps_to_live = t + 1
            break

    # 現在の学習状態を一時ファイルに保存
    save_checkpoint(CHECKPOINT_EPISODE_ID, CHECKPOINT_MODEL, CHECKPOINT_OPT, CHECKPOINT_EPS, e+1, Q_net, optimizer, EPS)
    save_replay_memory(CHECKPOINT_REPLAY_MEMORY, memory)

    avg_loss = sum_loss if n_data == 0 else sum_loss / n_data
    print('  episode finished after {0} steps'.format(steps_to_live))
    print('  train loss = {0:.6f}'.format(avg_loss))
    print('  total reward = {0}'.format(total_reward))
    print('')

torch.save(Q_net.state_dict(), 'CartPole_Q.pth')
