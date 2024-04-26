import torch
import torch.nn as nn
import torch.optim as optim
import gymnasium as gym
import numpy as np
from utils import EPSController, ReplayMemory, save_checkpoint, load_checkpoint, save_replay_memory, load_replay_memory
from networks import PendulumQPolNet, PendulumQValNet


##### 中断／再開に関する設定: ここから #####

# 前回の試行の続きを行いたい場合は True にする -> 再開モードになる
RESTART_MODE = False

# 中断／再開の際に用いる一時ファイル
CHECKPOINT_REPLAY_MEMORY = 'Pendulum_checkpoint_memory.pkl'
CHECKPOINT_EPISODE_ID = 'Pendulum_checkpoint_epoch.pkl'
CHECKPOINT_QP_MODEL = 'Pendulum_checkpoint_QP_model.pth'
CHECKPOINT_QV_MODEL = 'Pendulum_checkpoint_QV_model.pth'
CHECKPOINT_QP_OPT = 'Pendulum_checkpoint_QP_opt.pth'
CHECKPOINT_QV_OPT = 'Pendulum_checkpoint_QV_opt.pth'
CHECKPOINT_EPS = 'Pendulum_checkpoint_eps.pkl'

##### 中断／再開に関する設定: ここまで #####


##### 学習設定: ここから #####

# 使用するデバイス
DEVICE = 'cpu'

# 学習時のバッチサイズ
BATCH_SIZE = 128

# Replay memory に登録で可能なデータ数の最大値
REPLAY_MEMORY_CAPACITY = 100000

# Fixed target Q-network において target network のパラメータを何エピソードに 1 回の割合で更新するか
TARGET_UPDATE_FREQ = 10

# 何エピソード分，学習を実行するか
# 再開モードの場合も, このエピソード数の分だけ追加学習される（N_EPISODESは最終エピソード番号ではない）
N_EPISODES = 2000

# 1エピソードあたりの最大ステップ数
N_STEPS = 200

# 割引報酬和を計算する際の割引率
DISCOUNT_FACTOR = 0.98

# ε-greedy に関する設定値（ utils.py の関連部分を参照のこと）
EPS_START = 1.0
EPS_END = 0.01
EPS_DECAY = 500

##### 学習設定: ここまで #####


# Q-network を用意
QP_net = PendulumQPolNet().to(DEVICE) # 学習対象の Q-network
QV_net = PendulumQValNet().to(DEVICE) # 同上
TP_net = PendulumQPolNet().to(DEVICE) # Fixed target Q-network 用．学習対象の Q network とは別物として用意
TV_net = PendulumQValNet().to(DEVICE) # 同上
QP_net.train() # Q-network は学習モードで運用
QV_net.train() # 同上
TP_net.eval() # target network は学習対象ではないので，非学習モードで運用
TV_net.eval() # 同上

# 最適化アルゴリズムの指定
QP_optimizer = optim.Adam(QP_net.parameters())
QV_optimizer = optim.Adam(QV_net.parameters())

# 損失関数
loss_func = nn.SmoothL1Loss()

# Experience replay 用の replay memory を用意
memory = ReplayMemory(capacity=REPLAY_MEMORY_CAPACITY)

# ε-greedy における ε を決定するクラスを用意
EPS = EPSController(EPS_START, EPS_END, EPS_DECAY)

# 再開モードの場合は，前回チェックポイントから情報をロードして学習再開
if RESTART_MODE:
    INIT_EPISODE_ID, LAST_EPISODE_ID, QP_net, QP_optimizer, EPS = load_checkpoint(CHECKPOINT_EPISODE_ID, CHECKPOINT_QP_MODEL, CHECKPOINT_QP_OPT, CHECKPOINT_EPS, N_EPISODES, QP_net, QP_optimizer)
    _, _, QV_net, QV_optimizer, _ = load_checkpoint(CHECKPOINT_EPISODE_ID, CHECKPOINT_QV_MODEL, CHECKPOINT_QV_OPT, CHECKPOINT_EPS, N_EPISODES, QV_net, QV_optimizer)
    memory = load_replay_memory(CHECKPOINT_REPLAY_MEMORY)
    print('')
else:
    INIT_EPISODE_ID = 0
    LAST_EPISODE_ID = INIT_EPISODE_ID + N_EPISODES

# ゲーム環境を作成
# render_mode=None を指定するとゲーム画面が描画されなくなり，学習を高速化できる
env = gym.make('Pendulum-v1', render_mode='human')

# ここから学習．ゲームを N_EPISODES 回実行
for e in range(INIT_EPISODE_ID, LAST_EPISODE_ID):

    print('Episode {0}:'.format(e + 1))

    # Fixed target Q-network における target network のパラメータを
    # 学習対象の Q network のパラメータに同期させる
    if e % TARGET_UPDATE_FREQ == 0:
        TP_net.load_state_dict(QP_net.state_dict())
        TV_net.load_state_dict(QV_net.state_dict())

    # Replay memory に登録されているデータ数を確認する
    mem_size = len(memory)
    learn_flag = True
    if mem_size < BATCH_SIZE:
        learn_flag = False # 登録データ数が BATCH_SIZE に満たない間は学習せず，事例収集に専念する

    # ε-greedy の ε は同一エピソード内では固定にする
    if learn_flag:
        eps = EPS.get()
    else:
        eps = 1.0 # 事例収集に専念するときは，ε=1 として 100% ランダム行動するようにする
    print('  epsilon = {0}'.format(eps))

    # ゲームを初期化し，初期状態を取得
    current_state, info = env.reset()

    # 1エピソード分を実行
    n_data = 0
    sum_QP_loss = 0
    sum_QV_loss = 0
    total_reward = 0
    steps_to_live = N_STEPS
    for t in range(N_STEPS):

        # ε-greedy で AI の行動を決める
        p = np.random.rand()
        if p < eps:
            action = env.action_space.sample() # ランダムに行動を選択
        else:
            QP_net.eval() # 行動選択のため，Q-network を一時的に非学習モードにする
            s = torch.tensor(np.asarray([current_state]), dtype=torch.float32).to(DEVICE)
            action = QP_net(s).detach().reshape(-1).numpy() # 現在の Q-network の下で行動を選択
            QP_net.train()

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

            for param in QP_net.parameters():
                param.grad = None
            for param in QV_net.parameters():
                param.grad = None

            # Replay memory からランダムにデータをサンプリングしてミニバッチを構成
            C, A, R, N, D = memory.sample(batch_size=BATCH_SIZE)
            C = C.to(DEVICE)
            A = A.to(DEVICE)
            R = R.to(DEVICE)
            N = N.to(DEVICE)
            D = D.to(DEVICE)

            # QV_net の学習
            Q = QV_net(C, A)
            Am = TP_net(N)
            Qm = TV_net(N, Am) * D
            QV_loss = loss_func(Q, R + DISCOUNT_FACTOR * Qm)
            QV_loss.backward()
            QV_optimizer.step()
            sum_QV_loss += float(QV_loss) * len(C)
            n_data += len(C)

            # QP_net の学習
            QP_loss = -torch.mean(QV_net(C, QP_net(C)))
            QP_loss.backward()
            QP_optimizer.step()
            sum_QP_loss += float(QP_loss) * len(C)

        # 終了フラグが立った場合は次のエピソードに移行
        if done:
            steps_to_live = t + 1
            break

    # 現在の学習状態を一時ファイルに保存
    save_checkpoint(CHECKPOINT_EPISODE_ID, CHECKPOINT_QP_MODEL, CHECKPOINT_QP_OPT, CHECKPOINT_EPS, e+1, QP_net, QP_optimizer, EPS)
    save_checkpoint(CHECKPOINT_EPISODE_ID, CHECKPOINT_QV_MODEL, CHECKPOINT_QV_OPT, CHECKPOINT_EPS, e+1, QV_net, QV_optimizer, EPS)
    save_replay_memory(CHECKPOINT_REPLAY_MEMORY, memory)

    avg_QP_loss = sum_QP_loss if n_data == 0 else sum_QP_loss / n_data
    avg_QV_loss = sum_QV_loss if n_data == 0 else sum_QV_loss / n_data
    print('  episode finished after {0} steps'.format(steps_to_live))
    print('  train QPol loss = {0:.6f}'.format(avg_QP_loss))
    print('  train QVal loss = {0:.6f}'.format(avg_QV_loss))
    print('  total reward = {0}'.format(total_reward))
    print('')

torch.save(QP_net.state_dict(), 'Pendulum_QP.pth')
torch.save(QV_net.state_dict(), 'Pendulum_QV.pth')
