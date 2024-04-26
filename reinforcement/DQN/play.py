import sys
import numpy as np
import gymnasium as gym


# 何エピソード分，ゲームを実行するか
N_EPISODES = 10

# 1エピソードあたりの最大ステップ数
N_STEPS = 200


if len(sys.argv) < 2:
    print('usage: python play.py [game environment name]')
    exit()
else:
    env_name = sys.argv[1]


# ゲーム環境を作成
# ちなみに，render_mode=None を指定するとゲーム画面が描画されなくなる
env = gym.make(env_name, render_mode='human')

# 一旦ゲームを初期化し，環境状態の次元数や行動の種類数をチェックする
current_state, info = env.reset()
state_shape = np.asarray(current_state).shape
print('shape of state tensor:', state_shape)
if hasattr(env.action_space, 'n'):
    n_actions = env.action_space.n
    print('num. of actions:', n_actions)
else:
    action_shape = np.asarray(env.action_space.sample()).shape
    print('shape of action tensor:', action_shape)

# ゲームを N_EPISODES 回実行
for e in range(N_EPISODES):

    print('Episode {0}:'.format(e + 1))

    # まず，ゲームを初期化
    current_state, info = env.reset()

    # N_STEPS 分を1エピソードとして実行
    steps_to_live = N_STEPS
    for t in range(N_STEPS):

        # AIの行動をランダム選択
        action = env.action_space.sample()

        # 選択した行動を実行．戻り値の意味は次の通り
        #   - next_state: 次時刻の環境状態
        #   - reward: 即時報酬
        #   - done: 終了フラグ
        #   - truncated, info: このプログラムでは使用しない
        next_state, reward, done, truncated, info = env.step(action)

        # 現在状態の更新
        current_state = next_state

        # N_STEP分が完了する前にゲームが終了した場合は，すぐに次のエピソードに以降
        if done:
            steps_to_live = t + 1
            break

    print('  episode finished after {0} steps'.format(steps_to_live))
    print('')
