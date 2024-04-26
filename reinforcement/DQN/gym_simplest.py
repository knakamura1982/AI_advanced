import gymnasium as gym


# 環境の名称（タスク名）を指定
# ちなみに，render_mode=None を指定するとゲーム画面が描画されなくなる
env = gym.make('MountainCar-v0', render_mode='human')

# 初期化
state = env.reset()

# ひとまず1000ステップ分実行
N_STEPS = 1000
for t in range(N_STEPS):

    env.render() # 現在の状態を画面にレンダリング

    action = env.action_space.sample() # AIの行動をランダム選択

    # 選択した行動を実行．戻り値の意味は次の通り
    #   - state: 次時刻の環境状態
    #   - reward: 即時報酬
    #   - done: 終了フラグ
    #   - info: デバッグ情報（気にしなくてOK）
    state, reward, done, info = env.step(action)

    # 終了フラグが True になったら終了
    if done:
        break
