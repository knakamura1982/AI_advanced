import gym


# 何エピソード分，ゲームを実行するか
N_EPISODES = 10

# 1エピソードあたりの最大ステップ数
N_STEPS = 200

# 環境状態や行動，即時報酬などを標準出力に出力するか否か
PRINT_STATUS = False


# 倒立振子ゲームの環境を作成
env = gym.make('CartPole-v0')

# ゲームを N_EPISODES 回実行
for e in range(N_EPISODES):

    print('Episode {0}:'.format(e + 1))

    # まず，ゲームを初期化
    # current_state には環境の初期状態（台車の位置，台車の速度，棒の角度，棒の先端の速度，の4次元ベクトル）が保存される
    current_state = env.reset()

    # N_STEPS 分を1エピソードとして実行
    steps_to_live = N_STEPS
    for t in range(N_STEPS):

        # 現在のゲーム画面をレンダリング
        env.render()

        if PRINT_STATUS:
            # 現在の環境状態を出力
            print('s({0}) = {1}'.format(t, current_state))

        # AIの行動を決める．実行可能な行動は「台車を左に動かす」「台車を右に動かす」の2種類で，整数値で表現
        #   - 0: 台車を左に動かす
        #   - 1: 台車を右に動かす
        action = env.action_space.sample() # これはランダムに行動を選択する，という意味

        if PRINT_STATUS:
            # 選択した行動を出力
            print('a({0}) = {1}'.format(t, action))

        # 選択した行動を実行．戻り値の意味は次の通り
        #   - next_state: 次時刻の環境状態
        #   - reward: 即時報酬（このゲームでは常に +1 ．きちんと学習／制御するためには，プログラマが設定した報酬を加算することが望ましい）
        #   - done: 終了フラグ（N_STEP 分が完了していなくても，ある程度棒が倒れるとそこで終了となる）
        #   - info: デバッグ情報，気にしなくて良い
        next_state, reward, done, info = env.step(action)

        if PRINT_STATUS:
            # 即時報酬の値を出力
            print('r(s({0}), a({0})) = {1}'.format(t, reward))

        # 現在状態の更新
        current_state = next_state

        # このプログラムでは終了フラグは敢えて無視（警告メッセージが出力されるが，気にしない）
        if done:
            if steps_to_live == N_STEPS:
                steps_to_live = t + 1
            #break # 終了フラグを無視したくない場合は，この break をアンコメントすればよい

    print('  episode finished after {0} steps'.format(steps_to_live))
    print('')
