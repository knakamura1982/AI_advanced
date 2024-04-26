import os
import math
import torch
import pickle
import numpy as np


# ε-greedy における ε を制御するクラス
#   - eps_start: ε の初期値
#   - eps_end: ε の最終値（eps_start から開始して，eps_end まで徐々に減らす）
#   - eps_decay: ε を減衰させる速度を制御するパラメータ
#
# [具体的な計算]
#   s = eps_start, e = eps_end, d = eps_decay として，
#   n 回目の行動時における ε の値を ε = e + (s-e) * exp(-n/d) で計算する
class EPSController:

    def __init__(self, eps_start=0.9, eps_end=0.01, eps_decay=5000):
        self.steps = 0
        self.decay = eps_decay
        self.start = eps_start
        self.end = eps_end

    def reset(self):
        self.steps = 0

    def get(self):
            eps = self.end + (self.start - self.end) * math.exp(-self.steps / self.decay)
            self.steps += 1
            return eps


# Replay memory を実現するクラス
#   - capacity: 登録可能なデータの数の上限
class ReplayMemory(object):

    def __init__(self, capacity):
        self.capacity = capacity
        self.size = 0 # 現在の登録データ数

    def __len__(self):
        return self.size

    def size(self):
        return self.size

    # データを一つ追加
    #    - data: （現在状態，行動，報酬，次状態，終了フラグ）の 5 要素からなるリスト
    def append(self, data):
        cs = np.asarray(data[0]) # 現在状態
        ns = np.asarray(data[3]) # 次状態
        ac = np.asarray([data[1]]) # 行動
        rw = np.asarray([data[2]]) # 即時報酬
        fl = np.asarray([1.0 - float(data[4])]) # 終了フラグ
        cs = cs.reshape((1, *cs.shape))
        ns = ns.reshape((1, *ns.shape))
        ac = ac.reshape((1, -1))
        rw = rw.reshape((1, -1))
        fl = fl.reshape((1, -1))
        if self.size == 0:
            # 1番目のデータを replay memory に登録する場合
            self.curr_state = cs
            self.next_state = ns
            self.action = ac
            self.reward = rw
            self.flag = fl
            self.size += 1 # 現在データ数を 1 増やす
        else:
            # 2番目以降のデータを replay memory に登録する場合
            self.curr_state = np.concatenate([self.curr_state, cs], axis=0)
            self.next_state = np.concatenate([self.next_state, ns], axis=0)
            self.action = np.concatenate([self.action, ac], axis=0)
            self.reward = np.concatenate([self.reward, rw], axis=0)
            self.flag = np.concatenate([self.flag, fl], axis=0)
            if self.size > self.capacity:
                # 現在サイズが replay memory 自体のサイズを超えた場合は，最も古いデータを削除
                self.curr_state = self.curr_state[1:]
                self.next_state = self.next_state[1:]
                self.action = self.action[1:]
                self.reward = self.reward[1:]
                self.flag = self.flag[1:]
            else:
                # 古いデータを削除しなかった場合は，現在データ数を 1 増やす
                self.size += 1

    # batch_size 個のデータをランダムにサンプリング
    def sample(self, batch_size):
        if batch_size >= self.size:
            # バッチサイズ以上のデータがまだ登録されていない場合は全データを返すことにする
            C = torch.tensor(self.curr_state, dtype=torch.float32)
            N = torch.tensor(self.next_state, dtype=torch.float32)
            A = torch.tensor(self.action, dtype=torch.long)
            R = torch.tensor(self.reward, dtype=torch.float32)
            D = torch.tensor(self.flag, dtype=torch.float32)
        else:
            perm = np.random.permutation(self.size)[:batch_size]
            C = torch.tensor(self.curr_state[perm], dtype=torch.float32)
            N = torch.tensor(self.next_state[perm], dtype=torch.float32)
            A = torch.tensor(self.action[perm], dtype=torch.long)
            R = torch.tensor(self.reward[perm], dtype=torch.float32)
            D = torch.tensor(self.flag[perm], dtype=torch.float32)
        return C, A, R, N, D


def save_replay_memory(mem_file, memory):
    with open(mem_file, 'wb') as f:
        pickle.dump(memory, f)


def load_replay_memory(mem_file):
    if os.path.isfile(mem_file):
        with open(mem_file, 'rb') as f:
            memory = pickle.load(f)
        print('{} has been loaded.'.format(mem_file))
    return memory


def save_checkpoint(epoch_file, model_file, opt_file, eps_file, epoch, model, opt, eps_controller):
    for param in model.parameters():
        device = param.data.device
        break
    torch.save(model.to('cpu').state_dict(), model_file)
    torch.save(opt.state_dict(), opt_file)
    with open(epoch_file, 'wb') as f:
        pickle.dump(epoch, f)
    with open(eps_file, 'wb') as f:
        pickle.dump(eps_controller, f)
    model.to(device)


def load_checkpoint(epoch_file, model_file, opt_file, eps_file, n_epochs, model, opt):
    init_epoch = 0
    if os.path.isfile(model_file) and os.path.isfile(opt_file) and os.path.isfile(eps_file):
        model.load_state_dict(torch.load(model_file))
        opt.load_state_dict(torch.load(opt_file))
        print('{} has been loaded.'.format(model_file))
        print('{} has been loaded.'.format(opt_file))
        with open(epoch_file, 'rb') as f:
            init_epoch = pickle.load(f)
            last_epoch = n_epochs + init_epoch
        with open(eps_file, 'rb') as f:
            eps_controller = pickle.load(f)
    return init_epoch, last_epoch, model, opt, eps_controller
