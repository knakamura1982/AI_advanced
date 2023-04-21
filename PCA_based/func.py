import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm
from PIL import Image


# 画像ファイルリスト作成関数
def make_list(filename, item, data_dir='./'):
    df = pd.read_csv(filename)
    image_list = df[item].to_list()
    for i in range(len(image_list)):
        image_list[i] = os.path.join(data_dir, image_list[i])
    return image_list


# 単一画像読み込み関数
def read_single_image(image_list, ID):
    img = Image.open(image_list[ID]).convert('RGB')
    img = np.array(img)
    h, w, _ = img.shape
    if h > w:
        c = (h - w) // 2
        img = img[c:c+w,:,:] # 中央の部分だけを取り出す
    elif w > h:
        c = (w - h) // 2
        img = img[:,c:c+h,:] # 中央の部分だけを取り出す
    img = img.reshape(-1)
    return np.asarray([img], dtype=np.float32) / 255


# 画像データセット読み込み関数
def read_images(image_list, IDs):
    images = []
    for i in IDs:
        img = read_single_image(image_list, i)
        images.append(img[0])
    return np.asarray(images)


# 画像データを固有空間に射影して圧縮する関数
def compress(img, mean, eigenvectors):
    return eigenvectors @ (img - mean).T


# 圧縮表現から画像を生成する関数
def generate(c, mean, eigenvectors):
    img = c.T @ eigenvectors + mean
    return img


# 画像表示関数
def show_single_image(img, shape, window_name='no title'):
    img = img.reshape(shape)
    img = (255 * np.minimum(np.ones(img.shape), np.maximum(np.zeros(img.shape), img))).astype(np.uint8) # 画素値を 0～1 に丸め込んだ後で 255 倍する
    plt.subplots_adjust(left=0.04, right=0.96, bottom=0.04, top=0.96)
    plt.axis('off')
    plt.title(window_name)
    plt.imshow(img, cmap=cm.gray, interpolation='nearest')
    plt.show()
    plt.close()

# 複数画像表示関数
def show_images(imgs, shape, num, num_per_row=0, window_name='no title'):
    if num_per_row <= 0:
        num_per_row = int(np.ceil(np.sqrt(num)))
    n_total = min(imgs.shape[0], num) # 保存するデータの総数
    n_rows = int(np.ceil(n_total / num_per_row)) # 保存先画像においてデータを何行に分けて表示するか
    plt.figure(window_name, figsize=(1 + num_per_row * shape[1] / 128, 1 + n_rows * shape[0] / 128))
    plt.subplots_adjust(left=0.04, right=0.96, bottom=0.04, top=0.96)
    for i in range(0, n_total):
        img = imgs[i].reshape(shape)
        img = (255 * np.minimum(np.ones(img.shape), np.maximum(np.zeros(img.shape), img))).astype(np.uint8) # 画素値を 0～1 に丸め込んだ後で 255 倍する
        plt.subplot(n_rows, num_per_row, i+1)
        plt.axis('off')
        plt.imshow(img, cmap=cm.gray, interpolation='nearest')
    plt.show()
    plt.close()
