import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
import matplotlib.cm as cm
import cv2
from PIL import Image

def generate_random_data(data_length: int):
    """縦960, 横1280のうちランダムに1000点を選択し(x, y)の形式で格納

    Args:
        data_length (int): データの個数
    """
    x = np.random.randint(0, 1280, data_length)
    y = np.random.randint(0, 960, data_length)
    return x, y

def validate_data_length(x: np.array, y: np.array):
    """データの個数を確認（xとyの長さが一致していることを確かめる）

    Args:
        x (np.array): x座標の配列
        y (np.array): y座標の配列
    """
    if len(x) != len(y):
        raise ValueError('xとyのデータ数が異なります．')

def generate_heatmap(title: str, x_label: str, y_label: str, x: np.array, y: np.array, bins_x: int, bins_y: int, vmin: int, vmax: int):
    """ヒートマップ（グリッドタイプ）作成し.pngで保存

    Args:
        title (str): グラフのタイトル（.pngに表示される）
        x_label (str): 横軸のラベル
        y_label (str): 縦軸のラベル
        x (np.array): x座標の配列
        y (np.array): y座標の配列
    """
    validate_data_length(x, y)
    fig, ax = plt.subplots() # 返り値がタプルのため，figを明示的に指定している
    sns.heatmap(np.histogram2d(x, y, bins=[bins_x, bins_y])[0], annot=True, cbar=False, cmap='Reds', vmin = vmin, vmax = vmax)
    ax.axis("off")
    fig.subplots_adjust(left=0,bottom=0,right= 1,top=1)
    plt.savefig('detected_heatmap.png')
    plt.show()


#ヒートマップと画像を重ねる関数
def overlap_image(sample_img: str):
    img1 = cv2.imread(sample_img)
    img2 = cv2.imread('detected_heatmap.png')
    
    img2 = cv2.resize(img2,(img1.shape[1],img1.shape[0]))
    
    assert img1.shape == img2.shape
    
    alpha = 0.5 
    blended = cv2.addWeighted(img1, alpha, img2, 1 - alpha, 0)
    cv2.imwrite('heatmap_image.png', blended)
    
    fig, ax = plt.subplots()
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    plt.suptitle(title)
    im = Image.open('heatmap_image.png')
    im = np.array(im)
    heatmap = ax.imshow(im, alpha=1.0)
    plt.savefig('heatmap_image.png')


if __name__ == '__main__':
    title = 'Detected Heatmap'
    
    #読み込む画像
    sample_img = 'sample_image.png'

    #分割数
    bins_x = 10
    bins_y = 10

    #カラーバーの最大値と最小値
    vmin = 0
    vmax = 16

    # 軸のラベルはアルファベットのみ指定可能
    x_label = 'X'
    y_label = 'Y'

    x, y = generate_random_data(1000)

    generate_heatmap(title, x_label, y_label, x, y, bins_x, bins_y, vmin, vmax)
    
    #ヒートマップと画像を重ねる
    overlap_image(sample_img)

