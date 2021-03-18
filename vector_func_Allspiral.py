import numpy as np
import math
import time
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.axes3d import Axes3D

fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(1, 1, 1, projection='3d')


# ソレノイドの磁場をシミュレーション
def calc(X, Xs, Y, Ys, Z, Zs, Q, i):
    res = []
    tmp = []
    R = np.sqrt((X-Xs[0, i])**2+(Y-Ys[0, i])**2+(Z-Zs[0, i])**2)
    if i > 0:
        crX = (Ys[0, i]-Ys[0, i-1])*(Z-Zs[0, i]) - (Zs[0, i]-Zs[0, i-1])*(Y-Ys[0, i])
        crY = (Zs[0, i]-Zs[0, i-1])*(X-Xs[0, i]) - (Xs[0, i]-Xs[0, i-1])*(Z-Zs[0, i])
        crZ = (Xs[0, i]-Xs[0, i-1])*(Y-Ys[0, i]) - (Ys[0, i]-Ys[0, i-1])*(X-Xs[0, i])
    elif i == 0:
        crX = 0
        crY = 0
        crZ = 0
    tmp.append(
        np.divide(
            (Q*crX), R**3, out=np.zeros_like((Q*(X-Xs[0, i]))), where=R != 0
            )
        )
    tmp.append(
        np.divide(
            (Q*crY), R**3, out=np.zeros_like((Q*(Y-Ys[0, i]))), where=R != 0
            )
        )
    tmp.append(
        np.divide(
            (Q*crZ), R**3, out=np.zeros_like((Q*(Z-Zs[0, i]))), where=R != 0
            )
        )
    res.append(tmp)
    return res


def plot():
    start = time.time()
    # らせん上に点電荷を生成し、それらの磁場の和を考える
    # 格子点の生成
    LL = 8
    LX, LY, LZ = LL, LL, LL
    gridwidth = 2
    X, Y, Z = np.meshgrid(
        np.arange(-LX, LX+1, gridwidth),
        np.arange(-LY, LY+1, gridwidth),
        np.arange(-LZ, LZ+1, gridwidth)
        )

    # 置きなおし
    LXx = 1+math.floor(LX/gridwidth)*2
    LYy = 1+math.floor(LY/gridwidth)*2
    LZz = 1+math.floor(LZ/gridwidth)*2

    # らせんの表示
    Rs = 3
    Gn = 5000
    Nn = 32  # らせんの巻き数
    Ltheta = Nn*2*np.pi  # theta生成数
    theta = np.linspace(0, Ltheta, Gn)
    Xs = Rs*np.cos(theta)
    Ys = Rs*np.sin(theta)
    Zs = np.linspace(-LZ, LZ, Gn)
    ax.plot(Xs, Ys, Zs, linewidth=0.5)

    # 点電化の生成
    Gn = 5000  # 生成数
    theta = np.linspace(0, Ltheta, Gn)
    Xs = Rs*np.cos(theta)
    Ys = Rs*np.sin(theta)
    Zs = np.linspace(-LZ, LZ, Gn)
    Xs = np.reshape(Xs, (1, Gn))
    Ys = np.reshape(Ys, (1, Gn))
    Zs = np.reshape(Zs, (1, Gn))

    # 点電荷による磁場の計算
    Q = 1  # らせん上の全点電荷の合計
    Q = Q/Gn
    U = np.zeros((LXx, LXx, LXx), dtype=float)
    V = np.zeros((LYy, LYy, LYy), dtype=float)
    W = np.zeros((LZz, LZz, LZz), dtype=float)

    args_ary = []
    for i in range(Gn):
        args_ary.append((X, Xs, Y, Ys, Z, Zs, Q, i))
    for args in args_ary:
        results = calc(*args)
        for tmp in results:
            U = U + tmp[0]
            V = V + tmp[1]
            W = W + tmp[2]

    # 全ベクトルの大きさを合計
    UVW = np.nansum(
        np.sqrt(U**2)) + np.nansum(np.sqrt(V**2)) + np.nansum(np.sqrt(W**2))
    # print(UVW)
    FinalResize = 200/UVW  # 倍率
    ax.quiver(
        X, Y, Z, U*FinalResize, V*FinalResize, W*FinalResize,
        edgecolor='r', facecolor='None', linewidth=0.5
        )

    # 目盛り幅を揃える
    ax.set_xlim(-LX, LX)
    ax.set_ylim(-LY, LY)
    ax.set_zlim(-LZ, LZ)

    print(time.time()-start)


if __name__ == "__main__":
    plot()
    # グラフ描画
    plt.show()
