import numpy as np
import math
import time
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.axes3d import Axes3D


# 曲線上の電荷が作るベクトル場をシミュレーションする
LL = 8  # 計算領域の立方体の一辺の長さ
LX, LY, LZ = LL, LL, LL
gridwidth = 2  # 生成する格子点の間隔
Nn = 32  # らせんの巻き数
GnLight = 5000  # らせん表示用の生成数
Gn = 5000  # 点電荷生成数(計算用)
Qsum = 1  # らせん上の全点電荷の合計
Rs = 3  # らせん半径


class Vecal:
    def __init__(self):
        pass

    # 磁場用の計算メソッド
    def magCross(self, X, Xs, Y, Ys, Z, Zs, Q, i):
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

    # 電場用の計算メソッド
    def eleCal(self, X, Xs, Y, Ys, Z, Zs, Q, i):
        res = []
        tmp = []
        R = np.sqrt((X-Xs[0, i])**2+(Y-Ys[0, i])**2+(Z-Zs[0, i])**2)
        tmp.append(
            np.divide(
                (Q*(X-Xs[0, i])), R**3, out=np.zeros_like((Q*(X-Xs[0, i]))), where=R != 0
                )
            )
        tmp.append(
            np.divide(
                (Q*(Y-Ys[0, i])), R**3, out=np.zeros_like((Q*(Y-Ys[0, i]))), where=R != 0
                )
            )
        tmp.append(
            np.divide(
                (Q*(Z-Zs[0, i])), R**3, out=np.zeros_like((Q*(Z-Zs[0, i]))), where=R != 0
                )
            )
        res.append(tmp)
        return res


class Curveplot:
    def __init__(self):
        pass

    # らせんの３次元グラフ生成メソッド
    def plotSpiral(self):
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
        Ltheta = Nn*2*np.pi  # theta生成数
        theta = np.linspace(0, Ltheta, GnLight)
        Xs = Rs*np.cos(theta)
        Ys = Rs*np.sin(theta)
        Zs = np.linspace(-LZ, LZ, GnLight)
        ax.plot(Xs, Ys, Zs, linewidth=0.5)

        # 点電化の生成
        theta = np.linspace(0, Ltheta, Gn)
        Xs = Rs*np.cos(theta)
        Ys = Rs*np.sin(theta)
        Zs = np.linspace(-LZ, LZ, Gn)
        Xs = np.reshape(Xs, (1, Gn))
        Ys = np.reshape(Ys, (1, Gn))
        Zs = np.reshape(Zs, (1, Gn))

        # 点電荷によるベクトル場の計算
        Q = Qsum/Gn
        U = np.zeros((LXx, LXx, LXx), dtype=float)
        V = np.zeros((LYy, LYy, LYy), dtype=float)
        W = np.zeros((LZz, LZz, LZz), dtype=float)

        Ve = Vecal()  # インスタンス化
        args_ary = []
        for i in range(Gn):
            args_ary.append((X, Xs, Y, Ys, Z, Zs, Q, i))
        for args in args_ary:
            results = Ve.magCross(*args)  # 好みのベクトル計算メソッドを選択
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

        # グラフの見た目について
        ax.set_xlim(-LX, LX)
        ax.set_ylim(-LY, LY)
        ax.set_zlim(-LZ, LZ)

        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')

        # 目盛り幅を揃える
        max_range = np.array(
            [X.max()-X.min(), Y.max()-Y.min(), Z.max()-Z.min()]).max() * 0.5

        mid_x = (X.max()+X.min()) * 0.5
        mid_y = (Y.max()+Y.min()) * 0.5
        mid_z = (Z.max()+Z.min()) * 0.5
        ax.set_xlim(mid_x - max_range, mid_x + max_range)
        ax.set_ylim(mid_y - max_range, mid_y + max_range)
        ax.set_zlim(mid_z - max_range, mid_z + max_range)

    # ２重らせんの３次元グラフ生成メソッド
    def plotDspiral(self):
        X, Y, Z = np.meshgrid(
            np.arange(-LX, LX+1, gridwidth),
            np.arange(-LY, LY+1, gridwidth),
            np.arange(-LZ, LZ+1, gridwidth)
            )

        # 置きなおし
        LXx = 1+math.floor(LX/gridwidth)*2
        LYy = 1+math.floor(LY/gridwidth)*2
        LZz = 1+math.floor(LZ/gridwidth)*2

        # 内側らせんの表示
        Ltheta = Nn*2*np.pi  # theta生成数
        theta = np.linspace(0, Ltheta, GnLight)
        Xs = Rs*np.cos(theta)
        Ys = Rs*np.sin(theta)
        Zs = np.linspace(-LZ, LZ, GnLight)
        ax.plot(Xs, Ys, Zs, linewidth=0.5)

        # 内側点電化の生成
        theta = np.linspace(0, Ltheta, Gn)
        Xs = Rs*np.cos(theta)
        Ys = Rs*np.sin(theta)
        Zs = np.linspace(-LZ, LZ, Gn)
        Xs = np.reshape(Xs, (1, Gn))
        Ys = np.reshape(Ys, (1, Gn))
        Zs = np.reshape(Zs, (1, Gn))

        # 内側点電荷によるベクトル場の計算
        Q = Qsum/Gn
        Uin = np.zeros((LXx, LXx, LXx), dtype=float)
        Vin = np.zeros((LYy, LYy, LYy), dtype=float)
        Win = np.zeros((LZz, LZz, LZz), dtype=float)

        Ve = Vecal()  # インスタンス化
        args_ary = []
        for i in range(Gn):
            args_ary.append((X, Xs, Y, Ys, Z, Zs, Q, i))
        for args in args_ary:
            results = Ve.magCross(*args)  # 好みのベクトル計算メソッドを選択
            for tmp in results:
                Uin = Uin + tmp[0]
                Vin = Vin + tmp[1]
                Win = Win + tmp[2]

        # 外側らせんの表示
        Ltheta = Nn*2*np.pi  # theta生成数
        theta = np.linspace(0, Ltheta, GnLight)
        Xs = (Rs+2)*np.cos(theta)
        Ys = (Rs+2)*np.sin(theta)
        Zs = np.linspace(-LZ, LZ, GnLight)
        ax.plot(Xs, Ys, Zs, linewidth=0.5)

        # 外側点電化の生成
        theta = np.linspace(0, Ltheta, Gn)
        Xs = (Rs+2)*np.cos(theta)
        Ys = (Rs+2)*np.sin(theta)
        Zs = np.linspace(-LZ, LZ, Gn)
        Xs = np.reshape(Xs, (1, Gn))
        Ys = np.reshape(Ys, (1, Gn))
        Zs = np.reshape(Zs, (1, Gn))

        # 外側点電荷によるベクトル場の計算
        Q = Qsum/Gn
        Uout = np.zeros((LXx, LXx, LXx), dtype=float)
        Vout = np.zeros((LYy, LYy, LYy), dtype=float)
        Wout = np.zeros((LZz, LZz, LZz), dtype=float)

        args_ary = []
        for i in range(Gn):
            args_ary.append((X, Xs, Y, Ys, Z, Zs, Q, i))
        for args in args_ary:
            results = Ve.magCross(*args)  # 好みのベクトル計算メソッドを選択
            for tmp in results:
                Uout = Uout + tmp[0]
                Vout = Vout + tmp[1]
                Wout = Wout + tmp[2]

        # ２つのベクトル場の合計
        U = Uin + Uout
        V = Vin + Vout
        W = Win + Wout

        # 全ベクトルの大きさを合計
        UVW = np.nansum(
            np.sqrt(U**2)) + np.nansum(np.sqrt(V**2)) + np.nansum(np.sqrt(W**2))
        # print(UVW)
        FinalResize = 200/UVW  # 倍率
        ax.quiver(
            X, Y, Z, U*FinalResize, V*FinalResize, W*FinalResize,
            edgecolor='r', facecolor='None', linewidth=0.5
            )

        # グラフの見た目について
        ax.set_xlim(-LX, LX)
        ax.set_ylim(-LY, LY)
        ax.set_zlim(-LZ, LZ)

        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')

        # 目盛り幅を揃える
        max_range = np.array(
            [X.max()-X.min(), Y.max()-Y.min(), Z.max()-Z.min()]).max() * 0.5

        mid_x = (X.max()+X.min()) * 0.5
        mid_y = (Y.max()+Y.min()) * 0.5
        mid_z = (Z.max()+Z.min()) * 0.5
        ax.set_xlim(mid_x - max_range, mid_x + max_range)
        ax.set_ylim(mid_y - max_range, mid_y + max_range)
        ax.set_zlim(mid_z - max_range, mid_z + max_range)

    # らせんとZ軸の差分表示メソッド（電場用？）
    def plotDiffSpi(self):
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
        Ltheta = Nn*2*np.pi  # theta生成数
        theta = np.linspace(0, Ltheta, GnLight)
        Xs = Rs*np.cos(theta)
        Ys = Rs*np.sin(theta)
        Zs = np.linspace(-LZ, LZ, GnLight)
        ax.plot(Xs, Ys, Zs, linewidth=0.5)

        # 点電化の生成
        theta = np.linspace(0, Ltheta, Gn)
        Xs = Rs*np.cos(theta)
        print(Xs)
        Ys = Rs*np.sin(theta)
        Zs = np.linspace(-LZ, LZ, Gn)
        Xs = np.reshape(Xs, (1, Gn))
        Ys = np.reshape(Ys, (1, Gn))
        Zs = np.reshape(Zs, (1, Gn))

        # 点電荷によるベクトル場の計算
        Q = Qsum/Gn
        Us = np.zeros((LXx, LXx, LXx), dtype=float)
        Vs = np.zeros((LYy, LYy, LYy), dtype=float)
        Ws = np.zeros((LZz, LZz, LZz), dtype=float)

        Ve = Vecal()  # インスタンス化
        args_ary = []
        for i in range(Gn):
            args_ary.append((X, Xs, Y, Ys, Z, Zs, Q, i))
        for args in args_ary:
            results = Ve.eleCal(*args)  # 好みのベクトル計算メソッドを選択
            for tmp in results:
                Us = Us + tmp[0]
                Vs = Vs + tmp[1]
                Ws = Ws + tmp[2]

        U = Us
        V = Vs
        W = Ws

        # 全ベクトルの大きさを合計
        UVW = np.nansum(
            np.sqrt(U**2)) + np.nansum(np.sqrt(V**2)) + np.nansum(np.sqrt(W**2))
        # print(UVW)
        FinalResize = 200/UVW  # 倍率
        ax.quiver(
            X, Y, Z, U*FinalResize, V*FinalResize, W*FinalResize,
            edgecolor='r', facecolor='None', linewidth=0.5
            )

        # グラフの見た目について
        ax.set_xlim(-LX, LX)
        ax.set_ylim(-LY, LY)
        ax.set_zlim(-LZ, LZ)

        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')

        # 目盛り幅を揃える
        max_range = np.array(
            [X.max()-X.min(), Y.max()-Y.min(), Z.max()-Z.min()]).max() * 0.5

        mid_x = (X.max()+X.min()) * 0.5
        mid_y = (Y.max()+Y.min()) * 0.5
        mid_z = (Z.max()+Z.min()) * 0.5
        ax.set_xlim(mid_x - max_range, mid_x + max_range)
        ax.set_ylim(mid_y - max_range, mid_y + max_range)
        ax.set_zlim(mid_z - max_range, mid_z + max_range)


if __name__ == "__main__":
    start = time.time()

    Cu = Curveplot()  # インスタンス化
    fig = plt.figure(figsize=(16, 8))

    ax = fig.add_subplot(1, 2, 1, projection='3d')
    Cu.plotSpiral()  # 好みのプロットメソッドを指定

    ax = fig.add_subplot(1, 2, 2, projection='3d')
    Cu.plotDiffSpi()  # 好みのプロットメソッドを指定

    print(time.time()-start)
    # グラフ描画
    plt.show()
