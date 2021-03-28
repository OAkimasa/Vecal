import numpy as np
import math
import time
import matplotlib.pyplot as plt
import matplotlib.animation as animation
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
    def plotSpiral(self, selectFunc):
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

        if selectFunc == 0:
            Ve = Vecal()  # インスタンス化
            args_ary = []
            for i in range(Gn):
                args_ary.append((X, Xs, Y, Ys, Z, Zs, Q, i))
            for args in args_ary:
                results = Ve.magCross(*args)  # 磁場
                for tmp in results:
                    U = U + tmp[0]
                    V = V + tmp[1]
                    W = W + tmp[2]

        elif selectFunc == 1:
            Ve = Vecal()  # インスタンス化
            args_ary = []
            for i in range(Gn):
                args_ary.append((X, Xs, Y, Ys, Z, Zs, Q, i))
            for args in args_ary:
                results = Ve.eleCal(*args)  # 電場
                for tmp in results:
                    U = U + tmp[0]
                    V = V + tmp[1]
                    W = W + tmp[2]

        else:
            print('Forgot to choose a calculation method')

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

        if selectFunc == 0:
            plt.title('plotSpiral_magCross')
        elif selectFunc == 1:
            plt.title('plotSpiral_eleCal')
        else:
            plt.title('error')

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
    def plotDspiral(self, selectFunc):
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

        if selectFunc == 0:
            Ve = Vecal()  # インスタンス化
            args_ary = []
            for i in range(Gn):
                args_ary.append((X, Xs, Y, Ys, Z, Zs, Q, i))
            for args in args_ary:
                results = Ve.magCross(*args)  # 磁場
                for tmp in results:
                    Uin = Uin + tmp[0]
                    Vin = Vin + tmp[1]
                    Win = Win + tmp[2]

        elif selectFunc == 1:
            Ve = Vecal()  # インスタンス化
            args_ary = []
            for i in range(Gn):
                args_ary.append((X, Xs, Y, Ys, Z, Zs, Q, i))
            for args in args_ary:
                results = Ve.eleCal(*args)  # 電場
                for tmp in results:
                    Uin = Uin + tmp[0]
                    Vin = Vin + tmp[1]
                    Win = Win + tmp[2]

        else:
            print('Forgot to choose a calculation method')

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

        if selectFunc == 0:
            Ve = Vecal()  # インスタンス化
            args_ary = []
            for i in range(Gn):
                args_ary.append((X, Xs, Y, Ys, Z, Zs, Q, i))
            for args in args_ary:
                results = Ve.magCross(*args)  # 磁場
                for tmp in results:
                    Uout = Uout + tmp[0]
                    Vout = Vout + tmp[1]
                    Wout = Wout + tmp[2]

        elif selectFunc == 1:
            Ve = Vecal()  # インスタンス化
            args_ary = []
            for i in range(Gn):
                args_ary.append((X, Xs, Y, Ys, Z, Zs, Q, i))
            for args in args_ary:
                results = Ve.eleCal(*args)  # 電場
                for tmp in results:
                    Uout = Uout + tmp[0]
                    Vout = Vout + tmp[1]
                    Wout = Wout + tmp[2]

        else:
            print('Forgot to choose a calculation method')

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

        if selectFunc == 0:
            plt.title('plotDspiral_magCross')
        elif selectFunc == 1:
            plt.title('plotDspiral_eleCal')
        else:
            plt.title('error')

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
    def plotDiffSpi(self, selectFunc):
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

        # らせん上に点電化を生成
        theta = np.linspace(0, Ltheta, Gn)
        Xs0 = Rs*np.cos(theta)
        Ys0 = Rs*np.sin(theta)
        Zs0 = np.linspace(-LZ, LZ, Gn)
        Xs0 = np.reshape(Xs0, (1, Gn))
        Ys0 = np.reshape(Ys0, (1, Gn))
        Zs0 = np.reshape(Zs0, (1, Gn))

        # 点電荷によるベクトル場の計算
        Q = Qsum/Gn
        U0 = np.zeros((LXx, LXx, LXx), dtype=float)
        V0 = np.zeros((LYy, LYy, LYy), dtype=float)
        W0 = np.zeros((LZz, LZz, LZz), dtype=float)

        if selectFunc == 0:
            Ve = Vecal()  # インスタンス化
            args_ary = []
            for i in range(Gn):
                args_ary.append((X, Xs0, Y, Ys0, Z, Zs0, Q, i))
            for args in args_ary:
                results = Ve.magCross(*args)  # 磁場
                for tmp in results:
                    U0 = U0 + tmp[0]
                    V0 = V0 + tmp[1]
                    W0 = W0 + tmp[2]

        elif selectFunc == 1:
            Ve = Vecal()  # インスタンス化
            args_ary = []
            for i in range(Gn):
                args_ary.append((X, Xs0, Y, Ys0, Z, Zs0, Q, i))
            for args in args_ary:
                results = Ve.eleCal(*args)  # 電場
                for tmp in results:
                    U0 = U0 + tmp[0]
                    V0 = V0 + tmp[1]
                    W0 = W0 + tmp[2]

        else:
            print('Forgot to choose a calculation method')

        # Z軸上に点電化を生成
        Xs1 = np.zeros((1, Gn), dtype=float)
        Ys1 = np.zeros((1, Gn), dtype=float)
        Zs1 = np.linspace(-LZ, LZ, Gn)
        Zs1 = np.reshape(Zs1, (1, Gn))

        # 電荷によるベクトル場の計算
        Q = Qsum/Gn
        U1 = np.zeros((LXx, LXx, LXx), dtype=float)
        V1 = np.zeros((LYy, LYy, LYy), dtype=float)
        W1 = np.zeros((LZz, LZz, LZz), dtype=float)

        if selectFunc == 0:
            Ve = Vecal()  # インスタンス化
            args_ary = []
            for i in range(Gn):
                args_ary.append((X, Xs1, Y, Ys1, Z, Zs1, Q, i))
            for args in args_ary:
                results = Ve.magCross(*args)  # 磁場
                for tmp in results:
                    U1 = U1 + tmp[0]
                    V1 = V1 + tmp[1]
                    W1 = W1 + tmp[2]

        elif selectFunc == 1:
            Ve = Vecal()  # インスタンス化
            args_ary = []
            for i in range(Gn):
                args_ary.append((X, Xs1, Y, Ys1, Z, Zs1, Q, i))
            for args in args_ary:
                results = Ve.eleCal(*args)  # 電場
                for tmp in results:
                    U1 = U1 + tmp[0]
                    V1 = V1 + tmp[1]
                    W1 = W1 + tmp[2]

        else:
            print('Forgot to choose a calculation method')

        U = U0 - U1
        V = V0 - V1
        W = W0*0 - W1*0  # 無限に長い直線とみなす

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

        if selectFunc == 0:
            plt.title('plotDiffspi_magCross')
        elif selectFunc == 1:
            plt.title('plotDiffspi_eleCal')
        else:
            plt.title('error')

        # 目盛り幅を揃える
        max_range = np.array(
            [X.max()-X.min(), Y.max()-Y.min(), Z.max()-Z.min()]).max() * 0.5

        mid_x = (X.max()+X.min()) * 0.5
        mid_y = (Y.max()+Y.min()) * 0.5
        mid_z = (Z.max()+Z.min()) * 0.5
        ax.set_xlim(mid_x - max_range, mid_x + max_range)
        ax.set_ylim(mid_y - max_range, mid_y + max_range)
        ax.set_zlim(mid_z - max_range, mid_z + max_range)

    # 円環の３次元グラフ生成メソッド
    def plotTorus(self, selectFunc):
        X, Y, Z = np.meshgrid(
            np.arange(-LX, LX+1, gridwidth),
            np.arange(-LY, LY+1, gridwidth),
            np.arange(-LZ, LZ+1, gridwidth)
            )

        # 置きなおし
        LXx = 1+math.floor(LX/gridwidth)*2
        LYy = 1+math.floor(LY/gridwidth)*2
        LZz = 1+math.floor(LZ/gridwidth)*2

        # 円環の表示
        rs = Rs/4  # 小半径
        Ltheta = Nn*2*np.pi  # theta生成数
        Lphi = Nn*np.pi  # phi生成数
        theta = np.linspace(0, Ltheta, GnLight)
        phi = np.linspace(0, Lphi, GnLight)
        theta, phi = np.meshgrid(theta, phi)
        Xs = (Rs+rs*np.cos(phi))*np.cos(theta)
        Ys = (Rs+rs*np.cos(phi))*np.sin(theta)
        Zs = rs*np.sin(phi)
        ax.plot_wireframe(Xs, Ys, Zs, linewidth=0.05)

        # 点電化の生成
        rs = Rs/4  # 小半径
        theta = np.linspace(0, Ltheta, Gn)
        phi = np.linspace(0, Lphi, Gn)
        Xs = (Rs+rs*np.cos(phi))*np.cos(theta)
        Ys = (Rs+rs*np.cos(phi))*np.sin(theta)
        Zs = rs*np.sin(phi)
        Xs = np.reshape(Xs, (1, Gn))
        Ys = np.reshape(Ys, (1, Gn))
        Zs = np.reshape(Zs, (1, Gn))

        # 点電荷によるベクトル場の計算
        Q = Qsum/Gn
        U = np.zeros((LXx, LXx, LXx), dtype=float)
        V = np.zeros((LYy, LYy, LYy), dtype=float)
        W = np.zeros((LZz, LZz, LZz), dtype=float)

        if selectFunc == 0:
            Ve = Vecal()  # インスタンス化
            args_ary = []
            for i in range(Gn):
                args_ary.append((X, Xs, Y, Ys, Z, Zs, Q, i))
            for args in args_ary:
                results = Ve.magCross(*args)  # 磁場
                for tmp in results:
                    U = U + tmp[0]
                    V = V + tmp[1]
                    W = W + tmp[2]

        elif selectFunc == 1:
            Ve = Vecal()  # インスタンス化
            args_ary = []
            for i in range(Gn):
                args_ary.append((X, Xs, Y, Ys, Z, Zs, Q, i))
            for args in args_ary:
                results = Ve.eleCal(*args)  # 電場
                for tmp in results:
                    U = U + tmp[0]
                    V = V + tmp[1]
                    W = W + tmp[2]

        else:
            print('Forgot to choose a calculation method')

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

        if selectFunc == 0:
            plt.title('plotTorus_magCross')
        elif selectFunc == 1:
            plt.title('plotTorus_eleCal')
        else:
            plt.title('error')

        # 目盛り幅を揃える
        max_range = np.array(
            [X.max()-X.min(), Y.max()-Y.min(), Z.max()-Z.min()]).max() * 0.5

        mid_x = (X.max()+X.min()) * 0.5
        mid_y = (Y.max()+Y.min()) * 0.5
        mid_z = (Z.max()+Z.min()) * 0.5
        ax.set_xlim(mid_x - max_range, mid_x + max_range)
        ax.set_ylim(mid_y - max_range, mid_y + max_range)
        ax.set_zlim(mid_z - max_range, mid_z + max_range)

    # ２重円環の３次元グラフ生成メソッド
    def plotDtorus(self, selectFunc):
        X, Y, Z = np.meshgrid(
            np.arange(-LX, LX+1, gridwidth),
            np.arange(-LY, LY+1, gridwidth),
            np.arange(-LZ, LZ+1, gridwidth)
            )

        # 置きなおし
        LXx = 1+math.floor(LX/gridwidth)*2
        LYy = 1+math.floor(LY/gridwidth)*2
        LZz = 1+math.floor(LZ/gridwidth)*2

        # 内側円環の表示
        rs = Rs/4  # 小半径
        Ltheta = Nn*2*np.pi  # theta生成数
        Lphi = Nn*np.pi  # phi生成数
        theta = np.linspace(0, Ltheta, GnLight)
        phi = np.linspace(0, Lphi, GnLight)
        theta, phi = np.meshgrid(theta, phi)
        Xs = (Rs+rs*np.cos(phi))*np.cos(theta)
        Ys = (Rs+rs*np.cos(phi))*np.sin(theta)
        Zs = rs*np.sin(phi)
        ax.plot_wireframe(Xs, Ys, Zs, linewidth=0.05)

        # 内側点電化の生成
        rs = Rs/4  # 小半径
        theta = np.linspace(0, Ltheta, Gn)
        phi = np.linspace(0, Lphi, Gn)
        Xs = (Rs+rs*np.cos(phi))*np.cos(theta)
        Ys = (Rs+rs*np.cos(phi))*np.sin(theta)
        Zs = rs*np.sin(phi)
        Xs = np.reshape(Xs, (1, Gn))
        Ys = np.reshape(Ys, (1, Gn))
        Zs = np.reshape(Zs, (1, Gn))

        # 内側点電荷によるベクトル場の計算
        Q = Qsum/Gn
        Uin = np.zeros((LXx, LXx, LXx), dtype=float)
        Vin = np.zeros((LYy, LYy, LYy), dtype=float)
        Win = np.zeros((LZz, LZz, LZz), dtype=float)

        if selectFunc == 0:
            Ve = Vecal()  # インスタンス化
            args_ary = []
            for i in range(Gn):
                args_ary.append((X, Xs, Y, Ys, Z, Zs, Q, i))
            for args in args_ary:
                results = Ve.magCross(*args)  # 磁場
                for tmp in results:
                    Uin = Uin + tmp[0]
                    Vin = Vin + tmp[1]
                    Win = Win + tmp[2]

        elif selectFunc == 1:
            Ve = Vecal()  # インスタンス化
            args_ary = []
            for i in range(Gn):
                args_ary.append((X, Xs, Y, Ys, Z, Zs, Q, i))
            for args in args_ary:
                results = Ve.eleCal(*args)  # 電場
                for tmp in results:
                    Uin = Uin + tmp[0]
                    Vin = Vin + tmp[1]
                    Win = Win + tmp[2]

        else:
            print('Forgot to choose a calculation method')

        # 外側円環の表示
        rs = Rs/4  # 小半径
        Ltheta = Nn*2*np.pi  # theta生成数
        Lphi = Nn*np.pi  # phi生成数
        theta = np.linspace(0, Ltheta, GnLight)
        phi = np.linspace(0, Lphi, GnLight)
        theta, phi = np.meshgrid(theta, phi)
        Xs = ((Rs+3)+rs*np.cos(phi))*np.cos(theta)
        Ys = ((Rs+3)+rs*np.cos(phi))*np.sin(theta)
        Zs = rs*np.sin(phi)
        ax.plot_wireframe(Xs, Ys, Zs, linewidth=0.05)

        # 外側点電化の生成
        rs = Rs/4  # 小半径
        theta = np.linspace(0, Ltheta, Gn)
        phi = np.linspace(0, Lphi, Gn)
        Xs = ((Rs+3)+rs*np.cos(phi))*np.cos(theta)
        Ys = ((Rs+3)+rs*np.cos(phi))*np.sin(theta)
        Zs = rs*np.sin(phi)
        Xs = np.reshape(Xs, (1, Gn))
        Ys = np.reshape(Ys, (1, Gn))
        Zs = np.reshape(Zs, (1, Gn))

        # 外側点電荷によるベクトル場の計算
        Q = Qsum/Gn
        Uout = np.zeros((LXx, LXx, LXx), dtype=float)
        Vout = np.zeros((LYy, LYy, LYy), dtype=float)
        Wout = np.zeros((LZz, LZz, LZz), dtype=float)

        if selectFunc == 0:
            Ve = Vecal()  # インスタンス化
            args_ary = []
            for i in range(Gn):
                args_ary.append((X, Xs, Y, Ys, Z, Zs, Q, i))
            for args in args_ary:
                results = Ve.magCross(*args)  # 磁場
                for tmp in results:
                    Uout = Uout + tmp[0]
                    Vout = Vout + tmp[1]
                    Wout = Wout + tmp[2]

        elif selectFunc == 1:
            Ve = Vecal()  # インスタンス化
            args_ary = []
            for i in range(Gn):
                args_ary.append((X, Xs, Y, Ys, Z, Zs, Q, i))
            for args in args_ary:
                results = Ve.eleCal(*args)  # 電場
                for tmp in results:
                    Uout = Uout + tmp[0]
                    Vout = Vout + tmp[1]
                    Wout = Wout + tmp[2]

        else:
            print('Forgot to choose a calculation method')

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

        if selectFunc == 0:
            plt.title('plotDtorus_magCross')
        elif selectFunc == 1:
            plt.title('plotDtorus_eleCal')
        else:
            plt.title('error')

        # 目盛り幅を揃える
        max_range = np.array(
            [X.max()-X.min(), Y.max()-Y.min(), Z.max()-Z.min()]).max() * 0.5

        mid_x = (X.max()+X.min()) * 0.5
        mid_y = (Y.max()+Y.min()) * 0.5
        mid_z = (Z.max()+Z.min()) * 0.5
        ax.set_xlim(mid_x - max_range, mid_x + max_range)
        ax.set_ylim(mid_y - max_range, mid_y + max_range)
        ax.set_zlim(mid_z - max_range, mid_z + max_range)

    # 環状ソレノイドの３次元グラフ生成メソッド
    def plotCsol(self, selectFunc):
        X, Y, Z = np.meshgrid(
            np.arange(-LX, LX+1, gridwidth),
            np.arange(-LY, LY+1, gridwidth),
            np.arange(-LZ, LZ+1, gridwidth)
            )

        # 置きなおし
        LXx = 1+math.floor(LX/gridwidth)*2
        LYy = 1+math.floor(LY/gridwidth)*2
        LZz = 1+math.floor(LZ/gridwidth)*2

        # 環状ソレノイドの表示
        rs = Rs/4  # 小半径
        Ltheta = Nn*2*np.pi  # theta生成数
        theta = np.linspace(0, Ltheta, GnLight)
        Xs = (Rs+rs*np.cos(Nn*theta))*np.cos(theta)
        Ys = (Rs+rs*np.cos(Nn*theta))*np.sin(theta)
        Zs = rs*np.sin(Nn*theta)
        ax.plot(Xs, Ys, Zs, linewidth=0.05)

        # 点電化の生成
        theta = np.linspace(0, Ltheta, Gn)
        Xs = (Rs+rs*np.cos(Nn*theta))*np.cos(theta)
        Ys = (Rs+rs*np.cos(Nn*theta))*np.sin(theta)
        Zs = rs*np.sin(Nn*theta)
        Xs = np.reshape(Xs, (1, Gn))
        Ys = np.reshape(Ys, (1, Gn))
        Zs = np.reshape(Zs, (1, Gn))

        # 点電荷によるベクトル場の計算
        Q = Qsum/Gn
        U = np.zeros((LXx, LXx, LXx), dtype=float)
        V = np.zeros((LYy, LYy, LYy), dtype=float)
        W = np.zeros((LZz, LZz, LZz), dtype=float)

        if selectFunc == 0:
            Ve = Vecal()  # インスタンス化
            args_ary = []
            for i in range(Gn):
                args_ary.append((X, Xs, Y, Ys, Z, Zs, Q, i))
            for args in args_ary:
                results = Ve.magCross(*args)  # 磁場
                for tmp in results:
                    U = U + tmp[0]
                    V = V + tmp[1]
                    W = W + tmp[2]

        elif selectFunc == 1:
            Ve = Vecal()  # インスタンス化
            args_ary = []
            for i in range(Gn):
                args_ary.append((X, Xs, Y, Ys, Z, Zs, Q, i))
            for args in args_ary:
                results = Ve.eleCal(*args)  # 電場
                for tmp in results:
                    U = U + tmp[0]
                    V = V + tmp[1]
                    W = W + tmp[2]

        else:
            print('Forgot to choose a calculation method')

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

        if selectFunc == 0:
            plt.title('plotCsol_magCross')
        elif selectFunc == 1:
            plt.title('plotCsol_eleCal')
        else:
            plt.title('error')

        # 目盛り幅を揃える
        max_range = np.array(
            [X.max()-X.min(), Y.max()-Y.min(), Z.max()-Z.min()]).max() * 0.5

        mid_x = (X.max()+X.min()) * 0.5
        mid_y = (Y.max()+Y.min()) * 0.5
        mid_z = (Z.max()+Z.min()) * 0.5
        ax.set_xlim(mid_x - max_range, mid_x + max_range)
        ax.set_ylim(mid_y - max_range, mid_y + max_range)
        ax.set_zlim(mid_z - max_range, mid_z + max_range)

    # らせんの中心軸について、巻き数とノルムをプロット
    def plotCenter(self, selectFunc):
        def base(j, selectFunc):
            X, Y, Z = np.meshgrid(
                np.arange(-LX, LX+1, gridwidth),
                np.arange(-LY, LY+1, gridwidth),
                np.arange(-LZ, LZ+1, gridwidth)
                )

            # 置きなおし
            LXx = 1+math.floor(LX/gridwidth)*2
            LYy = 1+math.floor(LY/gridwidth)*2
            LZz = 1+math.floor(LZ/gridwidth)*2

            # 点電化の生成
            Ltheta = j*2*np.pi  # theta生成数
            theta = np.linspace(0, Ltheta, Gn)
            Xs = Rs*np.cos(theta)
            Ys = Rs*np.sin(theta)
            Zs = np.linspace(-LZ, LZ, Gn)
            Xs = np.reshape(Xs, (1, Gn))
            Ys = np.reshape(Ys, (1, Gn))
            Zs = np.reshape(Zs, (1, Gn))

            # ベクトル場の計算
            Q = Qsum/Gn
            U = np.zeros((LXx, LXx, LXx), dtype=float)
            V = np.zeros((LYy, LYy, LYy), dtype=float)
            W = np.zeros((LZz, LZz, LZz), dtype=float)

            if selectFunc == 0:
                Ve = Vecal()  # インスタンス化
                args_ary = []
                for i in range(Gn):
                    args_ary.append((X, Xs, Y, Ys, Z, Zs, Q, i))
                for args in args_ary:
                    results = Ve.magCross(*args)  # 磁場
                    for tmp in results:
                        U = U + tmp[0]
                        V = V + tmp[1]
                        W = W + tmp[2]

            elif selectFunc == 1:
                Ve = Vecal()  # インスタンス化
                args_ary = []
                for i in range(Gn):
                    args_ary.append((X, Xs, Y, Ys, Z, Zs, Q, i))
                for args in args_ary:
                    results = Ve.eleCal(*args)  # 電場
                    for tmp in results:
                        U = U + tmp[0]
                        V = V + tmp[1]
                        W = W + tmp[2]

            else:
                print('Forgot to choose a calculation method')

            # ソレノイド中心のベクトル
            Uc = U[int(LL/2) ,int(LL/2)]
            Vc = V[int(LL/2) ,int(LL/2)]
            Wc = W[int(LL/2) ,int(LL/2)]
            Cuvw = np.sqrt(Uc**2) + np.sqrt(Vc**2) + np.sqrt(Wc**2)
            return Cuvw

        Jpl = []
        arrnum = []
        norm = []
        for j in range(Nn):
            Jpl.append([j])
            Nnresult = base(j, selectFunc).tolist()
            norm.append(Nnresult)
            arrnum.append(np.arange(-LL, LL+1, 2).tolist())
            print(j)

        npJpl = np.array(Jpl)
        nparrnum = np.array(arrnum)
        npnorm = np.array(norm)
        ax.plot_wireframe(npJpl, nparrnum, npnorm, color='b')

        # グラフ表示設定
        if selectFunc == 0:
            plt.title('Center Norm_magCross')
        elif selectFunc == 1:
            plt.title('Center Norm_eleCal')
        else:
            plt.title('error')

        ax.set_xlabel('Winding Number')
        ax.set_ylabel('Z-axis')
        ax.set_zlabel('Norm')

    # らせんの巻き数を変化させるアニメーション
    def plotAspi(self, selectFunc):
        X, Y, Z = np.meshgrid(
            np.arange(-LX, LX+1, gridwidth),
            np.arange(-LY, LY+1, gridwidth),
            np.arange(-LZ, LZ+1, gridwidth)
            )

        # 置きなおし
        LXx = 1+math.floor(LX/gridwidth)*2
        LYy = 1+math.floor(LY/gridwidth)*2
        LZz = 1+math.floor(LZ/gridwidth)*2

        ims = []
        for j in range(Nn):
            # らせんの表示
            Ltheta = j*2*np.pi  # theta生成数
            theta = np.linspace(0, Ltheta, GnLight)
            Xs = Rs*np.cos(theta)
            Ys = Rs*np.sin(theta)
            Zs = np.linspace(-LZ, LZ, GnLight)
            Sim = ax.plot(Xs, Ys, Zs, linewidth=0.5, color='b')

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

            if selectFunc == 0:
                Ve = Vecal()  # インスタンス化
                args_ary = []
                for i in range(Gn):
                    args_ary.append((X, Xs, Y, Ys, Z, Zs, Q, i))
                for args in args_ary:
                    results = Ve.magCross(*args)  # 磁場
                    for tmp in results:
                        U = U + tmp[0]
                        V = V + tmp[1]
                        W = W + tmp[2]

            elif selectFunc == 1:
                Ve = Vecal()  # インスタンス化
                args_ary = []
                for i in range(Gn):
                    args_ary.append((X, Xs, Y, Ys, Z, Zs, Q, i))
                for args in args_ary:
                    results = Ve.eleCal(*args)  # 電場
                    for tmp in results:
                        U = U + tmp[0]
                        V = V + tmp[1]
                        W = W + tmp[2]

            else:
                print('Forgot to choose a calculation method')

            # 全ベクトルの大きさを合計
            UVW = np.nansum(
                np.sqrt(U**2)) + np.nansum(np.sqrt(V**2)) + np.nansum(np.sqrt(W**2))
            # print(UVW)
            FinalResize = 200/UVW  # 倍率
            Qim = [ax.quiver(
                    X, Y, Z, U*FinalResize, V*FinalResize, W*FinalResize,
                    edgecolor='r', facecolor='None', linewidth=0.5
                    )]
            #print(ims)
            im = Sim+Qim
            ims.extend([im])
            print(j)

        animation.ArtistAnimation(fig, ims, blit=True)

        # グラフの見た目について
        ax.set_xlim(-LX, LX)
        ax.set_ylim(-LY, LY)
        ax.set_zlim(-LZ, LZ)

        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')

        if selectFunc == 0:
            plt.title('plotAspi_magCross')
        elif selectFunc == 1:
            plt.title('plotAspi_eleCal')
        else:
            plt.title('error')

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
    Cu.plotSpiral(0)  # 好みのプロットメソッドを指定, 0:magCross, 1:eleCal

    ax = fig.add_subplot(1, 2, 2, projection='3d')
    Cu.plotAspi(0)  # 好みのプロットメソッドを指定, 0:magCross, 1:eleCal

    print(time.time()-start)
    # グラフ描画
    plt.show()
