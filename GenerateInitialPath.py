#初期パスを生成するファイル

import numpy as np
from scipy import interpolate
from param import Parameter as p

########
#rrt_pathから3次スプライン関数を生成し、状態量をサンプリングする
########

#3次スプライン関数の生成
def cubic_spline(rrt_path):   
    x, y = [], []
    for i in range(len(rrt_path)):
        x.append(rrt_path[i][0])
        y.append(rrt_path[i][1])
        
    tck,u = interpolate.splprep([x,y], k=3, s=0) 
    u = np.linspace(0, 1, num=p.N, endpoint=True)
    spline = interpolate.splev(u, tck)
    cubicX = spline[0]
    cubicY = spline[1]
    return cubicX, cubicY

#x, yからΘとφを生成する
def generate_initialpath(cubicX, cubicY, theta_start, theta_goal):
    #nd.arrayに変換
    x = np.array(cubicX)
    y = np.array(cubicY)
    
    #x, yの差分を計算
    deltax = np.diff(x)
    deltay = np.diff(y)
    
    #x, y の差分からthetaを計算
    #theta[0]を初期値に置き換え、配列の最後に終端状態を追加
    theta = np.arctan(deltay / deltax)
    theta[0] = theta_start
    theta = np.append(theta, theta_goal)
    
    #thetaの差分からphiを計算
    #phi[0]を初期値に置き換え配列の最後に終端状態を追加
    deltatheta = np.diff(theta)
    phi = deltatheta / p.dt
    phi[0] = p.initial_phi
    phi = np.append(phi, p.terminal_phi)
    
    #x,yの差分からvを計算
    #phi[0]を初期値に置き換え配列の最後に終端状態を追加
    v = np.sqrt((deltax ** 2 + deltay ** 2) / p.dt)
    v[0] = p.initial_v
    v = np.append(v, p.terminal_v)
    return x, y, theta, phi, v

