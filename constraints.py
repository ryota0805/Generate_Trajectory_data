#不等式制約、等式制約を定義する
from param import Parameter as p
import util
import numpy as np
import env

########
#制約条件を生成する関数
########
def generate_constraints(x_start, x_goal, theta_start, theta_goal):
    env_data = env.Env()
    obs_rectangle = env_data.obs_rectangle
    obs_circle = env_data.obs_circle

    #最初に不等式制約(K×N個)
    cons = ()
    
    #矩形の障害物に対する不等式制約
    for k in range(len(obs_rectangle)):
        for i in range(p.N):
            cons = cons + ({'type':'ineq', 'fun':lambda x, i = i, k = k: (((2*0.8/obs_rectangle[k][2]) ** 10) * (x[i] - (obs_rectangle[k][0] + obs_rectangle[k][2]/2)) ** 10 + ((2*0.95/obs_rectangle[k][3]) ** 10) * (x[i + p.N] - (obs_rectangle[k][1] + obs_rectangle[k][3]/2)) ** 10) - 1},)
    
    #円形の障害物に対する不等式制約
    for k in range(len(obs_circle)):
        for i in range(p.N):
            cons = cons + ({'type':'ineq', 'fun':lambda x, i = i, k = k: ((x[i] - obs_circle[k][0]) ** 2 + (x[i + p.N] - obs_circle[k][1]) ** 2) - (obs_circle[k][2] + p.robot_size) ** 2},)

    #次にモデルの等式制約(3×(N-1)個)
    #x
    for i in range(p.N-1):
        cons = cons + ({'type':'eq', 'fun':lambda x, i = i: x[i+1] - (x[i] + x[i + 4 * p.N] * np.cos(x[i + 2 * p.N]) * p.dt)},)
        
    #y
    for i in range(p.N-1):
        cons = cons + ({'type':'eq', 'fun':lambda x, i = i: x[i+1 + p.N] - (x[i + p.N] + x[i + 4 * p.N] * np.sin(x[i + 2 * p.N]) * p.dt)},)
        
    #theta
    for i in range(p.N-1):
        cons = cons + ({'type':'eq', 'fun':lambda x, i = i: x[i+1 + 2 * p.N] - (x[i + 2 * p.N] + x[i + 4 * p.N] * np.tan(x[i+ 3 * p.N]) * p.dt / p.L)},)

    #境界条件(10個)
    #境界条件が設定されている場合は制約条件に加える。
    #x初期条件
    if p.set_cons['initial_x'] == False:
        pass
    else:
        cons = cons + ({'type':'eq', 'fun':lambda x: x[0] - x_start[0]},)
        
    #x終端条件
    if p.set_cons['terminal_x'] == False:
        pass
    else:
        cons = cons + ({'type':'eq', 'fun':lambda x: x[p.N - 1] - x_goal[0]},)

    #y初期条件
    if p.set_cons['initial_y'] == False:
        pass
    else:
        cons = cons + ({'type':'eq', 'fun':lambda x: x[p.N] - x_start[1]},)
        
    #y終端条件
    if p.set_cons['terminal_y'] == False:
        pass
    else:
        cons = cons + ({'type':'eq', 'fun':lambda x: x[2*p.N - 1] - x_goal[1]},)
        
    #theta初期条件
    if p.set_cons['initial_theta'] == False:
        pass
    else:
        cons = cons + ({'type':'eq', 'fun':lambda x: x[2*p.N] - theta_start},)
        
    #theta終端条件
    if p.set_cons['terminal_theta'] == False:
        pass
    else:
        cons = cons + ({'type':'eq', 'fun':lambda x: x[3*p.N - 1] - theta_goal},)
        
    #phi初期条件
    if p.set_cons['initial_phi'] == False:
        pass
    else:
        cons = cons + ({'type':'eq', 'fun':lambda x: x[3*p.N] - p.initial_phi},)
        
    #phi終端条件
    if p.set_cons['terminal_phi'] == False:
        pass
    else:
        cons = cons + ({'type':'eq', 'fun':lambda x: x[4*p.N - 1] - p.terminal_phi},)
        
    #v初期条件
    if p.set_cons['initial_v'] == False:
        pass
    else:
        cons = cons + ({'type':'eq', 'fun':lambda x: x[4*p.N] - p.initial_v},)
        
    #v終端条件
    if p.set_cons['terminal_v'] == False:
        pass
    else:
        cons = cons + ({'type':'eq', 'fun':lambda x: x[5*p.N - 1] - p.terminal_v},)

    return cons


########
#bounds(変数の範囲)を設定する関数
########

#変数の数だけタプルのリストとして返す関数
def generate_bounds():
    
    #boundsのリストを生成
    bounds = []
    
    #xの範囲
    for i in range(p.N):
        bounds.append((p.x_min, p.x_max))
        
    #yの範囲
    for i in range(p.N):
        bounds.append((p.y_min, p.y_max))
        
    #thetaの範囲
    for i in range(p.N):
        bounds.append((p.theta_min, p.theta_max))
        
    #phiの範囲
    for i in range(p.N):
        bounds.append((p.phi_min, p.phi_max))
        
    #vの範囲
    for i in range(p.N):
        bounds.append((p.v_min, p.v_max))
        
    return bounds