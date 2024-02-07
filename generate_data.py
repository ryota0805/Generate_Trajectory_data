import rrt
import GenerateInitialPath
import objective_function
import constraints
import numpy as np
import util
import scipy.optimize as optimize
import plot
import csv
import random

for k in range(20000):
    x_start = (random.uniform(-2, 4), random.uniform(-9, 9))  # Starting node
    x_goal = (random.uniform(26, 32), random.uniform(-9, 9))  # Goal node
    
    if x_start[1] >= 6:
        theta_start = random.uniform(-np.pi/2, 0)
    elif -6 < x_start[1] < 6:
        theta_start = random.uniform(-np.pi/2, np.pi/2)
    else:
        theta_start = random.uniform(0, np.pi/2)
        
    if x_goal[1] >= 6:
        theta_goal = random.uniform(0, np.pi/2)
    elif -6 < x_goal[1] < 6:
        theta_goal = random.uniform(-np.pi/2, np.pi/2)
    else:
        theta_goal = random.uniform(-np.pi/2, 0)

    #0.05の確率でゴールのノードをサンプリング
    
    rrt_instance = rrt.Rrt(x_start, x_goal, 0.5, 0.05, 10000)
    path = rrt_instance.planning()
        
    print("RRTのノード数:{}".format(len(path)))
    
    """
    #アニメーションの作成
    if path:
        rrt_instance.plotting.animation(rrt_instance.vertex, path, "RRT", True)
        rrt_instance.plotting.animation(rrt_instance.vertex, processed_path, "RRT", False)
    else:
        print("No Path Found!")
    """
    #ノードの順番を反転させる
    rrt_path = []
    for i in range(len(path)):
        rrt_path.append(list(path[-i-1]))
        
    #print(rrt_path)

    #初期軌道作成
    #スプライン補間
    cubicX, cubicY = GenerateInitialPath.cubic_spline(rrt_path)
    
    #初期条件、終端条件を満たすそれらしい軌道を生成する
    initial_x, initial_y, initial_theta, initial_phi, initial_v = GenerateInitialPath.generate_initialpath(cubicX, cubicY, theta_start, theta_goal)
    
    trajectory_matrix = np.array([initial_x, initial_y, initial_theta, initial_phi, initial_v])
    trajectory_vector = util.matrix_to_vector(trajectory_matrix)

    #最適化のための目的関数、制約条件、境界条件を設定
    func = objective_function.objective_function
    cons = constraints.generate_constraints(x_start, x_goal, theta_start, theta_goal)
    bounds = constraints.generate_bounds()
    options = {'maxiter':1000}
    
    #最適化を実行
    result = optimize.minimize(func, trajectory_vector, method='SLSQP', constraints=cons, bounds=bounds, options=options)

    #最適化結果の表示
    #print(result)
    #plot.vis_path(trajectory_vector)
    #plot.compare_path(trajectory_vector, result.x)
    #plot.compare_history_theta(trajectory_vector, result.x, range_flag = True)
    #plot.compare_history_phi(trajectory_vector, result.x, range_flag = True)
    #plot.compare_history_v(trajectory_vector, result.x, range_flag = True)
    #plot.vis_history_theta(result.x, range_flag=True)
    #plot.vis_history_phi(result.x, range_flag=True)
    #plot.vis_history_v(result.x, range_flag = True)

    evaluation = func(result.x)
    list_evaluation = [evaluation]
    x, y, theta, phi, v = util.generate_result(result.x)
    
    print("csvファイルに書き込み中")
    
    with open('../data/env3/x.csv', 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(x)

    with open('../data/env3/y.csv', 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(y)
        
    with open('../data/env3/theta.csv', 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(theta)
        
    with open('../data/env3/phi.csv', 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(phi)
    
    with open('../data/env3/v.csv', 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(v)
    
    with open('../data/env3/evaluation.csv', 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(list_evaluation)
        
    del rrt_instance
    
    print("最適化完了数:{}".format(k))
