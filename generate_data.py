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

for k in range(1):
    x_start = (random.uniform(-2, 4), random.uniform(-3, 3))  # Starting node
    x_goal = (random.uniform(26, 32), random.uniform(-3, 3))  # Goal node

    theta_start = random.uniform(-np.pi/2, np.pi/2)
    theta_goal = random.uniform(-np.pi/2, np.pi/2)

    #0.05の確率でゴールのノードをサンプリング
    
    #後処理後のパスの長さが3になるならば、棄却しもう一度RRTを実行 
    while True:
        rrt_instance = rrt.Rrt(x_start, x_goal, 0.5, 0.05, 10000)
        path = rrt_instance.planning()
        processed_path = rrt_instance.utils.post_processing(path)
        
        if len(processed_path) == 3:
            del rrt_instance
            continue
        
        else:
            break
        
    print(len(processed_path))
    
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
    for i in range(len(processed_path)):
        rrt_path.append(list(processed_path[-i-1]))
        
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

    x, y, theta, phi, v = util.generate_result(result.x)
    print(x, y, theta, phi, v)


    with open('../data/practice/x.csv', 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(x)

    with open('../data/practice/y.csv', 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(y)
        
    with open('../data/practice/theta.csv', 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(theta)
        
    with open('../data/practice/phi.csv', 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(phi)
    
    with open('../data/practice/v.csv', 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(v)
        
    del rrt_instance
    
    print("{}完了".format(k))
