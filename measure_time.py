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
from param import Parameter as p
import time

point_number_list = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]

for point_number in point_number_list:
    p.N = point_number
    p.dt = 30/p.N
    time_list = []
    while True:
        
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
            
        p.initial_x, p.initial_y = x_start[0], x_start[1]
        p.terminal_x, p.terminal_y = x_goal[0], x_goal[1]
        p.initial_theta, p.terminal_theta = theta_start, theta_goal

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

        #目的関数の設定
        func = objective_function.objective_function
        jac_of_objective_function = objective_function.jac_of_objective_function

        #制約条件の設定
        cons = constraints.generate_cons_with_jac()

        #変数の範囲の設定
        bounds = constraints.generate_bounds()

        #オプションの設定
        options = {'maxiter':10000}

        print(p.N)
        #最適化を実行
        start_time = time.time()
        result = optimize.minimize(func, trajectory_vector, method='SLSQP', jac = jac_of_objective_function, constraints=cons, bounds=bounds, options=options)
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"実行時間: {elapsed_time}秒")
        print(result)
        if result.success == True:
            time_list.append(elapsed_time)
        else:
            pass
        
        if len(time_list) == 100:
            break
        else:
            pass
    
    with open('../data/env3/time.csv', 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(time_list)