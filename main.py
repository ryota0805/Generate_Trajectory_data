from param import Parameter as p
import GenerateInitialPath
import util
import constraints
import matplotlib.pyplot as plt
import numpy as np
import scipy.optimize as optimize
import objective_function 
import plot

#WayPointから設計変数の初期値を計算する
cubicX, cubicY = GenerateInitialPath.cubic_spline()
x, y, theta, phi, v = GenerateInitialPath.generate_initialpath(cubicX, cubicY)
trajectory_matrix = np.array([x, y, theta, phi, v])
trajectory_vector = util.matrix_to_vector(trajectory_matrix)

#目的関数の設定
func = objective_function.objective_function2

#制約条件の設定
cons = constraints.generate_constraints()

#変数の範囲の設定
bounds = constraints.generate_bounds()
print(bounds)
#オプションの設定
options = {'maxiter':1000}
print(type(func))
print(type(cons))
print(type(bounds))
#最適化を実行
result = optimize.minimize(func, trajectory_vector, method='SLSQP', constraints=cons, bounds=bounds, options=options)

#最適化結果の表示
print(result)
plot.vis_env()
plot.vis_path(trajectory_vector)
plot.compare_path(trajectory_vector, result.x)
plot.compare_history_theta(trajectory_vector, result.x, range_flag = True)
plot.compare_history_phi(trajectory_vector, result.x, range_flag = True)
plot.compare_history_v(trajectory_vector, result.x, range_flag = True)
plot.vis_history_theta(result.x, range_flag=True)
plot.vis_history_phi(result.x, range_flag=True)
plot.vis_history_v(result.x, range_flag = True)
plot.compare_path_rec(trajectory_vector, result.x)


