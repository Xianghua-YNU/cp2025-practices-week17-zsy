"""学生模板：ParallelPlateCapacitor
文件：parallel_plate_capacitor_student.py
重要：函数名称必须与参考答案一致！
"""
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import time

def solve_laplace_jacobi(xgrid, ygrid, w, d, tol=1e-5):
    """
    使用Jacobi迭代法求解拉普拉斯方程
    
    参数:
        xgrid (int): x方向网格点数
        ygrid (int): y方向网格点数
        w (int): 平行板宽度
        d (int): 平行板间距
        tol (float): 收敛容差
    
    返回:
        tuple: (potential_array, iterations, convergence_history)
    
    物理背景: 求解平行板电容器内部的电势分布，满足拉普拉斯方程 \(\nabla^2 V = 0\)。
    数值方法: 使用Jacobi迭代法，通过反复迭代更新每个网格点的电势值，直至收敛。
    
    实现步骤:
    1. 初始化电势网格，设置边界条件（极板电势）。
    2. 循环迭代，每次迭代根据周围点的电势更新当前点的电势。
    3. 记录每次迭代的最大变化量，用于收敛历史分析。
    4. 检查收敛条件，如果最大变化量小于容差，则停止迭代。
    5. 返回最终的电势分布、迭代次数和收敛历史。
    """
    u = np.zeros((ygrid, xgrid))

    xl = (xgrid - w) // 2
    xr = (xgrid + w) // 2
    yb = (ygrid - d) // 2
    yt = (ygrid + d) // 2

    # 边界
    u[yt, xl:xr+1] = 100.0  #上边界
    u[yb, xl:xr+1] = -100.0   #下边界

    iterations = 0
    max_iter = 10000
    convergence_history = []

    while iterations < max_iter:
        u_old = u.copy()

        # Jacobi迭代法
        u[1:-1, 1:-1] = 0.25 * (u_old[2:, 1:-1] + u_old[:-2, 1:-1] + u_old[1:-1, 2:] + u_old[1:-1, :-2])

        u[yt, xl:xr+1] = 100.0
        u[yb, xl:xr+1] = -100.0

        # 收敛度计算
        max_change = np.max(np.abs(u - u_old))
        convergence_history.append(max_change)
        
        iterations += 1
        if max_change < tol:
            break

    return u, iterations, convergence_history
    
def solve_laplace_sor(xgrid, ygrid, w, d, omega=1.25, Niter=1000, tol=1e-5):
    """
    实现SOR算法求解平行板电容器的电势分布
    
    参数:
        xgrid (int): x方向网格点数
        ygrid (int): y方向网格点数
        w (int): 平行板宽度
        d (int): 平行板间距
        omega (float): 松弛因子
        Niter (int): 最大迭代次数
        tol (float): 收敛容差
    返回:
        tuple: (电势分布数组, 迭代次数, 收敛历史)
    
    物理背景: 求解平行板电容器内部的电势分布，满足拉普拉斯方程 \(\nabla^2 V = 0\)。
    数值方法: 使用逐次超松弛（SOR）迭代法，通过引入松弛因子加速收敛。
    
    实现步骤:
    1. 初始化电势网格，设置边界条件（极板电势）。
    2. 循环迭代，每次迭代根据周围点和松弛因子更新当前点的电势。
    3. 记录每次迭代的最大变化量，用于收敛历史分析。
    4. 检查收敛条件，如果最大变化量小于容差，则停止迭代。
    5. 返回最终的电势分布、迭代次数和收敛历史。
    """
    u = np.zeros((ygrid, xgrid))

    xl = (xgrid - w) // 2
    xr = (xgrid + w) // 2
    yb = (ygrid - d) // 2
    yt = (ygrid + d) // 2

    u[yt, xl:xr+1] = 100.0  
    u[yb, xl:xr+1] = -100.0  

    iterations = 0
    convergence_history = []

    while iterations < Niter:
        u_old = u.copy()

        # Gauss-Seidel SOR 
        for i in range(1, ygrid-1):
            for j in range(1, xgrid-1):
                r_ij = 0.25 * (u_old[i+1, j] + u_old[i-1, j] + u_old[i, j+1] + u_old[i, j-1])
                u[i, j] = (1 - omega) * u_old[i, j] + omega * r_ij

    
        u[yt, xl:xr+1] = 100.0
        u[yb, xl:xr+1] = -100.0

        max_change = np.max(np.abs(u - u_old))
        convergence_history.append(max_change)


        iterations += 1
        if max_change < tol:
            break
    #大致同Jacobi迭代法
    
    return u, iterations, convergence_history

def plot_results(x, y, u, method_name):
    """
    绘制三维电势分布、等势线和电场线
    
    参数:
        x (array): X坐标数组
        y (array): Y坐标数组
        u (array): 电势分布数组
        method_name (str): 方法名称
    
    实现步骤:
    1. 创建包含两个子图的图形。
    2. 在第一个子图中绘制三维线框图显示电势分布以及在z方向的投影等势线。
    3. 在第二个子图中绘制等势线和电场线流线图。
    4. 设置图表标题、标签和显示(注意不要出现乱码)。
    """
    fig = plt.figure(figsize=(12, 8))

    ax1 = fig.add_subplot(121, projection='3d')
    X, Y = np.meshgrid(x, y)
    ax1.plot_surface(X, Y, u, cmap='viridis', edgecolor='none')
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Potential (V)')
    ax1.set_title('3D Potential Distribution\n({})'.format(method_name))
    plt.savefig('3D Potential Distribution')


    
    ax2 = fig.add_subplot(122)
    levels = np.linspace(u.min(), u.max(), 20)
    contour = ax2.contour(X, Y, u, levels=levels, colors='red', linestyles='dashed', linewidths=0.8)
    ax2.clabel(contour, inline=True, fontsize=8, fmt='%1.1f')

    Ey, Ex = np.gradient(-u)
    ax2.streamplot(X, Y, Ex, Ey, density=1.5, color='blue', linewidth=1, arrowstyle='-|>')

    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_title('Equipotential Lines & Electric Field Lines\n({})'.format(method_name))
    ax2.set_aspect('equal')
    plt.savefig('Equipotential Lines & Electric Field Lines')

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # 示例用法（学生可以取消注释并在此处测试他们的实现）
    xgrid, ygrid = 50, 50
    w, d = 20, 20  # plate width and separation
    tol = 1e-3

    # 坐标
    x = np.linspace(0, xgrid-1, xgrid)
    y = np.linspace(0, ygrid-1, ygrid)

    print("Solving Laplace equation for parallel plate capacitor...")
    print(f"Grid size: {xgrid} x {ygrid}")
    print(f"Plate width: {w}, separation: {d}")
    print(f"Tolerance: {tol}")

    # Jacobi
    print("\n1. Jacobi iteration method:")
    start_time = time.time()
    u_jacobi, iter_jacobi, conv_history_jacobi = solve_laplace_jacobi(xgrid, ygrid, w, d, tol=tol)
    time_jacobi = time.time() - start_time
    print(f"Converged in {iter_jacobi} iterations")
    print(f"Time: {time_jacobi:.3f} seconds")

    # SOR
    print("\n2. Gauss-Seidel SOR iteration method:")
    start_time = time.time()
    u_sor, iter_sor, conv_history_sor = solve_laplace_sor(xgrid, ygrid, w, d, tol=tol)
    time_sor = time.time() - start_time
    print(f"Converged in {iter_sor} iterations")
    print(f"Time: {time_sor:.3f} seconds")

    # Performance comparison
    print("\n3. Performance comparison:")
    print(f"Jacobi: {iter_jacobi} iterations, {time_jacobi:.3f}s")
    print(f"SOR: {iter_sor} iterations, {time_sor:.3f}s")
    print(f"Speedup: {iter_jacobi/iter_sor:.1f}x iterations, {time_jacobi/time_sor:.2f}x time")

    # Plot results
    plot_results(x, y, u_jacobi, "Jacobi Method")
    plot_results(x, y, u_sor, "SOR Method")

    # Plot convergence comparison
    plt.figure(figsize=(10, 6))
    plt.semilogy(range(len(conv_history_jacobi)), conv_history_jacobi, 'r-', label='Jacobi Method')
    plt.semilogy(range(len(conv_history_sor)), conv_history_sor, 'b-', label='SOR Method')
    plt.xlabel('Iteration')
    plt.ylabel('Maximum Change')
    plt.title('Convergence Comparison')
    plt.savefig('Convergence Comparison')

    plt.grid(True)
    plt.legend()
    plt.show()
