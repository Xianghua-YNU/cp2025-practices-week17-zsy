#!/usr/bin/env python3
"""
Module: Finite Thickness Parallel Plate Capacitor (Student Version)
"""

import numpy as np
import matplotlib.pyplot as plt

def solve_laplace_sor(nx, ny, plate_thickness, plate_separation, omega=1.9, max_iter=10000, tolerance=1e-6):
    """
    Solve 2D Laplace equation using SOR method for finite thickness parallel plate capacitor.
    
    Args:
        nx (int): Number of grid points in x direction
        ny (int): Number of grid points in y direction
        plate_thickness (int): Thickness of conductor plates in grid points
        plate_separation (int): Separation between plates in grid points
        omega (float): Relaxation factor (1.0 < omega < 2.0)
        max_iter (int): Maximum number of iterations
        tolerance (float): Convergence tolerance
        
    Returns:
        np.ndarray: 2D electric potential distribution
    """
    # TODO: Implement SOR iteration for Laplace equation
    potential = np.zeros((ny, nx))
    
    # 设置边界条件
    # 上导体板（顶部）
    potential[:plate_thickness, nx//2 - plate_separation//2 : nx//2 + plate_separation//2] = 100.0
    # 下导体板（底部）
    potential[-plate_thickness:, nx//2 - plate_separation//2 : nx//2 + plate_separation//2] = -100.0
    
    # 左右边界设为0V
    potential[:, 0] = 0.0
    potential[:, -1] = 0.0
    
    # SOR迭代
    for _ in range(max_iter):
        prev_potential = potential.copy()
        
        for i in range(1, ny-1):
            for j in range(1, nx-1):
                # 跳过导体板区域（已设为固定电势）
                if (i < plate_thickness and j >= nx//2 - plate_separation//2 and j < nx//2 + plate_separation//2) or \
                   (i >= ny - plate_thickness and j >= nx//2 - plate_separation//2 and j < nx//2 + plate_separation//2):
                    continue
                
                # SOR更新公式
                potential[i,j] = (1 - omega) * potential[i,j] + omega/4 * (potential[i+1,j] + potential[i-1,j] + potential[i,j+1] + potential[i,j-1])
        
        # 检查收敛
        if np.max(np.abs(potential - prev_potential)) < tolerance:
            break
    
    return potential

def calculate_charge_density(potential_grid, dx, dy):
    """
    Calculate charge density using Poisson equation.
    
    Args:
        potential_grid (np.ndarray): 2D electric potential distribution
        dx (float): Grid spacing in x direction
        dy (float): Grid spacing in y direction
        
    Returns:
        np.ndarray: 2D charge density distribution
    """
    # TODO: Calculate charge density from potential
    laplacian = (np.roll(potential_grid, 1, axis=0) - 2*potential_grid + np.roll(potential_grid, -1, axis=0)) / dy**2 + \
                (np.roll(potential_grid, 1, axis=1) - 2*potential_grid + np.roll(potential_grid, -1, axis=1)) / dx**2
    
    # 根据泊松方程 ∇²U = -4πρ 计算电荷密度
    charge_density = -laplacian / (4 * np.pi)
    
    return charge_density

def plot_results(potential, charge_density, x_coords, y_coords):
    """
    Create visualization of potential and charge density distributions.
    
    Args:
        potential (np.ndarray): 2D electric potential distribution
        charge_density (np.ndarray): Charge density distribution
        x_coords (np.ndarray): X coordinate array
        y_coords (np.ndarray): Y coordinate array
    """
    # TODO: Implement visualization
    plt.figure(figsize=(12, 6))
    
    # Potential contour plot
    plt.subplot(1, 2, 1)
    CS = plt.contourf(x_coords, y_coords, potential, levels=50, cmap='viridis')
    plt.colorbar(CS, label='Electric Potential (V)')
    plt.xlabel('X Position')
    plt.ylabel('Y Position')
    plt.title('Electric Potential Distribution')
    plt.savefig('potential_distribution.png')
    
    # Charge density contour plot
    plt.subplot(1, 2, 2)
    CS = plt.contourf(x_coords, y_coords, charge_density, levels=50, cmap='RdBu', extend='both')
    plt.colorbar(CS, label='Charge Density (C/m²)')
    plt.xlabel('X Position')
    plt.ylabel('Y Position')
    plt.title('Charge Density Distribution')
    plt.savefig('charge_density_distribution.png')
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # TODO: Set simulation parameters and call functions
    nx = 100  # Number of grid points in x direction
    ny = 100  # Number of grid points in y direction
    plate_thickness = 5  # Thickness of conductor plates in grid points
    plate_separation = 20  # Separation between plates in grid points
    omega = 1.9  # Relaxation factor
    max_iter = 10000  # Maximum number of iterations
    tolerance = 1e-6  # Convergence tolerance
    
    # Grid spacing (uniform)
    Lx = 1.0  # Domain length in x direction
    Ly = 1.0  # Domain length in y direction
    dx = Lx / (nx - 1)
    dy = Ly / (ny - 1)
    
    x_coords = np.linspace(0, Lx, nx)
    y_coords = np.linspace(0, Ly, ny)
    X, Y = np.meshgrid(x_coords, y_coords)
    
    potential = solve_laplace_sor(nx, ny, plate_thickness, plate_separation, omega, max_iter, tolerance)
    
    charge_density = calculate_charge_density(potential, dx, dy)

    plot_results(potential, charge_density, x_coords, y_coords)
