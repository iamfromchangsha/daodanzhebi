import numpy as np
import random
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

random.seed(42)
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
def yuanzhufenge():
    center = np.array([0, 200, 0])
    r = 7
    h = 10
    step = 5
    points = []
    z_c = np.arange(center[2], center[2] + h + step, step)
    angle_step = step / r
    total_angles = int(2 * np.pi / angle_step) + 1
    theta_coords = np.linspace(0, 2 * np.pi, total_angles, endpoint=True)
    for z in z_c:
        for theta in theta_coords:
            x = center[0] + r * np.cos(theta)
            y = center[1] + r * np.sin(theta)
            points.append([x, y, z])
    z_top = center[2] + h
    x_range = np.arange(-r, r + step, step)
    y_range = np.arange(-r, r + step, step)
    for dx in x_range:
        for dy in y_range:
            if dx**2 + dy**2 <= r**2 + 0.001:
                x = center[0] + dx
                y = center[1] + dy
                points.append([x, y, z_top])
    return points

def sudufenpei(v, x, y, z):
    k = y / x
    h = z / x
    v_x = (v**2 / (k**2 + h**2 + 1))**0.5
    v_y = k * v_x
    v_z = h * v_x
    return v_x, v_y, v_z

def distance(point, point_line, fangxiang):
    p = np.array(point)
    l = np.array(point_line)
    d = np.array(fangxiang)
    lp = p - l
    cr = np.cross(lp, d)
    return np.linalg.norm(cr) / np.linalg.norm(d)

def xiangliang(yanx,yany,yanz,mubiaox,mubiaoy,mubiaoz,daodan_x,daodan_y,daodan_z):
    
    a = np.array([yanx-mubiaox,yany-mubiaoy,yanz-mubiaoz])
    b = np.array([daodan_x-mubiaox,daodan_y-mubiaoy,daodan_z-mubiaoz])
    dd = np.linalg.norm(b)
    cross_product = np.dot(a, b)
    cross_norm = np.linalg.norm(cross_product) 
    touying = cross_norm/dd
    if touying <= dd+10:
        return True
    else:
        return False

def Kurisu_Makise(t, vx, vy, t1, t2, points):
    daodan_vx, daodan_vy, daodan_vz = sudufenpei(300, 20000, 0, 2000)
    daodan_x = 20000 - daodan_vx * t
    daodan_y = 0
    daodan_z = 2000 - daodan_vz * t

    yan_x = 17800 + vx * (t1 + t2)
    yan_y = vy * (t1 + t2)
    yan_z = 1800 - 0.5 * 9.8 * (t2**2) - 3 * (t - t1 - t2)
    point = [yan_x, yan_y, yan_z]
    line_point = [daodan_x, daodan_y, daodan_z]
    k = []
    k.clear()

    for sb in points:
        direction = [daodan_x - sb[0], daodan_y - sb[1], daodan_z - sb[2]]
        dist = distance(point, line_point, direction)
        if dist < 10 and xiangliang(yan_x,yan_y,yan_z,sb[0],sb[1],sb[2],daodan_x,daodan_y,daodan_z):
            k.append(1)
    if len(k) == len(points):
        return True
    else:
        return False

def Mon3tr(vx, vy, t1, t2, points):
    t = t1+t2
    time_count = 0
    while t < t1 +t2 +20:
        t += 0.1
        if Kurisu_Makise(t, vx, vy, t1, t2, points):
            time_count += 1
    return time_count * 0.1 

def pso_optimize():
    # 参数设置
    n_particles = 30      
    max_iter = 100  
    w = 0.7               
    c1 = 1.5              
    c2 = 1.5             

    vx_min, vx_max = 0, 140
    vy_min, vy_max = 0, 140
    t1_min, t1_max = 0.1, 13.82
    t2_min, t2_max = 0.1, 11.76

    particles = []
    velocities = []
    pbest_positions = []   
    pbest_fitness = []    

    points = yuanzhufenge() 
    history_gbest_fitness = []

    for _ in range(n_particles):
        while True:
            vx = random.uniform(vx_min, vx_max)
            vy = random.uniform(vy_min, vy_max)
            if 70 <= (vx**2 + vy**2)**0.5 <= 140:  
                break
        t1 = random.uniform(t1_min, t1_max)
        t2 = random.uniform(t2_min, t2_max)
        particles.append([vx, vy, t1, t2])
        velocities.append([
            random.uniform(-10, 10),
            random.uniform(-10, 10),
            random.uniform(-1, 1),
            random.uniform(-1, 1)
        ])
        pbest_positions.append([vx, vy, t1, t2])

    for i in range(n_particles):
        fitness = Mon3tr(*particles[i], points)
        pbest_fitness.append(fitness)

    gbest_index = np.argmax(pbest_fitness)
    gbest_position = pbest_positions[gbest_index]
    gbest_fitness = pbest_fitness[gbest_index]

    # 记录初始值
    history_gbest_fitness.append(gbest_fitness)
    print(f"初始全局最优适应度: {gbest_fitness:.2f}")

    for iter in range(max_iter):
        improved = False
        for i in range(n_particles):
            fitness = Mon3tr(*particles[i], points)

            if fitness > pbest_fitness[i]:
                pbest_fitness[i] = fitness
                pbest_positions[i] = particles[i].copy()
                if fitness > gbest_fitness:
                    gbest_fitness = fitness
                    gbest_position = particles[i].copy()
                    improved = True

        for i in range(n_particles):
            vel = velocities[i]
            pos = particles[i]
            pbest = pbest_positions[i]
            r1, r2 = random.random(), random.random()

            for d in range(4):  
                vel[d] = (w * vel[d] +
                          c1 * r1 * (pbest[d] - pos[d]) +
                          c2 * r2 * (gbest_position[d] - pos[d]))

            pos[0] += vel[0]  
            pos[1] += vel[1]  
            pos[2] += vel[2]  
            pos[3] += vel[3] 

            pos[2] = np.clip(pos[2], t1_min, t1_max)
            pos[3] = np.clip(pos[3], t2_min, t2_max)

            speed = (pos[0]**2 + pos[1]**2)**0.5
            if speed > 140:
                pos[0] = 140 * pos[0] / speed
                pos[1] = 140 * pos[1] / speed
            elif speed < 70:
                angle = random.uniform(0, 2*np.pi)
                pos[0] = 70 * np.cos(angle)
                pos[1] = 70 * np.sin(angle)

        # 每代记录一次全局最优
        history_gbest_fitness.append(gbest_fitness)

        if iter % 10 == 0 or improved:
            print(f"迭代 {iter}: 全局最优适应度 = {gbest_fitness:.2f}")

    print("\n=== 优化完成 ===")
    print(f"最优解: vx={gbest_position[0]:.2f}, vy={gbest_position[1]:.2f}, "
          f"t1={gbest_position[2]:.2f}, t2={gbest_position[3]:.2f}")
    print(f"最大遮挡时间: {gbest_fitness:.2f} 秒")

    return gbest_position, gbest_fitness, history_gbest_fitness  # 返回历史数据




def plot_convergence(history):
    """绘制收敛曲线"""
    plt.figure(figsize=(10, 6))
    plt.plot(history, 'b-o', markersize=3, label='Global Best Fitness')
    plt.title('粒子群优化算法收敛曲线')
    plt.xlabel('迭代')
    plt.ylabel('最大遮蔽时间（秒）')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()

def plot_3d_trajectory(best_params, points):
    vx, vy, t1, t2 = best_params
    daodan_vx, daodan_vy, daodan_vz = sudufenpei(300, 20000, 0, 2000)
    
    # 生成时间序列
    t_list = np.arange(0, 69, 0.5)
    
    missile_x, missile_y, missile_z = [], [], []
    smoke_x, smoke_y, smoke_z = [], [], []
    遮挡点_x, 遮挡点_y, 遮挡点_z = [], [], []

    for t in t_list:
        # 导弹位置
        mx = 20000 - daodan_vx * t
        my = 0
        mz = 2000 - daodan_vz * t
        missile_x.append(mx)
        missile_y.append(my)
        missile_z.append(mz)

        # 烟雾弹位置
        sx = 17800 + vx * (t1 + t2)
        sy = vy * (t1 + t2)
        sz = 1800 - 0.5 * 9.8 * (t2**2) - 3 * (t - t1 - t2)
        smoke_x.append(sx)
        smoke_y.append(sy)
        smoke_z.append(sz)

        # 判断是否遮挡
        if Kurisu_Makise(t, vx, vy, t1, t2, points):
            遮挡点_x.append(sx)
            遮挡点_y.append(sy)
            遮挡点_z.append(sz)

    # 创建 3D 图
    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection='3d')

    # 绘制导弹轨迹
    ax.plot(missile_x, missile_y, missile_z, 'r-', label='Missile Trajectory', alpha=0.7)

    # 绘制烟雾弹轨迹
    ax.plot(smoke_x, smoke_y, smoke_z, 'b-', label='Smoke Trajectory', alpha=0.7)

    # 绘制遮挡点（红色）
    if 遮挡点_x:
        ax.scatter(遮挡点_x, 遮挡点_y, 遮挡点_z, color='red', s=30, label='Obscuration Points', alpha=0.8)

    # 绘制圆柱障碍物（只画顶部和几个侧面点，简化显示）
    obstacle_x = [p[0] for p in points]
    obstacle_y = [p[1] for p in points]
    obstacle_z = [p[2] for p in points]
    ax.scatter(obstacle_x, obstacle_y, obstacle_z, color='gray', s=10, alpha=0.5, label='Cylinder Obstacle')

    # 设置标签和标题
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('3D Trajectory: Missile, Smoke, and Obscuration')
    ax.legend()
    ax.grid(True)

    # 调整视角
    ax.view_init(elev=20, azim=45)

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    best_params, best_fitness, history = pso_optimize()
    
    # 可视化
    plot_convergence(history)
    plot_3d_trajectory(best_params, yuanzhufenge())
