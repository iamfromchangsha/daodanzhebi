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

def Kurisu_Makise(t, vx, vy, t1, t2, points,Viviana):
    daodan_vx, daodan_vy, daodan_vz = sudufenpei(300, 20000, 0, 2000)
    daodan_x = 20000 - daodan_vx * t
    daodan_y = 0
    daodan_z = 2000 - daodan_vz * t
    if Viviana == 1:
        yan_x = 17800 + vx * (t1 + t2)
        yan_y = vy * (t1 + t2)
        yan_z = 1800 - 0.5 * 9.8 * (t2**2) - 3 * (t - t1 - t2)
    if Viviana == 2:
        yan_x = 12000 + vx*(t1+t2)
        yan_y = 1400 + vy*(t1+t2)
        yan_z = 1400 - 0.5*9.8*(t2**2)-3*(t - t1 - t2)
    if Viviana == 3:
        yan_x = 6000 + vx * (t1 + t2)
        yan_y = -3000 + vy *(t1+t2)
        yan_z = 700 - 0.5 * 9.8 * (t2**2) - 3 * (t - t1 - t2)
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

def Mon3tr(vx, vy,v2x,v2y,v3x,v3y, t1, t2,t3,t4,t5,t6, points):
    t = t1+t2
    time = []
    while t < t1 +t2 +20:
        t += 0.1
        Viviana = 1
        if Kurisu_Makise(t, vx, vy, t1, t2, points,Viviana):
            time.append(t)
    t = t3 + t4
    while t < t3 + t4 +20:
        Viviana = 2
        t += 0.1
        if Kurisu_Makise(t, v2x, v2y, t3, t4, points,Viviana):
            time.append(t)
    t = t5 + t6
    while t < t5 + t6 +20:
        Viviana = 3
        t += 0.1
        if Kurisu_Makise(t, v3x, v3y, t5, t6, points,Viviana):
            time.append(t)

    time = list(set(time))

    return len(time) * 0.1 
def lhs_sample(n, bounds):
    """拉丁超立方抽样"""
    sample = np.zeros((n, len(bounds)))
    for i, (low, high) in enumerate(bounds):
        intervals = np.linspace(low, high, n+1)
        samples = np.random.uniform(intervals[:-1], intervals[1:])
        np.random.shuffle(samples)
        sample[:, i] = samples
    return sample
def pso_optimize():
    # 参数设置
    n_particles = 30      
    max_iter = 100  
    w = 0.7               
    c1 = 1.5              
    c2 = 1.5          
    # 在 pso_optimize() 开头添加
    no_improve_count = 0
    max_no_improve = 15  # 最多容忍 15 代无改进   

    vx_min, vx_max = -140, 140
    vy_min, vy_max = 0, 140
    v2_min, v2_max = -140, 140
    v3_min, v3_max = -140, 140
    t1_min, t1_max = 0.1, 13.82
    t2_min, t2_max = 0.1, 11.76
    t3_min, t3_max = 0.1, 13.82
    t4_min, t4_max = 0.1, 11.76
    t5_min, t5_max = 0.1, 13.82
    t6_min, t6_max = 0.1, 11.76

    particles = []
    velocities = []
    pbest_positions = []   
    pbest_fitness = []    

    points = yuanzhufenge() 
    history_gbest_fitness = []

    # 在 pso_optimize() 中
    bounds = [
        (vx_min, vx_max), (vy_min, vy_max),
        (v2_min, v2_max), (vy_min, vy_max),
        (v3_min, v3_max), (vy_min, vy_max),
        (t1_min, t1_max), (t2_min, t2_max),
        (t3_min, t3_max), (t4_min, t4_max),
        (t5_min, t5_max), (t6_min, t6_max)
    ]

    lhs_samples = lhs_sample(n_particles, bounds)

    for i in range(n_particles):
        while True:
            sample = lhs_samples[i]
            vx, vy, v2x, v2y, v3x, v3y, t1, t2, t3, t4, t5, t6 = sample

            # 检查速度模长
            if not (70 <= (vx**2+vy**2)**0.5 <= 140): continue
            if not (70 <= (v2x**2+v2y**2)**0.5 <= 140): continue
            if not (70 <= (v3x**2+v3y**2)**0.5 <= 140): continue

            # 检查时间组合
            if t1+t2 > 13.87 or t3+t4 > 13.87 or t5+t6 > 13.87: continue

            break  # 满足所有条件

        particles.append([vx, vy, v2x,v2y,v3x,v3y,t1, t2, t3, t4, t5, t6])
        velocities.append([
            random.uniform(-10, 10),
            random.uniform(-10, 10),
            random.uniform(-10, 10),
            random.uniform(-10, 10),
            random.uniform(-10, 10),
            random.uniform(-10, 10),
            random.uniform(-1, 1),
            random.uniform(-1, 1),
            random.uniform(-1, 1),
            random.uniform(-1, 1),
            random.uniform(-1, 1),
            random.uniform(-1, 1),
        ])
        pbest_positions.append([vx, vy, v2x,v2y,v3x,v3y, t1, t2, t3, t4, t5, t6])

    for i in range(n_particles):
        fitness = Mon3tr(*particles[i], points)
        pbest_fitness.append(fitness)

    gbest_index = np.argmax(pbest_fitness)
    gbest_position = pbest_positions[gbest_index]
    gbest_fitness = pbest_fitness[gbest_index]

    # 记录初始值
    history_gbest_fitness.append(gbest_fitness)
    if improved:
        no_improve_count = 0
    else:
        no_improve_count += 1

    # 如果长时间无改进，对部分粒子进行扰动
    if no_improve_count > max_no_improve:
        for i in range(n_particles // 5):  # 扰动 20% 粒子
            idx = random.randint(0, n_particles - 1)
            # 重置位置（保持约束）
            particles[idx][0] = random.uniform(vx_min, vx_max)
            particles[idx][1] = random.uniform(vy_min, vy_max)
            speed = (particles[idx][0]**2 + particles[idx][1]**2)**0.5
            if speed > 140 or speed < 70:
                angle = random.uniform(0, 2*np.pi)
                particles[idx][0] = random.choice([70, 140]) * np.cos(angle)
                particles[idx][1] = random.choice([70, 140]) * np.sin(angle)

            # 时间部分随机但满足组合约束
            t1 = random.uniform(t1_min, t1_max)
            t2 = random.uniform(t2_min, min(t2_max, 13.87 - t1))
            particles[idx][6] = t1
            particles[idx][7] = t2

            # 其他时间也类似处理...
            t3 = random.uniform(t3_min, t3_max)
            t4 = random.uniform(t4_min, min(t4_max, 13.87 - t3))
            particles[idx][8] = t3
            particles[idx][9] = t4

            t5 = random.uniform(t5_min, t5_max)
            t6 = random.uniform(t6_min, min(t6_max, 13.87 - t5))
            particles[idx][10] = t5
            particles[idx][11] = t6

            # 速度也重置
            velocities[idx] = [random.uniform(-10,10) for _ in range(12)]
        print(f">>> 第 {iter} 代：无改进 {no_improve_count} 代，执行粒子扰动！")
        no_improve_count = 0  # 重置计数
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

        # === 更新所有粒子 ===
        for i in range(n_particles):
            vel = velocities[i]
            pos = particles[i]
            pbest = pbest_positions[i]
            r1, r2 = random.random(), random.random()

            # === 更新所有 12 个维度的速度和位置 ===
            for d in range(12):
                vel[d] = (w * vel[d] +
                          c1 * r1 * (pbest[d] - pos[d]) +
                          c2 * r2 * (gbest_position[d] - pos[d]))
                pos[d] += vel[d]

            # === 单独参数边界约束 ===
            # 时间参数 t1~t6 的独立上下界
            time_bounds = [
                (t1_min, t1_max),
                (t2_min, t2_max),
                (t3_min, t3_max),
                (t4_min, t4_max),
                (t5_min, t5_max),
                (t6_min, t6_max)
            ]
            for i, (t_min, t_max) in enumerate(time_bounds):
                pos[6 + i] = np.clip(pos[6 + i], t_min, t_max)

            # === 组合约束：t1+t2 <= 13.87, t3+t4 <= 13.87, t5+t6 <= 13.87 ===
            max_sum = 13.87

            # 修复 t1 + t2
            if pos[6] + pos[7] > max_sum:
                # 方法1：按比例缩放
                scale = max_sum / (pos[6] + pos[7])
                pos[6] *= scale
                pos[7] *= scale
                # 可选：也可只减去超出部分的一半
                # excess = (pos[6] + pos[7]) - max_sum
                # pos[6] -= excess * 0.6
                # pos[7] -= excess * 0.4

            # 修复 t3 + t4
            if pos[8] + pos[9] > max_sum:
                scale = max_sum / (pos[8] + pos[9])
                pos[8] *= scale
                pos[9] *= scale

            # 修复 t5 + t6
            if pos[10] + pos[11] > max_sum:
                scale = max_sum / (pos[10] + pos[11])
                pos[10] *= scale
                pos[11] *= scale

            # === 速度向量模长约束 ===
            def constrain_speed(x_idx, y_idx):
                speed = (pos[x_idx]**2 + pos[y_idx]**2)**0.5
                if speed > 140:
                    pos[x_idx] = 140 * pos[x_idx] / speed
                    pos[y_idx] = 140 * pos[y_idx] / speed
                elif speed < 70:
                    angle = random.uniform(0, 2*np.pi)
                    pos[x_idx] = 70 * np.cos(angle)
                    pos[y_idx] = 70 * np.sin(angle)

            constrain_speed(0, 1)   # vx, vy
            constrain_speed(2, 3)   # v2x, v2y
            constrain_speed(4, 5)   # v3x, v3y

        # 记录当前最优
        history_gbest_fitness.append(gbest_fitness)

        if iter % 10 == 0 or improved:
            vx, vy, v2x, v2y, v3x, v3y, t1, t2, t3, t4, t5, t6 = gbest_position
            print(f"[迭代 {iter:3d}] "
                f"适应度={gbest_fitness:5.2f}s | "
                f"t1+t2={t1+t2:5.2f}, t3+t4={t3+t4:5.2f}, t5+t6={t5+t6:5.2f} | "
                f"v1=({vx:5.1f},{vy:5.1f}), v2=({v2x:5.1f},{v2y:5.1f}), v3=({v3x:5.1f},{v3y:5.1f})")
            print("\n=== 优化完成 ===")
    vx, vy, v2x, v2y, v3x, v3y, t1, t2, t3, t4, t5, t6 = gbest_position

    print("\n=== 最优参数 ===")
    print(f"烟雾弹 1 发射速度: vx1 = {vx:8.2f}, vy1 = {vy:8.2f}  → 速度模长 = {np.hypot(vx, vy):.2f}")
    print(f"烟雾弹 1 发射时间: t1  = {t1:8.2f}, t2  = {t2:8.2f}  → 总延迟 = {t1+t2:.2f}s")
    print(f"烟雾弹 2 发射速度: vx2 = {v2x:8.2f}, vy2 = {v2y:8.2f}  → 速度模长 = {np.hypot(v2x, v2y):.2f}")
    print(f"烟雾弹 2 发射时间: t3  = {t3:8.2f}, t4  = {t4:8.2f}  → 总延迟 = {t3+t4:.2f}s")
    print(f"烟雾弹 3 发射速度: vx3 = {v3x:8.2f}, vy3 = {v3y:8.2f}  → 速度模长 = {np.hypot(v3x, v3y):.2f}")
    print(f"烟雾弹 3 发射时间: t5  = {t5:8.2f}, t6  = {t6:8.2f}  → 总延迟 = {t5+t6:.2f}s")
    print(f"\n最大遮挡时间: {gbest_fitness:.2f} 秒")
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
    vx, vy, v2x, v2y, v3x, v3y, t1, t2, t3, t4, t5, t6 = best_params
    daodan_vx, daodan_vy, daodan_vz = sudufenpei(300, 20000, 0, 2000)
    
    t_list = np.arange(0, 69, 0.5)
    
    missile_x, missile_y, missile_z = [], [], []
    smoke_x, smoke_y, smoke_z = [], [], []
    obs_x, obs_y, obs_z = [], [], []

    for t in t_list:
        # 导弹轨迹
        mx = 20000 - daodan_vx * t
        my = 0
        mz = 2000 - daodan_vz * t
        missile_x.append(mx)
        missile_y.append(my)
        missile_z.append(mz)

        # 判断属于哪一段烟雾弹
        if t >= t1 + t2 and t < t1 + t2 + 20:
            # 第一段
            sx = 17800 + vx * (t1 + t2)
            sy = vy * (t1 + t2)
            sz = 1800 - 0.5 * 9.8 * (t2**2) - 3 * (t - t1 - t2)
            if Kurisu_Makise(t, vx, vy, t1, t2, points, 1):
                obs_x.append(sx); obs_y.append(sy); obs_z.append(sz)
        elif t >= t3 + t4 and t < t3 + t4 + 20:
            # 第二段
            sx = 12000 + v2x * (t3 + t4)
            sy = 1400 + v2y * (t3 + t4)
            sz = 1400 - 0.5 * 9.8 * (t4**2) - 3 * (t - t3 - t4)
            if Kurisu_Makise(t, v2x, v2y, t3, t4, points, 2):
                obs_x.append(sx); obs_y.append(sy); obs_z.append(sz)
        elif t >= t5 + t6 and t < t5 + t6 + 20:
            # 第三段
            sx = 6000 + v3x * (t5 + t6)
            sy = -3000 + v3y * (t5 + t6)
            sz = 700 - 0.5 * 9.8 * (t6**2) - 3 * (t - t5 - t6)
            if Kurisu_Makise(t, v3x, v3y, t5, t6, points, 3):
                obs_x.append(sx); obs_y.append(sy); obs_z.append(sz)
        else:
            sx = sy = sz = np.nan  # 不发射

        smoke_x.append(sx)
        smoke_y.append(sy)
        smoke_z.append(sz)

    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection='3d')

    ax.plot(missile_x, missile_y, missile_z, 'r-', label='Missile', alpha=0.8)
    ax.plot(smoke_x, smoke_y, smoke_z, 'b--', label='Smoke Clouds', alpha=0.6)

    if obs_x:
        ax.scatter(obs_x, obs_y, obs_z, color='purple', s=40, label='Obscuration Points', alpha=0.9)

    # 障碍物
    obstacle_pts = np.array(points)
    ax.scatter(obstacle_pts[:,0], obstacle_pts[:,1], obstacle_pts[:,2], 
               color='gray', s=10, alpha=0.4, label='Cylinder')

    ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z')
    ax.set_title('3D Trajectory: Missile, Multi-Smoke & Obscuration')
    ax.legend()
    ax.view_init(elev=20, azim=45)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    best_params, best_fitness, history = pso_optimize()
    
    # 可视化
    plot_convergence(history)
    plot_3d_trajectory(best_params, yuanzhufenge())