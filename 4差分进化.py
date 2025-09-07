import numpy as np
import random
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

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

def Mon3tr(vx, vy, v2x, v2y, v3x, v3y, t1, t2, t3, t4, t5, t6, points):
    times = []  # 用于记录所有触发时刻（去重后为总时间）

    # === 烟雾弹 1 ===
    t = t1 + t2
    detection1 = []
    while t < t1 + t2 + 20:
        t += 0.1
        if Kurisu_Makise(t, vx, vy, t1, t2, points, Viviana=1):
            detection1.append(t)
    time1 = len(detection1) * 0.1

    # === 烟雾弹 2 ===
    t = t3 + t4
    detection2 = []
    while t < t3 + t4 + 20:
        t += 0.1
        if Kurisu_Makise(t, v2x, v2y, t3, t4, points, Viviana=2):
            detection2.append(t)
    time2 = len(detection2) * 0.1

    # === 烟雾弹 3 ===
    t = t5 + t6
    detection3 = []
    while t < t5 + t6 + 20:
        t += 0.1
        if Kurisu_Makise(t, v3x, v3y, t5, t6, points, Viviana=3):
            detection3.append(t)
    time3 = len(detection3) * 0.1

    # 合并所有时间点并去重（防止重叠部分重复计算）
    all_times = sorted(set(detection1 + detection2 + detection3))
    total_time = len(all_times) * 0.1  # 总遮蔽时间（并集）

    return total_time, time1, time2, time3

def de_optimize(bounds, n_pop=50, F=0.8, CR=0.9, max_iter=100):

    n_dim = len(bounds)
    # 初始化种群
    pop = np.zeros((n_pop, n_dim))
    for i in range(n_dim):
        min_val, max_val = bounds[i]
        pop[:, i] = np.random.uniform(min_val, max_val, n_pop)

    fitness = np.array([Mon3tr(*ind, points) for ind in pop])
    best_idx = np.argmax(fitness)
    best_fitness = fitness[best_idx]
    best_individual = pop[best_idx].copy()
    
    history = [best_fitness]
    print(f"初始最优适应度: {best_fitness:.2f}")

    for gen in range(max_iter):
        improved = False
        for i in range(n_pop):
            # === 1. 变异：随机选三个不同个体 a, b, c
            idxs = list(range(n_pop))
            idxs.remove(i)
            a, b, c = np.random.choice(idxs, 3, replace=False)
            mutant = pop[a] + F * (pop[b] - pop[c])
            
            # === 2. 交叉：二项式交叉
            cross_mask = np.random.rand(n_dim) <= CR
            trial = np.where(cross_mask, mutant, pop[i])
            
            # === 3. 约束修复 ===
            trial = repair_constraints(trial, bounds)
            
            # === 4. 评估与选择 ===
            f_trial = Mon3tr(*trial, points)
            if f_trial > fitness[i]:
                pop[i] = trial
                fitness[i] = f_trial
                if f_trial > best_fitness:
                    best_fitness = f_trial
                    best_individual = trial.copy()
                    improved = True
        
        history.append(best_fitness)
        
        if gen % 10 == 0 or improved:
            t1, t2, t3, t4, t5, t6 = best_individual[6:12]
            print(f"[代 {gen:3d}] 适应度={best_fitness:5.2f}s | "
                  f"t1+t2={t1+t2:5.2f}, t3+t4={t3+t4:5.2f}, t5+t6={t5+t6:5.2f}")
    
    print("\n=== DE优化完成 ===")
    return best_individual, best_fitness, history


def repair_constraints(individual, bounds):
    """
    修复个体满足所有约束
    """
    vx, vy, v2x, v2y, v3x, v3y, t1, t2, t3, t4, t5, t6 = individual

    # === 速度模长约束：70 <= |v| <= 140 ===
    def fix_speed(x, y, low=70, high=140):
        speed = np.hypot(x, y)
        if speed > high:
            x = x * high / speed
            y = y * high / speed
        elif speed < low:
            angle = np.random.uniform(0, 2*np.pi)
            x = low * np.cos(angle)
            y = low * np.sin(angle)
        return x, y

    vx, vy = fix_speed(vx, vy)
    v2x, v2y = fix_speed(v2x, v2y)
    v3x, v3y = fix_speed(v3x, v3y)

    # === 时间边界约束 ===
    t1 = np.clip(t1, bounds[6][0], bounds[6][1])
    t2 = np.clip(t2, bounds[7][0], bounds[7][1])
    t3 = np.clip(t3, bounds[8][0], bounds[8][1])
    t4 = np.clip(t4, bounds[9][0], bounds[9][1])
    t5 = np.clip(t5, bounds[10][0], bounds[10][1])
    t6 = np.clip(t6, bounds[11][0], bounds[11][1])

    # === 组合时间约束 ===
    if t1 + t2 > 13.87:
        scale = 13.87 / (t1 + t2)
        t1 *= scale
        t2 *= scale
    if t3 + t4 > 50.63:
        scale = 50.63 / (t3 + t4)
        t3 *= scale
        t4 *= scale
    if t5 + t6 > 45.81:
        scale = 45.81 / (t5 + t6)
        t5 *= scale
        t6 *= scale

    return [vx, vy, v2x, v2y, v3x, v3y, t1, t2, t3, t4, t5, t6]

if __name__ == "__main__":
    # 先生成点云
    points = yuanzhufenge()

    # 定义参数边界
    bounds = [
        (-140, 140), (0, 140),           # v1x, v1y
        (-140, 140), (-140, 0),          # v2x, v2y (v2y 负向)
        (-140, 140), (0, 140),           # v3x, v3y
        (0.1, 13.82), (0.1, 11.76),      # t1, t2
        (8.52, 17.04), (0.1, 16.9),     # t3, t4
        (22.90, 45.81), (0.1, 11.95),   # t5, t6
    ]

    # 使用差分进化优化
    best_params, best_fitness, history = de_optimize(bounds, n_pop=50, F=0.7, CR=0.8, max_iter=200)

    # 输出最优参数
    vx, vy, v2x, v2y, v3x, v3y, t1, t2, t3, t4, t5, t6 = best_params
    print("\n=== 最优参数 ===")
    print(f"烟雾弹 1 发射速度: vx1 = {vx:8.2f}, vy1 = {vy:8.2f}  → 模长 = {np.hypot(vx, vy):.2f}")
    print(f"烟雾弹 1 发射时间: t1  = {t1:8.2f}, t2  = {t2:8.2f}  → 总延迟 = {t1+t2:.2f}s")
    print(f"烟雾弹 2 发射速度: vx2 = {v2x:8.2f}, vy2 = {v2y:8.2f}  → 模长 = {np.hypot(v2x, v2y):.2f}")
    print(f"烟雾弹 2 发射时间: t3  = {t3:8.2f}, t4  = {t4:8.2f}  → 总延迟 = {t3+t4:.2f}s")
    print(f"烟雾弹 3 发射速度: vx3 = {v3x:8.2f}, vy3 = {v3y:8.2f}  → 模长 = {np.hypot(v3x, v3y):.2f}")
    print(f"烟雾弹 3 发射时间: t5  = {t5:8.2f}, t6  = {t6:8.2f}  → 总延迟 = {t5+t6:.2f}s")
    print(f"\n最大遮挡时间: {best_fitness:.2f} 秒")
        # === 计算每颗烟的遮蔽时间 ===
    total_time, time1, time2, time3 = Mon3tr(vx, vy, v2x, v2y, v3x, v3y, t1, t2, t3, t4, t5, t6, points)

    print(f"\n=== 每颗烟雾弹的遮蔽时间 ===")
    print(f"烟雾弹 1 遮蔽时间: {time1:.2f} 秒")
    print(f"烟雾弹 2 遮蔽时间: {time2:.2f} 秒")
    print(f"烟雾弹 3 遮蔽时间: {time3:.2f} 秒")
    print(f"总遮蔽时间（去重）: {total_time:.2f} 秒")

    # 绘制收敛曲线
    plt.figure(figsize=(10, 6))
    plt.plot(history, 'g-o', markersize=3)
    plt.title('差分进化算法收敛曲线')
    plt.xlabel('代数')
    plt.ylabel('最大遮蔽时间（秒）')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()