import numpy as np
import random

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

def cos(a, b, c):
    cos_val = (a**2 + b**2 - c**2) / (2 * a * b)
    return cos_val <= 0

# =================== 适应度函数保持不变 ===================

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

    for sb in points:
        direction = [daodan_x - sb[0], daodan_y - sb[1], daodan_z - sb[2]]
        dist = distance(point, line_point, direction)
        d1 = np.linalg.norm([yan_x - sb[0], yan_y - sb[1], yan_z - sb[2]])
        d2 = np.linalg.norm([daodan_x - yan_x, daodan_y - yan_y, daodan_z - yan_z])
        d3 = np.linalg.norm([daodan_x - sb[0], daodan_y - sb[1], daodan_z - sb[2]])
        if dist < 10 and cos(d1, d2, d3):
            return True
    return False

def Mon3tr(vx, vy, t1, t2, points):
    t = 0
    time_count = 0
    while t < 69:
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
    t1_min, t1_max = 0, 13.87
    t2_min, t2_max = 0, 11.76

    particles = []
    velocities = []
    pbest_positions = []   
    pbest_fitness = []    

    points = yuanzhufenge() 

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

        if iter % 10 == 0 or improved:
            print(f"迭代 {iter}: 全局最优适应度 = {gbest_fitness:.2f}")

    print("\n=== 优化完成 ===")
    print(f"最优解: vx={gbest_position[0]:.2f}, vy={gbest_position[1]:.2f}, "
          f"t1={gbest_position[2]:.2f}, t2={gbest_position[3]:.2f}")
    print(f"最大遮挡时间: {gbest_fitness:.2f} 秒")

    return gbest_position, gbest_fitness


if __name__ == "__main__":
    best_params, best_fitness = pso_optimize()