import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from matplotlib.animation import FuncAnimation

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

def distance(point,point_line,fangxiang):
    p = np.array(point)
    l = np.array(point_line)
    d = np.array(fangxiang)
    lp = p - l
    cr = np.cross(lp,d)
    distance = np.linalg.norm(cr) / np.linalg.norm(d)
    return distance
def yuanzhufenge():
    center = np.array([0,200,0])
    r = 7
    h = 10
    step = 1
    points = []
    z_c = np.arange(center[2],center[2]+h+step,step)
    angle_step = step/r
    total_angles = int(2*np.pi/angle_step) +1
    theta_coords = np.linspace(0,2*np.pi,total_angles,endpoint=True)
    for z in z_c:
        for theta in theta_coords:
            x = center[0] + r*np.cos(theta)
            y = center[1] + r*np.sin(theta)
            points.append([x,y,z])
    z_top = center[2]+h
    x_range = np.arange(-r, r + step, step)
    y_range = np.arange(-r, r + step, step)
    for dx in x_range:
        for dy in y_range:
            if dx**2 + dy**2 <= r**2+ 0.001:
                x = center[0] + dx
                y = center[1] + dy
                points.append([x,y,z_top])
    return points



def panduan(qiut):
    FY_x =[17800]
    FY_y =[0]
    FY_z = [1800] 
    d = 1.5*120
    bao_x = FY_x[0] - d
    bao_y = FY_y[0]
    bao_z = FY_z[0]
    d = 120 * 3.6   
    bao_x = bao_x - d
    bao_y = bao_y
    bao_z = bao_z - 0.5*9.8*(3.6**2)
    time = []
    x = bao_x 
    y = 0
    z = bao_z - 3*qiut #球圆心的坐标
    t = qiut +3.6+1.5
    daodan_vx,daodan_vy,daodan_vz = sudufenpei(300,20000,0,2000)
    daodan_x = 20000 - daodan_vx*t
    daodan_y = 0
    daodan_z = 2000 - daodan_vz*t
    point = [x,y,z]
    line_point = [daodan_x,daodan_y,daodan_z]
    points = yuanzhufenge()
    k= []
    k.clear()
    for sb in points:
        direction = [daodan_x-sb[0],daodan_y-sb[1],daodan_z-sb[2]]
        dist = distance(point,line_point,direction)
        if dist < 10 and x**2+ (y-200)**2+z**2 <= (daodan_x-sb[0])**2+ (daodan_y-sb[1])**2+(daodan_z-sb[2])**2+20:
            k.append(1)
    if len(k)== len(points):
        return True
    else:
        return False

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# 静态点数据
x_static = [0, 17800, 0]
y_static = [0, 0, 200]
z_static = [0, 1800, 0]

# 绘制静态点
ax.scatter(x_static, y_static, z_static, c='r', marker='o')

# 添加标签
labels = ['原点', '无人机FY1', '真目标']
for i in range(len(x_static)):
    text_x, text_y, text_z = x_static[i], y_static[i], z_static[i] - 1000
    ax.text(text_x, text_y, text_z, labels[i])
    # 连线
    ax.plot([x_static[i], x_static[i]], [y_static[i], y_static[i]], 
            [z_static[i], text_z], linestyle='--', color='gray')

# 绘制参考线
line_x = [0, 20000]
line_y = [0, 0]
line_z = [0, 2000]
ax.plot(line_x, line_y, line_z, color='b')



# === 创建可动画的点（使用 scatter）===
# 注意：scatter 返回的是 PathCollection，不能直接 set_data，要用 _offsets3d
scat = ax.scatter([], [], [], c='g', marker='o', s=50, label='运动的导弹')
# 使用 ax.text 创建3D文本而不是 ax.text2D
text = ax.text(0, 0, 0, '导弹M1', transform=ax.transData)  # 初始文本位置
scat2 = ax.scatter([], [], [], c='b', marker='o', s=50, label='云团中心')
text2 = ax.text(0, 0, 0, '云团中心', transform=ax.transData)  # 初始文本位置
prompt_text = ax.text2D(0.02, 0.95, '', transform=ax.transAxes, color='darkred')

# === 创建导弹与真目标之间的连线 ===
# 真目标位置
true_target = (0, 200, 0)

# 创建一条空的线对象，用于连接导弹和真目标
line, = ax.plot([], [], [], color='r', linestyle='-', linewidth=2)
line2, = ax.plot([], [], [], color='orange', linestyle='-', linewidth=2)

# 速度分解函数
def sudufenpei(v, x, y, z):
    k = y / x if x != 0 else 0
    h = z / x if x != 0 else 0
    v_x = (v**2 / (k**2 + h**2 + 1))**0.5
    v_y = k * v_x
    v_z = h * v_x
    return v_x, v_y, v_z

# 初始化函数
def init():
    scat._offsets3d = ([], [], [])  # 清空
    return scat,

# 动画函数
def animate(t):
    # 导弹速度 300 m/s，从 (20000,0,2000) 飞向原点
    daodan_vx, daodan_vy, daodan_vz = sudufenpei(300, 20000, 0, 2000)
    
    # 当前位置（t 是帧索引，假设每帧 0.1 秒）
    dt = 0.1
    time = t * dt
    x = max(20000 - daodan_vx * time, 0)  # 到原点停止
    y = 0
    z = max(2000 - daodan_vz * time, 0)
    
    # 更新 scatter 的位置
    scat._offsets3d = (np.array([x]), np.array([y]), np.array([z]))
    
    # 更新文本标签的位置
    text_x, text_y, text_z = x, y, z + 500
    text.set_position_3d((text_x, text_y, text_z))
    text.set_text('导弹M1')
    
    # 更新导弹到真目标的连线
    line_x_data = [x, true_target[0]]
    line_y_data = [y, true_target[1]]
    line_z_data = [z, true_target[2]]
    line.set_data(line_x_data, line_y_data)
    line.set_3d_properties(line_z_data)

    if t >=51 :
        x2 = 17188
        y2 = 0
        z2 = 1736.5 - 3*time
        scat2._offsets3d = (np.array([x2]), np.array([y2]), np.array([z2]))
        text_x2, text_y2, text_z2 = x2, y2, z2 + 1000
        text2.set_position_3d((text_x2, text_y2, text_z2))
        text2.set_text('云团中心')
    else:
        x2, y2, z2 = 114514,114514,114514
    
    ssee = panduan(time-5.1)

    if ssee == True:
        prompt_text.set_text('目标已有效遮蔽')
    else:
        prompt_text.set_text('目标未有效遮蔽')



    return scat, text, line, scat2, text2,
    

# 创建动画
ani = FuncAnimation(fig, animate, init_func=init, frames=200, interval=50, blit=False)

# 设置视角
ax.view_init(elev=20, azim=60)
ax.legend()

plt.show()