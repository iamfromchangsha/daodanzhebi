import numpy as np


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

def sudufenpei(v,x,y,z):
    k = y/x
    h = z/x
    v_x=(v**2/(k**2+h**2+1))**0.5
    v_y = k*v_x
    v_z = h*v_x
    return v_x,v_y,v_z

def distance(point,point_line,fangxiang):
    p = np.array(point)
    l = np.array(point_line)
    d = np.array(fangxiang)
    lp = p - l
    cr = np.cross(lp,d)
    distance = np.linalg.norm(cr) / np.linalg.norm(d)
    return distance

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
print("初始位置：",bao_x,bao_y,bao_z)
for qiut in np.arange(0,20,0.01):
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
        if dist < 10 and xiangliang(x,y,z,sb[0],sb[1],sb[2],daodan_x,daodan_y,daodan_z):
            k.append(1)
    if len(k)== len(points):
        time.append(t)

print(time)
for i in range(len(time)):
    if i == 0 :
        aaa = 1
    else:
        if time[i] - time[i-1] <= 0.15:
            aaa = aaa + 1
            if aaa == len(time):
                print("有效遮蔽时间断{}~{}秒".format(time[0],time[-1]))

print("有效遮蔽时间：",round((len(time)-1)*0.01,2),"秒")
            
