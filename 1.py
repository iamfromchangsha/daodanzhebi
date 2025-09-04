import numpy as np
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
    direction = [daodan_x,daodan_y-200,daodan_z]
    dist = distance(point,line_point,direction)
    if dist < 10 and x**2+ (y-200)**2+z**2 <= daodan_x**2+ (daodan_y-200)**2+daodan_z**2+20:
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

print("有效遮蔽时间：",round(len(time)*0.01,2),"秒")
            
