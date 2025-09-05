import numpy as np
import random

random.seed(1)
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

def cos(a,b,c):
    cos = (a**2+b**2-c**2)/(2*a*b)
    if cos <= 0:
        return True
    else:
        return False
    
vx = None
vy = None
t1 = None
t2 = None



def Kurisu_Makise(t,vx,vy,t1,t2):#适应度函数,判断是否遮挡
   # time = []
   # time.clear()
    daodan_vx,daodan_vy,daodan_vz = sudufenpei(300,20000,0,2000)
    daodan_x = 20000 - daodan_vx*t
    daodan_y = 0
    daodan_z = 2000 - daodan_vz*t
    points = yuanzhufenge()
    k = []
    k.clear()
    yan_x = 17800+vx*(t1+t2)
    yan_y = vy*(t1+t2)
    yan_z = 1800-0.5*9.8*(t2**2)-3*(t-t1-t2)
    point = [yan_x,yan_y,yan_z]
    line_point = [daodan_x,daodan_y,daodan_z]
    for sb in points:
        direction = [daodan_x-sb[0],daodan_y-sb[1],daodan_z-sb[2]]
        dist = distance(point,line_point,direction)
        if dist < 10 and cos(((yan_x-sb[0])**2+(yan_y-sb[1])**2+(yan_z-sb[2])**2)**0.5,((daodan_x-yan_x)**2+(daodan_y-yan_y)**2+(daodan_z-yan_z)**2)**0.5,((daodan_x-sb[0])**2+(daodan_y-sb[1])**2+(daodan_z-sb[2])**2)**0.5):
            k.append(1)
        if len(k)== len(points):
           # time.append(t)
            return True

def Raidian(n):
    geti = []
    while len(geti)<=n:
        vx = random.uniform(0,140)
        vy = random.uniform(0,140)
        t1 = random.uniform(0,30)
        t2 = random.uniform(0,11.76)
        if vx**2+vy**2 <= 140**2:
            geti.append([vx,vy,t1,t2])
    return geti

def Mon3tr(vx,vy,t1,t2):#求个体的适应度
    t = 0
    time = []
    time.clear()
    while t < 69:
        t = t + 0.01
        buer = Kurisu_Makise(t,vx,vy,t1,t2)
        if buer == True:
            time.append(t)
    total_time = 0.01*len(time)
    return total_time

def Viviana(youxiugeti):#全随机交换
    n=len(youxiugeti)
    simple = []
    simple.clear()
    for i in range(len(youxiugeti)):
        simple.append(i)
    select1 = random.sample(simple,len(youxiugeti))
    select2 = random.sample(simple,len(youxiugeti))
    select3 = random.sample(simple,len(youxiugeti))
    select4 = random.sample(simple,len(youxiugeti))
    fengchuang = True
    good_huan = []
    willclear = []
    i =0
    while i < n:
        geti = []
        geti.clear()
        x = youxiugeti[select1[i]][0]
        y = youxiugeti[select2[i]][1]
        r_squared = x**2 + y**2
        if r_squared > 140**2 or r_squared < 70**2:
            if i + 1 < n:
                select1[i], select1[i+1] = select1[i+1], select1[i]
                if fengchuang == False:
                    willclear.append(i)
                    print(f"索引 {i} 无法满足条件")
                    geti.append(youxiugeti[select1[i]][0])
                    geti.append(youxiugeti[select2[i]][1])
                    geti.append(youxiugeti[select3[i]][2])
                    geti.append(youxiugeti[select4[i]][3])
                    good_huan.append(geti)
                    i += 1
                    fengchuang = True
                    continue
                fengchuang = False
                continue
            else:
                if fengchuang == False:
                    willclear.append(i)
                    print(f"索引 {i} 无法满足条件")
                    geti.append(youxiugeti[select1[i]][0])
                    geti.append(youxiugeti[select2[i]][1])
                    geti.append(youxiugeti[select3[i]][2])
                    geti.append(youxiugeti[select4[i]][3])
                    good_huan.append(geti)
                    i += 1
                    fengchuang = True
                fengchuang = False
                continue
        else:
            geti.append(youxiugeti[select1[i]][0])
            geti.append(youxiugeti[select2[i]][1])
            geti.append(youxiugeti[select3[i]][2])
            geti.append(youxiugeti[select4[i]][3])
            good_huan.append(geti)
            i += 1
    print(good_huan)
    for i in sorted(willclear, reverse=True):
        del good_huan[i]
    print('最终交换完个体：',good_huan)
    return good_huan

def Suzuran(youxiugeti):#特定1位交换
    n=len(youxiugeti)
    simple = []
    simple.clear()
    for i in range(len(youxiugeti)):
        simple.append(i)
    select1 = random.sample(simple,len(youxiugeti))
    #select2 = random.sample(simple,len(youxiugeti))
    #select3 = random.sample(simple,len(youxiugeti))
    #select4 = random.sample(simple,len(youxiugeti))
    fengchuang = True
    good_huan = []
    willclear = []
    i = 0
    while i < n:
        geti = []
        geti.clear()
        x = youxiugeti[select1[i]][0]
        y = youxiugeti[i][1]
        r_squared = x**2 + y**2
        if r_squared > 140**2 or r_squared < 70**2:
            if i + 1 < n:
                select1[i], select1[i+1] = select1[i+1], select1[i]
                if fengchuang == False:
                    willclear.append(i)
                    print(f"索引 {i} 无法满足条件")
                    geti.append(youxiugeti[select1[i]][0])
                    geti.append(youxiugeti[i][1])
                    geti.append(youxiugeti[i][2])
                    geti.append(youxiugeti[i][3])
                    good_huan.append(geti)
                    i += 1
                    fengchuang = True
                    continue
                fengchuang = False
                continue
            else:
                if fengchuang == False:
                    willclear.append(i)
                    print(f"索引 {i} 无法满足条件")
                    geti.append(youxiugeti[select1[i]][0])
                    geti.append(youxiugeti[i][1])
                    geti.append(youxiugeti[i][2])
                    geti.append(youxiugeti[i][3])
                    good_huan.append(geti)
                    i += 1
                    fengchuang = True
                fengchuang = False
                continue
        else:
            geti.append(youxiugeti[select1[i]][0])
            geti.append(youxiugeti[i][1])
            geti.append(youxiugeti[i][2])
            geti.append(youxiugeti[i][3])
            good_huan.append(geti)
            i += 1
    print(good_huan)
    for i in sorted(willclear, reverse=True):
        del good_huan[i]
    print('最终交换完个体：',good_huan)
    return good_huan

def Myrtle(youxiugeti):#特定第二位交换
    n=len(youxiugeti)
    simple = []
    simple.clear()
    for i in range(len(youxiugeti)):
        simple.append(i)
    #select1 = random.sample(simple,len(youxiugeti))
    select2 = random.sample(simple,len(youxiugeti))
    #select3 = random.sample(simple,len(youxiugeti))
    #select4 = random.sample(simple,len(youxiugeti))
    fengchuang = True
    good_huan = []
    willclear = []
    i = 0
    while i < n:
        geti = []
        geti.clear()
        x = youxiugeti[i][0]
        y = youxiugeti[select2[i]][1]
        r_squared = x**2 + y**2
        if r_squared > 140**2 or r_squared < 70**2:
            if i + 1 < n:
                select2[i], select2[i+1] = select2[i+1], select2[i]
                if fengchuang == False:
                    willclear.append(i)
                    print(f"索引 {i} 无法满足条件")
                    geti.append(youxiugeti[i][0])
                    geti.append(youxiugeti[select2[i]][1])
                    geti.append(youxiugeti[i][2])
                    geti.append(youxiugeti[i][3])
                    good_huan.append(geti)
                    i += 1
                    fengchuang = True
                    continue
                fengchuang = False
                continue
            else:
                if fengchuang == False:
                    willclear.append(i)
                    print(f"索引 {i} 无法满足条件")
                    geti.append(youxiugeti[i][0])
                    geti.append(youxiugeti[select2[i]][1])
                    geti.append(youxiugeti[i][2])
                    geti.append(youxiugeti[i][3])
                    good_huan.append(geti)
                    i += 1
                    fengchuang = True
                fengchuang = False
                continue
        else:
            geti.append(youxiugeti[i][0])
            geti.append(youxiugeti[select2[i]][1])
            geti.append(youxiugeti[i][2])
            geti.append(youxiugeti[i][3])
            good_huan.append(geti)
            i += 1
    print(good_huan)
    for i in sorted(willclear, reverse=True):
        del good_huan[i]
    print('最终交换完个体：',good_huan)
    return good_huan

def Skadi(youxiugeti):#特定第三位交换
    n=len(youxiugeti)
    simple = []
    simple.clear()
    for i in range(len(youxiugeti)):
        simple.append(i)
    #select1 = random.sample(simple,len(youxiugeti))
    #select2 = random.sample(simple,len(youxiugeti))
    select3 = random.sample(simple,len(youxiugeti))
    #select4 = random.sample(simple,len(youxiugeti))
    fengchuang = True
    good_huan = []
    willclear = []
    i = 0
    while i < n:
        geti = []
        geti.clear()
        x = youxiugeti[i][0]
        y = youxiugeti[i][1]
        r_squared = x**2 + y**2
        if r_squared > 140**2 or r_squared < 70**2:
            if i + 1 < n:
                youxiugeti[i][0], youxiugeti[i+1][0] = youxiugeti[i+1][0], youxiugeti[i][0]
                if fengchuang == False:
                    willclear.append(i)
                    print(f"索引 {i} 无法满足条件")
                    geti.append(youxiugeti[i][0])
                    geti.append(youxiugeti[i][1])
                    geti.append(youxiugeti[select3[i]][2])
                    geti.append(youxiugeti[i][3])
                    good_huan.append(geti)
                    i += 1
                    fengchuang = True
                    continue
                fengchuang = False
                continue
            else:
                if fengchuang == False:
                    willclear.append(i)
                    print(f"索引 {i} 无法满足条件")
                    geti.append(youxiugeti[i][0])
                    geti.append(youxiugeti[i][1])
                    geti.append(youxiugeti[select3[i]][2])
                    geti.append(youxiugeti[i][3])
                    good_huan.append(geti)
                    i += 1
                    fengchuang = True
                fengchuang = False
                continue
        else:
            geti.append(youxiugeti[i][0])
            geti.append(youxiugeti[i][1])
            geti.append(youxiugeti[select3[i]][2])
            geti.append(youxiugeti[i][3])
            good_huan.append(geti)
            i += 1
    print(good_huan)
    for i in sorted(willclear, reverse=True):
        del good_huan[i]
    print('最终交换完个体：',good_huan)
    return good_huan

def Amiya(youxiugeti):#特定第四位交换
    n=len(youxiugeti)
    simple = []
    simple.clear()
    for i in range(len(youxiugeti)):
        simple.append(i)
    #select1 = random.sample(simple,len(youxiugeti))
    #select2 = random.sample(simple,len(youxiugeti))
    #select3 = random.sample(simple,len(youxiugeti))
    select4 = random.sample(simple,len(youxiugeti))
    fengchuang = True
    good_huan = []
    willclear = []
    i = 0
    while i < n:
        geti = []
        geti.clear()
        x = youxiugeti[i][0]
        y = youxiugeti[i][1]
        r_squared = x**2 + y**2
        if r_squared > 140**2 or r_squared < 70**2:
            if i + 1 < n:
                youxiugeti[i][0], youxiugeti[i+1][0] = youxiugeti[i+1][0], youxiugeti[i][0]
                if fengchuang == False:
                    willclear.append(i)
                    print(f"索引 {i} 无法满足条件")
                    geti.append(youxiugeti[i][0])
                    geti.append(youxiugeti[i][1])
                    geti.append(youxiugeti[i][2])
                    geti.append(youxiugeti[select4[i]][3])
                    good_huan.append(geti)
                    i += 1
                    fengchuang = True
                    continue
                fengchuang = False
                continue
            else:
                if fengchuang == False:
                    willclear.append(i)
                    print(f"索引 {i} 无法满足条件")
                    geti.append(youxiugeti[i][0])
                    geti.append(youxiugeti[i][1])
                    geti.append(youxiugeti[i][2])
                    geti.append(youxiugeti[select4[i]][3])
                    good_huan.append(geti)
                    i += 1
                    fengchuang = True
                fengchuang = False
                continue
        else:
            geti.append(youxiugeti[i][0])
            geti.append(youxiugeti[i][1])
            geti.append(youxiugeti[i][2])
            geti.append(youxiugeti[select4[i]][3])
            good_huan.append(geti)
            i += 1
    print(good_huan)
    for i in sorted(willclear, reverse=True):
        del good_huan[i]
    print('最终交换完个体：',good_huan)
    return good_huan

def Typhon(youxiugeti):#特定锁后两位交换
    n=len(youxiugeti)
    simple = []
    simple.clear()
    for i in range(len(youxiugeti)):
        simple.append(i)
    select1 = random.sample(simple,len(youxiugeti))
    select2 = random.sample(simple,len(youxiugeti))
    #select3 = random.sample(simple,len(youxiugeti))
    #select4 = random.sample(simple,len(youxiugeti))
    fengchuang = True
    good_huan = []
    willclear = []
    i = 0
    while i < n:
        geti = []
        geti.clear()
        x = youxiugeti[select1[i]][0]
        y = youxiugeti[select2[i]][1]
        r_squared = x**2 + y**2
        if r_squared > 140**2 or r_squared < 70**2:
            if i + 1 < n:
                select1[i], select1[i+1] = select1[i+1], select1[i]
                if fengchuang == False:
                    willclear.append(i)
                    print(f"索引 {i} 无法满足条件")
                    geti.append(youxiugeti[select1[i]][0])
                    geti.append(youxiugeti[select2[i]][1])
                    geti.append(youxiugeti[i][2])
                    geti.append(youxiugeti[i][3])
                    good_huan.append(geti)
                    i += 1
                    fengchuang = True
                    continue
                fengchuang = False
                continue
            else:
                if fengchuang == False:
                    willclear.append(i)
                    print(f"索引 {i} 无法满足条件")
                    geti.append(youxiugeti[select1[i]][0])
                    geti.append(youxiugeti[select2[i]][1])
                    geti.append(youxiugeti[i][2])
                    geti.append(youxiugeti[i][3])
                    good_huan.append(geti)
                    i += 1
                    fengchuang = True
                fengchuang = False
                continue
        else:
            geti.append(youxiugeti[select1[i]][0])
            geti.append(youxiugeti[select2[i]][1])
            geti.append(youxiugeti[i][2])
            geti.append(youxiugeti[i][3])
            good_huan.append(geti)
            i += 1
    print(good_huan)
    for i in sorted(willclear, reverse=True):
        del good_huan[i]
    print('最终交换完个体：',good_huan)
    return good_huan

def Virtuosa(youxiugeti):#特定锁死前两位交换
    n=len(youxiugeti)
    simple = []
    simple.clear()
    for i in range(len(youxiugeti)):
        simple.append(i)
    #select1 = random.sample(simple,len(youxiugeti))
    #select2 = random.sample(simple,len(youxiugeti))
    select3 = random.sample(simple,len(youxiugeti))
    select4 = random.sample(simple,len(youxiugeti))
    fengchuang = True
    good_huan = []
    willclear = []
    i = 0
    while i < n:
        geti = []
        geti.clear()
        x = youxiugeti[i][0]
        y = youxiugeti[i][1]
        r_squared = x**2 + y**2
        if r_squared > 140**2 or r_squared < 70**2:
            if i + 1 < n:
                youxiugeti[i][0], youxiugeti[i+1][0] = youxiugeti[i+1][0], youxiugeti[i][0]
                if fengchuang == False:
                    willclear.append(i)
                    print(f"索引 {i} 无法满足条件")
                    geti.append(youxiugeti[i][0])
                    geti.append(youxiugeti[i][1])
                    geti.append(youxiugeti[select3[i]][2])
                    geti.append(youxiugeti[select4[i]][3])
                    good_huan.append(geti)
                    i += 1
                    fengchuang = True
                    continue
                fengchuang = False
                continue
            else:
                if fengchuang == False:
                    willclear.append(i)
                    print(f"索引 {i} 无法满足条件")
                    geti.append(youxiugeti[i][0])
                    geti.append(youxiugeti[i][1])
                    geti.append(youxiugeti[select3[i]][2])
                    geti.append(youxiugeti[select4[i]][3])
                    good_huan.append(geti)
                    i += 1
                    fengchuang = True
                fengchuang = False
                continue
        else:
            geti.append(youxiugeti[i][0])
            geti.append(youxiugeti[i][1])
            geti.append(youxiugeti[select3[i]][2])
            geti.append(youxiugeti[select4[i]][3])
            good_huan.append(geti)
            i += 1
    print(good_huan)
    for i in sorted(willclear, reverse=True):
        del good_huan[i]
    print('最终交换完个体：',good_huan)
    return good_huan

def main():
    geti = Raidian(100)
    bijiao = []
    bijiao.clear()
    for i in geti:
        time = Mon3tr(i[0],i[1],i[2],i[3])
        bijiao.append(time)
    sorted_list = sorted(zip(bijiao, geti), reverse=True)
    result_list = [item for _, item in sorted_list]
    result_list = result_list[:20]
    quansuiji = Viviana(result_list)
    yiweisuiji = Suzuran(result_list)
    erweisuiji = Myrtle(result_list)
    sanweisuiji = Skadi(result_list)
    siweisuiji = Amiya(result_list)
    housuosi = Typhon(result_list)
    qiansuosi = Virtuosa(result_list)
    totaljiaohuan = quansuiji + yiweisuiji + erweisuiji + sanweisuiji + siweisuiji + housuosi + qiansuosi














    



