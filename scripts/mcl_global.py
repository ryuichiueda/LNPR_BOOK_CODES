#!/usr/bin/env python
# coding: utf-8

# In[1]:


import sys 
sys.path.append('../scripts/')
from robot import *
from scipy.stats import multivariate_normal
import random #追加
import copy


# In[2]:


class Particle:  ###mclglobal1particle
    def __init__(self, init_pose, weight):
        if init_pose is None:
            self.pose = np.array([
                np.random.uniform(-5.0, 5.0),
                np.random.uniform(-5.0, 5.0),
                np.random.uniform(-math.pi, math.pi)]
            ).T
        else:
            self.pose = init_pose
            
        self.weight = weight
        
    def motion_update(self, nu, omega, time, noise_rate_pdf): 
        ns = noise_rate_pdf.rvs()
        pnu = nu + ns[0]*math.sqrt(abs(nu)/time) + ns[1]*math.sqrt(abs(omega)/time)
        pomega = omega + ns[2]*math.sqrt(abs(nu)/time) + ns[3]*math.sqrt(abs(omega)/time)
        self.pose = IdealRobot.state_transition(pnu, pomega, time, self.pose)
        
    def observation_update(self, observation, envmap, distance_dev_rate, direction_dev):  #変更
        for d in observation:
            obs_pos = d[0]
            obs_id = d[1]
            
            ##パーティクルの位置と地図からランドマークの距離と方角を算出##
            pos_on_map = envmap.landmarks[obs_id].pos
            particle_suggest_pos = IdealCamera.relative_polar_pos(self.pose, pos_on_map)
            
            ##尤度の計算##
            distance_dev = distance_dev_rate*particle_suggest_pos[0]
            cov = np.diag(np.array([distance_dev**2, direction_dev**2]))
            self.weight *= multivariate_normal(mean=particle_suggest_pos, cov=cov).pdf(obs_pos)


# In[3]:


class Mcl:  
    def __init__(self, envmap, init_pose, num, motion_noise_stds,                  distance_dev_rate=0.14, direction_dev=0.05):
        self.particles = [Particle(init_pose, 1.0/num) for i in range(num)]
        self.map = envmap
        self.distance_dev_rate = distance_dev_rate
        self.direction_dev = direction_dev

        v = motion_noise_stds
        c = np.diag([v["nn"]**2, v["no"]**2, v["on"]**2, v["oo"]**2])
        self.motion_noise_rate_pdf = multivariate_normal(cov=c)
        
        self.ml = self.particles[0] #追加
        
    def set_ml(self): #追加
        i = np.argmax([p.weight for p in self.particles])
        self.ml = self.particles[i]
        
    def motion_update(self, nu, omega, time): 
        for p in self.particles: p.motion_update(nu, omega, time, self.motion_noise_rate_pdf)
            
    def observation_update(self, observation): 
        for p in self.particles:
            p.observation_update(observation, self.map, self.distance_dev_rate, self.direction_dev) 
        self.set_ml() #リサンプリング前に実行
        self.resampling() 
            
    def resampling(self): 
        ws = [e.weight for e in self.particles]    # 重みのリストを作る
        
        #重みの和がゼロに丸め込まれるとサンプリングできなくなるので小さな数を足しておく
        if sum(ws) < 1e-100: ws = [e + 1e-100 for e in ws]
        
        # パーティクルのリストから、weightsのリストの重みに比例した確率で、num個選ぶ    
        ps = random.choices(self.particles, weights=ws, k=len(self.particles))  
        
        # 選んだリストからパーティクルを取り出し、重みを均一に
        self.particles = [copy.deepcopy(e) for e in ps]
        for p in self.particles: p.weight = 1.0/len(self.particles)
        
    def draw(self, ax, elems):  
        xs = [p.pose[0] for p in self.particles]
        ys = [p.pose[1] for p in self.particles]
        vxs = [math.cos(p.pose[2])*p.weight*len(self.particles) for p in self.particles] #重みを要素に反映
        vys = [math.sin(p.pose[2])*p.weight*len(self.particles) for p in self.particles]  #重みを要素に反映
        elems.append(ax.quiver(xs, ys, vxs, vys,                                angles='xy', scale_units='xy', scale=1.5, color="blue", alpha=0.5)) #変更


# In[4]:


class MclAgent(Agent): 
    def __init__(self, time_interval, nu, omega, particle_pose, envmap, particle_num=100,                 motion_noise_stds={"nn":0.19, "no":0.001, "on":0.13, "oo":0.2}): #2行目にenvmapを追加
        super().__init__(nu, omega)
        self.pf = Mcl(envmap, particle_pose, particle_num, motion_noise_stds) #envmapを追加
        self.time_interval = time_interval
        
        self.prev_nu = 0.0
        self.prev_omega = 0.0
        
    def decision(self, observation=None): 
        self.pf.motion_update(self.prev_nu, self.prev_omega, self.time_interval)
        self.prev_nu, self.prev_omega = self.nu, self.omega
        self.pf.observation_update(observation)
        return self.nu, self.omega
        
    def draw(self, ax, elems):
        self.pf.draw(ax, elems)
        x, y, t = self.pf.ml.pose #以下追加
        s = "({:.2f}, {:.2f}, {})".format(x,y,int(t*180/math.pi)%360)
        elems.append(ax.text(x, y+0.1, s, fontsize=8))


# In[5]:


def trial(animation): ###mclglobal1test（下のTrialのところまで）
    time_interval = 0.1
    world = World(30, time_interval, debug=not animation) 

    ## 地図を生成して3つランドマークを追加 ##
    m = Map()                                  
    m.append_landmark(Landmark(-4,2))
    m.append_landmark(Landmark(2,-3))
    m.append_landmark(Landmark(3,3))
    world.append(m)

    ## ロボットを作る ##
    init_pose = np.array([
        np.random.uniform(-5.0, 5.0),
        np.random.uniform(-5.0, 5.0),
        np.random.uniform(-math.pi, math.pi)]
    ).T
        
    a = MclAgent(time_interval, 0.2, 10.0/180*math.pi, None, m, particle_num=1000) #初期姿勢をNoneに
    r = Robot(init_pose, sensor=Camera(m), agent=a, color="red")
    world.append(r)

    world.draw()
    
    return (r.pose, a.pf.ml.pose)


# In[6]:


if __name__ == '__main__':
    ok = 0 ###mclglobal1exec
    for i in range(1000):
        actual, estm = trial(False)
        diff = math.sqrt((actual[0]-estm[0])**2 + (actual[1]-estm[1])**2)
        print(i, "真値:", actual, "推定値", estm, "誤差:", diff)
        if diff <= 1.0:
            ok += 1

    ok


# In[ ]:




