from snow import cfg, Snow
from random import random as r
from audio_wrpper import Audio_wrapper
import pickle
import numpy as np
import cv2
from tqdm import tqdm



class SnowFall:
    def __init__(self, time_len = None, policy = None, gen_fps = 1000, rend_fps = 30, path = None) -> None: 
        if path is not None: 
            obj = self.load(path)
            for attr, value in obj.__dict__.items(): 
                self.__setattr__(attr, value)
            return; 
        
        assert time_len is not None
        if policy is None: 
            policy = lambda _: 200, [1.]

        self.policy = policy
        self.gen_step = 1. / gen_fps
        self.rend_fps = rend_fps

        snowy_time = time_len - cfg.time["melt_g"] / 2

        gen_time = []
        snowflakes = []
        for i in range(int(snowy_time / self.gen_step)): 
            t = i * self.gen_step
            r = self.get_radius(t)
            if r == 0: 
                continue
            gen_time.append(t)
            snowflakes.append(Snow(r = r))
        
        event_list = []
        num_rend_frm = int(rend_fps * time_len)
        for i in range(num_rend_frm): 
            event_list.append([])

        for i in range(len(gen_time)): 
            for j in range(int(gen_time[i] * rend_fps) + 1, int((gen_time[i] + snowflakes[i].final_t) * rend_fps) + 1): 
                if j < num_rend_frm: 
                    event_list[j].append(i)
        
        self.gen_time = gen_time
        self.snowflakes = snowflakes
        self.event_list = event_list
    
    def load(self, path): 
        assert path[-5:].lower() == ".sfev"
        with open(path, "rb") as f: 
            obj = pickle.load(f)
        return obj

    
    
    def get_radius(self, t): 
        dense, proba = self.policy(t)
        proba = proba[::-1]
        if r() > dense * self.gen_step/ cfg.time["fall"]: 
            return 0
        rr = r()
        for i in range(len(proba)): 
            if rr < proba[i]: 
                return i + 10
            else: 
                rr -= proba[i]
        return i + 7
    
    def dump(self, path): 
        assert path[-5:].lower() == ".sfev"
        with open(path, "wb") as f: 
            pickle.dump(self, f)
    

    
    def gen_frm(self, frm_idx): 
        time = frm_idx / self.rend_fps
        bw_frm = np.zeros((cfg.screen["height"], cfg.screen["width"]))
        for i in self.event_list[frm_idx]: 
            re = self.snowflakes[i].iter(time - self.gen_time[i])
            if re is None:
                continue
            field, dense = re
            delta = bw_frm[field[0] : field[1], field[2] : field[3]] - dense
            bw_frm[field[0] : field[1], field[2] : field[3]] = dense + np.clip(delta, 0., 1.01)
        return bw_frm
    
    def rend(self, output_path): 
        fourcc = cv2.VideoWriter_fourcc(*"XVID")
        out = cv2.VideoWriter(output_path, fourcc, self.rend_fps, (cfg.screen["width"], cfg.screen["height"]), True)
        im_arr = np.zeros((cfg.screen["height"], cfg.screen["width"], 3))
        print("渲染中...")
        for i in tqdm(range(len(self.event_list))): 
            arr = self.gen_frm(i)
            for i in range(3): 
                im_arr[:, :, i] = arr * 255
            out.write(np.uint8(im_arr))
        out.release()





def au2snow(path): 
    name, snow_fall = get_snowfall(path)
    snow_fall.rend(name + "avi")

def get_snowfall(path): 
    if path[-5:] == ".sfev": 
        return path[:-4], SnowFall(path = path)
    if path[-5:] == ".awrp": 
        wrapper = Audio_wrapper(wrapper = path)
        while path[-1] != '.': path = path[:-1]
    else: 
        wrapper = Audio_wrapper(sound = path)
        while path[-1] != '.': path = path[:-1]
        wrapper.dump(path + "awrp")
    
    policy = wrapper.get_status

    snow_fall = SnowFall(wrapper.num_frm / 10 + 10, policy)
    snow_fall.dump(path + "sfev")
    return path, snow_fall