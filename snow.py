from configs import snow_config
import numpy as np
from numpy.random import rand, randn
from math import log, exp



cfg = snow_config()
rnd_mul = lambda sigma: exp(randn() / 1.645 * log (1 + sigma))
r_decay = lambda x: exp( - 2.5 * x) * (1 - x ** 2)

class Snow: 
    def __init__(self, r = 10, pos = None) -> None:
        self.fall_t = cfg.time["fall"] * rnd_mul(cfg.sigma["vy_mul"])
        self.mel_t = cfg.time["melt_g"] * rnd_mul(cfg.sigma["mg"])
        
        self.r = r
        self.p = rand() * cfg.screen["width"] if pos is None else pos
        
        self.vy = cfg.screen["height"] / self.fall_t
        self.vx = randn() / 1.645 * cfg.sigma["vx"] * self.vy / (1. + r / 10.) ** 3

        

        self.final_p = cfg.screen["height"] - r / 2., self.p + self.vx / self.vy * cfg.screen["height"]
        self.final_s = self.final_p[1] < cfg.screen["width"] + r and self.final_p[1] > - r
        if self.final_s:
            self.final_t = self.fall_t + self.mel_t
        elif self.final_p[1] < -r: 
            self.final_t = self.fall_t * (1 - (r + self.final_p[1]) / (self.final_p[1] - self.p))+ self.mel_t
        else: 
            self.final_t = self.fall_t * (1 - (self.final_p[1] - cfg.screen["width"] - r) / (self.final_p[1] - self.p))+ self.mel_t
    


    def t_ex(self, t): assert t <= self.final_t

    def radius(self, t): 
        self.t_ex(t)
        if t < self.fall_t: 
            return self.r
        else: 
            return self.r * r_decay((t - self.fall_t) / self.mel_t)
    
    def posX(self, t): 
        self.t_ex(t)
        if t < self.fall_t: 
            return self.p + self.vx * t
        else: 
            return self.final_p[1]
    
    def posY(self, t): 
        self.t_ex(t)
        if t < self.fall_t: 
            return self.vy * t - self.radius(t) / 2
        else: 
            return cfg.screen["height"] - self.radius(t) / 2
    


    def dense(self, mtx, mty, t): 
        r2 = (mtx - self.posX(t)) ** 2 + (mty - self.posY(t)) ** 2
        return np.exp( - r2 / self.radius(t) ** 2 * 4)
    
    def field(self, t): 
        x = self.posX(t) - self.radius(t), self.posX(t) + self.radius(t)
        y = self.posY(t) - self.radius(t), self.posY(t) + self.radius(t)
        return (
            int(max(0, y[0])), 
            int(min(cfg.screen["height"], y[1] + 1)), 
            int(max(0, x[0])), 
            int(min(cfg.screen["width"], x[1] + 1)))
    
    def iter(self, t): 
        if self.radius(t) < 1.: 
            return None
        yh, yl, xl, xr = self.field(t)
        if min(yh, yl, xl, xr) < 0: 
            return None
        mtx = np.array([[i for i in range(xl, xr)]] * (yl - yh))
        mty = np.array([[i for i in range(yh, yl)]] * (xr - xl)).transpose()
        return (yh, yl, xl, xr), self.dense(mtx, mty, t)