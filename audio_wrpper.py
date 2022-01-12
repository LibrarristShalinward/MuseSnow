from au_frm import audio2frame
import numpy as np
from scipy.ndimage import convolve
from tqdm import tqdm
import pickle
import threading



class Audio_wrapper: 
    def __init__(self, wrapper = None, sound = None) -> None:
        if wrapper is not None: 
            return self.__init_file(wrapper)
        if sound is not None: 
            return self.__init_sound(sound)
        assert False; 
    
    def __gen_conv(self): 
        fres = np.exp(np.linspace(np.log(10), np.log(4000), 21))
        freb = np.array([(fres[i], fres[i + 5]) for i in range(16)])
        conv_table = np.zeros((16, 800))
        base = np.array([i for i in range(800)])
        for i in (range(1, 801)): 
            Cos = np.cos(i * np.pi / 4000 * base)
            idx = 0
            while idx < 16 and i * 10 > freb[idx, 1]: idx += 1
            while idx < 16 and i * 10 >= freb[idx, 0]: 
                conv_table[idx, :] += Cos
                idx += 1

        for i in range(16): 
            conv_table[i, :] /= conv_table[i, 0]
        
        self.conv = conv_table

    def __init_sound(self, sound, n = 10): 
        print("音频读取...")
        frms = audio2frame(sound, frame = 800)
        self.num_frm = len(frms)

        print("音频分析...")
        en = (frms ** 2).sum(axis = -1)
        dense = (1. + np.log10(en)) * 100
        self.dense = np.clip(dense, 5, 300).tolist()

        self.__gen_conv()

        # proba = []
        # for i in tqdm(range(frms.shape[0])): 
        #     proba_frm = []
        #     for j in range(16): 
        #         re = convolve(frms[i, :], self.conv[j, :])
        #         proba_frm.append((re ** 2).sum())
        #     proba.append(proba_frm)
        # proba_a = np.log10(np.array(proba))
        # proba_a -= np.matmul(proba_a.min(axis = -1).reshape(-1, 1), np.ones((1, 16)))
        # proba_a /= np.matmul(proba_a.sum(axis = -1).reshape(-1, 1), np.ones((1, 16)))
        # self.freq_proba = proba_a.tolist()

        self.freq_proba = []
        for i in range(frms.shape[0]): 
            self.freq_proba.append([])
        self.frms = frms

        thrds = []
        for i in range(frms.shape[0]): 
            thrds.append(threading.Thread(target = self.get_proba, args = [i, ]))
        for i in tqdm(range(frms.shape[0])): 
            thrds[i].start()
        for i in tqdm(range(frms.shape[0])): 
            thrds[i].join()
        self.freq_proba = np.log10(np.array(self.freq_proba))
        self.freq_proba -= np.matmul(self.freq_proba.min(axis = -1).reshape(-1, 1), np.ones((1, 16)))
        self.freq_proba /= np.matmul(self.freq_proba.sum(axis = -1).reshape(-1, 1), np.ones((1, 16)))
        self.freq_proba = self.freq_proba.tolist()
    
    def get_proba(self, i): 
        proba_frm = []
        for j in range(16): 
            re = convolve(self.frms[i, :], self.conv[j, :])
            proba_frm.append((re ** 2).sum())
        self.freq_proba[i] = proba_frm

    
    def __init_file(self, path): 
        assert path[-5:].lower() == ".awrp"
        with open(path, "rb") as f: 
            self.dense, self.freq_proba = pickle.load(f)
        self.num_frm = len(self.dense)
    


    def dump(self, path): 
        assert path[-5:].lower() == ".awrp"
        with open(path, "wb") as f: 
            pickle.dump((self.dense, self.freq_proba), f)
    


    def get_status(self, t): 
        frm = self.get_frm(t)
        return self.dense[frm], self.freq_proba[frm]
    
    def get_frm(self, t): 
        frm = int(t * 10.)
        return frm if frm < self.num_frm else self.num_frm - 1