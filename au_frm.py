import librosa
import numpy as np

def enframe(signal, nw, inc, winfunc):
    '''将音频信号转化为帧。
    参数含义：
    signal:原始音频型号
    nw:每一帧的长度(这里指采样点的长度，即采样频率乘以时间间隔)
    inc:相邻帧的间隔（同上定义）
    '''
    signal_length = len(signal) #信号总长度
    if signal_length <= nw: #若信号长度小于一个帧的长度，则帧数定义为1
        nf = 1
    else: #否则，计算帧的总长度
        nf = int(np.ceil((1.0 * signal_length - nw + inc)/inc))
    pad_length = int((nf-1 ) * inc + nw) #所有帧加起来总的铺平后的长度
    zeros = np.zeros((pad_length-signal_length,)) #不够的长度使用0填补，类似于FFT中的扩充数组操作
    pad_signal = np.concatenate((signal,zeros)) #填补后的信号记为pad_signal
    indices = np.tile(np.arange(0, nw),(nf, 1)) + np.tile(np.arange(0, nf*inc, inc),(nw,1)).T  #相当于对所有帧的时间点进行抽取，得到nf*nw长度的矩阵
    indices = np.array(indices,dtype=np.int32) #将indices转化为矩阵
    frames = pad_signal[indices] #得到帧信号
    win = np.tile(winfunc, (nf, 1))  #window窗函数，这里默认取1
    return frames*win   #返回帧信号矩阵

def zero_rate(x):
    result = 0
    
    for i in range(len(x) - 1):
        result += abs(np.sign(x[i+1] - x[i]))
    return result / 2

def audio2frame(audio_file, frame=100, frameshift=30, window=1): 
    y, _ = librosa.load(audio_file, sr=8000)
    # 小于某个阈值的直接置0
    # y = np.where(np.abs(y) / np.max(y) < 0.1, 0, y)
    inc = frame - frameshift
    
    if window==0:
        z = 1
    elif window==1:
        z = np.hanning(frame)
    elif window==2:
        z = np.hamming(frame)
    elif window==3:
        z = np.ones(frame)
    
    return enframe(y, frame, inc, z)

def audio2vector(audio_file, sr=None, frame=100, frameshift=30, window=1):
    y, sr = librosa.load(audio_file, sr=8000)
    # 小于某个阈值的直接置0
    y = np.where(np.abs(y) / np.max(y) < 0.1, 0, y)
    inc = frame - frameshift
    
    if window==0:
        z = 1
    elif window==1:
        z = np.hanning(frame)
    elif window==2:
        z = np.hamming(frame)
    elif window==3:
        z = np.ones(frame)
    
    x = enframe(y, frame, inc, z)
    amplitude, energy, rate = np.zeros(x.shape[0]), np.zeros(x.shape[0]), np.zeros(x.shape[0])
    
    for i in range(x.shape[0]):
        amplitude[i] = np.sum(abs(x[i]))
        energy[i] =  np.sum(x[i]**2)
        rate[i] = zero_rate(x[i])
    
    #幅度，能量归一化
    amplitude = amplitude / np.max(amplitude)
    energy = energy / np.max(energy)
    
    return amplitude, energy, rate

def thres( amplitude, energy, rate, thres_energy_high = .5, thres_energy_low = .3, thres_rate = 10):
    if thres_energy_low > thres_energy_high:
        print('thres_energy_low must be lower than thres_energy_high')
        return
    
    n = [0 for i in range(6)]
    
    #能量高阈值搜索
    flag = True
    n[0] = len(energy) - 1
    for i in range(len(energy)):
        if energy[i] > thres_energy_high and flag:
            n[0] = i
            flag = False

    flag = True
    n[1] = 0
    for i in np.arange(len(energy) - 1, 0, -1):
        if energy[i] > thres_energy_high and flag:
            n[1] = i
            flag = False
    
    #能量低阈值搜索
    flag = True
    for i in np.arange(n[0], 0, -1):
        if energy[i] < thres_energy_low and flag:
            n[2] = i
            flag = False
        
    flag = True
    n[3] = len(energy) - 1
    for i in np.arange(n[1], len(energy) - 1):
        if energy[i] < thres_energy_low and flag:
            n[3] = i
            flag = False
    
    #过零率阈值搜索
    flag = True
    for i in np.arange(n[2], 0, -1):
        if rate[i] < thres_rate and flag:
            n[4] = i
            flag = False
        
    flag = True
    n[5] = len(energy) - 1
    for i in np.arange(n[3], len(energy) - 1):
        if rate[i] < thres_rate and flag:
            n[5] = i
            flag = False
            
    amplitude_c = np.zeros(len(energy))
    energy_c = np.zeros(len(energy))
    rate_c = np.zeros(len(energy))
    
    amplitude_c[0:n[5] - n[4]] = amplitude[n[4]:n[5]]
    energy_c[0:n[5] - n[4]] = energy[n[4]:n[5]]
    rate_c[0:n[5] - n[4]] = rate[n[4]:n[5]]
    
    return n[4], n[5], amplitude_c, energy_c, rate_c

def get_vector(path):
    amplitude, energy, rate = audio_to_vector(path)
    _, _, amplitude_c, energy_c, rate_c = thres(amplitude, energy, rate)
    return amplitude_c, energy_c, rate_c