# MuseSnow
基于python的落雪视频生成程序，并用于音频可视化

## 效果展示

[![示例视频](./运行截图.jpg)](./僕が持ってるものなら.avi)

## 运行原理

|||
|:--:|:---|
|1. 音频分析|对[`.wav`音频文件](./僕が持ってるものなら.wav)进行分析，对音频分帧后计算每帧能量；同时，通过一组数字带通滤波器分析每一帧在16个不同频带上的能量,将结果保存为[`.awrp`文件](./僕が持ってるものなら.awrp)|
|2. 事件生成|将音频/视频以更高频率分帧，对每一帧计算生成雪花的概率及雪花半径的概率分布，并根据随机函数决定是否生成雪花。将各雪花事件保存为[`.sfev`文件](./僕が持ってるものなら.sfev)|
|3. 视频渲染|根据雪花事件，对视频进行逐帧渲染，生成[`.avi`视频文件](./僕が持ってるものなら.avi)|

## 运行环境

本生成代码的运行需要安装以下依赖库：

- librosa，用于生成音频
- numpy，用于相关矩阵运算
- tqdm，用于进度条显示
- pickle，用于`.awrp`文件与`.sfev`文件生成
- python-OpenCV，用于视频生成

## 运行方法

运行以下命令以生成视频：

```
python main.py [输入文件]
```

其中输入文件支持`.wav`、`.awrp`、`.sfev`。例如：

```
python main.py 僕が持ってるものなら.wav
python main.py 僕が持ってるものなら.awrp
python main.py 僕が持ってるものなら.sfev
```