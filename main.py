import argparse

from snow_fall import au2snow

if __name__ == "__main__": 
    parser = argparse.ArgumentParser("落雪特效")
    parser.add_argument("-f", "--file", help = "将.wav文件转化为落雪视频（.avi格式），或将中间文件.awrp或.sfev转化为视频")
    args = parser.parse_args()
    au2snow(args.file)