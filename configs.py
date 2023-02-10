import yaml as y



class snow_config:
    def __init__(self) -> None:
        with open("configs.yaml", "r") as f:
            # self.yaml = y.load(f)
            # 由于YAML 5.1版本后弃用了yaml.load(file)
            # 出现问题：TypeError: load() missing 1 required positional argument: 'Loader'。
            # 使用如下替换
            self.yaml = y.load(f, Loader=y.FullLoader)  
            
        for k, v in self.yaml.items():
            self.__setattr__(k, v)
