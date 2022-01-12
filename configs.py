import yaml as y



class snow_config:
    def __init__(self) -> None:
        with open("configs.yaml", "r") as f:
            self.yaml = y.load(f)
        
        for k, v in self.yaml.items():
            self.__setattr__(k, v)