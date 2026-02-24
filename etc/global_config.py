
import argparse
import yaml
config = {}


def _init():
    """
    初始化
    """
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--f', type=str, default="denonet/etc/config.yaml", help='--config file')

    opt = parser.parse_args()
    config_path = opt.f
    global config
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
        f.close()

def get_global_conf():
    _init()
    try:
        return config
    except:
        print('读取失败\r\n')
        return {}

config = get_global_conf()