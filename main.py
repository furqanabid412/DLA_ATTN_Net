import warnings
warnings.filterwarnings('ignore')
import yaml
from train import train,validate



if __name__ == '__main__':
    config =  yaml.safe_load(open('configs/config.yaml', 'r'))
    train(cfg=config,is_pretrained=True)
    # validate(cfg=config)