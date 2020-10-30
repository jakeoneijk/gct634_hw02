from PreProcessing import Preprocessing
from SpecDataset import SpecDataset
from enum import Enum,unique
import torch
import argparse
import json
import librosa

@unique 
class AppMode(Enum):
    PREPROCESSING = -1
    TRAIN = 0

class AppController():
    def __init__(self,app_config):
        self.system_check()
        self.app_mode = app_config['app_mode']
        self.data_path = app_config["data_path"]
        self.device = torch.device('cuda:3')
        
    
    def system_check(self):
        if not torch.cuda.is_available():
            raise SystemError('Gpu device not found!')
        print(f'Found GPU at {torch.cuda.get_device_name()}')
        print(f'PyTorch version: {torch.__version__}')
        print(f'Librosa version: {librosa.__version__}')
    
    def run(self):
        print("run this app")
        if self.app_mode == AppMode.PREPROCESSING.value:
            preprocessor = Preprocessing(self.data_path)
            preprocessor.preprocess()
        elif self.app_mode == AppMode.TRAIN.value:
            preprocessor = Preprocessing(self.data_path)
            train_data_path = preprocessor.get_train_path_list()
            dataset = SpecDataset(train_data_path,genre_dict=preprocessor.genres_dict)
            dataset.normalize_data(dataset)
            print("train end")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='hw2 parser')
    parser.add_argument('-c','--config',type=str,default='./default_config.json', help='configfile')
    args = parser.parse_args()
    config = args.config
    with open(config,'r') as config_file:
        config = json.load(config_file)

    app_controller = AppController(config)
    app_controller.run()