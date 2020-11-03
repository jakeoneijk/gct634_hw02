from PreProcessing import Preprocessing
from Dataset.SpecDataset import SpecDataset
from Dataset.EmbedDataset import EmbedDataset
from Dataset.ChunkDataset import ChunkDataset
from DataloaderBuilderChunk import DataloaderBuilderChunk
from Model.HW2Model import HW2Model
from Model.HW2Q1Model import HW2Q1Model
from Model.HW2Q2Model import HW2Q2Model
from Trainer import Trainer
from enum import Enum,unique
import torch
from torch.utils.data import DataLoader
import argparse
import json
import librosa
import os

@unique 
class AppMode(Enum):
    PREPROCESSING = -1
    TRAIN = 0

class AppController():
    def __init__(self,app_config):
        self.system_check()
        self.app_mode = app_config['app_mode']
        self.question_num = app_config['question_num']
        self.data_path = app_config["data_path"]
        self.batch_size = app_config["batch_size"]
        self.device = torch.device('cuda:3') if torch.cuda.is_available() else torch.device('cpu')
        self.preprocessor = Preprocessing(self.data_path)

    def system_check(self):
        if not torch.cuda.is_available():
            raise SystemError('Gpu device not found!')
        print(f'Found GPU at {torch.cuda.get_device_name()}')
        print(f'PyTorch version: {torch.__version__}')
        print(f'Librosa version: {librosa.__version__}')

    def run(self):
        print("run this app")
        if self.app_mode == AppMode.PREPROCESSING.value:
            self.preprocessor.preprocess()
            self.preprocessor.preprocess(musiccnn=True)
            self.preprocessor.preprocess_chunk()
        elif self.app_mode == AppMode.TRAIN.value and (self.question_num==0 or self.question_num==1):
            train_data_path,valid_data_path,test_data_path = self.preprocessor.get_path_list()

            dataset_train = SpecDataset(train_data_path,genre_dict=self.preprocessor.genres_dict)
            dataset_train.normalize_data(dataset_train)
            dataset_valid = SpecDataset(valid_data_path,genre_dict=self.preprocessor.genres_dict,
                                        mean=dataset_train.mean,std=dataset_train.std,
                                       time_dim_size=dataset_train.time_dim_size)
            dataset_test = SpecDataset(test_data_path,genre_dict=self.preprocessor.genres_dict,
                                        mean=dataset_train.mean,std=dataset_train.std,
                                       time_dim_size=dataset_train.time_dim_size)
            
            loader_train,loader_valid,loader_test = self.make_dataloader(dataset_train,dataset_valid,dataset_test)
            
            self.train(model=HW2Model(self.preprocessor.number_mel,len(self.preprocessor.genres)),valid_dataloader=loader_valid,train_dataloader=loader_train,test_dataloader=loader_test)

            if self.question_num == 1:
                self.train(model=HW2Q1Model(self.preprocessor.number_mel,len(self.preprocessor.genres)), train_dataloader=loader_train,valid_dataloader=loader_valid,test_dataloader=loader_test)
            print("train end")

        elif self.app_mode == AppMode.TRAIN.value and (self.question_num==2):
            train_data_path,valid_data_path,test_data_path = self.preprocessor.get_path_list(musiccnn=True)
            dataset_train = EmbedDataset(train_data_path,self.preprocessor.genres_dict)
            dataset_valid = EmbedDataset(valid_data_path,self.preprocessor.genres_dict)
            dataset_test = EmbedDataset(test_data_path,self.preprocessor.genres_dict)
            loader_train,loader_valid,loader_test = self.make_dataloader(dataset_train,dataset_valid,dataset_test)
            embed_size = dataset_train[0][0].shape[0]
            self.train(model=HW2Q2Model(feature_size=embed_size, num_genres=len(self.preprocessor.genres)), train_dataloader=loader_train,valid_dataloader=loader_valid,test_dataloader=loader_test)
            print("train end")

        elif self.app_mode == AppMode.TRAIN.value and (self.question_num==3.1):
            chunk_sec = 4
            print(f"use spec chunk {chunk_sec} dataset")
            dataset_train = ChunkDataset(f'{self.data_path}/spec_chunk{chunk_sec}/train')
            dataset_test = ChunkDataset(f'{self.data_path}/spec_chunk{chunk_sec}/test')
            dataloader_builder = DataloaderBuilderChunk(dataset_train,dataset_test,self.batch_size)
            loader_train,loader_valid,loader_test = dataloader_builder.get_data_loader()
            embed_size = dataset_train[0][0]
            embed_size = dataset_train[1][0]
            print("debug")

    def make_dataloader(self,train_dataset,valid_dataset,test_dataset):
        num_workers = os.cpu_count()
        loader_train = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=num_workers, drop_last=True)
        loader_valid = DataLoader(valid_dataset, batch_size=self.batch_size, shuffle=True, num_workers=num_workers, drop_last=True)
        loader_test = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=num_workers, drop_last=False)
        return loader_train,loader_valid,loader_test
        
    def train(self,model,train_dataloader,valid_dataloader,test_dataloader):
        trainer = Trainer(model=model, device=self.device)
        trainer.set_dataloader(train=train_dataloader,valid=valid_dataloader,test=test_dataloader)
        trainer.fit()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='hw2 parser')
    parser.add_argument('-c','--config',type=str,default='./default_config.json', help='configfile')
    args = parser.parse_args()
    config = args.config
    with open(config,'r') as config_file:
        config = json.load(config_file)

    app_controller = AppController(config)
    app_controller.run()