from PreProcessing import Preprocessing

from Dataset.SpecDataset import SpecDataset
from Dataset.EmbedDataset import EmbedDataset
from Dataset.ChunkDataset import ChunkDataset
from Dataset.ChunkTestDataset import ChunkTestDataset
from Dataset.ChunkEmbededDataset import ChunkEmbededDataset
from Dataset.ChunkEmbededTestDataset import ChunkEmbededTestDataset
from Dataset.ChunkEmbededAugmentDataset import ChunkEmbededAugmentDataset
from Dataset.ChunkTrainValidDivide import ChunkTrainValidDivide

from DataloaderBuilderChunk import DataloaderBuilderChunk
from Model.HW2Model import HW2Model
from Model.HW2Q1Model import HW2Q1Model
from Model.HW2Q2Model import HW2Q2Model
from Model.HW2Q31Model import HW2Q31Model
from Model.HW2Q32ResnetModel import HW2Q32ResnetModel
from Model.HW2Q33ResnetPlusEmbedModel import HW2Q33ResnetPlusEmbedModel

from Trainer import Trainer
from enum import Enum,unique
import torch
from torch.utils.data import DataLoader
import argparse
import json
import librosa
import numpy as np
import os

@unique 
class AppMode(Enum):
    PREPROCESSING = -1
    TRAIN = 0
    TEST = 1

class AppController():
    def __init__(self,app_config):
        self.system_check()
        self.app_mode = app_config['app_mode']
        self.question_num = app_config['question_num']
        self.data_path = app_config["data_path"]
        self.batch_size = app_config["batch_size"]
        self.device = torch.device('cuda:3') if torch.cuda.is_available() else torch.device('cpu')
        self.preprocessor = Preprocessing(self.data_path)
        self.dataloaders = {"spec":None,"embed":None,"chunk":None,"chunk_augment":None,"chunk_embed":None,"chunk_embed_augment":None} #train , valid, test
        

    def system_check(self):
        if not torch.cuda.is_available():
            raise SystemError('Gpu device not found!')
        print(f'Found GPU at {torch.cuda.get_device_name()}')
        print(f'PyTorch version: {torch.__version__}')
        print(f'Librosa version: {librosa.__version__}')

    def run(self):
        print("run this app")
        if self.app_mode != AppMode.PREPROCESSING.value:
            print("make dataloader")
            self.make_all_dataloader()

        if self.app_mode == AppMode.PREPROCESSING.value:
            self.preprocessor.preprocess()
            self.preprocessor.preprocess(musiccnn=True)
            self.preprocessor.preprocess_chunk()

        elif self.app_mode == AppMode.TRAIN.value and (self.question_num==0):
            print("base line")
            self.train(model=HW2Model(self.preprocessor.number_mel,len(self.preprocessor.genres)),train_dataloader=self.dataloaders["spec"]["train"],valid_dataloader=self.dataloaders["spec"]["valid"],test_dataloader=self.dataloaders["spec"]["test"])
            print("train end")

        elif self.app_mode == AppMode.TRAIN.value and (self.question_num==1):
            print("Q1")
            self.train(model=HW2Q1Model(self.preprocessor.number_mel,len(self.preprocessor.genres)),train_dataloader=self.dataloaders["spec"]["train"],valid_dataloader=self.dataloaders["spec"]["valid"],test_dataloader=self.dataloaders["spec"]["test"])
            print("train end")

        elif self.app_mode == AppMode.TRAIN.value and (self.question_num==2):
            print("Q2")
            self.train(model=HW2Q2Model(feature_size=753, num_genres=len(self.preprocessor.genres)),train_dataloader=self.dataloaders["embed"]["train"],valid_dataloader=self.dataloaders["embed"]["valid"],test_dataloader=self.dataloaders["embed"]["test"])
            print("train end")

        elif self.app_mode == AppMode.TRAIN.value and (self.question_num==3.1): 
            print("use chunk")           
            self.train(model=HW2Q31Model(self.preprocessor.number_mel,len(self.preprocessor.genres)), train_dataloader=self.dataloaders["chunk"]["train"],valid_dataloader=self.dataloaders["chunk"]["valid"],test_dataloader=self.dataloaders["chunk"]["test"],chunk=True)
            print("debug")
        
        elif self.app_mode == AppMode.TRAIN.value and (self.question_num==3.2):
            print("use chunk and resnet")
            test_acc_chunk, confusion_matrix_chunk = self.train(model=HW2Q32ResnetModel(len(self.preprocessor.genres)), train_dataloader=self.dataloaders["chunk"]["train"],valid_dataloader=self.dataloaders["chunk"]["valid"],test_dataloader=self.dataloaders["chunk"]["test"],chunk=True)
            test_acc_chunk_augment, confusion_matrix_chunk_augment = self.train(model=HW2Q32ResnetModel(len(self.preprocessor.genres)), train_dataloader=self.dataloaders["chunk_augment"]["train"],valid_dataloader=self.dataloaders["chunk_augment"]["valid"],test_dataloader=self.dataloaders["chunk_augment"]["test"],chunk=True)
            self.result_print(model_name="Resnet",test_acc=test_acc_chunk,confusinon_matrix=confusion_matrix_chunk,dataset="chunk")
            self.result_print(model_name="Resnet",test_acc=test_acc_chunk_augment,confusinon_matrix=confusion_matrix_chunk_augment,dataset="chunk & augment")
        
        elif self.app_mode == AppMode.TRAIN.value and (self.question_num==3.3):
            print("chunk +pretrain")
            test_acc_chunk, confusion_matrix_chunk =self.train(model=HW2Q33ResnetPlusEmbedModel(len(self.preprocessor.genres)), train_dataloader=self.dataloaders["chunk_embed"]["train"],valid_dataloader=self.dataloaders["chunk_embed"]["valid"],test_dataloader=self.dataloaders["chunk_embed"]["test"],chunk=True)
            test_acc_chunk_augment, confusion_matrix_chunk_augment =self.train(model=HW2Q33ResnetPlusEmbedModel(len(self.preprocessor.genres)), train_dataloader=self.dataloaders["chunk_embed_augment"]["train"],valid_dataloader=self.dataloaders["chunk_embed_augment"]["valid"],test_dataloader=self.dataloaders["chunk_embed_augment"]["test"],chunk=True)
            self.result_print(model_name="Resnet+pretrain",test_acc=test_acc_chunk,confusinon_matrix=confusion_matrix_chunk,dataset="chunk")
            self.result_print(model_name="Resnet+pretrain",test_acc=test_acc_chunk_augment,confusinon_matrix=confusion_matrix_chunk_augment,dataset="chunk & augment")
               
        elif self.app_mode == AppMode.TEST.value:
            trainer = Trainer(model=HW2Q33ResnetPlusEmbedModel(len(self.preprocessor.genres)), device=self.device,num_input=2)
            trainer.only_test("./best_models/HW2Q33ResnetPlusEmbedModel_augmentation.pth",self.dataloaders["chunk_embed"]["test"] ,chunk=True,message="HW2Q33ResnetPlusEmbedModel, use data augmentation, use chunk")
            
    def make_all_dataloader(self):
        #{"spec":None,"embed":None,"chunk":None,"chunk_augment":None,"chunk_embed":None,"chunk_embed_augment":None}
        train_data_path,valid_data_path,test_data_path = self.preprocessor.get_path_list()
        print("spec dataloader make")
        spec_dataset_train = SpecDataset(train_data_path,genre_dict=self.preprocessor.genres_dict)
        spec_dataset_train.normalize_data(spec_dataset_train)
        spec_dataset_valid = SpecDataset(valid_data_path,genre_dict=self.preprocessor.genres_dict,
                                        mean=spec_dataset_train.mean,std=spec_dataset_train.std,
                                       time_dim_size=spec_dataset_train.time_dim_size)
        spec_dataset_test = SpecDataset(test_data_path,genre_dict=self.preprocessor.genres_dict,
                                        mean=spec_dataset_train.mean,std=spec_dataset_train.std,
                                       time_dim_size=spec_dataset_train.time_dim_size)
        spec_loader_train,spec_loader_valid,spec_loader_test = self.make_dataloader(spec_dataset_train,spec_dataset_valid,spec_dataset_test)
        self.dataloaders["spec"] = {"train":spec_loader_train,"valid":spec_loader_valid,"test":spec_loader_test}
        
        print("embed dataloader make")
        embed_dataset_train = EmbedDataset(train_data_path,self.preprocessor.genres_dict)
        embed_dataset_valid = EmbedDataset(valid_data_path,self.preprocessor.genres_dict)
        embed_dataset_test = EmbedDataset(test_data_path,self.preprocessor.genres_dict)
        embed_loader_train,embed_loader_valid,embed_loader_test = self.make_dataloader(embed_dataset_train,embed_dataset_valid,embed_dataset_test)
        self.dataloaders["embed"] = {"train":embed_loader_train,"valid":embed_loader_valid,"test":embed_loader_test}

        print("chunk dataloader make")
        chunk_train_valid_divide= ChunkTrainValidDivide(f'{self.data_path}/spec_chunk{self.preprocessor.chunk_sec}/train',train_data_path,valid_data_path,augmentation=False)
        chunk_train_path, chunk_valid_path = chunk_train_valid_divide.divide()

        chunk_dataset_train = ChunkDataset(chunk_train_path,self.preprocessor.genres_dict)
        chunk_dataset_valid = ChunkDataset(chunk_valid_path,self.preprocessor.genres_dict)
        chunk_dataset_test = ChunkTestDataset(f'{self.data_path}/spec_chunk{self.preprocessor.chunk_sec}/test',self.preprocessor.genres_dict)
        dataloader_builder = DataloaderBuilderChunk(chunk_dataset_train,chunk_dataset_valid,chunk_dataset_test,self.batch_size)
        chunk_loader_train,chunk_loader_valid,chunk_loader_test = dataloader_builder.get_data_loader()
        self.dataloaders["chunk"] = {"train":chunk_loader_train,"valid":chunk_loader_valid,"test":chunk_loader_test}

        print("chunk_augment dataloader make")
        chunk_train_valid_divide= ChunkTrainValidDivide(f'{self.data_path}/spec_chunk{self.preprocessor.chunk_sec}/train',train_data_path,valid_data_path,augmentation=True)
        chunk_augment_train_path, chunk_augment_valid_path = chunk_train_valid_divide.divide()
        
        chunk_augment_dataset_train = ChunkDataset(chunk_augment_train_path,self.preprocessor.genres_dict)
        chunk_augment_dataset_valid = ChunkDataset(chunk_augment_valid_path,self.preprocessor.genres_dict)
        dataloader_builder = DataloaderBuilderChunk(chunk_augment_dataset_train,chunk_augment_dataset_valid,chunk_dataset_test,self.batch_size)
        chunk_augment_loader_train,chunk_augment_loader_valid,chunk_augment_loader_test = dataloader_builder.get_data_loader()
        self.dataloaders["chunk_augment"] = {"train":chunk_augment_loader_train,"valid":chunk_augment_loader_valid,"test":chunk_augment_loader_test}

        print("chunk_embed dataloader make")
        chunk_embed_dataset_train = ChunkEmbededDataset(chunk_train_path,self.preprocessor.genres_dict)
        chunk_embed_dataset_valid= ChunkEmbededDataset(chunk_valid_path,self.preprocessor.genres_dict)
        chunk_embed_dataset_test = ChunkEmbededTestDataset(f'{self.data_path}/spec_chunk{self.preprocessor.chunk_sec}/test',self.preprocessor.genres_dict)
        dataloader_builder = DataloaderBuilderChunk(chunk_embed_dataset_train,chunk_embed_dataset_valid,chunk_embed_dataset_test,self.batch_size)
        chunk_embed_loader_train,chunk_embed_loader_valid,chunk_embed_loader_test = dataloader_builder.get_data_loader()
        self.dataloaders["chunk_embed"] = {"train":chunk_embed_loader_train,"valid":chunk_embed_loader_valid,"test":chunk_embed_loader_test}

        print("chunk_embed_augment dataloader make")
        chunk_embed_augment_dataset_train = ChunkEmbededAugmentDataset(chunk_augment_train_path,self.preprocessor.genres_dict)
        chunk_embed_augment_dataset_valid = ChunkEmbededAugmentDataset(chunk_augment_valid_path,self.preprocessor.genres_dict)
        dataloader_builder = DataloaderBuilderChunk(chunk_embed_augment_dataset_train,chunk_embed_augment_dataset_valid,chunk_embed_dataset_test,self.batch_size)
        chunk_embed_augment_loader_train,chunk_embed_augment_loader_valid,chunk_embed_augment_loader_test = dataloader_builder.get_data_loader()
        self.dataloaders["chunk_embed_augment"] = {"train":chunk_embed_augment_loader_train,"valid":chunk_embed_augment_loader_valid,"test":chunk_embed_augment_loader_test}
            


    def make_dataloader(self,train_dataset,valid_dataset,test_dataset):
        num_workers = os.cpu_count()
        loader_train = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=num_workers, drop_last=True)
        loader_valid = DataLoader(valid_dataset, batch_size=self.batch_size, shuffle=True, num_workers=num_workers, drop_last=True)
        loader_test = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=num_workers, drop_last=False)
        return loader_train,loader_valid,loader_test
        
    def train(self,model,train_dataloader,valid_dataloader,test_dataloader,num_input = 1,chunk=False):
        trainer = Trainer(model=model, device=self.device,num_input=num_input)
        trainer.set_dataloader(train=train_dataloader,valid=valid_dataloader,test=test_dataloader)
        return trainer.fit(chunk=chunk)
    
    def result_print(self,model_name,test_acc,confusinon_matrix,dataset:str):
        print("\n========================================================\n")
        print(f"Model Name: {model_name}")
        print(f"Dataset: {dataset}")
        print(f"Test acc: {test_acc}")
        print("Confusion Matrix")
        print(confusinon_matrix)
        print("\n========================================================\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='hw2 parser')
    parser.add_argument('-c','--config',type=str,default='./default_config.json', help='configfile')
    args = parser.parse_args()
    config = args.config
    with open(config,'r') as config_file:
        config = json.load(config_file)

    app_controller = AppController(config)
    app_controller.run()