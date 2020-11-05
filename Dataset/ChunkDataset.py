import torch.utils.data.dataset as dataset
import os
import pickle

class ChunkDataset(dataset.Dataset):
    def __init__(self,file_list,genre_dict:dict):
        self.file_list = file_list
        self.genre_dict = genre_dict
    
    def __getitem__(self,i):
        with open(self.file_list[i],'rb') as read_dict:
            feature_genre_dict = pickle.load(read_dict)
        return feature_genre_dict["feature"], self.genre_dict[feature_genre_dict["genre"]]
    
    def __len__(self):
        return len(self.file_list)