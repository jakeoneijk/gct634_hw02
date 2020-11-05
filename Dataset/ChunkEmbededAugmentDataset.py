import torch.utils.data.dataset as dataset
import os
import pickle
import numpy as np

class ChunkEmbededAugmentDataset(dataset.Dataset):
    def __init__(self,file_list,genre_dict:dict):
        self.file_list = file_list
        self.genre_dict = genre_dict
    
    def __getitem__(self,i):
        with open(self.file_list[i],'rb') as read_dict:
            feature_genre_dict = pickle.load(read_dict)
        debug = self.file_list[i]
        file_name = (self.file_list[i].split('/')[-1]).split('_')[0]
        if file_name[-1] == ")":
            file_name = file_name.split('(')[0]
        embeded_file_path = './gtzan/embed/'+feature_genre_dict["genre"]+"/"+file_name+".npy"
        embeded = np.load(embeded_file_path)
        return {"spec":feature_genre_dict["feature"],"embed":embeded}, self.genre_dict[feature_genre_dict["genre"]]
    
    def __len__(self):
        return len(self.file_list)