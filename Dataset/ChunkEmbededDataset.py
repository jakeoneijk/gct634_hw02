import torch.utils.data.dataset as dataset
import os
import pickle
import numpy as np

class ChunkEmbededDataset(dataset.Dataset):
    def __init__(self,data_root,genre_dict:dict):
        self.file_list = [os.path.join(data_root,f_name) for f_name in os.listdir(data_root)]
        self.genre_dict = genre_dict
    
    def __getitem__(self,i):
        with open(self.file_list[i],'rb') as read_dict:
            feature_genre_dict = pickle.load(read_dict)

        file_name = (self.file_list[i].split('/')[-1]).split('_')[0]
        embeded_file_path = './gtzan/embed/'+feature_genre_dict["genre"]+"/"+file_name+".npy"
        embeded = np.load(embeded_file_path)
        return {"spec":feature_genre_dict["feature"],"embed":embeded}, self.genre_dict[feature_genre_dict["genre"]]
    
    def __len__(self):
        return len(self.file_list)