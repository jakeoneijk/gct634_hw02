import torch.utils.data.dataset as dataset
import os
import pickle
import numpy as np

class ChunkTestDataset(dataset.Dataset):
    def __init__(self,data_root,genre_dict:dict):
        self.file_list = [os.path.join(data_root,f_name) for f_name in os.listdir(data_root)]
        self.genre_dict = genre_dict
    
    def __getitem__(self,i):
        with open(self.file_list[i],'rb') as read_dict:
            feature_genre_dict = pickle.load(read_dict)
        features = np.expand_dims(np.array(feature_genre_dict[0]['feature']),axis=0)
        genre = self.genre_dict[feature_genre_dict[0]["genre"]]
        for i in range(1,len(feature_genre_dict)):
            features = np.concatenate((features,np.expand_dims(np.array(feature_genre_dict[i]['feature']),axis=0)),axis=0)
        return features,genre
    
    def __len__(self):
        return len(self.file_list)