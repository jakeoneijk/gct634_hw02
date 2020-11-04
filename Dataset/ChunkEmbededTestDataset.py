import torch.utils.data.dataset as dataset
import os
import pickle
import numpy as np

class ChunkEmbededTestDataset(dataset.Dataset):
    def __init__(self,data_root,genre_dict:dict):
        self.file_list = [os.path.join(data_root,f_name) for f_name in os.listdir(data_root)]
        self.genre_dict = genre_dict
    
    def __getitem__(self,i):
        with open(self.file_list[i],'rb') as read_dict:
            feature_genre_dict = pickle.load(read_dict)
        features = np.expand_dims(np.array(feature_genre_dict[0]['feature']),axis=0)
        genre = self.genre_dict[feature_genre_dict[0]["genre"]]

        file_name = (self.file_list[i].split('/')[-1]).split('_')[0]
        embeded_file_path = './gtzan/embed/'+feature_genre_dict[0]["genre"]+"/"+file_name+".npy"
        embeded_features = np.expand_dims(np.load(embeded_file_path),axis=0)

        for i in range(1,len(feature_genre_dict)):
            features = np.concatenate((features,np.expand_dims(np.array(feature_genre_dict[i]['feature']),axis=0)),axis=0)
            embeded_features = np.concatenate((embeded_features,np.expand_dims(np.load(embeded_file_path),axis=0)),axis=0)
        return {"spec":features,"embed":embeded_features},genre
    
    def __len__(self):
        return len(self.file_list)