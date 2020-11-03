import torch.utils.data.dataset as dataset
import os
import pickle

class ChunkDataset(dataset.Dataset):
    def __init__(self,data_root):
        self.file_list = [os.path.join(data_root,f_name) for f_name in os.listdir(data_root)]
    
    def __getitem__(self,i):
        with open(self.file_list[i],'rb') as read_dict:
            feature_genre_dict = pickle.load(read_dict)
        return feature_genre_dict["feature"], feature_genre_dict["genre"]
    
    def __len__(self):
        return len(self.file_list)