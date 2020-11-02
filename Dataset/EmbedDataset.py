from torch.utils.data import Dataset
import numpy as np

class EmbedDataset(Dataset):
    def __init__(self, paths,genre_dict:dict):
        self.paths = paths
        self.genre_dict = genre_dict
    
    def __getitem__(self,i):
        path = self.paths[i]
        genre = path.split('/')[-2]
        label = self.genre_dict[genre]
        embeded = np.load(path)
        return embeded, label
    
    def __len__(self):
        return len(self.paths)
