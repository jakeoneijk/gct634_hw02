from torch.utils.data import Dataset
import numpy as np

class SpecDataset(Dataset):
    def __init__(self, paths, genre_dict:dict, mean:int = 0, std:int = 1, time_dim_size = None):
        self.paths = paths
        self.genre_dict = genre_dict
        self.mean = mean
        self.std = std
        self.time_dim_size = time_dim_size
    
    def __getitem__(self,i):
        path = self.paths[i]

        genre = path.split('/')[-2]
        label = self.genre_dict[genre]

        spec = np.load(path)
        if self.time_dim_size is not None:
            spec = spec[:,:self.time_dim_size]
        
        spec = (spec - self.mean) / self.std
        return spec, label
    
    def __len__(self):
        return len(self.paths)
    
    def normalize_data(self, dataset:Dataset):
        specs = [spec for spec,_ in dataset]
        time_dims = [spec.shape[1] for spec in specs]
        self.time_dim_size = min(time_dims)

        specs = [spec[:,:self.time_dim_size] for spec in specs]
        specs = np.stack(specs)
        self.mean = specs.mean()
        self.std = specs.std()
        print("finish normalizing")
        

