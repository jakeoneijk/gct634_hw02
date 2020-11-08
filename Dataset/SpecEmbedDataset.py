from torch.utils.data import Dataset
from .SpecDataset import SpecDataset
from .EmbedDataset import EmbedDataset

class SpecEmbedDataset(Dataset):
    def __init__(self,paths,genre_dict:dict, mean:int = 0, std:int = 1, time_dim_size = None):
        self.spec_dset = SpecDataset(paths,genre_dict,mean,std,time_dim_size)
        embed_path = [path.replace('/spec/','/embed/') for path in paths]
        self.embd_dset = EmbedDataset(embed_path,genre_dict)

    def __getitem__(self,i):
        spec,label_s = self.spec_dset[i]
        embed,label_e = self.embd_dset[i]
        assert(label_e == label_s),"something wrong with spec embed dataset"
        return {"spec":spec,"embed":embed}, label_s
    
    def __len__(self):
        return len(self.spec_dset)
