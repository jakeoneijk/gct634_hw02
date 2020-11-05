import random
import math
from torch.utils.data import DataLoader, Dataset, SubsetRandomSampler

class DataloaderBuilderChunk():
    def __init__(self,train_dset:Dataset,valid_dset:Dataset,test_dset:Dataset,batch_size,num_workers=0):
        self.train_dataset = train_dset
        self.valid_dataset = valid_dset
        self.test_dataset = test_dset
        self.batch_size = batch_size
        self.num_workers = num_workers
    
    def get_data_loader(self):
        train_dataloader = DataLoader(self.train_dataset
                                        ,pin_memory=True,drop_last=True,num_workers=self.num_workers,batch_size=self.batch_size)
        valid_dataloader = DataLoader(self.valid_dataset
                                        ,pin_memory=True,drop_last=True,num_workers=self.num_workers,batch_size=self.batch_size)
        test_dataloader = DataLoader(self.test_dataset, batch_size=1, shuffle=False, num_workers=self.num_workers, drop_last=False)
        return train_dataloader,valid_dataloader,test_dataloader
        
