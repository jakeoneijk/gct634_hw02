import os
class ChunkTrainValidDivide():
    def __init__(self,data_root,train_name_list,valid_name_list,augmentation=False):
        self.file_list = [os.path.join(data_root,f_name) for f_name in os.listdir(data_root)]
        if augmentation:
            augmentation_root = data_root.split("/")[0] +"/" + data_root.split("/")[1] +"/" + data_root.split("/")[2] +"/augmentaion"
            augmentation_list = [os.path.join(augmentation_root,f_name) for f_name in os.listdir(augmentation_root)]
            self.file_list = self.file_list + augmentation_list
        self.train_name_list = [(train_name.split("/")[-1]).replace(".npy","") for train_name in train_name_list]
        self.valid_name_list = [(valid_name.split("/")[-1]).replace(".npy","") for valid_name in valid_name_list]
        print("debug")
    
    def divide(self):
        train_file_path = []
        valid_file_path = []
        for file in self.file_list:
            if "(" in file:
                file_name = (file.split("/")[-1]).split("(")[0]
            else:
                file_name = (file.split("/")[-1]).split("_")[0]
            if file_name in self.train_name_list:
                train_file_path.append(file)
            elif file_name in self.valid_name_list:
                valid_file_path.append(file)
            else:
                print("something wrong")
                return
        return train_file_path, valid_file_path