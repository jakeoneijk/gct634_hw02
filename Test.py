from Model.HW2Model import HW2Model
from Model.HW2Q1Model import HW2Q1Model
from Model.HW2Q31Model import HW2Q31Model
from Model.HW2Q32ResnetModel import HW2Q32ResnetModel
from Model.HW2Q33ResnetPlusEmbedModel import HW2Q33ResnetPlusEmbedModel
from Model.HW2Q3Conv2d import HW2Q3Conv2d
from Model.HW2Q3Conv2dEmbed import HW2Q3Conv2dEmbed


import torch
class TestController():
    def __init__(self):
        print("=====test start=====")
        self.test_batch_size = 5
        self.models = {"HW2Model":HW2Model(num_mels=96,num_genres=8)}
        self.test_input = {"HW2Model_input":torch.randn((self.test_batch_size,96,936))}
        self.models["HW2Q1Model"] = HW2Q1Model(num_mels=96, num_genres=8)
        self.test_input["HW2Q1Model_input"] = torch.randn((self.test_batch_size, 96, 936))
        self.models["HW2Q31Model"] = HW2Q31Model(num_mels=96, num_genres=8)
        self.test_input["HW2Q31Model_input"] = torch.randn((self.test_batch_size, 96, 157))
        self.models["HW2Q32ResnetModel"] = HW2Q32ResnetModel(num_genres=8)
        self.test_input["HW2Q32ResnetModel_input"] = torch.randn((self.test_batch_size, 96, 157))
        self.models["HW2Q3Conv2d"] = HW2Q3Conv2d()
        self.test_input["HW2Q3Conv2d_input"] = torch.randn((self.test_batch_size, 96, 157))

        self.models_use_embed = {"HW2Q33ResnetPlusEmbedModel": HW2Q33ResnetPlusEmbedModel(num_genres=8)}
        self.test_input_use_embed = {"HW2Q33ResnetPlusEmbedModel_input":[torch.randn((self.test_batch_size, 96, 157)),torch.randn((self.test_batch_size, 753))]}
        self.models_use_embed["HW2Q3Conv2dEmbed"] =  HW2Q3Conv2dEmbed()
        self.test_input_use_embed["HW2Q3Conv2dEmbed_input"] = [torch.randn((self.test_batch_size, 96, 157)),torch.randn((self.test_batch_size, 753))]

    def model_size_test(self):
        print("model size test")
        for model_name in self.models:
            print(f'====={model_name} test=====')
            output = self.models[model_name](self.test_input[model_name+"_input"])
            print(output.shape)
            print(output)
            print('\n')

        for model_name in self.models_use_embed:
            print(f'====={model_name} test=====')
            output = self.models_use_embed[model_name](self.test_input_use_embed[model_name+"_input"][0],self.test_input_use_embed[model_name+"_input"][1])
            print(output.shape)
            print(output)
            print('\n')

    def test(self):
        self.model_size_test()

if __name__ == "__main__":
    test_controller = TestController()
    test_controller.test()