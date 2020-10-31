from Model.HW2Model import HW2Model
from Model.HW2Q1Model import HW2Q1Model
import torch
class TestController():
    def __init__(self):
        print("=====test start=====")
        self.test_batch_size = 5
        self.models = {"HW2Model":HW2Model(num_mels=96,num_genres=8)}
        self.test_input = {"HW2Model_input":torch.randn((self.test_batch_size,96,936))}
        self.models["HW2Q1Model"] = HW2Q1Model(num_mels=96, num_genres=8)
        self.test_input["HW2Q1Model_input"] = torch.randn((self.test_batch_size, 96, 936))

    def model_size_test(self):
        print("model size test")
        for model_name in self.models:
            print(f'====={model_name} test=====')
            output = self.models[model_name](self.test_input[model_name+"_input"])
            print(output.shape)
            print(output)
            print('\n')

    def test(self):
        self.model_size_test()

if __name__ == "__main__":
    test_controller = TestController()
    test_controller.test()