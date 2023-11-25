"""Python file to instantite the model and the transform that goes with it."""
from model import Net, Net2, SketchDNN
from data import data_transforms, data_transforms_transformer, data_transforms_sketchdnn
import torch
import torchvision


class ModelFactory:
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.supported_models = ["basic_cnn", "basic_cnn2", "resnet18", "resnet50", "resnet18empty", "resnet152", "vitB32", "sketchdnn"]
        self.model = self.init_model()
        self.transform = self.init_transform()
    def init_model(self):
        if self.model_name not in self.supported_models:
            raise NotImplementedError("Model not implemented")
        elif self.model_name == "basic_cnn":
            return Net()
        elif self.model_name == "basic_cnn2":
            return Net2()
        elif self.model_name == "resnet18":
            return torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=True)
        elif self.model_name == "resnet50":
            return torch.hub.load('pytorch/vision:v0.10.0', 'resnet50', pretrained=True)
        elif self.model_name == "resnet18empty":
            return torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=False)
        elif self.model_name == "resnet152":
            return torch.hub.load('pytorch/vision:v0.10.0', 'resnet152', pretrained=True)
        elif self.model_name == "vitB32":
            model = torchvision.models.vit_b_32(torchvision.models.ViT_B_32_Weights.DEFAULT)
            return model
        elif self.model_name == "sketchdnn":
            return SketchDNN()
    def init_transform(self):
        if self.model_name not in self.supported_models:
            raise NotImplementedError("Model not implemented")
        elif self.model_name in ["basic_cnn", "basic_cnn2", "resnet18", "resnet18empty", "resnet50", "resnet152"]:
            return data_transforms
        elif self.model_name in ["vitB32"]:
            return data_transforms_transformer
        elif self.model_name in ["sketchdnn"]:
            return data_transforms_sketchdnn


    def get_model(self):
        return self.model

    def get_transform(self):
        return self.transform

    def get_all(self):
        return self.model, self.transform
