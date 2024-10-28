import torch

import torchvision.models as models

def load_resnet18_model(model_path):
    # Load the ResNet18 model
    model = models.resnet18()
    whole_dict = torch.load(model_path)
    model.load_state_dict(whole_dict['model'])
    model.eval()
    return model, whole_dict

if __name__ == "__main__":
    model_path = 'run_femnist/fanchuaneval_result.pth.tar'  # Adjust the path as necessary
    model = load_resnet18_model(model_path)
    print("Model loaded successfully.")