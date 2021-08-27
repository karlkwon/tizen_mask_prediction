import torch
import torchvision
from torchvision import transforms, datasets, models
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from PIL import Image


# device = torch.device('cpu')
device = torch.device('cuda')

class MaskDataset(object):
    def __init__(self, transforms, file):
        self.transforms = transforms
        self.file = file

    def __getitem__(self, idx):
        img = Image.open(self.file).convert("RGB")
        
        if self.transforms is not None:
            img = self.transforms(img)

        img = img.to(device)

        return img, None

    def __len__(self):
        return 1


data_transform = transforms.Compose([
        transforms.ToTensor(), 
    ])

def collate_fn(batch):
    return tuple(zip(*batch))

def get_source(img):
    dataset = MaskDataset(data_transform, img)
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=4, collate_fn=collate_fn)

    return data_loader

def get_model_instance_segmentation(num_classes):
    # load an instance segmentation model pre-trained pre-trained on COCO
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    # get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    return model

# model_path: E:\\DATA\\[promakers] mask\\model.pt
def load_model(model_path):
    model2 = get_model_instance_segmentation(4)
    
    model2.load_state_dict(torch.load(model_path, map_location=device))
    model2.eval()
    model2.to(device)

    return model2


if __name__=='__main__':
    model2 = get_model_instance_segmentation(4)
    
    model2.load_state_dict(torch.load('E:\\DATA\\[promakers] mask\\model.pt', map_location=device))
    model2.eval()
    model2.to(device)

    print(model2)
