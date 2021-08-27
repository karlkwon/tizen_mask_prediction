import io

import torchvision.transforms as transforms
from PIL import Image

from torchvision import models
import json

import kaggle_mask

from flask import Flask, jsonify, request
app = Flask(__name__)


imagenet_class_index = json.load(open('imagenet_class_index.json'))

# model = kaggle_mask.load_model('E:\\DATA\\[promakers] mask\\model.pt')
model = kaggle_mask.load_model('/workspace/mask/tizen_mask_prediction/model.pt')
# model = kaggle_mask.load_model('/workspace/mask/model.pt')

# def transform_image(image_bytes):
#     my_transforms = transforms.Compose([
#         transforms.Resize(255),
#         transforms.CenterCrop(224),
#         transforms.ToTensor(),
#         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
#     ])

#     image = Image.open(io.BytesIO(image_bytes))
#     return my_transforms(image).unsqueeze(0)

# def get_prediction(image_bytes):
#     tensor = transform_image(image_bytes=image_bytes)
#     outputs = model.forward(tensor)
#     _, y_hat = outputs.max(1)
#     predicted_idx = str(y_hat.item())
#     return imagenet_class_index[predicted_idx]

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        file = request.files['file']
        print(file)

        data_loader = kaggle_mask.get_source(file)

        pred2 = None
        for imgs, _ in data_loader:
            pred2 = model(imgs)
            break

        mask_cnt = list(pred2[0]['labels'].size())[0]
        print(pred2)
        print(mask_cnt)

        return jsonify({'score': pred2[0]['scores'].tolist(), 'type':pred2[0]['labels'].tolist()})


if __name__ == '__main__':
    with open('E:\\DATA\\[promakers] mask\\archive\\images\\maksssksksss6.png', 'rb') as f:
        data_loader = kaggle_mask.get_source(f)

        pred2 = None
        for imgs, _ in data_loader:
            pred2 = model(imgs)
            break

        print(pred2)
