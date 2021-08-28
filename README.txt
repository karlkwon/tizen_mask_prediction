++ mask detection train
https://www.kaggle.com/andrewmvd/face-mask-detection
https://www.kaggle.com/daniel601/pytorch-fasterrcnn
: 마스크 안쓴 얼굴은 인식 못하는 것 같다.


https://flask-restful.readthedocs.io/en/latest/quickstart.html#a-minimal-api

python flask_pytorch_test.py --host 0.0.0.0



https://blog.miguelgrinberg.com/post/handling-file-uploads-with-flask


++ pytorch tutorial (restapi)
https://pytorch.org/tutorials/intermediate/flask_rest_api_tutorial.html
: example project in official pytorch site.

pip install Flask==2.0.1 torchvision==0.10.0

set FLASK_ENV=development
set FLASK_APP=app.py
flask run

[linux]
export FLASK_ENV=development
export FLASK_APP=app_mask.py
flask run --host 0.0.0.0
python test_send.py

[windows]
set FLASK_ENV=development
set FLASK_APP=app_mask.py
flask run --host 0.0.0.0
python test_send.py

