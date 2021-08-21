import requests

resp = requests.post("http://localhost:5000/predict",
                     files={"file": open('C:\\Users\\mrthi\\Pictures\\1622195325.jpg','rb')})

print(resp.json())