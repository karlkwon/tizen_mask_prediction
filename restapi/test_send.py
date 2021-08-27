import requests


#fname = 'E:\\DATA\\[promakers] mask\\archive\\images\\maksssksksss6.png'
#fname = 'E:\\DATA\\[promakers] mask\\archive\\images\\maksssksksss48.png'
fname = '/workspace/mask/archive/images/maksssksksss48.png'

#resp = requests.post("http://localhost:5000/predict",
#                     files={"file": open('C:\\Users\\mrthi\\Pictures\\1622195325.jpg','rb')})
resp = requests.post("http://localhost:5000/predict",
                     files={"file": open(fname,'rb')})

print(resp.json())