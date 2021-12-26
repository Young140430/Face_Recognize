import os

from PIL import Image

from dataset import tf

dict1={}
for filename in os.listdir("F:/face_data_test1"):
    dict1[filename]=tf(Image.open("F:/face_data_test1/"+filename).convert("RGB"))
#print(dict1)
