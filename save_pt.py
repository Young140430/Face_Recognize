from face import FaceNet
from PIL import Image
from dataset import tf
from torch.nn.functional import normalize
import torch,os

path=r"F:\face_data_test1"
net=FaceNet()
net.load_state_dict(torch.load("params/face.pt"))
net.eval()
list1=[]
for i in os.listdir(path):
    pic_list = torch.Tensor([])
    for filename in os.listdir(fr"{path}\{i}"):
        data=tf(Image.open(fr"{path}\{i}\{filename}"))
        feat=net.encode(data[None,...])
        pic_list=torch.cat((pic_list,feat))
        print(f"{i} {filename} done!")
    list1.append(pic_list)
list1=torch.stack(list1)
print(list1)
torch.save(list1, fr"params\all_feature.pt")


