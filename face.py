import torchvision.models as models
from torch import nn
import torch
from torch.nn import functional as F
from dataset import *
from torch import optim
from torch.utils.data import DataLoader
import torch.jit as jit
from PIL import Image


class Arcsoftmax(nn.Module):
    def __init__(self, feature_num, cls_num):
        super().__init__()
        self.w = nn.Parameter(torch.randn((feature_num, cls_num)))
        self.func = nn.Softmax()

    def forward(self, x, s=1, m=0.2):
        x_norm = F.normalize(x, dim=1)
        w_norm = F.normalize(self.w, dim=0)

        cosa = torch.matmul(x_norm, w_norm) / 10
        a = torch.acos(cosa)

        arcsoftmax = torch.exp(
            s * torch.cos(a + m) * 10) / (torch.sum(torch.exp(s * cosa * 10), dim=1, keepdim=True) - torch.exp(
            s * cosa * 10) + torch.exp(s * torch.cos(a + m) * 10))

        return arcsoftmax


class FaceNet(nn.Module):

    def __init__(self):
        super(FaceNet, self).__init__()
        self.sub_net = nn.Sequential(
            models.densenet121(pretrained=True),

        )
        self.feature_net = nn.Sequential(
            nn.LeakyReLU(0.1),
            nn.Linear(1000, 512, bias=False),
        )
        self.arc_softmax = Arcsoftmax(512, 12)

    def forward(self, x):
        y = self.sub_net(x)
        feature = self.feature_net(y)
        return feature, self.arc_softmax(feature, 1, 1)

    def encode(self, x):
        return self.feature_net(self.sub_net(x))


def compare(face1, face2):
    face1_norm = F.normalize(face1)
    face2_norm = F.normalize(face2)
    print(face1_norm.shape)
    print(face2_norm.shape)
    cosa = torch.matmul(face1_norm, face2_norm.t())
    return cosa


loss_fn = nn.NLLLoss()

if __name__ == '__main__':

    # 训练过程
    net = FaceNet().cuda()

    optimizer = optim.Adam(net.parameters())

    dataset = MyDataset("F:/face_data")
    dataloader = DataLoader(dataset=dataset, batch_size=50, shuffle=True)
    max_loss=11
    for epoch in range(100000):
        for xs, ys in dataloader:
            feature, cls = net(xs.cuda())
            loss = loss_fn(torch.log(cls), ys.cuda())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        # print(torch.argmax(cls, dim=1), ys)
        print(str(epoch)+" loss====> "+str(loss.item()))
        if loss.item()<max_loss:
            max_loss=loss.item()
            torch.save(net.state_dict(), "params/face_best.pt")
            print(str(epoch) + " 最好的参数保存成功")
        if epoch%50==0:
            torch.save(net.state_dict(), "params/face.pt")
            print(str(epoch)+" 参数保存成功")

    # 使用
    '''net = FaceNet().cuda()
    net.load_state_dict(torch.load("params/face.pt"))
    net.eval()

    person1 = tf(Image.open("data/1.jpg").convert("RGB")).cuda()
    person1_feature = net.encode(person1[None, ...])

    person2 = tf(Image.open("data/1.jpg").convert("RGB")).cuda()
    person2_feature = net.encode(person2[None, ...])

    siam = compare(person1_feature, person2_feature)
    print(siam)'''

    # 把模型和参数进行打包，以便C++或PYTHON调用
    # x = torch.Tensor(1, 3, 112, 112)
    # traced_script_module = jit.trace(net, x)
    # traced_script_module.save("model.cpt")
