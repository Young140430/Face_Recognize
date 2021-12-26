# 训练P网络

import nets
import train

if __name__ == '__main__':
    net = nets.PNet()

    trainer = train.Trainer(net, 'E:/param1/pnet.pt', r"F:\celeba_1\12") # 网络、保存参数、训练数据
    trainer.train()                                                    # 调用训练方法
