import torch, util, nets, time, os, cv2, math
from torchvision import transforms
from face import FaceNet
from dataset import tf
from PIL import Image
from torch.nn.functional import normalize

# P网络:
p_cls = 0.6 #原为0.6
p_nms = 0.5 #原为0.5
# R网络：
r_cls = 0.6 #原为0.6
r_nms = 0.5 #原为0.5
# O网络：
o_cls = 0.97 #原为0.97
o_nms = 0.7 #原为0.7

name = ["杨霖萱", "钟静", "刘泉", "鲁思炜","候捷","不认识"]
name_eng=["YangLinxuan","ZhongJing","LiuQuan","LuSiwei","HouJie","NONE"]
class Detector:
    def __init__(self, param_pnet=r"E:\param\pnet.pt",
                 param_rnet=r"E:\param\rnet.pt",
                 param_onet=r"E:\param\onet.pt", isCuda=True):
        self.isCuda = isCuda

        self.pnet = nets.PNet()
        self.rnet = nets.RNet()
        self.onet = nets.ONet()

        if self.isCuda:
            self.pnet.cuda()
            self.rnet.cuda()
            self.onet.cuda()

        self.pnet.load_state_dict(torch.load(param_pnet))
        self.rnet.load_state_dict(torch.load(param_rnet))
        self.onet.load_state_dict(torch.load(param_onet))

        # 训练网络中有BN(批归一化时),要调用eval方法
        self.pnet.eval()
        self.rnet.eval()
        self.onet.eval()

        # Compose()类会将transforms列表里面的transform操作进行遍历
        self.image_transform = transforms.Compose([transforms.ToTensor()])

    def box_to_real(self, start_index, _offset, cls, scale, w, h, stride=2, side_len=12):
        _x1 = (start_index[:, 1] * stride) / scale
        _y1 = (start_index[:, 0] * stride) / scale
        _x2 = (start_index[:, 1] * stride + side_len - 1) / scale
        _y2 = (start_index[:, 0] * stride + side_len - 1) / scale

        ow = _x2 - _x1
        oh = _y2 - _y1

        x1 = torch.maximum(torch.Tensor([0]).cuda(), _offset[0] * ow + _x1)
        y1 = torch.maximum(torch.Tensor([0]).cuda(), _offset[1] * oh + _y1)
        x2 = torch.minimum(torch.Tensor([w]).cuda(), _offset[2] * ow + _x2)
        y2 = torch.minimum(torch.Tensor([h]).cuda(), _offset[3] * oh + _y2)

        return torch.stack([x1, y1, x2, y2, cls]).T

    def pnet_detect(self, image):

        img = image
        h, w, c = img.shape
        min_side_len = min(w, h)

        scale = 1
        if self.isCuda:
            boxes = torch.tensor([]).cuda()
        while min_side_len > 12:
            img_data = self.image_transform(img)
            if self.isCuda:
                img_data = img_data.cuda()
            img_data = img_data.unsqueeze(0)  # 训练时nchw,所以测试一张图片时需要升维
            _cls, _offset = self.pnet(img_data)

            cls = _cls[0][0]
            offset = _offset[0]

            idxs = torch.nonzero(torch.gt(cls, p_cls))
            boxes = torch.cat(
                (self.box_to_real(idxs, offset[:, idxs[:, 0], idxs[:, 1]], cls[idxs[:, 0], idxs[:, 1]], scale, w, h),
                 boxes))

            scale *= 0.6
            # scale *= 0.6
            _w = int(w * scale)
            _h = int(h * scale)

            img = cv2.resize(img, (_w, _h))
            min_side_len = min(_w, _h)

        return util.nms(boxes, p_nms)

    def rnet_detect(self, image, pnet_boxes):
        img_dataset = []
        _pnet_boxes = util.to_square(pnet_boxes)
        for _box in _pnet_boxes:
            _x1 = int(_box[0])
            _y1 = int(_box[1])
            _x2 = int(_box[2])
            _y2 = int(_box[3])

            h, w, c = image.shape
            img = image[_y1:_y2, _x1:_x2]
            img = cv2.resize(img, (24, 24), interpolation=cv2.INTER_CUBIC)
            img_data = self.image_transform(img)
            img_dataset.append(img_data)

        img_dataset = torch.stack(img_dataset)
        if self.isCuda:
            img_dataset = img_dataset.cuda()

        cls, offset = self.rnet(img_dataset)

        idxs, _ = torch.where(cls > r_cls)

        _box = _pnet_boxes[idxs]
        _x1 = _box[:, 0]
        _y1 = _box[:, 1]
        _x2 = _box[:, 2]
        _y2 = _box[:, 3]

        ow = _x2 - _x1
        oh = _y2 - _y1

        x1 = torch.maximum(torch.Tensor([0]).cuda(), offset[idxs, 0] * ow + _x1)
        y1 = torch.maximum(torch.Tensor([0]).cuda(), offset[idxs, 1] * oh + _y1)
        x2 = torch.minimum(torch.Tensor([w]).cuda(), offset[idxs, 2] * ow + _x2)
        y2 = torch.minimum(torch.Tensor([h]).cuda(), offset[idxs, 3] * oh + _y2)

        boxes = torch.stack([x1, y1, x2, y2, cls[idxs, 0]]).T

        return util.nms(boxes, r_nms)

    def onet_detect(self, image, rnet_boxes):
        img_dataset = []
        h, w, c = image.shape
        _rnet_boxes = util.to_square(rnet_boxes)
        for _box in _rnet_boxes:
            _x1 = int(_box[0])
            _y1 = int(_box[1])
            _x2 = int(_box[2])
            _y2 = int(_box[3])

            img = image[_y1:_y2, _x1:_x2]
            img = cv2.resize(img, (48, 48), interpolation=cv2.INTER_CUBIC)
            img_data = self.image_transform(img)
            img_dataset.append(img_data)

        img_dataset = torch.stack(img_dataset)

        if self.isCuda:
            img_dataset = img_dataset.cuda()

        cls, offset = self.onet(img_dataset)

        idxs, _ = torch.where(cls > o_cls)
        _box = _rnet_boxes[idxs]
        _x1 = _box[:, 0]
        _y1 = _box[:, 1]
        _x2 = _box[:, 2]
        _y2 = _box[:, 3]

        ow = _x2 - _x1
        oh = _y2 - _y1

        x1 = torch.maximum(torch.Tensor([0]).cuda(), offset[idxs, 0] * ow + _x1)
        y1 = torch.maximum(torch.Tensor([0]).cuda(), offset[idxs, 1] * oh + _y1)
        x2 = torch.minimum(torch.Tensor([w]).cuda(), offset[idxs, 2] * ow + _x2)
        y2 = torch.minimum(torch.Tensor([h]).cuda(), offset[idxs, 3] * oh + _y2)

        boxes = torch.stack([x1, y1, x2, y2, cls[idxs, 0]]).T

        return util.nms(boxes, o_nms, isMin=True)

    def detect(self, image):

        # P网络
        start_time = time.time()
        pnet_boxes = self.pnet_detect(image)
        if pnet_boxes.shape[0] == 0:
            return torch.Tensor([])
        end_time = time.time()
        t_pnet = end_time - start_time

        # R网络
        start_time = time.time()
        rnet_boxes = self.rnet_detect(image, pnet_boxes)
        if rnet_boxes.shape[0] == 0:
            return torch.Tensor([])
        end_time = time.time()
        t_rnet = end_time - start_time

        # O网络
        start_time = time.time()
        onet_boxes = self.onet_detect(image, rnet_boxes)
        if onet_boxes.shape[0] == 0:
            return torch.Tensor([])
        end_time = time.time()
        t_onet = end_time - start_time

        # t_sum = t_pnet + t_rnet + t_onet
        # print("total:{0} pnet:{1} rnet:{2} onet:{3}".format(t_sum, t_pnet, t_rnet, t_onet))

        return onet_boxes


if __name__ == '__main__':
    detector = Detector()
    database_path = r"params\all_feature.pt"
    all_feat = torch.permute(torch.load(database_path), (0, 2, 1)).cuda()
    net = FaceNet().cuda()
    net.load_state_dict(torch.load("params/face.pt"))
    net.eval()
    cap = cv2.VideoCapture(r"E:\face_video\face_v\3\4.mp4")
    while True:
        ret, img = cap.read()
        if ret:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.flip(img, 1)
            boxes = detector.detect(img)
            boxes_square = util.to_square_max(boxes)
            # print(len(boxes_square))
            try:
                if len(boxes_square) == 1:
                    for box in boxes_square:
                        x1 = int(box[0])
                        y1 = int(box[1])
                        x2 = int(box[2])
                        y2 = int(box[3])
                        conf = box[4]
                        face = tf(Image.fromarray(img[y1:y2, x1:x2])).cuda()
                        feat = net.encode(face[None, ...])
                        feat_norm = normalize(feat, dim=1)
                        all_feat_norm = normalize(all_feat, dim=1)
                        out = torch.matmul(feat_norm, all_feat_norm)
                        # print(torch.argmax(out))
                        if torch.max(out).cpu().item()>0.99:
                            print(f"{name[torch.nonzero(out == torch.max(out))[0][0].cpu().item()]}:{torch.max(out).cpu().item()}")
                            cv2.putText(img, f"{name_eng[torch.nonzero(out == torch.max(out))[0][0].cpu().item()]}", (x1, y1),
                                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                        # else:
                        #     cv2.putText(img, f"{name_eng[-1]}",(x1, y1),
                        #                 cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
            except:
                print("-")
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            cv2.imshow("img", img)
            cv2.waitKey(1)
        else:
            print("读取视频结束")
            break
    cap.release()
    cv2.destroyAllWindows()