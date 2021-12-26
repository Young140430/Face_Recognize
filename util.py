import torch


def iou(box, boxes, isMin=False):
    box_area = (box[2] - box[0]) * (box[3] - box[1])
    area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])

    xx1 = torch.maximum(box[0], boxes[:, 0])
    yy1 = torch.maximum(box[1], boxes[:, 1])
    xx2 = torch.minimum(box[2], boxes[:, 2])
    yy2 = torch.minimum(box[3], boxes[:, 3])

    w = torch.maximum(torch.Tensor([0]).cuda(), xx2 - xx1)
    h = torch.maximum(torch.Tensor([0]).cuda(), yy2 - yy1)

    inv = w * h

    if isMin:
        ovr = torch.true_divide(inv, torch.maximum(box_area, area))
    else:
        ovr = torch.true_divide(inv, (box_area + area - inv))
    return ovr


def nms(boxes, thresh=0.3, isMin=False):
    if boxes.shape[0] == 0:
        return torch.Tensor([])
    _boxes = boxes[(-boxes[:, 4]).argsort()]
    r_boxes = []

    while _boxes.shape[0] > 1:
        a_box = _boxes[0]
        b_boxes = _boxes[1:]

        r_boxes.append(a_box)
        index = torch.where(iou(a_box, b_boxes, isMin) < thresh)

        _boxes = boxes[index]

    if _boxes.shape[0] > 0:
        r_boxes.append(_boxes[0])

    return torch.stack(r_boxes)


def to_square(bbox):
    square_bbox = bbox.clone()
    if bbox.shape[0] == 0:
        return torch.Tensor([])

    w = bbox[:, 2] - bbox[:, 0]
    h = bbox[:, 3] - bbox[:, 1]
    max_side = torch.minimum(w, h)

    square_bbox[:, 0] = bbox[:, 0] + w * 0.5 - max_side * 0.5
    square_bbox[:, 1] = bbox[:, 1] + h * 0.5 - max_side * 0.5
    square_bbox[:, 2] = square_bbox[:, 0] + max_side
    square_bbox[:, 3] = square_bbox[:, 1] + max_side
    return square_bbox

def to_square_max(bbox):
    square_bbox = bbox.clone()
    if bbox.shape[0] == 0:
        return torch.Tensor([])

    w = bbox[:, 2] - bbox[:, 0]
    h = bbox[:, 3] - bbox[:, 1]
    max_side = torch.maximum(w, h)

    square_bbox[:, 0] = bbox[:, 0] + w * 0.5 - max_side * 0.5
    square_bbox[:, 1] = bbox[:, 1] + h * 0.5 - max_side * 0.5
    square_bbox[:, 2] = square_bbox[:, 0] + max_side
    square_bbox[:, 3] = square_bbox[:, 1] + max_side
    return square_bbox

def prewhiten(x):
    mean = torch.mean(x)
    std = torch.std(x)
    std_adj = torch.maximum(std, 1.0 / torch.sqrt(x.size))
    y = torch.multiply(torch.subtract(x, mean), 1 / std_adj)
    return y
