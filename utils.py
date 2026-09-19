import torch
from torchvision.ops import box_area
def box_cxcywh_to_xyxy(x):
    x_c, y_c, w, h = x.unbind(-1)
    w = w.clamp(min=0)
    h = h.clamp(min=0)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
         (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=-1)

def box_xyxy_to_cxcywh(x):
    x0, y0, x1, y1 = x.unbind(-1)
    b = [(x0 + x1) / 2, (y0 + y1) / 2,
         (x1 - x0), (y1 - y0)]
    return torch.stack(b, dim=-1)

def box_iou(boxes1, boxes2, eps=1e-7):
    area1 = box_area(boxes1)
    area2 = box_area(boxes2)
    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])
    wh = (rb - lt).clamp(min=0)
    inter = wh[:, :, 0] * wh[:, :, 1]
    union = area1[:, None] + area2 - inter + eps
    iou = inter / union
    return iou, union

def generalized_box_iou(boxes1, boxes2, eps=1e-7):
    """
    Generalized IoU from https://giou.stanford.edu/
    Input:
        boxes1: (N, 4) xyxy
        boxes2: (M, 4) xyxy
    """
    iou, union = box_iou(boxes1, boxes2, eps=eps)

    lt = torch.min(boxes1[:, None, :2], boxes2[:, :2])
    rb = torch.max(boxes1[:, None, 2:], boxes2[:, 2:])

    wh = (rb - lt).clamp(min=0)  # [N,M,2]
    area = wh[:, :, 0] * wh[:, :, 1] + eps

    return iou - (area - union) / area
