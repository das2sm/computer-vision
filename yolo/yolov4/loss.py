import torch
import torch.nn as nn
import math

# CIoU calculation for bounding box regression
# Considers the overlap, the distance between box centers, and the aspect ratio consistency
def bbox_iou(box1, box2, CIoU=True):
    # box1: [N, 4] (x, y, w, h) - predicted
    # box2: [N, 4] (x, y, w, h) - ground truth
    
    # Convert to x1, y1, x2, y2
    b1_x1, b1_x2 = box1[:, 0] - box1[:, 2] / 2, box1[:, 0] + box1[:, 2] / 2
    b1_y1, b1_y2 = box1[:, 1] - box1[:, 3] / 2, box1[:, 1] + box1[:, 3] / 2
    b2_x1, b2_x2 = box2[:, 0] - box2[:, 2] / 2, box2[:, 0] + box2[:, 2] / 2
    b2_y1, b2_y2 = box2[:, 1] - box2[:, 3] / 2, box2[:, 1] + box2[:, 3] / 2

    # Intersection area
    inter = (torch.min(b1_x2, b2_x2) - torch.max(b1_x1, b2_x1)).clamp(0) * \
            (torch.min(b1_y2, b2_y2) - torch.max(b1_y1, b2_y1)).clamp(0)

    # Union Area
    union = box1[:, 2] * box1[:, 3] + box2[:, 2] * box2[:, 3] - inter + 1e-7
    iou = inter / union

    if CIoU:
        # Smallest enclosing box
        cw = torch.max(b1_x2, b2_x2) - torch.min(b1_x1, b2_x1)
        ch = torch.max(b1_y2, b2_y2) - torch.min(b1_y1, b2_y1)
        c2 = cw**2 + ch**2 + 1e-7
        
        # Distance between centers
        rho2 = (box1[:, 0] - box2[:, 0])**2 + (box1[:, 1] - box2[:, 1])**2
        
        # Aspect ratio term
        v = (4 / math.pi**2) * torch.pow(torch.atan(box2[:, 2] / (box2[:, 3] + 1e-7)) - torch.atan(box1[:, 2] / (box1[:, 3] + 1e-7)), 2)
        with torch.no_grad():
            alpha = v / (1 - iou + v + 1e-7)
        return iou - (rho2 / c2 + v * alpha)

    return iou

class YOLOLoss(nn.Module):
    def __init__(self, num_classes=80):
        super().__init__()
        self.num_classes = num_classes
        self.bce_obj = nn.BCEWithLogitsLoss()
        self.bce_cls = nn.BCEWithLogitsLoss()
        self.stride = [32, 16, 8] 
        
        # YOLOv4 Default Anchors (scaled for 416)
        self.anchors = torch.tensor([
            [[116, 90], [156, 198], [373, 326]],  # Scale 1 (13x13)
            [[30, 61], [62, 45], [59, 119]],      # Scale 2 (26x26)
            [[10, 13], [16, 30], [33, 23]]       # Scale 3 (52x52)
        ]).float()

    def forward(self, predictions, targets):
        device = targets.device
        
        # Reshape predictions from [batch, 3*(5+nc), H, W] to [batch, 3, H, W, 5+nc]
        reshaped_predictions = []
        for pred in predictions:
            batch_size = pred.shape[0]
            # unfolds [batch, 3*(5+num_classes), H, W] -> [batch, 3, 5+num_classes, H, W]
            pred_reshaped = pred.view(batch_size, 3, self.num_classes + 5, pred.shape[2], pred.shape[3])
            # transpose to [batch, 3, H, W, 5+num_classes]
            pred_reshaped = pred_reshaped.permute(0, 1, 3, 4, 2)
            reshaped_predictions.append(pred_reshaped)
        predictions = reshaped_predictions
        
        lcls, lbox, lobj = torch.zeros(1, device=device), torch.zeros(1, device=device), torch.zeros(1, device=device)
        
        # Build targets: matches ground truth to anchors/grids
        tcls, tbox, indices, anchor_list = self.build_targets(predictions, targets)

        for i, pi in enumerate(predictions):
            b, a, gj, gi = indices[i]
            tobj = torch.zeros_like(pi[..., 0], device=device)  # target objectness

            n = b.shape[0]
            if n:
                # Get the prediction corresponding to the target cell/anchor
                # Shape: [n, 5+num_classes] (batch, anchor, grid indices select cell; : selects all channels)
                ps = pi[b, a, gj, gi, :]

                # 1. Regression Loss (CIoU)
                pxy = ps[:, 0:2].sigmoid() * 2. - 0.5
                pwh = (ps[:, 2:4].sigmoid() * 2)**2 * anchor_list[i]
                pbox = torch.cat((pxy, pwh), 1)
                iou = bbox_iou(pbox, tbox[i], CIoU=True)
                lbox += (1.0 - iou).mean()

                # 2. Objectness Target (IoU aware)
                tobj[b, a, gj, gi] = iou.detach().clamp(0).type(tobj.dtype)

                # 3. Classification Loss
                if self.num_classes > 1:
                    t = torch.zeros_like(ps[:, 5:], device=device)
                    t[range(n), tcls[i]] = 1
                    lcls += self.bce_cls(ps[:, 5:], t)

            # 4. Objectness Loss (calculated for all cells, not just those with targets)
            lobj += self.bce_obj(pi[..., 4], tobj)

        loss = lbox * 0.05 + lobj * 1.0 + lcls * 0.5
        return loss * predictions[0].shape[0], torch.cat((lbox, lobj, lcls, loss)).detach()

    def build_targets(self, p, targets):
        na, nt = 3, targets.shape[0]
        tcls, tbox, indices, anch = [], [], [], []
        
        # anchor index assistant
        ai = torch.arange(na, device=targets.device).float().view(na, 1).repeat(1, nt)
        # targets: [anchor_idx, img_idx, cls, x, y, w, h, ai]
        targets = torch.cat((targets.repeat(na, 1, 1), ai[:, :, None]), 2)

        for i in range(3):
            anchors = self.anchors[i].to(targets.device) / self.stride[i]
            gain = torch.ones(7, device=targets.device)
            gain[2:6] = torch.tensor(p[i].shape)[[3, 2, 3, 2]] # grid w, h

            t = targets * gain
            if nt:
                # Match targets to anchors based on width/height ratio
                r = t[:, :, 4:6] / anchors[:, None]
                j = torch.max(r, 1. / r).max(2)[0] < 4.0 
                t = t[j]

            # Only process if we have matched targets after filtering
            if t.shape[0] == 0 or t.numel() == 0:
                # No targets for this scale
                indices.append((torch.tensor([], dtype=torch.long, device=targets.device), 
                               torch.tensor([], dtype=torch.long, device=targets.device),
                               torch.tensor([], dtype=torch.long, device=targets.device),
                               torch.tensor([], dtype=torch.long, device=targets.device)))
                tbox.append(torch.zeros((0, 4), device=targets.device))
                anch.append(torch.zeros((0, 2), device=targets.device))
                tcls.append(torch.tensor([], dtype=torch.long, device=targets.device))
                continue

            # Unpack indices
            # image and class indices are first two columns
            b = t[:, 0].long()  # batch index
            c = t[:, 1].long()  # class index
            gxy = t[:, 2:4]
            gwh = t[:, 4:6]
            # grid xy indices (integer)
            gij = gxy.long()
            # unpack grid indices
            gi = gij[:, 0]
            gj = gij[:, 1]

            # clamp grid indices to valid range (convert bounds to integers)
            max_y = int(gain[3].item() - 1)
            max_x = int(gain[2].item() - 1)
            indices.append((b, t[:, 6].long(), gj.clamp_(0, max_y), gi.clamp_(0, max_x)))
            tbox.append(torch.cat((gxy - gij, gwh), 1))
            anch.append(anchors[t[:, 6].long()])
            tcls.append(c)

        return tcls, tbox, indices, anch