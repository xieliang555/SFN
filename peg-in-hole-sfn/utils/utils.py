import numpy as np
import math


def get_position_gt(o):
    img = o['img']
    gt_mask = o['gt']
    dxy = o['dxy']

    # 世界坐标系转图像坐标系
    # !!!!!!!!!!!!!
    dx = np.clip(np.around(-dxy[:,0]*1000)+10, 0, 20)
    dy = np.clip(np.around(dxy[:,1]*1000)+10, 0, 20)
    # dx = np.clip((np.around(-dxy[:,0]*1000)+5)*24, 0, 250)
    # dy = np.clip((np.around(dxy[:,1]*1000)+5)*19, 0, 200)

    # print('dxy', dxy, 'dx', dx, 'dy', dy)

    position_gts = []
    for i in range(img.shape[0]):
        # 用numpy array表示图像时，行是y轴，列是x轴
        # !!!!!!!!!!!!!
        position_gt = np.zeros((21,21), dtype=np.float32)
        # position_gt = np.zeros((200,250), dtype=np.float32)
        position_gt[int(dy[i])][int(dx[i])] = 1.0
        position_gts.append(position_gt)
    return np.array(position_gts)



# def get_position_gt(o):
#         img = o['img']
#         gt_mask = o['gt']
#         peg_xy = o['peg_xy']
#         hole_xy = o['hole_xy']

#         # get peg_gt and hole_gt
#         peg_gt = np.zeros_like(gt_mask, dtype=np.float32)
#         sigma = 3
#         n,h,w = peg_gt.shape
#         for idx in range(n):
#             x_min = int(max(0, peg_xy[idx][0]-sigma))
#             x_max = int(min(w-1, peg_xy[idx][0]+sigma))
#             y_min = int(max(0, peg_xy[idx][1]-sigma))
#             y_max = int(min(h-1, peg_xy[idx][1]+sigma))
#             for i in range(x_min,x_max):
#                 for j in range(y_min,y_max):
#                     dx = peg_xy[idx][0]-i
#                     dy = peg_xy[idx][1]-j
#                     v = math.exp(-(pow(dx,2)+pow(dy,2))/(2*pow(sigma,2)))
#                     if v < 0.7:
#                         continue
#                     peg_gt[idx][j][i] = v

#         hole_gt = np.zeros_like(gt_mask, dtype=np.float32)
#         sigma = 3
#         n,h,w = hole_gt.shape
#         for idx in range(n):
#             x_min = int(max(0, hole_xy[idx][0]-sigma))
#             x_max = int(min(w-1, hole_xy[idx][0]+sigma))
#             y_min = int(max(0, hole_xy[idx][1]-sigma))
#             y_max = int(min(h-1, hole_xy[idx][1]+sigma))
#             for i in range(x_min,x_max):
#                 for j in range(y_min,y_max):
#                     dx = hole_xy[idx][0]-i
#                     dy = hole_xy[idx][1]-j
#                     v = math.exp(-(pow(dx,2)+pow(dy,2))/(2*pow(sigma,2)))
#                     if v < 0.7:
#                         continue
#                     hole_gt[idx][j][i] = v

#         return peg_gt, hole_gt



def _fast_hist(label_true, label_pred, n_class):
    mask = (label_true >= 0) & (label_true < n_class)
    hist = np.bincount(
        n_class * label_true[mask].astype(int) +
        label_pred[mask], minlength=n_class ** 2).reshape(n_class, n_class)
    return hist


def label_accuracy_score(label_trues, label_preds, n_class):
    """Returns accuracy score evaluation result as defined in FCN

      - overall accuracy
      - mean accuracy
      - mean IU
      - fwavacc
    """
    hist = np.zeros((n_class, n_class))
    for lt, lp in zip(label_trues, label_preds):
        hist += _fast_hist(lt.flatten(), lp.flatten(), n_class)
    acc = np.diag(hist).sum() / hist.sum()
    with np.errstate(divide='ignore', invalid='ignore'):
        acc_cls = np.diag(hist) / hist.sum(axis=1)
    acc_cls = np.nanmean(acc_cls)
    with np.errstate(divide='ignore', invalid='ignore'):
        iu = np.diag(hist) / (
            hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist)
        )
    mean_iu = np.nanmean(iu)
    freq = hist.sum(axis=1) / hist.sum()
    fwavacc = (freq[freq > 0] * iu[freq > 0]).sum()
    return acc, acc_cls, mean_iu, fwavacc





class Buffer(object):
    def __init__(self, buffer_size):
        super(Buffer, self).__init__()
        self.seg_pred_buf = np.zeros(shape=(buffer_size, 200, 250), dtype=np.float32)
        self.seg_gt_buf = np.zeros(shape=(buffer_size, 200, 250), dtype=np.int32)
        self.peg_gt_buf = np.zeros(shape=(buffer_size, 200, 250), dtype=np.float32)
        self.hole_gt_buf = np.zeros(shape=(buffer_size, 200, 250), dtype=np.float32)
        self.position_gt_buf = np.zeros(shape=(buffer_size, 21, 21), dtype=np.float32)
        self.ptr = 0

    def store(self, seg_pred, seg_gt, peg_gt=None, hole_gt=None, position_gt=None):
        self.seg_pred_buf[self.ptr] = seg_pred
        self.seg_gt_buf[self.ptr] = seg_gt
        self.peg_gt_buf[self.ptr] = peg_gt
        self.hole_gt_buf[self.ptr] = hole_gt
        self.position_gt_buf[self.ptr] = position_gt
        self.ptr += 1

    def get(self):
        self.ptr = 0
        data = dict(seg_pred=self.seg_pred_buf, 
                    seg_gt=self.seg_gt_buf,
                    peg_gt=self.peg_gt_buf,
                    hole_gt=self.hole_gt_buf,
                    position_gt=self.position_gt_buf)
        return data



