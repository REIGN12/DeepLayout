# modified from https://github.com/ktrk115/const_layout/metric.py
import torch
from torch import Tensor
from einops import rearrange

def compute_overlap(bboxes:Tensor,bbox_nums:Tensor)->float:
    """
    bboxes: (B,N,4) xywh
    bbox_nums:(B,) indicate real length
    >>> bboxes =  torch.tensor([
    ...     [[0,0,2,2],[1,2,1,1],[1,1,1,1]]
    ... ])
    >>> bbox_nums = torch.tensor([3,])
    >>> compute_overlap(bboxes,bbox_nums) # gt is 1.25/3~0.41666..., some float number error
    0.4166666269302368
    >>> bboxes =  torch.tensor([
    ...     [[0,0,2,2],[1,2,1,1],[1,1,1,1]],
    ...     [[0,0,2,2],[1,2,1,1],[1,1,1,1]],
    ... ])
    >>> bbox_nums = torch.tensor([3,2])
    >>> compute_overlap(bboxes,bbox_nums) # gt is 1.25/6~0.20833..., some float number error
    0.2083333134651184
    """
    bboxes = bboxes.clone()
    B,N,_ = bboxes.shape
    # xywh -> x1,y1,x2,y2
    bboxes[...,[2,3]] += bboxes[...,[0,1]]
    # l = max{0,|min(end) - max(front)|}
    front_bound = rearrange(bboxes[...,[0,1]],'b n c -> b c n')
    end_bound = rearrange(bboxes[...,[2,3]],'b n c -> b c n')
    front_max = torch.maximum(front_bound[...,None],front_bound[...,None,:])
    end_min = torch.minimum(end_bound[...,None],end_bound[...,None,:])
    length = end_min-front_max; length[length<=0] = 0
    areas = torch.prod(length,dim=1) # prod xl, yl; (B,N,N)

    res = 0.
    for i,n in enumerate(bbox_nums):
        eff_areas = areas[i][:n,:n] # (n,n)
        diag = eff_areas.diag()
        res +=  ( (eff_areas.sum(dim=-1) / diag ).mean().item() - 1 )

    return res / B

if __name__ == "__main__":
    import doctest
    doctest.testmod()