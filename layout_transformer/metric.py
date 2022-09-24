# modified from https://github.com/ktrk115/const_layout/metric.py
import torch
from torch import Tensor
from einops import rearrange

from typing import Tuple

# cp from https://github.com/rwightman/pytorch-image-models/timm/utils/metrics.py
class AverageMeter:
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def seqs2bboxes(seqs:Tensor,eos:int)->Tuple[Tensor,Tensor]:
    """
    seqs -> bboxs, bbox_nums
    seqs:(B,TL)
    bboxs: (B,N,4)
    bbox_nums:(B,)
    >>> seqs = torch.tensor([
    ... [100,1,1,1,1,1,101,101,0,0,0,0],
    ... [100,1,1,1,1,1,2,2,2,2,2,101],
    ... ])
    >>> eos = 101
    >>> bboxs,bbox_nums = seqs2bboxes(seqs,eos)
    >>> bboxs
    tensor([[[  1,   1,   1,   1],
             [101,   0,   0,   0]],
    <BLANKLINE>
            [[  1,   1,   1,   1],
             [  2,   2,   2,   2]]])
    >>> bbox_nums
    tensor([[1],
            [2]])
    """
    B,TL = seqs.shape
    seqs = seqs.clone()[:,1:] # rm <bos>
    seq_lengths = ( (seqs==eos).int() * reversed(torch.arange(1,TL,device=seqs.device))[None,:] ).argmax(-1,keepdim=True)

    if torch.any(seq_lengths % 5 != 0):
        import warnings
        warnings.warn("some seq_len % 5 is not 0!",RuntimeWarning)
    bbox_nums = seq_lengths // 5 

    N = torch.max(bbox_nums).item()
    bboxes = rearrange(seqs[:,:N * 5],'b (n c) -> b n c',c=5)[...,1:]
    return bboxes,bbox_nums


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
        # note div zero condition
        non_zero_mask = diag>0
        res +=  ( (eff_areas.sum(dim=-1)[non_zero_mask] / diag[non_zero_mask] ).mean().item() - 1 )

    return res / B

if __name__ == "__main__":
    import doctest
    doctest.testmod()