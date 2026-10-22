from .segformer_head import MLP, SegFormerHead
from .ulp_segformer_head import ULPSegFormerHead
from .prada_ulp_segformer_head import PrAdaULPSegFormerHead
from ..builder import HEADS
# Public paper name; preserve module/tensor names for exact checkpoint loading.
HEADS.register_module(name='MARESegHead', module=PrAdaULPSegFormerHead)
__all__ = ['SegFormerHead', 'ULPSegFormerHead', 'PrAdaULPSegFormerHead']
