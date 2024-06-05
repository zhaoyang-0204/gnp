import jax_models
from jax_models.models import swin_transformer
from gnp.models import _register_model
from typing import Iterable, Optional
import flax.linen as nn


class SwinTransformer(swin_transformer.SwinTransformer):

    @nn.compact
    def __call__(self, inputs, train):
        return super().__call__(inputs, not train)
   
# @_register_model("Swin_Tiny")
# def swin_tiny(num_outputs : int):
#     return swin_transformer.swin_tiny_224(
#         num_classes=num_outputs
#     )
        
@_register_model("Swin_Tiny")
def swin_tiny(num_outputs : int):
    return SwinTransformer(
        patch_size=2,
        emb_dim=96,
        depths=(2, 2, 6, 2),
        num_heads=(3, 6, 12, 24),
        window_size=4,
        mlp_ratio=4,
        use_att_bias=True,
        dropout=0.0,
        att_dropout=0.1,
        drop_path=0.0,
        use_abs_pos_emb=False,
        attach_head=True,
        num_classes=num_outputs,
    )

@_register_model("Swin_Tiny_Patch4")
def swin_tiny(num_outputs : int):
    return SwinTransformer(
        patch_size=4,
        emb_dim=96,
        depths=(2, 2, 6, 2),
        num_heads=(3, 6, 12, 24),
        window_size=4,
        mlp_ratio=4,
        use_att_bias=True,
        dropout=0.0,
        att_dropout=0.1,
        drop_path=0.0,
        use_abs_pos_emb=False,
        attach_head=True,
        num_classes=num_outputs,
    )

@_register_model("Swin_Tiny_224")
def swin_tiny_224(num_outputs : int):
    return SwinTransformer(
        patch_size=4,
        emb_dim=96,
        depths=(2, 2, 6, 2),
        num_heads=(3, 6, 12, 24),
        window_size=7,
        mlp_ratio=4,
        use_att_bias=True,
        dropout=0.0,
        att_dropout=0.1,
        drop_path=0.0,
        use_abs_pos_emb=False,
        attach_head=True,
        num_classes=num_outputs,
    )

@_register_model("Swin_Small")
def swin_small(num_outputs : int):
    return SwinTransformer(
        patch_size=2,
        emb_dim=96,
        depths=(2, 2, 18, 2),
        num_heads=(3, 6, 12, 24),
        window_size=4,
        mlp_ratio=4,
        use_att_bias=True,
        dropout=0.0,
        att_dropout=0.1,
        drop_path=0.0,
        use_abs_pos_emb=False,
        attach_head=True,
        num_classes=num_outputs,
    )

@_register_model("Swin_Small_Patch4")
def swin_small(num_outputs : int):
    return SwinTransformer(
        patch_size=4,
        emb_dim=96,
        depths=(2, 2, 18, 2),
        num_heads=(3, 6, 12, 24),
        window_size=4,
        mlp_ratio=4,
        use_att_bias=True,
        dropout=0.0,
        att_dropout=0.1,
        drop_path=0.0,
        use_abs_pos_emb=False,
        attach_head=True,
        num_classes=num_outputs,
    )

@_register_model("Swin_Base")
def swin_base(num_outputs : int):
    return SwinTransformer(
        patch_size=2,
        emb_dim=128,
        depths=(2, 2, 18, 2),
        num_heads=(4, 8, 16, 32),
        window_size=4,
        mlp_ratio=4,
        use_att_bias=True,
        dropout=0.0,
        att_dropout=0.1,
        drop_path=0.0,
        use_abs_pos_emb=False,
        attach_head=True,
        num_classes=num_outputs,
    )


@_register_model("Swin_Base_Patch4")
def swin_base(num_outputs : int):
    return SwinTransformer(
        patch_size=4,
        emb_dim=128,
        depths=(2, 2, 18, 2),
        num_heads=(4, 8, 16, 32),
        window_size=4,
        mlp_ratio=4,
        use_att_bias=True,
        dropout=0.0,
        att_dropout=0.1,
        drop_path=0.0,
        use_abs_pos_emb=False,
        attach_head=True,
        num_classes=num_outputs,
    )