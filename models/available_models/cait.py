import jax_models
from jax_models.models import cait
from gnp.models import _register_model
from typing import Iterable, Optional
import flax.linen as nn


class CaiT(cait.CaiT):

    @nn.compact
    def __call__(self, inputs, train):
        return super().__call__(inputs, not train)
   

@_register_model("CaiT_XXS12")
def cait_xxs12(num_outputs : int):
    return CaiT(
        patch_size=4,
        embed_dim=192,
        depth=12,
        num_heads=4,
        init_scale=1e-5,
        attach_head=True,
        num_classes=num_outputs,
        drop=0.0,
    )


@_register_model("CaiT_XXS24")
def cait_xxs24(num_outputs : int):
    return CaiT(
        patch_size=4,
        embed_dim=192,
        depth=24,
        num_heads=4,
        init_scale=1e-5,
        attach_head=True,
        num_classes=num_outputs,
        drop=0.0,
    )


@_register_model("CaiT_XXS36")
def cait_xxs36(num_outputs : int):
    return CaiT(
        patch_size=4,
        embed_dim=192,
        depth=36,
        num_heads=4,
        init_scale=1e-5,
        attach_head=True,
        num_classes=num_outputs,
        drop=0.0,
    )


@_register_model("CaiT_XS12")
def cait_xs12(num_outputs : int):
    return CaiT(
        patch_size=4,
        embed_dim=288,
        depth=12,
        num_heads=6,
        init_scale=1e-5,
        attach_head=True,
        num_classes=num_outputs,
        drop=0.0,
    )

@_register_model("CaiT_XS24")
def cait_xs24(num_outputs : int):
    return CaiT(
        patch_size=4,
        embed_dim=288,
        depth=24,
        num_heads=6,
        init_scale=1e-5,
        attach_head=True,
        num_classes=num_outputs,
        drop=0.0,
    )

@_register_model("CaiT_S12")
def cait_s12(num_outputs : int):
    return CaiT(
        patch_size=4,
        embed_dim=384,
        depth=12,
        num_heads=8,
        init_scale=1e-5,
        attach_head=True,
        num_classes=num_outputs,
        drop=0.0,
    )


@_register_model("CaiT_S24")
def cait_s24(num_outputs : int):
    return CaiT(
        patch_size=4,
        embed_dim=384,
        depth=24,
        num_heads=8,
        init_scale=1e-5,
        attach_head=True,
        num_classes=num_outputs,
        drop=0.0,
    )

@_register_model("CaiT_S36")
def cait_s36(num_outputs : int):
    return CaiT(
        patch_size=4,
        embed_dim=384,
        depth=36,
        num_heads=8,
        init_scale=1e-5,
        attach_head=True,
        num_classes=num_outputs,
        drop=0.0,
    )


@_register_model("CaiT_M24")
def cait_m24(num_outputs : int):
    return CaiT(
        patch_size=4,
        embed_dim=768,
        depth=24,
        num_heads=16,
        init_scale=1e-6,
        attach_head=True,
        num_classes=num_outputs,
        drop=0.0,
    )

@_register_model("CaiT_M36")
def cait_m36(num_outputs : int):
    return CaiT(
        patch_size=4,
        embed_dim=768,
        depth=36,
        num_heads=16,
        init_scale=1e-6,
        attach_head=True,
        num_classes=num_outputs,
        drop=0.0,
    )


@_register_model("CaiT_M48")
def cait_m48(num_outputs : int):
    return CaiT(
        patch_size=4,
        embed_dim=768,
        depth=48,
        num_heads=16,
        init_scale=1e-6,
        attach_head=True,
        num_classes=num_outputs,
        drop=0.0,
    )

