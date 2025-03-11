'''
Thanks Zeyuan
https://github.com/VILA-Lab/SRe2L/tree/main/SRe2L
'''

import torch
import torch.nn as nn
import timm
from timm.models.vision_transformer import VisionTransformer, Block


class BN_bnc(nn.BatchNorm1d):
    """
    BN_bnc: BatchNorm1d on hidden feature with (B,N,C) dimension
    """

    def forward(self, x):
        B, N, C = x.shape
        x = x.reshape(B * N, C)  # (B,N,C) -> (B*N,C)
        x = super().forward(x)   # apply batch normalization
        x = x.reshape(B, N, C)   # (B*N,C) -> (B,N,C)
        return x


class BN_MLP(timm.layers.Mlp):
    """
    BN_MLP: add BN_bnc in-between 2 linear layers in MLP module
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.norm = BN_bnc(kwargs['hidden_features'])

    def forward(self, x):
        x = self.fc1(x)
        x = self.norm(x)  # apply batch normalization before activation
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x


def replace_BN(model):
    if isinstance(model, VisionTransformer):
        model.norm = BN_bnc(model.norm.normalized_shape)
    else:
        raise NotImplementedError(
            'replace_BN only supports timm VisionTransformer')

    for name, module in model.named_modules():
        if isinstance(module, Block):
            module.norm1 = BN_bnc(module.norm1.normalized_shape)
            module.norm2 = BN_bnc(module.norm2.normalized_shape)
            module.mlp = BN_MLP(in_features=module.mlp.fc1.in_features,
                                hidden_features=module.mlp.fc1.out_features,
                                out_features=module.mlp.fc2.out_features,
                                act_layer=module.mlp.act.__class__,
                                bias=module.mlp.fc1.bias,
                                drop=module.mlp.drop1.p)

    return model