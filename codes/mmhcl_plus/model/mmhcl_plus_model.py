import copy

import torch.nn as nn


class MMHCLPlus(nn.Module):
    def __init__(self, hyper_encoder, bip_encoder, fusion, projector_u2u):
        super().__init__()
        self.hyper_encoder = hyper_encoder
        self.bip_encoder = bip_encoder
        self.fusion = fusion
        self.projector_u2u = projector_u2u
        self.ema_teacher = copy.deepcopy(bip_encoder)
        for p in self.ema_teacher.parameters():
            p.requires_grad = False

    def forward(self, h0, checkpoint_fn=None):
        h_hyper, hyper_embs = self.hyper_encoder(h0, checkpoint_fn)
        h_bip, bip_embs = self.bip_encoder(h0, checkpoint_fn)
        h_fused = self.fusion(h_hyper, h_bip)
        return {
            "h_fused": h_fused,
            "hyper_final": h_hyper,
            "bip_final": h_bip,
            "hyper_layers": hyper_embs,
            "bip_layers": bip_embs,
        }
