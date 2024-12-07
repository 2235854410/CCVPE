import time

import torch

from utils import center_padding, tokens_to_output
from torch.nn.functional import interpolate
import torch.nn as nn
import torch.nn.functional as F


class DINO(torch.nn.Module):
    def __init__(
        self,
        dino_name="dinov2",
        model_name="vitb14",
        output="dense",
        layer=-1,
        return_multilayer=True,
        output_channels=112,
        hidden_channels1=544,
        hidden_channels2=465,
        down_sample=False,
    ):
        super().__init__()
        # dinov2 only have patch size=14
        feat_dims = {
            "vitb8": 768,
            "vitb16": 768,
            "vitb14": 768,
            "vitb14_reg": 768,
            "vitl14": 1024,
            "vitg14": 1536,
        }

        # get model
        self.model_name = dino_name
        self.down_sample = down_sample
        self.checkpoint_name = f"{dino_name}_{model_name}"
        torch.hub._validate_not_a_forked_repo = lambda a, b, c: True
        dino_vit = torch.hub.load(f"facebookresearch/{dino_name}", self.checkpoint_name)
        self.vit = dino_vit.eval().to(torch.float32)
        for param in dino_vit.parameters():
            param.requires_grad = False
        self.has_registers = "_reg" in model_name

        self.flatten = nn.Flatten(2)
        self.unflatten = nn.Unflatten(2, (37, 37))
        self.conv_down = nn.Conv2d(in_channels=320, out_channels=320, kernel_size=3, stride=2, padding=1)
        self.relu = nn.ReLU()



        assert output in ["cls", "gap", "dense", "dense-cls"]
        self.output = output
        self.patch_size = self.vit.patch_embed.proj.kernel_size[0]

        feat_dim = feat_dims[model_name]
        feat_dim = feat_dim * 2 if output == "dense-cls" else feat_dim

        num_layers = len(self.vit.blocks)
        # print(f"Number of layers in DINOv2: {num_layers}")
        # dinov2b14 has 12 layers in total
        multilayers = [
            num_layers // 4 - 2, # 1
            num_layers // 4, # 3
            num_layers // 2 - 1, # 5
            num_layers // 2 + 1, #7
            num_layers // 4 * 3, # 9
            num_layers - 1, #11
        ]

        if return_multilayer:
            self.feat_dim = [feat_dim, feat_dim, feat_dim, feat_dim]
            self.multilayers = multilayers
        else:
            self.feat_dim = feat_dim
            layer = multilayers[-1] if layer == -1 else layer
            self.multilayers = [layer]


        # define layer name (for logging)
        self.layer = "-".join(str(_x) for _x in self.multilayers)

    def forward(self, images):

        images = center_padding(images, self.patch_size)
        h, w = images.shape[-2:]
        # print(f"patch size: {self.patch_size}")
        h, w = h // self.patch_size, w // self.patch_size

        if self.model_name == "dinov2":
            x = self.vit.prepare_tokens_with_masks(images, None)
        else:
            x = self.vit.prepare_tokens(images)

        embeds = []
        for i, blk in enumerate(self.vit.blocks):
            x = blk(x)
            if i in self.multilayers:
                embeds.append(x)
                if len(embeds) == len(self.multilayers):
                    break

        num_spatial = h * w
        outputs = []
        for i, x_i in enumerate(embeds):
            cls_tok = x_i[:, 0]
            # ignoring register tokens
            spatial = x_i[:, -1 * num_spatial :]
            x_i = tokens_to_output(self.output, spatial, cls_tok, (h, w))
            outputs.append(x_i)

        # res = self.dpt(outputs)
        # x = F.interpolate(res, size=(16,16), mode='bilinear', align_corners=True)


        # return outputs[0], x
        return outputs[0] if len(outputs) == 1 else outputs
