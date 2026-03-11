import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image
from typing import Any, Dict, Tuple

class VisualEvidenceNet(nn.Module):
    """
    Vision backbone -> multi-scale pyramid -> attention pooling -> gated aggregation -> evidence e ∈ R_{>=0}^K
    Outputs:
      - z_v: compact visual representation (vector)
      - e: non-negative class-wise evidence vector
      - debug dict: attention maps, scale gates, etc.
    """

    def __init__(self, num_classes: int = 3,
                 pretrained: bool = True, backbone="resnet50",
                 fpn_dim: int = 256, proj_dim: int = 256):
        super().__init__()
        self.K = num_classes
        self.fpn_dim: int = fpn_dim,
        self.proj_dim: int = proj_dim,
        self.backbone = build_backbone(backbone, pretrained=pretrained)
        c2_ch, c3_ch, c4_ch, c5_ch = self.backbone.out_channels

        # lateral 1x1
        self.lat2 = nn.Conv2d(c2_ch, fpn_dim, 1)
        self.lat3 = nn.Conv2d(c3_ch, fpn_dim, 1)
        self.lat4 = nn.Conv2d(c4_ch, fpn_dim, 1)
        self.lat5 = nn.Conv2d(c5_ch, fpn_dim, 1)

        # refinement 3x3
        self.ref2 = nn.Conv2d(fpn_dim, fpn_dim, 3, padding=1)
        self.ref3 = nn.Conv2d(fpn_dim, fpn_dim, 3, padding=1)
        self.ref4 = nn.Conv2d(fpn_dim, fpn_dim, 3, padding=1)
        self.ref5 = nn.Conv2d(fpn_dim, fpn_dim, 3, padding=1)

        def make_att(fdim: int):
            return nn.Sequential(
                nn.Conv2d(fdim, fdim, 3, padding=1),
                nn.Tanh(),
                nn.Conv2d(fdim, 1, 1),
            )

        self.att2 = make_att(fpn_dim)
        self.att3 = make_att(fpn_dim)
        self.att4 = make_att(fpn_dim)
        self.att5 = make_att(fpn_dim)

        # cross-scale gating
        self.gate = nn.Linear(fpn_dim, 1)

        # projection + evidence head
        self.proj = nn.Sequential(nn.Linear(fpn_dim, proj_dim), nn.GELU())
        self.head = nn.Sequential(
            nn.Linear(proj_dim, proj_dim),
            nn.GELU(),
            nn.Linear(proj_dim, self.K),
        )

        input_size = getattr(self.backbone, "input_size", 224)
        mean = getattr(self.backbone, "mean", [0.485, 0.456, 0.406])
        std = getattr(self.backbone, "std", [0.229, 0.224, 0.225])
        self.transform = transforms.Compose([
            transforms.Resize((input_size, input_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])

    def forward(self, I_explicit: Dict[str, Any]) -> Dict[str, Any]:
        device = next(self.parameters()).device
        img_obj = I_explicit.get("image")
        bbox = I_explicit.get("bbox", None)

        x = self._load_and_preprocess(img_obj, bbox, device)  # [1,3,224,224]
        c2, c3, c4, c5 = self.backbone(x)

        # pyramid fusion
        p5 = self.lat5(c5)
        p4 = self.lat4(c4) + F.interpolate(p5, size=c4.shape[-2:], mode="nearest")
        p3 = self.lat3(c3) + F.interpolate(p4, size=c3.shape[-2:], mode="nearest")
        p2 = self.lat2(c2) + F.interpolate(p3, size=c2.shape[-2:], mode="nearest")

        p5 = self.ref5(p5); p4 = self.ref4(p4); p3 = self.ref3(p3); p2 = self.ref2(p2)

        g2, a2 = self._att_pool(p2, self.att2)
        g3, a3 = self._att_pool(p3, self.att3)
        g4, a4 = self._att_pool(p4, self.att4)
        g5, a5 = self._att_pool(p5, self.att5)

        G = torch.stack([g2, g3, g4, g5], dim=1)  # [1,4,C]
        scores = self.gate(G).squeeze(-1)         # [1,4]
        beta = F.softmax(scores, dim=1)           # [1,4]
        z = (G * beta.unsqueeze(-1)).sum(dim=1)   # [1,C]

        z_v = self.proj(z)                        # [1,proj_dim]
        logits = self.head(z_v)                   # [1,K]
        e = F.softplus(logits)                    # [1,K]  (non-negative evidence)

        return {
            "z_v": z_v.squeeze(0),
            "e": e.squeeze(0),
            "debug": {
                "beta_scales": beta.squeeze(0).detach(),
                "att_maps": {
                    "p2": a2.detach(),
                    "p3": a3.detach(),
                    "p4": a4.detach(),
                    "p5": a5.detach(),
                }
            }
        }

    def _att_pool(self, feat: torch.Tensor, att_net: nn.Module):
        B, C, H, W = feat.shape
        s = att_net(feat).view(B, -1)            # [B,HW]
        a = F.softmax(s, dim=1).view(B, H, W)    # [B,H,W]
        g = (feat * a.unsqueeze(1)).sum(dim=(2, 3))
        return g, a

    def _load_and_preprocess(self, img_obj, bbox, device):
        if isinstance(img_obj, str):
            img = Image.open(img_obj).convert("RGB")
        else:
            img = img_obj.convert("RGB") if isinstance(img_obj, Image.Image) else img_obj

        if bbox is not None and isinstance(bbox, (list, tuple)) and len(bbox) == 4:
            try:
                x1, y1, x2, y2 = bbox
                x1, y1 = max(0, int(x1)), max(0, int(y1))
                x2, y2 = max(x1 + 1, int(x2)), max(y1 + 1, int(y2))
                img = img.crop((x1, y1, x2, y2))
            except Exception:
                pass

        x = self.transform(img).unsqueeze(0).to(device)
        return x


# class ResNet50Backbone(nn.Module):
#     """Extract C2-C5 feature maps from torchvision ResNet-50 (pretrained)."""
#
#     def __init__(self, pretrained: bool = True):
#         super().__init__()
#         m = models.resnet50(weights=models.ResNet50_Weights.DEFAULT if pretrained else None)
#         self.conv1 = m.conv1
#         self.bn1 = m.bn1
#         self.relu = m.relu
#         self.maxpool = m.maxpool
#         self.layer1 = m.layer1  # C2 (256)
#         self.layer2 = m.layer2  # C3 (512)
#         self.layer3 = m.layer3  # C4 (1024)
#         self.layer4 = m.layer4  # C5 (2048)
#
#     def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
#         x = self.conv1(x); x = self.bn1(x); x = self.relu(x); x = self.maxpool(x)
#         c2 = self.layer1(x)
#         c3 = self.layer2(c2)
#         c4 = self.layer3(c3)
#         c5 = self.layer4(c4)
#         return c2, c3, c4, c5

# -------------------------
# Backbone base + factory
# -------------------------
class BackboneBase(nn.Module):
    """
    All backbones must implement:
      - self.out_channels: tuple[int,int,int,int] = (c2,c3,c4,c5) channels
      - self.input_size: int  (e.g., 224 or 299)
      - forward(x) -> (c2,c3,c4,c5) feature maps (stride ~4/8/16/32 preferred)
    """
    out_channels: tuple
    input_size: int
    mean: list
    std: list


def build_backbone(name: str, pretrained: bool = True) -> BackboneBase:
    name = name.lower().strip()
    if name in ["resnet50", "res50", "resnet-50"]:
        return ResNet50Backbone(pretrained=pretrained)
    if name in ["vgg16", "vgg-16"]:
        return VGG16Backbone(pretrained=pretrained)
    if name in ["inceptionv3", "inception_v3", "inception-v3"]:
        return InceptionV3Backbone(pretrained=pretrained)
    if name in ["mobilenetv2", "mobilenet_v2", "mobilenet-v2", "mbv2"]:
        return MobileNetV2Backbone(pretrained=pretrained)
    if name in ["xception"]:
        return XceptionBackbone(pretrained=pretrained)  # uses timm
    raise ValueError(f"Unknown backbone: {name}")


# -------------------------
# ResNet-50
# -------------------------
class ResNet50Backbone(BackboneBase):
    """Extract C2-C5 feature maps from torchvision ResNet-50."""
    def __init__(self, pretrained: bool = True):
        super().__init__()
        m = models.resnet50(weights=models.ResNet50_Weights.DEFAULT if pretrained else None)
        self.conv1 = m.conv1
        self.bn1 = m.bn1
        self.relu = m.relu
        self.maxpool = m.maxpool
        self.layer1 = m.layer1  # C2 (256)
        self.layer2 = m.layer2  # C3 (512)
        self.layer3 = m.layer3  # C4 (1024)
        self.layer4 = m.layer4  # C5 (2048)
        self.out_channels = (256, 512, 1024, 2048)
        self.input_size = 224
        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]

    def forward(self, x: torch.Tensor):
        x = self.conv1(x); x = self.bn1(x); x = self.relu(x); x = self.maxpool(x)
        c2 = self.layer1(x)
        c3 = self.layer2(c2)
        c4 = self.layer3(c3)
        c5 = self.layer4(c4)
        return c2, c3, c4, c5


# -------------------------
# VGG16 (use outputs after pool2/3/4/5)
# -------------------------
class VGG16Backbone(BackboneBase):
    """
    VGG16 has only a sequential feature extractor.
    We'll take feature maps after pool2/pool3/pool4/pool5 as c2-c5.
    Channel dims at these points: 128, 256, 512, 512
    """
    def __init__(self, pretrained: bool = True):
        super().__init__()
        m = models.vgg16(weights=models.VGG16_Weights.DEFAULT if pretrained else None)
        self.features = m.features
        self.out_channels = (128, 256, 512, 512)
        self.input_size = 224
        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]

        # Indices of MaxPool layers in vgg16.features:
        # pools at 4, 9, 16, 23, 30
        self._pool_indices = [9, 16, 23, 30]  # pool2..pool5

    def forward(self, x: torch.Tensor):
        outs = []
        for i, layer in enumerate(self.features):
            x = layer(x)
            if i in self._pool_indices:
                outs.append(x)
        c2, c3, c4, c5 = outs  # pool2..pool5
        return c2, c3, c4, c5


# -------------------------
# MobileNetV2 (pick stages by spatial stride changes)
# -------------------------
class MobileNetV2Backbone(BackboneBase):
    """
    MobileNetV2.features is a list of blocks.
    We pick feature maps at strides approximately 4,8,16,32 as c2-c5.
    Common channel dims for torchvision mobilenet_v2:
      stride~4 : 24
      stride~8 : 32
      stride~16: 96
      stride~32: 1280 (after last 1x1 conv)
    """
    def __init__(self, pretrained: bool = True):
        super().__init__()
        m = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT if pretrained else None)
        self.features = m.features
        self.out_channels = (32, 64, 160, 1280)
        self.input_size = 224
        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]

        # 固定 stage 索引
        self.idx_c2 = 6  # 32  @ 1/4 (28x28)
        self.idx_c3 = 10  # 64  @ 1/8 (14x14)
        self.idx_c4 = 14  # 160 @ 1/16 (7x7)
        self.idx_c5 = 18  # 1280@ 1/32 (7x7)

    def forward(self, x):
        c2 = c3 = c4 = c5 = None
        for i, layer in enumerate(self.features):
            x = layer(x)
            if i == self.idx_c2:
                c2 = x
            elif i == self.idx_c3:
                c3 = x
            elif i == self.idx_c4:
                c4 = x
            elif i == self.idx_c5:
                c5 = x

        return c2, c3, c4, c5

# -------------------------
# Inception v3 (features from Mixed blocks)
# -------------------------
class InceptionV3Backbone(BackboneBase):
    """
    Inception v3 prefers input 299.
    We'll extract:
      c2: Mixed_5d   (approx stride 8,  channels 288)
      c3: Mixed_6e   (approx stride 16, channels 768)
      c4: Mixed_7b   (approx stride 32, channels 1280)
      c5: Mixed_7c   (approx stride 32, channels 2048)  (same stride as c4, but deeper)
    Note: c4/c5 having same stride is OK for your FPN; it still provides two semantic levels.
    """
    def __init__(self, pretrained: bool = True):
        super().__init__()
        m = models.inception_v3(
            weights=models.Inception_V3_Weights.DEFAULT if pretrained else None,
            init_weights=False
        )
        # stem
        self.Conv2d_1a_3x3 = m.Conv2d_1a_3x3
        self.Conv2d_2a_3x3 = m.Conv2d_2a_3x3
        self.Conv2d_2b_3x3 = m.Conv2d_2b_3x3
        self.maxpool1 = m.maxpool1
        self.Conv2d_3b_1x1 = m.Conv2d_3b_1x1
        self.Conv2d_4a_3x3 = m.Conv2d_4a_3x3
        self.maxpool2 = m.maxpool2
        # inception blocks
        self.Mixed_5b = m.Mixed_5b
        self.Mixed_5c = m.Mixed_5c
        self.Mixed_5d = m.Mixed_5d
        self.Mixed_6a = m.Mixed_6a
        self.Mixed_6b = m.Mixed_6b
        self.Mixed_6c = m.Mixed_6c
        self.Mixed_6d = m.Mixed_6d
        self.Mixed_6e = m.Mixed_6e
        self.Mixed_7a = m.Mixed_7a
        self.Mixed_7b = m.Mixed_7b
        self.Mixed_7c = m.Mixed_7c

        self.out_channels = (288, 768, 1280, 2048)
        self.input_size = 299
        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]

    def forward(self, x: torch.Tensor):
        # stem
        x = self.Conv2d_1a_3x3(x)
        x = self.Conv2d_2a_3x3(x)
        x = self.Conv2d_2b_3x3(x)
        x = self.maxpool1(x)
        x = self.Conv2d_3b_1x1(x)
        x = self.Conv2d_4a_3x3(x)
        x = self.maxpool2(x)

        # mixed 5
        x = self.Mixed_5b(x)
        x = self.Mixed_5c(x)
        c2 = self.Mixed_5d(x)

        # mixed 6
        x = self.Mixed_6a(c2)
        x = self.Mixed_6b(x)
        x = self.Mixed_6c(x)
        x = self.Mixed_6d(x)
        c3 = self.Mixed_6e(x)

        # mixed 7
        c4 = self.Mixed_7a(c3)  # 1280
        x = self.Mixed_7b(c4)
        c5 = self.Mixed_7c(x)  # 2048

        return c2, c3, c4, c5


# -------------------------
# Xception (timm optional)
# -------------------------
class XceptionBackbone(BackboneBase):
    """
    Requires: pip install timm
    Uses timm features_only to return multi-scale feature maps.
    We'll map out_indices (1,2,3,4) to c2-c5.
    """
    def __init__(self, pretrained: bool = True):
        super().__init__()
        try:
            import timm
        except Exception as e:
            raise ImportError(
                "XceptionBackbone requires 'timm'. Install with: pip install timm"
            ) from e

        # features_only gives a list of feature maps at different stages
        self.m = timm.create_model(
            "xception",
            pretrained=pretrained,
            features_only=True,
            out_indices=(1, 2, 3, 4),
        )
        chs = self.m.feature_info.channels()
        self.out_channels = tuple(chs)  # 4 stages
        cfg = self.m.default_cfg
        self.input_size = cfg.get("input_size", (3, 299, 299))[1]
        self.mean = list(cfg.get("mean", [0.485, 0.456, 0.406]))
        self.std = list(cfg.get("std", [0.229, 0.224, 0.225]))

    def forward(self, x: torch.Tensor):
        feats = self.m(x)  # list of 4 tensors
        c2, c3, c4, c5 = feats
        return c2, c3, c4, c5