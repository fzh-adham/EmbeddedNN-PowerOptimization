import torch
import torch.nn.functional as F
from torch import nn
import ai8x
import ai8x_blocks
from typing import Sequence, Tuple, Optional


class AI85FaceIDNet_112(nn.Module):
    """
    FaceID Network for MAX78000 with flexible input size.
    - Bottleneck settings format: (num_repeat, in_channels, out_channels, stride, expansion_factor)
    - Produces L2-normalized embeddings.
    """

    def __init__(
        self,
        pre_layer_stride: int,
        bottleneck_settings: Sequence[Sequence[int]],
        last_layer_width: int,
        emb_dimensionality: int,
        num_channels: int = 3,
        avg_pool_size: Tuple[int, int] = (7, 7),  # legacy default, replaced by AdaptiveAvgPool2d
        bias: bool = False,
        depthwise_bias: bool = False,
        reduced_depthwise_bias: bool = False,
        use_adaptive_pool: bool = True,
        **kwargs
    ):
        super().__init__()

        self.num_channels = num_channels
        self.use_adaptive_pool = use_adaptive_pool

        # Pre-stage conv: 3x3 conv -> ReLU (fused implementation)
        self.pre_stage = ai8x.FusedConv2dReLU(
            num_channels, bottleneck_settings[0][1], kernel_size=3,
            padding=1, stride=pre_layer_stride, bias=False, **kwargs
        )

        # pre_stage_2: MaxPool -> Conv -> ReLU (fused)
        self.pre_stage_2 = ai8x.FusedMaxPoolConv2dReLU(
            bottleneck_settings[0][1],
            bottleneck_settings[0][1],
            kernel_size=3,
            padding=1,
            stride=1,
            pool_size=2,
            pool_stride=2,
            bias=False,
            **kwargs
        )

        # Feature extraction bottlenecks (list of nn.Sequential stages)
        self.feature_stage = nn.ModuleList()
        for setting in bottleneck_settings:
            self._create_bottleneck_stage(setting, bias, depthwise_bias, reduced_depthwise_bias, **kwargs)

        # 1x1 conv projection after bottlenecks
        in_ch_post = bottleneck_settings[-1][2]
        self.post_stage = ai8x.FusedConv2dReLU(
            in_ch_post, last_layer_width, kernel_size=1, padding=0, stride=1, bias=False, **kwargs
        )

        # small conv before pooling (keeps original design)
        self.pre_avg = ai8x.Conv2d(last_layer_width, last_layer_width, kernel_size=3, padding=1, stride=1, bias=False, **kwargs)

        # Use adaptive pooling unless explicitly disabled
        if use_adaptive_pool:
            self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        else:
            # keep legacy fixed-size avg pool if requested (expects shape matches)
            self.avg_pool = ai8x.AvgPool2d(avg_pool_size, stride=1)

        # Final linear layer -> embedding
        self.linear = ai8x.Linear(last_layer_width, emb_dimensionality, bias=bias, **kwargs)

    def _create_bottleneck_stage(
        self,
        setting: Sequence[int],
        bias: bool,
        depthwise_bias: bool,
        reduced_depthwise_bias: bool,
        **kwargs
    ):
        """
        Create an nn.Sequential bottleneck stage.
        `setting` expected as (num_repeat, in_channels, out_channels, stride, expansion_factor)
        """
        num_repeat, in_channels, out_channels, stride, expansion = setting

        stage_layers = []
        if num_repeat <= 0:
            # nothing to add, but keep API consistent
            self.feature_stage.append(nn.Sequential(*stage_layers))
            return

        # First one may have stride != 1 and changes channels
        stage_layers.append(
            ai8x_blocks.ConvResidualBottleneck(
                in_channels=in_channels,
                out_channels=out_channels,
                stride=stride,
                expansion_factor=expansion,
                bias=bias,
                depthwise_bias=depthwise_bias,
                **kwargs
            )
        )

        # Remaining repeats (if any) have stride=1 and in/out channels equal
        for i in range(1, num_repeat):
            if reduced_depthwise_bias:
                # alternate depthwise_bias usage for reduced memory/compute
                use_dw_bias = (i % 2 == 0) and depthwise_bias
            else:
                use_dw_bias = depthwise_bias

            stage_layers.append(
                ai8x_blocks.ConvResidualBottleneck(
                    in_channels=out_channels,
                    out_channels=out_channels,
                    stride=1,
                    expansion_factor=expansion,
                    bias=bias,
                    depthwise_bias=use_dw_bias,
                    **kwargs
                )
            )

        self.feature_stage.append(nn.Sequential(*stage_layers))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass. Input shape: (B, C, H, W) where C matches `num_channels`.
        Produces L2-normalized embeddings of shape (B, emb_dimensionality).
        """
        # If input has extra channels (rare), we make an explicit decision:
        if x.shape[1] != self.num_channels:
            # If input has more channels, keep the first `num_channels` channels.
            # If it has fewer, this will naturally throw an informative error.
            if x.shape[1] > self.num_channels:
                x = x[:, :self.num_channels, :, :]
            else:
                raise ValueError(f"Expected input with {self.num_channels} channels, got {x.shape[1]}")

        x = self.pre_stage(x)
        x = self.pre_stage_2(x)

        for stage in self.feature_stage:
            x = stage(x)

        x = self.post_stage(x)
        x = self.pre_avg(x)
        x = self.avg_pool(x)         # if adaptive -> (B, C, 1, 1)
        x = x.view(x.size(0), -1)    # (B, C)
        x = self.linear(x)
        x = F.normalize(x, p=2, dim=1)
        return x

    def quantize_layer(self, layer_index: Optional[int] = None, qat_policy: Optional[dict] = None):
        """
        Initialize QAT params for a specific layer index. If no layer_index or qat_policy,
        this is a no-op.
        """
        if layer_index is None or qat_policy is None:
            return

        layer_buf = list(self.children())
        if layer_index < 0 or layer_index >= len(layer_buf):
            raise IndexError("layer_index out of range")

        layer = layer_buf[layer_index]
        # assume modules exposing `init_module` (ai8x QuantizationAwareModule)
        layer.init_module(qat_policy['weight_bits'], qat_policy['bias_bits'], True)

    def fuse_bn_layer(self, layer_index: Optional[int] = None):
        """
        Fuses BatchNorm parameters into a conv-like module for faster inference.
        Works on ai8x.QuantizationAwareModule-style modules that have `op` and `bn`.
        """
        if layer_index is None:
            return

        layer_buf = list(self.children())
        if layer_index < 0 or layer_index >= len(layer_buf):
            raise IndexError("layer_index out of range")

        layer = layer_buf[layer_index]

        # Only fuse if module has bn attribute and it's not already fused
        if not (hasattr(layer, "bn") and layer.bn is not None):
            return

        if not hasattr(layer, "op"):
            # not a conv-like module we can fuse
            return

        # conv weight / bias
        w = layer.op.weight.data
        b = layer.op.bias.data if layer.op.bias is not None else torch.zeros(w.shape[0], device=w.device)

        # bn stats
        bn = layer.bn
        r_mean = bn.running_mean
        r_var = bn.running_var
        r_std = torch.sqrt(r_var + 1e-20).to(w.device)

        # bn affine parameters (gamma / beta)
        gamma = bn.weight if bn.weight is not None else torch.ones_like(r_mean).to(w.device)  # scale
        beta = bn.bias if bn.bias is not None else torch.zeros_like(r_mean).to(w.device)       # shift

        # w_new = w * (gamma / r_std).reshape((C_out,1,1, ...))
        scale = (gamma / r_std).reshape((w.shape[0],) + (1,) * (len(w.shape) - 1))
        w_new = w * scale

        # b_new = (b - r_mean) / r_std * gamma + beta
        b_new = ((b - r_mean.to(b.device)) / r_std) * gamma + beta

        # write back
        layer.op.weight.data = w_new
        if layer.op.bias is None:
            # create bias tensor on op if necessary
            layer.op.bias = nn.Parameter(b_new.clone())
        else:
            layer.op.bias.data = b_new
        # mark bn fused
        layer.bn = None

    def trace_shapes(self, input_shape=(1, 3, 112, 112)):
        """
        Utility that runs a forward pass with dummy input and prints shapes at key points.
        Useful when aligning to a paper's expected shapes.
        """
        x = torch.randn(input_shape)
        print(f"input: {x.shape}")
        x = self.pre_stage(x)
        print(f"after pre_stage: {x.shape}")
        x = self.pre_stage_2(x)
        print(f"after pre_stage_2: {x.shape}")
        for i, stage in enumerate(self.feature_stage):
            x = stage(x)
            print(f"after feature_stage[{i}]: {x.shape}")
        x = self.post_stage(x)
        print(f"after post_stage: {x.shape}")
        x = self.pre_avg(x)
        print(f"after pre_avg: {x.shape}")
        x = self.avg_pool(x)
        print(f"after avg_pool: {x.shape}")
        x = x.view(x.size(0), -1)
        print(f"after flatten: {x.shape}")
        x = self.linear(x)
        print(f"after linear (embedding): {x.shape}")
        x = F.normalize(x, p=2, dim=1)
        print(f"after normalize: {x.shape}")
        return x


def ai85faceidnet_112(pretrained: bool = False, **kwargs) -> AI85FaceIDNet_112:
    """
    Constructs an AI85FaceIDNet_112 model with the default bottleneck configuration.
    """
    assert not pretrained, "pretrained weights not supported in this factory"

    bottleneck_settings = [
        # (num_repeat, in_channels, out_channels, stride, expansion)
        (1, 32, 48, 2, 2),
        (1, 48, 64, 2, 4),
        (1, 64, 64, 1, 2),
        (1, 64, 96, 2, 4),
        (1, 96, 128, 1, 2),
    ]

    return AI85FaceIDNet_112(
        pre_layer_stride=1,
        bottleneck_settings=bottleneck_settings,
        last_layer_width=128,
        emb_dimensionality=64,
        avg_pool_size=(7, 7),
        depthwise_bias=True,
        reduced_depthwise_bias=True,
        **kwargs
    )


# registration metadata (keeps original structure)
models = [
    {
        "name": "ai85faceidnet_112",
        "min_input": 1,
        "dim": 3,
    }
]


# -------------------------
# Example quick sanity test
# -------------------------
if __name__ == "__main__":
    # create model and run a dummy forward to verify shapes
    model = ai85faceidnet_112(num_channels=3, use_adaptive_pool=True)
    model.eval()
    # trace shapes (prints intermediate shapes)
    embeddings = model.trace_shapes(input_shape=(2, 3, 112, 112))
    assert embeddings.shape == (2, 64), f"unexpected embedding shape: {embeddings.shape}"
    print("Sanity test passed: embeddings shape is correct.")

