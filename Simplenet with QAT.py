###Simplenet with QAT


import torch
import torch.nn.functional as F
from torch import nn
import ai8x


class SimpleNetFaceID(nn.Module):
    """
    SimpleNet-style network adapted for face/ID embeddings on MAX78000.

    - Produces an L2-normalized embedding vector (e.g. 64-D).
    - Includes quantization and BN fusion hooks for deployment.
    """

    def __init__(
        self,
        emb_dimensionality: int = 64,
        num_channels: int = 1,
        dimensions: tuple = (28, 28),
        planes: int = 20,
        bias: bool = False,
        **kwargs
    ):
        super().__init__()

        # Track spatial dimension
        dim = dimensions[0]

        # Conv1: 5x5 conv + ReLU (no padding)
        self.conv1 = ai8x.FusedConv2dReLU(
            num_channels, planes, kernel_size=5, padding=0, bias=bias, **kwargs
        )
        dim -= 4  # 28 -> 24

        # Conv2: MaxPool + 5x5 conv + ReLU (no padding)
        self.conv2 = ai8x.FusedMaxPoolConv2dReLU(
            planes, planes * 2,
            kernel_size=5, padding=0,
            pool_size=2, pool_stride=2,
            bias=bias, **kwargs
        )
        dim = (dim - 4) // 2  # 24 -> 20 -> pooled -> 10

        # Flattened size for FC
        fc_input_dim = planes * 2 * dim * dim

        # Projection fully connected layer -> embedding
        self.fc1 = ai8x.Linear(fc_input_dim, emb_dimensionality, bias=bias, wide=True, **kwargs)

        # Initialization (conv layers)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass: produces L2-normalized embeddings."""
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size(0), -1)  # flatten
        x = self.fc1(x)
        x = F.normalize(x, p=2, dim=1)
        return x

    ###QAT / Deployment API
  
    def quantize_layer(self, layer_index=None, qat_policy=None):
        """
        Initialize QAT params for a specific layer index.
        If no layer_index or qat_policy, this is a no-op.
        """
        if layer_index is None or qat_policy is None:
            return

        layer_buf = list(self.children())
        if layer_index < 0 or layer_index >= len(layer_buf):
            raise IndexError("layer_index out of range")

        layer = layer_buf[layer_index]
        if hasattr(layer, "init_module"):
            layer.init_module(qat_policy["weight_bits"], qat_policy["bias_bits"], True)

    def fuse_bn_layer(self, layer_index=None):
        """
        Fuses BatchNorm parameters into a conv-like module for faster inference.
        Works on ai8x.QuantizationAwareModule-style modules.
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
            return  # not a conv-like module

        # Extract conv weights/bias
        
        w = layer.op.weight.data
        b = layer.op.bias.data if layer.op.bias is not None else torch.zeros(w.shape[0], device=w.device)

        # BatchNorm stats
        
        bn = layer.bn
        r_mean, r_var = bn.running_mean, bn.running_var
        r_std = torch.sqrt(r_var + 1e-20).to(w.device)

        # Affine params
        
        gamma = bn.weight if bn.weight is not None else torch.ones_like(r_mean).to(w.device)
        beta = bn.bias if bn.bias is not None else torch.zeros_like(r_mean).to(w.device)

        # Fuse
        
        scale = (gamma / r_std).reshape((w.shape[0],) + (1,) * (len(w.shape) - 1))
        w_new = w * scale
        b_new = ((b - r_mean.to(b.device)) / r_std) * gamma + beta

        # Write back
        
        layer.op.weight.data = w_new
        if layer.op.bias is None:
            layer.op.bias = nn.Parameter(b_new.clone())
        else:
            layer.op.bias.data = b_new

        # Remove BN
        
        layer.bn = None


def simplenet_faceid(pretrained: bool = False, **kwargs) -> SimpleNetFaceID:
    """Factory function to build SimpleNetFaceID."""
    assert not pretrained, "pretrained weights not supported"
    return SimpleNetFaceID(**kwargs)


models = [
    {
        "name": "simplenet_faceid",
        "min_input": 1,
        "dim": 2,
    }
]


### Example quick sanity test

if __name__ == "__main__":
    model = simplenet_faceid(num_channels=1, dimensions=(28, 28), emb_dimensionality=64)
    model.eval()

    dummy = torch.randn(2, 1, 28, 28)
    embeddings = model(dummy)

    print("Embeddings shape:", embeddings.shape)
    print("Norms:", embeddings.norm(dim=1))

