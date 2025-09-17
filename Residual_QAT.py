### Residual

from torch import nn
import ai8x

class AI85ResidualSimpleNet(nn.Module):
    """
    Residual SimpleNet v1 Model for CIFAR-10 classification.
    Uses custom AI85 hardware optimized layers.
    """

    def __init__(self, num_classes=10, num_channels=3, dimensions=(32, 32), bias=False, **kwargs):
        super().__init__()

        # Define the layers with custom ai8x layers
        self.conv1 = ai8x.FusedConv2dReLU(num_channels, 16, 3, stride=1, padding=1, bias=bias, **kwargs)
        self.conv2 = ai8x.FusedConv2dReLU(16, 20, 3, stride=1, padding=1, bias=bias, **kwargs)
        self.conv3 = ai8x.FusedConv2dReLU(20, 20, 3, stride=1, padding=1, bias=bias, **kwargs)
        self.conv4 = ai8x.FusedConv2dReLU(20, 20, 3, stride=1, padding=1, bias=bias, **kwargs)
        self.resid1 = ai8x.Add()

        self.conv5 = ai8x.FusedMaxPoolConv2dReLU(20, 20, 3, pool_size=2, pool_stride=2, stride=1, padding=1, bias=bias, **kwargs)
        self.conv6 = ai8x.FusedConv2dReLU(20, 20, 3, stride=1, padding=1, bias=bias, **kwargs)
        self.resid2 = ai8x.Add()

        self.conv7 = ai8x.FusedConv2dReLU(20, 44, 3, stride=1, padding=1, bias=bias, **kwargs)
        self.conv8 = ai8x.FusedMaxPoolConv2dReLU(44, 48, 3, pool_size=2, pool_stride=2, stride=1, padding=1, bias=bias, **kwargs)
        self.conv9 = ai8x.FusedConv2dReLU(48, 48, 3, stride=1, padding=1, bias=bias, **kwargs)
        self.resid3 = ai8x.Add()

        self.conv10 = ai8x.FusedMaxPoolConv2dReLU(48, 96, 3, pool_size=2, pool_stride=2, stride=1, padding=1, bias=bias, **kwargs)
        self.conv11 = ai8x.FusedMaxPoolConv2dReLU(96, 512, 1, pool_size=2, pool_stride=2, padding=0, bias=bias, **kwargs)
        self.conv12 = ai8x.FusedConv2dReLU(512, 128, 1, stride=1, padding=0, bias=bias, **kwargs)
        self.conv13 = ai8x.FusedMaxPoolConv2dReLU(128, 128, 3, pool_size=2, pool_stride=2, stride=1, padding=1, bias=bias, **kwargs)
        self.conv14 = ai8x.Conv2d(128, num_classes, 1, stride=1, padding=0, bias=bias, wide=True, **kwargs)

    def forward(self, x):
        """Forward pass through the model."""
        # Convolution + Residual connections
        
        x = self.conv1(x)  # 16x32x32
        x_res = self.conv2(x)  # 20x32x32
        x = self.conv3(x_res)  # 20x32x32
        x = self.resid1(x, x_res)  # 20x32x32
        x = self.conv4(x)  # 20x32x32
        x_res = self.conv5(x)  # 20x16x16
        x = self.conv6(x_res)  # 20x16x16
        x = self.resid2(x, x_res)  # 20x16x16
        x = self.conv7(x)  # 44x16x16
        x_res = self.conv8(x)  # 48x8x8
        x = self.conv9(x_res)  # 48x8x8
        x = self.resid3(x, x_res)  # 48x8x8
        x = self.conv10(x)  # 96x4x4
        x = self.conv11(x)  # 512x2x2
        x = self.conv12(x)  # 128x2x2
        x = self.conv13(x)  # 128x1x1
        x = self.conv14(x)  # num_classesx1x1
        
        # Flatten the output to match expected dimensions (batch_size, num_classes)
        
        x = x.view(x.size(0), -1)
        return x

    def quantize_layer(self, layer_index=None, qat_policy=None):
        """Apply quantization to the given layer."""
        if layer_index is None or qat_policy is None:
            return
        layer_buf = list(self.children())
        layer = layer_buf[layer_index]
        layer.init_module(qat_policy['weight_bits'], qat_policy['bias_bits'], True)

    def fuse_bn_layer(self, layer_index=None):
        """Fuse batch normalization parameters into the preceding convolution layer."""
        if layer_index is None:
            return
        layer_buf = list(self.children())
        layer = layer_buf[layer_index]
        if isinstance(layer, QuantizationAwareModule) and layer.bn is not None:
            w = layer.op.weight.data
            b = layer.op.bias.data
            device = w.device
            r_mean = layer.bn.running_mean
            r_var = layer.bn.running_var
            r_std = torch.sqrt(r_var + 1e-20)
            beta = layer.bn.weight
            gamma = layer.bn.bias
            if beta is None:
                beta = torch.ones(w.shape[0]).to(device)
            if gamma is None:
                gamma = torch.zeros(w.shape[0]).to(device)

            w_new = w * (beta / r_std).reshape((w.shape[0],) + (1,) * (len(w.shape) - 1))
            b_new = (b - r_mean) / r_std * beta + gamma
            layer.op.weight.data = w_new
            layer.op.bias.data = b_new
            layer.bn = None

def ai85ressimplenet(pretrained=False, **kwargs):
    """Factory function to create an AI85 Residual SimpleNet model."""
    assert not pretrained  # This version of the model doesn't support pre-trained weights.
    return AI85ResidualSimpleNet(**kwargs)

# Model configuration

models = [
    {
        'name': 'ai85ressimplenet',
        'min_input': 1,  # Minimum input channels
        'dim': 2,        # Number of dimensions
    },
]

