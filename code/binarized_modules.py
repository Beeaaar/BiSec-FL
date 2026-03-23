import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function


# ============================================================
#  Binary Operators
# ============================================================

class BinaryWeight(Function):
    """
    Weight-only binarization with optional layer-wise scaling.
    Forward:  sign(w) * alpha
    Backward: Straight-Through Estimator (STE)
    """

    @staticmethod
    def forward(ctx, weight, scale=True):
        if scale:
            # Layer-wise scaling factor (mean absolute value)
            alpha = weight.abs().mean()
        else:
            alpha = weight.new_tensor(1.0)

        ctx.save_for_backward(weight, alpha)
        return weight.sign() * alpha

    @staticmethod
    def backward(ctx, grad_output):
        weight, alpha = ctx.saved_tensors
        # STE: pass gradients unchanged
        grad_weight = grad_output.clone()
        return grad_weight, None


class BinaryActivation(Function):
    """
    Optional activation binarization.
    Not used in main experiments.
    """

    @staticmethod
    def forward(ctx, input):
        return input.sign()

    @staticmethod
    def backward(ctx, grad_output):
        # STE
        return grad_output.clone()


# ============================================================
#  Binary Layers (Drop-in Replacements)
# ============================================================

class BinaryLinear(nn.Linear):
    """
    Binary Linear Layer:
    - Weight: binarized (with optional scaling)
    - Activation: full precision by default
    """

    def __init__(self, *args, binary_scale=True, binary_activation=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.binary_scale = binary_scale
        self.binary_activation = binary_activation

    def forward(self, input):
        if self.binary_activation:
            input = BinaryActivation.apply(input)

        bw = BinaryWeight.apply(self.weight, self.binary_scale)
        return F.linear(input, bw, self.bias)


class BinaryConv2d(nn.Conv2d):
    """
    Binary Convolution Layer:
    - Weight: binarized (with optional scaling)
    - Activation: full precision by default
    """

    def __init__(self, *args, binary_scale=True, binary_activation=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.binary_scale = binary_scale
        self.binary_activation = binary_activation

    def forward(self, input):
        if self.binary_activation:
            input = BinaryActivation.apply(input)

        bw = BinaryWeight.apply(self.weight, self.binary_scale)
        return F.conv2d(
            input, bw, self.bias,
            self.stride, self.padding,
            self.dilation, self.groups
        )


# ============================================================
#  Utilities
# ============================================================

def convert_to_binary(module, binary_scale=True, binary_activation=False):
    """
    Utility function to convert a given nn.Module to its binary counterpart
    (only Conv2d and Linear layers).

    Example:
        model = convert_to_binary(model)
    """
    for name, child in module.named_children():
        if isinstance(child, nn.Conv2d):
            binary_layer = BinaryConv2d(
                child.in_channels,
                child.out_channels,
                kernel_size=child.kernel_size,
                stride=child.stride,
                padding=child.padding,
                dilation=child.dilation,
                groups=child.groups,
                bias=(child.bias is not None),
                binary_scale=binary_scale,
                binary_activation=binary_activation,
            )
            binary_layer.weight.data.copy_(child.weight.data)
            if child.bias is not None:
                binary_layer.bias.data.copy_(child.bias.data)
            setattr(module, name, binary_layer)

        elif isinstance(child, nn.Linear):
            binary_layer = BinaryLinear(
                child.in_features,
                child.out_features,
                bias=(child.bias is not None),
                binary_scale=binary_scale,
                binary_activation=binary_activation,
            )
            binary_layer.weight.data.copy_(child.weight.data)
            if child.bias is not None:
                binary_layer.bias.data.copy_(child.bias.data)
            setattr(module, name, binary_layer)

        else:
            convert_to_binary(child, binary_scale, binary_activation)

    return module
