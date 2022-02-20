from torch.nn.functional import conv2d, pad

from e2cnn.nn import R2Conv
from e2cnn.nn import init
from e2cnn.nn import FieldType
from e2cnn.nn import GeometricTensor
from e2cnn.gspaces import *

from e2cnn.nn.modules.equivariant_module import EquivariantModule

from e2cnn.nn.modules.r2_conv.basisexpansion import BasisExpansion
from e2cnn.nn.modules.r2_conv.basisexpansion_blocks import BlocksBasisExpansion
from e2cnn.nn.modules.r2_conv.r2convolution import compute_basis_params

from typing import Callable, Union, Tuple, List

import torch
from torch.nn import Parameter
import numpy as np
import math

class R2ConvDF(torch.nn.Module):
    def __init__(self,
                 in_type: FieldType,
                 out_type: FieldType,
                 kernel_size: int,
                 padding: int = 0,
                 stride: int = 1,
                 dilation: int = 1,
                 groups: int = 1,
                 bias: bool = True,
                 ):
        super().__init__()
        # super().__init__(in_type, out_type, kernel_size, padding, stride, dilation, padding_mode, groups, bias, initialize=False)

        self.space = in_type.gspace
        self.in_type = in_type
        self.out_type = out_type

        self.kernel_size = kernel_size
        self.stride = stride
        self.dilation = dilation
        self.padding = padding
        self.groups = groups

        if bias:
            # bias can be applied only to trivial irreps inside the representation
            # to apply bias to a field we learn a bias for each trivial irreps it contains
            # and, then, we transform it with the change of basis matrix to be able to apply it to the whole field
            # this is equivalent to transform the field to its irreps through the inverse change of basis,
            # sum the bias only to the trivial irrep and then map it back with the change of basis

            # count the number of trivial irreps
            trivials = 0
            for r in self.out_type:
                for irr in r.irreps:
                    if self.out_type.fibergroup.irreps[irr].is_trivial():
                        trivials += 1

            # if there is at least 1 trivial irrep
            if trivials > 0:

                # matrix containing the columns of the change of basis which map from the trivial irreps to the
                # field representations. This matrix allows us to map the bias defined only over the trivial irreps
                # to a bias for the whole field more efficiently
                bias_expansion = torch.zeros(self.out_type.size, trivials)

                p, c = 0, 0
                for r in self.out_type:
                    pi = 0
                    for irr in r.irreps:
                        irr = self.out_type.fibergroup.irreps[irr]
                        if irr.is_trivial():
                            bias_expansion[p:p + r.size, c] = torch.tensor(r.change_of_basis[:, pi])
                            c += 1
                        pi += irr.size
                    p += r.size

                self.bias_expansion = bias_expansion
            else:
                self.bias = None
                self.expanded_bias = None
        else:
            self.bias = None
            self.expanded_bias = None

        grid, basis_filter, rings, sigma, maximum_frequency = compute_basis_params(kernel_size,
                                                                                   None,
                                                                                   None,
                                                                                   None,
                                                                                   dilation,
                                                                                   None)
        self._basisexpansion = BlocksBasisExpansion(in_type, out_type,
                                                    grid,
                                                    sigma=sigma,
                                                    rings=rings,
                                                    maximum_offset=None,
                                                    maximum_frequency=maximum_frequency,
                                                    basis_filter=basis_filter,
                                                    recompute=False)

        # self.weights = None

    @property
    def basisexpansion(self) -> BasisExpansion:
        r"""
        Submodule which takes care of building the filter.

        It uses the learnt ``weights`` to expand a basis and returns a filter in the usual form used by conventional
        convolutional modules.
        It uses the learned ``weights`` to expand the kernel in the G-steerable basis and returns it in the shape
        :math:`(c_\text{out}, c_\text{in}, s^2)`, where :math:`s` is the ``kernel_size``.

        """
        return self._basisexpansion

    def forwardDynamicFilter(self, input, weights, bias=None):
        _filter = self.basisexpansion(weights)
        _filter = _filter.reshape(_filter.shape[0], _filter.shape[1], self.kernel_size, self.kernel_size)

        if bias is None:
            _bias = None
        else:
            _bias = self.bias_expansion.to(bias.device) @ bias

        output = conv2d(input.tensor, _filter,
                        stride=self.stride,
                        padding=self.padding,
                        dilation=self.dilation,
                        groups=self.groups,
                        bias=_bias)
        return GeometricTensor(output, self.out_type)
