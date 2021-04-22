#   Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# TODO: define random functions  

from ..fluid import core
from ..fluid.framework import in_dygraph_mode, Variable, convert_np_dtype_to_dtype_
from ..fluid.layer_helper import LayerHelper
from ..fluid.data_feeder import check_variable_and_dtype, check_type, check_dtype, check_shape
from ..fluid.layers import utils
import paddle


def read_file(filename, name=None):
    """
    This OP returns a Tensor filled with random values sampled from a uniform
    distribution in the range [0, 1), with ``shape`` and ``dtype``.

    Args:
        shape (list|tuple|Tensor): The shape of the output Tensor. If ``shape``
            is a list or tuple, the elements of it should be integers or Tensors
            (with the shape [1], and the data type int32 or int64). If ``shape``
            is a Tensor, it should be a 1-D Tensor(with the data type int32 or
            int64).
        dtype (str|np.dtype, optional): The data type of the output Tensor.
            Supported data types: float32, float64.
            Default is None, use global default dtype (see ``get_default_dtype``
            for details).
        name (str, optional): The default value is None. Normally there is no
            need for user to set this property. For more information, please
            refer to :ref:`api_guide_Name`.

    Returns:
        Tensor: A Tensor filled with random values sampled from a uniform
        distribution in the range [0, 1), with ``shape`` and ``dtype``.

    Examples:
        .. code-block:: python

            import paddle

    """
    # if dtype is None:
    #     dtype = paddle.framework.get_default_dtype()
    #     if dtype not in ['float32', 'float64']:
    #         raise TypeError(
    #             "uniform/rand only supports [float32, float64], but the default dtype is {}".
    #             format(dtype))

    # if not isinstance(dtype, core.VarDesc.VarType):
    #     dtype = convert_np_dtype_to_dtype_(dtype)

    if in_dygraph_mode():
        # shape = utils.convert_shape_to_list(shape)
        return core.ops.read_file('filename', filename)

    # check_type(shape, 'shape', (list, tuple, Variable), 'uniform/rand')
    # check_dtype(dtype, 'dtype', ('float32', 'float64'), 'uniform/rand')

    inputs = dict()
    attrs = {'filename': filename}
    # utils.get_shape_tensor_inputs(
    #     inputs=inputs, attrs=attrs, shape=shape, op_type='uniform/rand')

    helper = LayerHelper("read_file", **locals())
    out = helper.create_variable_for_type_inference('uint8')
    helper.append_op(
        type="read_file", inputs=inputs, attrs=attrs, outputs={"Out": out})

    return out


def decode_jpeg(x, name=None):
    """
    This OP returns a Tensor filled with random values sampled from a uniform
    distribution in the range [0, 1), with ``shape`` and ``dtype``.

    Args:
        shape (list|tuple|Tensor): The shape of the output Tensor. If ``shape``
            is a list or tuple, the elements of it should be integers or Tensors
            (with the shape [1], and the data type int32 or int64). If ``shape``
            is a Tensor, it should be a 1-D Tensor(with the data type int32 or
            int64).
        dtype (str|np.dtype, optional): The data type of the output Tensor.
            Supported data types: float32, float64.
            Default is None, use global default dtype (see ``get_default_dtype``
            for details).
        name (str, optional): The default value is None. Normally there is no
            need for user to set this property. For more information, please
            refer to :ref:`api_guide_Name`.

    Returns:
        Tensor: A Tensor filled with random values sampled from a uniform
        distribution in the range [0, 1), with ``shape`` and ``dtype``.

    Examples:
        .. code-block:: python

            import paddle

    """
    # if dtype is None:
    #     dtype = paddle.framework.get_default_dtype()
    #     if dtype not in ['float32', 'float64']:
    #         raise TypeError(
    #             "uniform/rand only supports [float32, float64], but the default dtype is {}".
    #             format(dtype))

    # if not isinstance(dtype, core.VarDesc.VarType):
    #     dtype = convert_np_dtype_to_dtype_(dtype)

    if in_dygraph_mode():
        # shape = utils.convert_shape_to_list(shape)
        print('enter cpp code!!')
        print('x:', x)
        return core.ops.decode_jpeg(x)

    # check_type(shape, 'shape', (list, tuple, Variable), 'uniform/rand')
    # check_dtype(dtype, 'dtype', ('float32', 'float64'), 'uniform/rand')

    inputs = {'X': x}
    attrs = {}
    # utils.get_shape_tensor_inputs(
    #     inputs=inputs, attrs=attrs, shape=shape, op_type='uniform/rand')

    helper = LayerHelper("decode_jpeg", **locals())
    out = helper.create_variable_for_type_inference('uint8')
    helper.append_op(
        type="decode_jpeg", inputs=inputs, attrs=attrs, outputs={"Out": out})

    return out
