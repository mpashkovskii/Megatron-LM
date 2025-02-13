# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
from functools import partial
from typing import Callable

import pytest
import torch
from torch.optim import Adam

from megatron.core.model_parallel_config import ModelParallelConfig
from megatron.core.tensor_parallel.layers import ColumnParallelLinear, linear_with_frozen_weight
from megatron.core.tensor_parallel.mappings import _gather_along_first_dim, gather_from_tensor_model_parallel_region
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed

from tests.unit_tests.test_utilities import Utils


@pytest.mark.parametrize("tensor_parallel,allreduce_dgrad", [(1, False), (8, True)])
def test_LinearWithFrozenWeight(tensor_parallel, allreduce_dgrad):
    Utils.initialize_model_parallel(tensor_parallel, 1)

    size_per_partition = int(8 / tensor_parallel)

    # Input is an 8x8 identity matrix.
    input_data = torch.eye(8).cuda()
    input_data.requires_grad = True

    # Weight is an 8x8 matrix of all ones. If tensor parallelism > 1, the weight is partitioned evenly across GPUs.
    weight = torch.ones((size_per_partition, 8)).cuda()

    # Bias is a vector of length 8 of all zeros. If tensor parallelism > 1, the bias is partitioned evenly across GPUs
    bias = torch.zeros((size_per_partition)).cuda()

    gradient_accumulation_fusion = False
    sequence_parallel = False
    grad_output_buffer = None
    wgrad_deferral_limit = None

    output_parallel = linear_with_frozen_weight(
        input_data,
        weight,
        bias,
        gradient_accumulation_fusion,
        allreduce_dgrad,
        sequence_parallel,
        grad_output_buffer,
        wgrad_deferral_limit,
    )
    output = gather_from_tensor_model_parallel_region(
        output_parallel
    )  # no-op if tensor_parallel == 1.
    output.sum().backward()

    expected_output = torch.ones(8).cuda()
    expected_grad = 8 * torch.ones(8).cuda()

    assert torch.allclose(output, expected_output)
    assert torch.allclose(input_data.grad, expected_grad)

    Utils.destroy_model_parallel()


# 
# Command to run that set of test:
#  torchrun --nproc_per_node=2 -m pytest --color=yes -k test_Linear_classes tests/unit_tests/tensor_parallel/test_layers.py
# 
@pytest.mark.parametrize("constructor", [
    # Linear,
    partial(ColumnParallelLinear, gather_output=True),
    # RowParallelLinear,
])
@pytest.mark.parametrize("tensor_model_parallel_size,sequence_parallel", [(2, False), (2, True)])
def test_Linear_classes(constructor: Callable, tensor_model_parallel_size: int, sequence_parallel: bool):
    Utils.initialize_model_parallel(
        expert_tensor_parallel_size=1,
        pipeline_model_parallel_size=1,
        tensor_model_parallel_size=tensor_model_parallel_size,
    )
    model_parallel_cuda_manual_seed(123)

    config = ModelParallelConfig(
        expert_model_parallel_size=1,
        pipeline_model_parallel_size=1,
        tensor_model_parallel_size=tensor_model_parallel_size,
        sequence_parallel=sequence_parallel,
    )
    
    input_size = 4
    output_size = 8
    input_data = torch.rand(input_size).cuda()

    model = constructor(
        input_size=input_size,
        output_size=output_size,
        config=config,
        init_method=torch.nn.init.ones_,
        bias=False,
        skip_bias_add=True,
    )
    original_weight = model.weight.clone().detach()

    output, _ = model(input_data)
    # Do we need to gather output? ColumnParallelLinear with gather_output=True does it already.
    # What about RowLinearParallel and Linear?
    # output = gather_from_tensor_model_parallel_region(output)

    # Following line crashes with:
    # - sequence_parallel: False
    #     FAILED tests/unit_tests/tensor_parallel/test_layers.py::test_Linear_classes[2-False-constructor0] - RuntimeError:
    #     Function LinearWithGradAccumulationAndAsyncCommunicationBackward returned an invalid gradient at index 1 - got []
    #     but expected shape compatible with [4, 4]
    # - sequence_parallel: True
    #     RuntimeError: mat1 and mat2 shapes cannot be multiplied (1x8 and 4x4)
    output.sum().backward()
    
    optimizer = Adam(model.parameters(), lr=0.01)
    optimizer.step()

    full_weight = _gather_along_first_dim(model.weight)
    assert not torch.allclose(original_weight, model.weight), "Local weight wasn't updated"
    for idx, weight in enumerate(torch.split(full_weight, model.weight.shape[0])):
        assert torch.allclose(weight, model.weight), f"Weight on rank {idx} doesn't match"
    
    Utils.destroy_model_parallel()