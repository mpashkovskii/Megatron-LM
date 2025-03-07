# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.

from functools import partial
from typing import Callable

import pytest
import torch
from torch.optim import Adam

from code.TransformerEngine.transformer_engine.pytorch.utils import init_method_normal
from megatron.core.extensions.transformer_engine import (
    TEColumnParallelLinear,
    TELinear,
)
from megatron.core.tensor_parallel.mappings import _gather_along_first_dim
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer.transformer_config import TransformerConfig

from tests.unit_tests.test_utilities import Utils


# torchrun --nproc_per_node=8 -m pytest --color=yes tests/unit_tests/extensions/test_transformer_engine.py
@pytest.mark.parametrize("constructor", [
    partial(TELinear, parallel_mode=None, skip_weight_param_allocation=False),
    partial(TEColumnParallelLinear, gather_output=False, is_expert=False),
    partial(TEColumnParallelLinear, gather_output=False, is_expert=True),
    # partial(TERowParallelLinear),  # Requires a separate setup (input)
])
@pytest.mark.parametrize(
    "tensor_model_parallel_size, sequence_parallel", 
    [
        (2, False),
        (2, True)
    ]
)
def test_linear_column_parallel_classes(constructor: Callable, tensor_model_parallel_size: int, sequence_parallel: bool) -> None: 
    Utils.initialize_model_parallel(
        expert_tensor_parallel_size=1,
        pipeline_model_parallel_size=1,
        tensor_model_parallel_size=tensor_model_parallel_size,
    )
    model_parallel_cuda_manual_seed(123)

    config = TransformerConfig(
        # mandatory params
        num_layers=1,
        add_bias_linear=False,
        hidden_size=12,
        num_attention_heads=4,
        pipeline_dtype=torch.float32,

        # parallelization params
        expert_model_parallel_size=1,
        pipeline_model_parallel_size=1,
        tensor_model_parallel_size=tensor_model_parallel_size,
        sequence_parallel=sequence_parallel,
    )
    
    batch_size = 1
    input_size = 4
    output_size = 8
    input_data = torch.rand(batch_size, input_size).cuda()

    model = constructor(
        input_size=input_size,
        output_size=output_size,
        config=config,
        init_method=config.init_method,
        bias=False,
        skip_bias_add=True,
    )
    # Sync model weights across all ranks.
    # - TE does that in TransformerEngine/tests/pytorch/distributed/run_numerics.py
    # - Who is doing that in Megatron-LM?

    original_weight = model.weight.clone().detach()

    output, _ = model(input_data)
    (1 - output.mean()).backward()
    Adam(model.parameters()).step()

    full_weight = _gather_along_first_dim(model.weight)
    assert not torch.allclose(original_weight, model.weight), "Local weight wasn't updated"
    for idx, weight in enumerate(torch.split(full_weight, model.weight.shape[0])):
        assert torch.allclose(weight, model.weight), f"Weight on rank {idx} doesn't match"
    
    Utils.destroy_model_parallel()

