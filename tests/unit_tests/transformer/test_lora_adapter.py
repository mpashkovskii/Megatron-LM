# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.

from functools import partial
from typing import Any, Callable, Generator

import pytest
import torch
from torch.optim import Adam, SGD
import transformer_engine.pytorch as te

from megatron.core.extensions.transformer_engine import (
    TEColumnParallelLinear,
    TELayerNormColumnParallelLinear,
    TERowParallelLinear,
)
from megatron.core.tensor_parallel import get_cuda_rng_tracker
from megatron.core.tensor_parallel.layers import ColumnParallelLinear, RowParallelLinear
from megatron.core.tensor_parallel.mappings import _gather_along_first_dim
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer.lora_adapter import LoraAdapter
from megatron.core.transformer.transformer_config import TransformerConfig

from tests.unit_tests.test_utilities import Utils


@pytest.mark.parametrize('pipeline_model_parallel_size', [
    1,
    2,
])
@pytest.mark.parametrize(
    "expert_tensor_parallel_size, tensor_model_parallel_size, sequence_parallel",
    [
        # tp=1, Can not use sequence paralllelism without tensor parallelism
        (1, 1, False), 
        (2, 1, False),

        # tp=2
        (1, 2, False),
        (1, 2, True),
        (2, 2, True),  # When using expert parallelism and tensor parallelism, sequence parallelism _must_ be used

        # (1, 2, True),  # For column parallel
        # (1, 2, False),  # For row parallel
    ]
)
@pytest.mark.parametrize("base_layer", [
    partial(ColumnParallelLinear),
    partial(TEColumnParallelLinear, gather_output=False),
    partial(TELayerNormColumnParallelLinear, gather_output=False),
    partial(RowParallelLinear, input_is_parallel=True),
    partial(TERowParallelLinear, input_is_parallel=True),
])
@pytest.mark.parametrize("is_expert", [
    False,

    # - Sequence paralellism has to be off for is_expert=True?
    #     tests/unit_tests/transformer/test_lora_adapter.py::TestLoraAdapterWithLoraLayers::test_forward[True-base_layer_constructor0-1-2-True-2]
    #     /workspace/Megatron-LM/megatron/core/tensor_parallel/layers.py:837: UserWarning: `sequence_parallel` is set to `True`, but tensor model parallel size is 1. Disabling sequence parallel.
    #       warnings.warn(
    #     FAILED tests/unit_tests/transformer/test_lora_adapter.py::TestLoraAdapterWithLoraLayers::test_forward[False-base_layer_constructor0-1-2-True-1] - AssertionError: Weight on rank 1 doesn't match for lora_a
    # - 'Transformer Engine linear layers do not yet support MoE' in TELayerNormColumnParallelLinear
    # True,
])
class TestLoraAdapterWithLoraLayers:

    @pytest.fixture(scope='function', autouse=True)
    def setup_and_teardown(self, expert_tensor_parallel_size: int, pipeline_model_parallel_size: int, tensor_model_parallel_size: int, sequence_parallel: bool, base_layer: Callable, is_expert: bool) -> Generator[Any, Any, Any]:
        Utils.initialize_model_parallel(
            expert_tensor_parallel_size=expert_tensor_parallel_size,
            pipeline_model_parallel_size=pipeline_model_parallel_size,
            tensor_model_parallel_size=tensor_model_parallel_size,
        )
        model_parallel_cuda_manual_seed(123)
        self.input_size = 4
        self.output_size = 8
        self.rank = 2
        self.alpha = 32
        self.config = TransformerConfig(
            # mandatory params
            num_layers=1,
            add_bias_linear=False,
            hidden_size=12,
            num_attention_heads=4,
            num_moe_experts=expert_tensor_parallel_size if expert_tensor_parallel_size > 1 else None,
            pipeline_dtype=torch.float32,
            
            # parallelization params
            expert_model_parallel_size=expert_tensor_parallel_size,
            pipeline_model_parallel_size=pipeline_model_parallel_size,
            tensor_model_parallel_size=tensor_model_parallel_size,
            sequence_parallel=sequence_parallel,
        )

        base_layer = base_layer(
            input_size=self.input_size,
            output_size=self.output_size,
            config=self.config,
            bias=False,
            skip_bias_add=True,
            init_method=torch.nn.init.zeros_,
            is_expert=is_expert,
        )
        self.lora_adapter = LoraAdapter(
            base_layer,
            config=self.config,
            rank=self.rank,
            alpha=self.alpha,
            dropout=0.01,
        )
        yield
        Utils.destroy_model_parallel()

    def test_constructor(self) -> None:
        parallel_output_size = int(self.output_size / self.config.tensor_model_parallel_size)
        assert _get_nparams(self.lora_adapter) == self.input_size * parallel_output_size \
            + self.input_size * self.rank \
            + self.rank * parallel_output_size
    
        INPUT_INDEX = 1
        OUTPUT_INDEX = 0
        assert self.lora_adapter.base_layer.weight.shape[INPUT_INDEX] == self.lora_adapter.lora_a.weight.shape[INPUT_INDEX]
        assert self.lora_adapter.base_layer.weight.shape[OUTPUT_INDEX] == self.lora_adapter.lora_b.weight.shape[OUTPUT_INDEX]

    def test_load_state_dict(self) -> None:
        model = torch.nn.Module()
        model.add_module("output_layer", self.lora_adapter)
        parallel_output_size = int(self.output_size / self.config.tensor_model_parallel_size)
        state_dict = {"output_layer.weight": torch.ones(parallel_output_size, self.input_size)}
        missing_keys, unexpected_keys = model.load_state_dict(state_dict)

        assert missing_keys == []
        assert unexpected_keys == []
        assert self.lora_adapter.base_layer.weight.all()
        assert torch.any(self.lora_adapter.lora_a.weight)
        assert self.lora_adapter.lora_b.weight.sum() == 0

    # Requires:
    #   if ctx.parallel_mode is None and get_distributed_world_size(ctx.tp_group) > 1:
    #       torch.distributed.all_reduce(wgrad, group=ctx.tp_group, op=torch.distributed.ReduceOp.AVG)
    # in /opt/conda/envs/py_3.10/lib/python3.10/site-packages/transformer_engine/pytorch/module/linear.py:569
    # 
    # To run use:
    #   CUDA_DEVICE_MAX_CONNECTIONS=1 torchrun --nproc_per_node=8 -m pytest --color=yes -k test_forward tests/unit_tests/transformer/test_lora_adapter.py
    def test_forward(self) -> None:
        batch_size = 1
        sequence_length = (
            self.input_size // self.config.tensor_model_parallel_size
            if type(self.lora_adapter.base_layer) in [RowParallelLinear, TERowParallelLinear]
            else self.input_size
        )
        input_data = torch.rand(batch_size, sequence_length).cuda()

        model = self.lora_adapter
        
        synced_layers = []
        if type(self.lora_adapter.base_layer) in [ColumnParallelLinear, TEColumnParallelLinear, TELayerNormColumnParallelLinear]:
            # Linear has to be synced
            synced_layers.append("lora_a")
            if self.config.sequence_parallel:
                # Column/Row parallel layers has be synced only for sequence_parallel
                synced_layers.append("lora_b")
        
        if type(self.lora_adapter.base_layer) in [RowParallelLinear, TERowParallelLinear]:
            # Linear has to be synced
            synced_layers.append("lora_b")
            if self.config.sequence_parallel:
                # Column/Row parallel layers has be synced only for sequence_parallel
                synced_layers.append("lora_a")
        
        original_weights = {
            layer: getattr(model, layer).weight.clone().detach()
            for layer in synced_layers
        }
        # optimizer = SGD(model.parameters())  # Causes fails! We have to use MegatronOptimizer sub classes instead of torch.optim.Optimizer
        optimizer = Adam(model.parameters())  # We have to use MegatronOptimizer sub classes instead of torch.optim.Optimizer
        
        # To propagate gradients through the zero-initialized weights we need at least two iterations.
        # Let's do 100 to see any error accumulation.
        for idx in range(100):
            optimizer.zero_grad()
            output, _ = model(input_data)
            (1 - output.mean()).backward()
            print(f"{model.lora_a.weight.grad.data=}")
            optimizer.step()

        assert model.base_layer.weight.sum() == 0, "Base layer frozen weights were updated"
        assert torch.all(model.lora_b.weight), "LoRA B layer weights weren't updated"
        
        for layer in synced_layers:
            original_weight = original_weights[layer]
            current_local_weight = getattr(model, layer).weight
            assert not torch.allclose(original_weight, current_local_weight), f"Local weight wasn't updated for {layer}"
            
            full_weight = _gather_along_first_dim(current_local_weight)
            print(f"FINAL: {current_local_weight.data}")
            print(f"FULL: {full_weight.data}\n")
            for idx, weight in enumerate(torch.split(full_weight, current_local_weight.shape[0])):
                assert torch.allclose(weight, current_local_weight), f"Weight on rank {idx} doesn't match for {layer}"


class TestLoraAdapterWithUnknownBaseLayer:

    @pytest.fixture(scope='function', autouse=True)
    def setup_and_teardown(self):
        Utils.initialize_model_parallel(1, 1)
        model_parallel_cuda_manual_seed(123)

        self.input_size = 64
        self.output_size = 64
        self.rank = 16
        self.alpha = 32
        self.transformer_config = TransformerConfig(
            num_layers=1,
            add_bias_linear=False,
            hidden_size=12,
            num_attention_heads=4,
        )
        self.base_layer = torch.nn.Linear(self.input_size, self.output_size, bias=False)
        self.lora_adapter = LoraAdapter(
            self.base_layer,
            config=self.transformer_config,
            rank=self.rank,
            alpha=self.alpha,
            dropout=0.01,
        )
        yield
        Utils.destroy_model_parallel()

    def test_constructor(self):
        assert _get_nparams(self.lora_adapter) == self.input_size * self.output_size


def _get_nparams(module: torch.nn.Module):
    return sum([p.numel() for p in module.parameters()])
