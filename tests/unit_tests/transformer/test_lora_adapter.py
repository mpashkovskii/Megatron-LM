# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.

from functools import partial
from typing import Any, Callable, Generator

import pytest
import torch
from torch.optim import Adam, SGD
import transformer_engine

from megatron.core.extensions.transformer_engine import (
    TEColumnParallelLinear,
)
from megatron.core.model_parallel_config import ModelParallelConfig
from megatron.core.tensor_parallel import get_cuda_rng_tracker
from megatron.core.tensor_parallel.layers import ColumnParallelLinear
from megatron.core.tensor_parallel.mappings import _gather_along_first_dim
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer.lora_adapter import LoraAdapter
from megatron.core.transformer.transformer_config import TransformerConfig

from tests.unit_tests.test_utilities import Utils


@pytest.mark.parametrize('pipeline_model_parallel_size', [1, 2])
@pytest.mark.parametrize("expert_tensor_parallel_size, tensor_model_parallel_size, sequence_parallel", [
    (1, 1, False),
    (2, 1, False),
    (1, 2, False),
    (1, 2, True),
    (2, 2, True),
])
@pytest.mark.parametrize("base_layer_constructor", [
    partial(ColumnParallelLinear),
    partial(TEColumnParallelLinear, gather_output=False, is_expert=False),
])
class TestLoraAdapterWithLoraLayers:

    @pytest.fixture(scope='function', autouse=True)
    def setup_and_teardown(self, expert_tensor_parallel_size: int, pipeline_model_parallel_size: int, tensor_model_parallel_size: int, sequence_parallel: bool, base_layer_constructor: Callable) -> Generator[Any, Any, Any]:
        Utils.initialize_model_parallel(
            expert_tensor_parallel_size=expert_tensor_parallel_size,
            pipeline_model_parallel_size=pipeline_model_parallel_size,
            tensor_model_parallel_size=tensor_model_parallel_size,
        )
        model_parallel_cuda_manual_seed(123)
        self.input_size = 2
        self.output_size = 2
        self.rank = 2
        self.alpha = 32
        self.config = TransformerConfig(
            num_layers=1,
            add_bias_linear=False,
            hidden_size=12,
            num_attention_heads=4,
            num_moe_experts=expert_tensor_parallel_size if expert_tensor_parallel_size > 1 else None,
            expert_model_parallel_size=expert_tensor_parallel_size,
            pipeline_model_parallel_size=pipeline_model_parallel_size,
            tensor_model_parallel_size=tensor_model_parallel_size,
            sequence_parallel=sequence_parallel,
            pipeline_dtype=torch.float32,
        )
        
        # Why do we need to call Linear constructor to make tests work?
        transformer_engine.pytorch.Linear(
            in_features=1,
            out_features=1,
            sequence_parallel=sequence_parallel,
            tp_size=tensor_model_parallel_size,

            # get_rng_state_tracker=None  # BREAKS!
            get_rng_state_tracker=(
                get_cuda_rng_tracker if get_cuda_rng_tracker().is_initialized() else None
            ),
            
            # # Probably not important params
            # fuse_wgrad_accumulation=self.config.gradient_accumulation_fusion,
            # tp_group=get_tensor_model_parallel_group(check_initialized=False),
            # init_method=None,
            # bias=False,
            # return_bias=False,
            # parallel_mode=None,
            # params_dtype=torch.float32,
            # device=torch.cuda.current_device(),
            # rng_tracker_name=None,
        )

        base_layer = base_layer_constructor(
            input_size=self.input_size,
            output_size=self.output_size,
            config=self.config,
            bias=False,
            skip_bias_add=True,
            init_method=torch.nn.init.zeros_,
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

    # torchrun --nproc_per_node=2 -m pytest --color=yes -k test_forward tests/unit_tests/transformer/test_lora_adapter.py
    @pytest.mark.parametrize("optimizer_constructor", [
        # We have to use MegatronOptimizer sub classes instead of torch.optim.Optimizer
        Adam,
        # SGD,
    ])
    def test_forward(self, optimizer_constructor: torch.optim.Optimizer) -> None:
        batch_size = 1
        # input_data = torch.rand(batch_size, self.input_size).cuda()  # WORKS UNSTABLE (FLAKY)! Why random values doesn't work?
        val = torch.distributed.get_rank() + 1.
        input_data = torch.full((batch_size, self.input_size), val).cuda()

        model = self.lora_adapter
        layers_to_check = [
            # "lora_a",
            "lora_b",
        ]
        original_weights = {
            layer: getattr(model, layer).weight.clone().detach()
            for layer in layers_to_check
        }
        optimizer = optimizer_constructor(model.parameters())
        
        # https://stackoverflow.com/questions/54447084/how-to-properly-update-the-weights-in-pytorch
        # https://hmkcode.com/ai/backpropagation-step-by-step/
        for idx in range(2):
            optimizer.zero_grad()
            output, _ = model(input_data)
            (1 - output.mean()).backward()
            optimizer.step()

        assert model.base_layer.weight.sum() == 0, "Base layer frozen weights were updated"
        assert torch.all(model.lora_b.weight), "LoRA B layer weights weren't updated"
        
        for layer in layers_to_check:
            original_weight = original_weights[layer]
            current_local_weight = getattr(model, layer).weight
            assert not torch.allclose(original_weight, current_local_weight), f"Local weight wasn't updated for {layer}"
            
            full_weight = _gather_along_first_dim(current_local_weight)
            print(f"FINAL: {current_local_weight}")
            print(f"FULL:  {full_weight.data}\n")
            for idx, weight in enumerate(torch.split(full_weight, current_local_weight.shape[0])):
                assert torch.allclose(weight, current_local_weight), f"Weight on rank {idx} doesn't match for {layer}"
        
        # assert False, "DEBUG!"


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


# torchrun --nproc_per_node=2 -m pytest --color=yes -k test_Linear_classes tests/unit_tests/transformer/test_lora_adapter.py
@pytest.mark.parametrize("constructor", [
    # Linear,
    partial(ColumnParallelLinear, gather_output=False),
    # RowParallelLinear,
])
@pytest.mark.parametrize("tensor_model_parallel_size,sequence_parallel", [(2, False), (2, True)])
@pytest.mark.parametrize("optimizer_constructor", [Adam, SGD])
def test_Linear_classes(constructor: Callable, tensor_model_parallel_size: int, sequence_parallel: bool, optimizer_constructor: torch.optim.Optimizer) -> None: 
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
    
    batch_size = 1
    input_size = 4
    output_size = 8
    input_data = torch.rand(batch_size, input_size).cuda()

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
    (1 - output.mean()).backward()
    optimizer_constructor(model.parameters()).step()

    full_weight = _gather_along_first_dim(model.weight)
    assert not torch.allclose(original_weight, model.weight), "Local weight wasn't updated"
    for idx, weight in enumerate(torch.split(full_weight, model.weight.shape[0])):
        assert torch.allclose(weight, model.weight), f"Weight on rank {idx} doesn't match"
    
    Utils.destroy_model_parallel()


def _get_nparams(module: torch.nn.Module):
    return sum([p.numel() for p in module.parameters()])