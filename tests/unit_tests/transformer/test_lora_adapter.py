# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.

import pytest
import torch
from torch.optim import SGD

from megatron.core.tensor_parallel.layers import ColumnParallelLinear
from megatron.core.tensor_parallel.mappings import _gather_along_first_dim
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer.lora_adapter import LoraAdapter
from megatron.core.transformer.transformer_config import TransformerConfig

from tests.unit_tests.test_utilities import Utils


@pytest.mark.parametrize(
    "expert_tensor_parallel_size, pipeline_model_parallel_size, tensor_model_parallel_size, sequence_parallel",
    [
        (1, 1, 2, False),
    ]
)
class TestLoraAdapterWithLoraLayers:

    @pytest.fixture(scope='function', autouse=True)
    def setup_and_teardown(self, expert_tensor_parallel_size: int, pipeline_model_parallel_size: int, tensor_model_parallel_size: int, sequence_parallel: bool):
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
        base_layer = ColumnParallelLinear(
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

    def test_constructor(self):
        parallel_output_size = int(self.output_size / self.config.tensor_model_parallel_size)
        assert _get_nparams(self.lora_adapter) == self.input_size * parallel_output_size \
            + self.input_size * self.rank \
            + self.rank * parallel_output_size
    
        INPUT_INDEX = 1
        OUTPUT_INDEX = 0
        assert self.lora_adapter.base_layer.weight.shape[INPUT_INDEX] == self.lora_adapter.lora_a.weight.shape[INPUT_INDEX]
        assert self.lora_adapter.base_layer.weight.shape[OUTPUT_INDEX] == self.lora_adapter.lora_b.weight.shape[OUTPUT_INDEX]

    def test_load_state_dict(self):
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

    def test_forward(self):
        batch_size = 1
        input_data = torch.rand(batch_size, self.input_size).cuda()
        print(f"INPUT: {input_data}\n")

        def print_adapter(adapter: LoraAdapter) -> str:
            return f"""
  A:
    weight = {adapter.lora_a.weight.data}
    grad = {adapter.lora_a.weight.grad.data}

  B:
    weight = {adapter.lora_b.weight.data}
    grad = {adapter.lora_b.weight.grad.data}

"""

        original_weights = {
            "lora_a": self.lora_adapter.lora_a.weight.clone().detach(),
            "lora_b": self.lora_adapter.lora_b.weight.clone().detach(),
        }
        
        # https://stackoverflow.com/questions/54447084/how-to-properly-update-the-weights-in-pytorch
        # https://hmkcode.com/ai/backpropagation-step-by-step/
        optimizer = SGD(self.lora_adapter.parameters())
        print(list(self.lora_adapter.parameters()))
        for idx in range(2):
            optimizer.zero_grad()
            # forward_backward_no_pipelining = get_forward_backward_func()
            # See docs for `get_forward_backward_func`
            # forward_backward_no_pipelining()
            output, _ = self.lora_adapter(input_data)
            (1 - output.mean()).backward()
            # backward_step()
            # finalize_model_grads() ?
            print(f"ITERATION {idx} -----------------------------------------------")
            print(f"OUTPUT: {output.data}\n{print_adapter(self.lora_adapter)}")
            optimizer.step()
        
        print(f"FINAL:{print_adapter(self.lora_adapter)}")

        assert self.lora_adapter.base_layer.weight.sum() == 0, "Base layer frozen weights were updated"
        assert torch.all(self.lora_adapter.lora_b.weight), "LoRA B layer weights weren't updated"
        
        for layer in [
                # "lora_a",
                "lora_b"
        ]:
            original_weight = original_weights[layer]
            current_local_weight = self.lora_adapter.__getattr__(layer).weight
            assert not torch.allclose(original_weight, current_local_weight), f"Local weight wasn't updated for {layer}"
            
            full_weight = _gather_along_first_dim(current_local_weight)
            print(f"FULL:\n  {full_weight.data}\n")
            for idx, weight in enumerate(torch.split(full_weight, current_local_weight.shape[0])):
                assert torch.allclose(weight, current_local_weight), f"Weight on rank {idx} doesn't match for {layer}"
        
        assert False, "DEBUG!"


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



# # Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.

# from functools import partial
# from typing import Callable

# import pytest
# import torch
# from torch.optim import Adam

# from megatron.core.model_parallel_config import ModelParallelConfig
# from megatron.core.tensor_parallel.layers import ColumnParallelLinear
# from megatron.core.tensor_parallel.mappings import _gather_along_first_dim
# from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed

# from tests.unit_tests.test_utilities import Utils

# # 
# # Command to run that set of test:
# #  torchrun --nproc_per_node=2 -m pytest --color=yes -k test_forward tests/unit_tests/transformer/test_lora_adapter.py
# # 
# @pytest.mark.parametrize("constructor", [
#     # Linear,
#     partial(ColumnParallelLinear, gather_output=False),
#     # RowParallelLinear,
# ])
# @pytest.mark.parametrize("tensor_model_parallel_size,sequence_parallel", [(2, False), (2, True)])
# def test_Linear_classes(constructor: Callable, tensor_model_parallel_size: int, sequence_parallel: bool):
#     Utils.initialize_model_parallel(
#         expert_tensor_parallel_size=1,
#         pipeline_model_parallel_size=1,
#         tensor_model_parallel_size=tensor_model_parallel_size,
#     )
#     model_parallel_cuda_manual_seed(123)

#     config = ModelParallelConfig(
#         expert_model_parallel_size=1,
#         pipeline_model_parallel_size=1,
#         tensor_model_parallel_size=tensor_model_parallel_size,
#         sequence_parallel=sequence_parallel,
#     )
    
#     batch_size = 1
#     input_size = 4
#     output_size = 8
#     input_data = torch.rand(batch_size, input_size).cuda()

#     model = constructor(
#         input_size=input_size,
#         output_size=output_size,
#         config=config,
#         init_method=torch.nn.init.ones_,
#         bias=False,
#         skip_bias_add=True,
#     )
#     original_weight = model.weight.clone().detach()

#     output, _ = model(input_data)
#     # Do we need to gather output? ColumnParallelLinear with gather_output=True does it already.
#     # What about RowLinearParallel and Linear?
#     # output = gather_from_tensor_model_parallel_region(output)

#     # Following line crashes with:
#     # - sequence_parallel: False
#     #     FAILED tests/unit_tests/tensor_parallel/test_layers.py::test_Linear_classes[2-False-constructor0] - RuntimeError:
#     #     Function LinearWithGradAccumulationAndAsyncCommunicationBackward returned an invalid gradient at index 1 - got []
#     #     but expected shape compatible with [4, 4]
#     # - sequence_parallel: True
#     #     RuntimeError: mat1 and mat2 shapes cannot be multiplied (1x8 and 4x4)
#     output.sum().backward()
    
#     optimizer = Adam(model.parameters(), lr=0.01)
#     optimizer.step()

#     full_weight = _gather_along_first_dim(model.weight)
#     assert not torch.allclose(original_weight, model.weight), "Local weight wasn't updated"
#     for idx, weight in enumerate(torch.split(full_weight, model.weight.shape[0])):
#         assert torch.allclose(weight, model.weight), f"Weight on rank {idx} doesn't match"
    
#     Utils.destroy_model_parallel()