# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# Experiments to perform GPU code generation using reinforcement learning:
# - https://arxiv.org/pdf/2402.03300
# - https://huggingface.co/docs/trl/main/en/grpo_trainer
# - https://sakana.ai/ai-cuda-engineer/
# - https://predibase.com/blog/teaching-ai-to-write-gpu-code-a-deep-dive-into-reinforcement-fine-tuning

import os
import sys
from functools import partial

sys.path.append(
    os.path.abspath(
        os.path.join(
            os.path.dirname(__file__),
            os.path.pardir,
            os.path.pardir
        )
    )
)

import torch
from argparse import ArgumentParser
from typing import Callable

from megatron.core.enums import ModelType
from megatron.core.models.gpt import GPTModel
from megatron.training import pretrain

# LoRA not neccesarily needed
from lora_mixtral import add_lora_args, lora_model_provider
from pretrain_gpt import SPIKY_LOSS_PERC, get_batch, train_valid_test_datasets_provider


# To be implemented
# - RewardFunc = Callable[[List[?], List[?]], List[float]]
# - `def train_valid_test_datasets_provider` or `get_batch` that has to repeat
#   the sample for num_generations times


def grpo_loss_func(loss_mask: torch.Tensor, output_tensor: torch.Tensor) -> tuple[torch.Tensor, int, dict[str, tuple[float, float]]]:
    """Loss function.

    TODO:
      - implement https://github.com/huggingface/trl/blob/main/trl/trainer/grpo_trainer.py#L871
      - use pretrain_gpt::loss_func as a reference
    """
    raise NotImplementedError('GRPO loss function is not implemented yet')


def forward_step(data_iterator, model: GPTModel) -> tuple[torch.Tensor, Callable]:
    tokens, labels, loss_mask, attention_mask, position_ids = get_batch(data_iterator)
    output_tensor = model(tokens, position_ids, attention_mask, labels=labels)
    return output_tensor, partial(grpo_loss_func, loss_mask)


def add_grpo_args(parser: ArgumentParser) -> ArgumentParser:
    group = parser.add_argument_group(title='GRPO data preprocessing')
    group.add_argument('--num-generations', type=int,
                       help='Number of generations to sample. The global batch size'
                            '(num_processes * per_device_batch_size)')

    group = parser.add_argument_group(title='GRPO training')
    group.add_argument('--grpo-beta', type=float, help='beta scale for KL divergence')

    return add_lora_args(parser)


if __name__ == "__main__":
    args = add_grpo_args(ArgumentParser()).parse_known_args()
    if args.grpo_beta is not None:
        raise ValueError('GRPO beta is not supported yet')

    train_valid_test_datasets_provider.is_distributed = True
    pretrain(
        train_valid_test_datasets_provider,
        lora_model_provider,
        ModelType.encoder_or_decoder,
        forward_step,
        extra_args_provider=add_grpo_args,
    )
