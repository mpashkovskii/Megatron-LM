# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# Experiments to perform GPU code generation using reinforcement learning:
# - https://arxiv.org/pdf/2402.03300
# - https://huggingface.co/docs/trl/main/en/grpo_trainer
# - https://sakana.ai/ai-cuda-engineer/
# - https://predibase.com/blog/teaching-ai-to-write-gpu-code-a-deep-dive-into-reinforcement-fine-tuning

import os
import sys

sys.path.append(
    os.path.abspath(
        os.path.join(
            os.path.dirname(__file__),
            os.path.pardir,
            os.path.pardir
        )
    )
)

from argparse import ArgumentParser
from functools import partial
from typing import Callable

import torch

from megatron.core.datasets.blended_megatron_dataset_builder import BlendedMegatronDatasetBuilder
from megatron.core.datasets.gpt_dataset import GPTDataset
from megatron.core.datasets.megatron_dataset import MegatronDataset
from megatron.core.enums import ModelType
from megatron.core.models.gpt import GPTModel
from megatron.training import pretrain
from megatron.training.global_vars import get_args
from megatron.training.utils import print_rank_0

# LoRA not neccesarily needed
# from lora_mixtral import add_lora_args, lora_model_provider
from pretrain_gpt import SPIKY_LOSS_PERC, core_gpt_dataset_config_from_args, get_batch, is_dataset_built_on_rank, model_provider


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


def train_valid_test_datasets_provider(train_val_test_num_samples: int) -> MegatronDataset:
    """Build the train test and validation datasets.

    Args:
        train_val_test_num_samples : A list containing the number of samples in train test and validation.
    """
    args = get_args()
    if args.mock_data:
        raise NotImplementedError('Mock data is not supported')

    config = core_gpt_dataset_config_from_args(args)

    print_rank_0("> building train, validation, and test datasets for GPT ...")

    train_ds, valid_ds, test_ds = BlendedMegatronDatasetBuilder(
        GPTDataset,
        train_val_test_num_samples,
        is_dataset_built_on_rank,
        config
    ).build()

    print_rank_0("> finished creating GPT datasets ...")

    print()
    raise NotImplementedError('GRPO dataset provider is not implemented yet')

    return train_ds, valid_ds, test_ds


def add_grpo_args(parser: ArgumentParser) -> ArgumentParser:
    group = parser.add_argument_group(title='GRPO data preprocessing')
    group.add_argument('--num-generations', type=int, default=int(os.getenv("WORLD_SIZE", '8')),
                       help='Number of generations to sample. The global batch size '
                            '(num_processes * per_device_batch_size)')

    group = parser.add_argument_group(title='GRPO training')
    group.add_argument('--grpo-beta', type=float, default=0,
                       help='beta scale for KL divergence')

    return parser
    # return add_lora_args(parser)


if __name__ == "__main__":
    args, _ = add_grpo_args(ArgumentParser()).parse_known_args()
    if args.grpo_beta != 0:
        raise ValueError('GRPO beta is not supported yet')

    train_valid_test_datasets_provider.is_distributed = True
    pretrain(
        train_valid_test_datasets_provider,
        model_provider,
        # lora_model_provider,
        ModelType.encoder_or_decoder,
        forward_step,
        extra_args_provider=add_grpo_args,
    )
