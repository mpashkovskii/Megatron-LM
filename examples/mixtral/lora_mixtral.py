# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.

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

import torch
from argparse import ArgumentParser

from megatron.core import mpu
from megatron.core.enums import ModelType
from megatron.core.models.gpt import GPTModel
from megatron.core.transformer.lora_adapter import LoraAdapter
from megatron.inference.text_generation.api import beam_search_and_post_process, generate_and_post_process
from megatron.inference.text_generation_server import MegatronServer
from megatron.training import get_args, pretrain
from megatron.training.checkpointing import load_checkpoint
from megatron.training.initialize import initialize_megatron
from megatron.training.training import get_model

from pretrain_gpt import train_valid_test_datasets_provider, forward_step, model_provider


def lora_model_provider(pre_process: bool = True, post_process: bool = True) -> GPTModel:
    args = get_args()
    rank = args.lora_rank
    alpha = args.lora_alpha
    assert rank > 0 and alpha > 0, "LoRA rank and alpha have to be greater than zero"
    
    model = model_provider(pre_process, post_process)
    common_args = {
        "config": model.config,
        "rank": rank,
        "alpha": alpha,
        "dropout": args.lora_dropout,
    }
    model.embedding.word_embeddings = LoraAdapter(model.embedding.word_embeddings, **common_args)
    for layer in model.decoder.layers:
        layer.self_attention.linear_qkv = LoraAdapter(layer.self_attention.linear_qkv, **common_args)
        layer.self_attention.linear_proj = LoraAdapter(layer.self_attention.linear_proj, **common_args)
        layer.mlp.router = LoraAdapter(layer.mlp.router, **common_args)
        for fc in layer.mlp.experts.local_experts:
            fc.linear_fc1 = LoraAdapter(fc.linear_fc1, is_expert=True, **common_args)
            fc.linear_fc2 = LoraAdapter(fc.linear_fc2, is_expert=True, **common_args)
    model.output_layer = LoraAdapter(model.output_layer, **common_args)
    return model


def add_lora_args(parser: ArgumentParser) -> ArgumentParser:
    group = parser.add_argument_group(title='LoRA')
    group.add_argument('--lora-rank', default=16, type=int,
                       help='LoRA rank')
    group.add_argument('--lora-alpha', default=32.0, type=float,
                       help='LoRA alpha')
    group.add_argument('--lora-dropout', default=0.1, type=float,
                       help='LoRA dropout')
    group.add_argument('--lora-task', default="train", type=str, choices=['train', 'serve'],
                       help='LoRA task')
    return parser


if __name__ == "__main__":
    args, _ = add_lora_args(ArgumentParser()).parse_known_args()
    if args.lora_task == "train":
        train_valid_test_datasets_provider.is_distributed = True
        pretrain(
            train_valid_test_datasets_provider,
            lora_model_provider,
            ModelType.encoder_or_decoder,
            forward_step,
            extra_args_provider=add_lora_args,
        )
    else:
        initialize_megatron(extra_args_provider=add_lora_args)
        model = get_model(lora_model_provider, wrap_with_ddp=False)
        load_checkpoint(model, None, None)
        assert len(model) == 1
        model = model[0]
        model.eval()

        if mpu.is_pipeline_first_stage() and mpu.get_tensor_model_parallel_rank() == 0 and mpu.get_expert_model_parallel_rank() == 0:
            server = MegatronServer(model)
            server.run("0.0.0.0", port=5000)

        while True:
            choice = torch.tensor(1, dtype=torch.long, device='cuda')
            torch.distributed.broadcast(choice, 0)
            if choice.item() == 0:
                try:
                    generate_and_post_process(model)
                except ValueError as ve:
                    pass
            elif choice.item() == 1:
                try:
                    beam_search_and_post_process(model)
                except ValueError as ve:
                    pass
