#!/bin/bash

set -x
export HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

PYTEST_MARKERS="(not flaky and not flaky_in_dev and not internal and not failing_on_rocm and not failing_on_upstream or test_on_rocm)"

if [[ "$HIP_ARCHITECTURES" == "gfx90a" ]]; then
    PYTEST_MARKERS="$PYTEST_MARKERS and not failing_on_rocm_mi250"
fi

torchrun --nproc_per_node=8 -m pytest -m "$PYTEST_MARKERS" tests/unit_tests
# torchrun --nproc_per_node=8 -m pytest -m "$PYTEST_MARKERS" tests/unit_tests/transformer
# torchrun --nproc_per_node=8 -m pytest -m "$PYTEST_MARKERS" /workspace/Megatron-LM/tests/unit_tests/transformer/moe/test_routers.py
# torchrun --nproc_per_node=8 -m pytest -m "$PYTEST_MARKERS" -k test_forward /workspace/Megatron-LM/tests/unit_tests/transformer/test_lora_adapter.py
