import os
from pathlib import Path

import pytest
import torch
import torch.distributed

from megatron.core.utils import is_te_min_version
from tests.unit_tests.dist_checkpointing import TempNamedDir
from tests.unit_tests.test_utilities import Utils


def pytest_sessionfinish(session, exitstatus):
    if exitstatus == 5:
        session.exitstatus = 0

def get_allocated_memory(before: float = 0, test_name: str = "") -> int | None:
    # free, total = torch.cuda.mem_get_info()
    # allocated = total - free
    allocated = torch.cuda.memory_reserved()
    # allocated = torch.cuda.memory_allocated()

    after = round(allocated / 1024 ** 3, 2)
    if before == 0:
        return after
    
    leaked = round(after - before, 2)
    # raise RuntimeError(f"asd {leaked}, {before}, {used_mem_gb}")
    assert leaked == 0, f"CUDA {torch.cuda.current_device()}: {test_name}: {leaked} GB leaked during the test run!"
    # if used_mem_gb > before:
    #     leaked = round(used_mem_gb - before, 2)
    #     raise RuntimeError(f"{test_name}: {leaked} GB leaked on {dev} device during the test run!")


@pytest.fixture(scope="session", autouse=True)
def cleanup():
    # before = get_allocated_memory()
    yield
    if torch.distributed.is_initialized():
        torch.distributed.barrier()
        torch.distributed.destroy_process_group()
    # get_allocated_memory(before)


@pytest.fixture(scope="function", autouse=True)
def set_env(request):
    dev = torch.cuda.current_device()
    if dev == 0:
        BYTES_IN_GB = 1024 ** 3
        free_b, total_b = torch.cuda.mem_get_info(dev)
        free_gb = round(free_b / BYTES_IN_GB, 2)
        total_gb = round(total_b / BYTES_IN_GB, 2)
        print(f"{request.node.nodeid} on cuda:{dev}: Free: {free_gb}, Total: {total_gb}")
    if is_te_min_version("1.3"):
        os.environ['NVTE_FLASH_ATTN'] = '0'
        os.environ['NVTE_FUSED_ATTN'] = '0'


@pytest.fixture(scope="session")
def tmp_path_dist_ckpt(tmp_path_factory) -> Path:
    """Common directory for saving the checkpoint.

    Can't use pytest `tmp_path_factory` directly because directory must be shared between processes.
    """

    tmp_dir = tmp_path_factory.mktemp('ignored', numbered=False)
    tmp_dir = tmp_dir.parent.parent / 'tmp_dist_ckpt'

    if Utils.rank == 0:
        with TempNamedDir(tmp_dir, sync=False):
            yield tmp_dir

    else:
        yield tmp_dir
