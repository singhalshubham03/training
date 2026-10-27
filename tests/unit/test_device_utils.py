# SPDX-License-Identifier: Apache-2.0
"""Tests for Intel XPU support: device selection, flash-attn fallback, teardown."""

# Standard
from pathlib import Path
from unittest.mock import patch
import os
import signal
import subprocess
import sys
import textwrap

# Third Party
from torch.utils.data import DataLoader
import pytest
import torch

# First Party
from instructlab.training import device_utils
from instructlab.training.main_ds import _shutdown_dataloader_workers
from instructlab.training.model import Model

_HAS_XPU = hasattr(torch, "xpu") and torch.xpu.is_available()


def requires_xpu(obj):
    """Needs a real Intel XPU: ``xpu`` marker + skip when absent."""
    return pytest.mark.xpu(
        pytest.mark.skipif(not _HAS_XPU, reason="Intel XPU not available")(obj)
    )


@pytest.fixture(autouse=True)
def _no_forced_device(monkeypatch):
    monkeypatch.delenv("INSTRUCTLAB_TRAINING_DEVICE", raising=False)


def _fake(monkeypatch, *, cuda: bool, xpu: bool):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: cuda)
    monkeypatch.setattr(torch.xpu, "is_available", lambda: xpu)


@pytest.mark.parametrize(
    "cuda,xpu,expected",
    [
        (True, False, "cuda"),
        (True, True, "cuda"),  # CUDA wins when both are present
        (False, True, "xpu"),
        (False, False, "cuda"),  # CPU-only hosts keep the original CUDA path
    ],
)
def test_device_selection(monkeypatch, cuda, xpu, expected):
    _fake(monkeypatch, cuda=cuda, xpu=xpu)
    assert device_utils.get_device_type() == expected
    assert device_utils.is_xpu() == (expected == "xpu")
    assert device_utils.get_device_module() is getattr(torch, expected)
    assert device_utils.get_device(2) == torch.device(expected, 2)


def test_env_override(monkeypatch):
    _fake(monkeypatch, cuda=False, xpu=True)
    monkeypatch.setenv("INSTRUCTLAB_TRAINING_DEVICE", "CUDA")
    assert device_utils.get_device_type() == "cuda"


def test_get_device_defaults_to_local_rank(monkeypatch):
    _fake(monkeypatch, cuda=False, xpu=True)
    monkeypatch.setenv("LOCAL_RANK", "1")
    assert device_utils.get_device() == torch.device("xpu", 1)


def test_num_alloc_retries_missing_on_xpu(monkeypatch):
    # torch.xpu.memory_stats() has no "num_alloc_retries" key; reading it
    # directly raised KeyError on the first logged step.
    _fake(monkeypatch, cuda=False, xpu=True)
    monkeypatch.setattr(
        torch.xpu, "memory_stats", lambda: {"allocated_bytes.all.current": 0}
    )
    assert device_utils.num_alloc_retries() == 0
    monkeypatch.setattr(torch.xpu, "memory_stats", lambda: {"num_alloc_retries": 3})
    assert device_utils.num_alloc_retries() == 3


def test_flash_attn_disabled_on_xpu_without_probing_cuda(monkeypatch):
    _fake(monkeypatch, cuda=False, xpu=True)
    with patch(
        "torch.cuda.get_device_capability",
        side_effect=AssertionError("Torch not compiled with CUDA enabled"),
    ):
        assert Model.check_flash_attn_enabled(False) is False
        assert Model.check_flash_attn_enabled(True) is False


def test_shutdown_dataloader_workers_stops_persistent_workers():
    loader = DataLoader(
        list(range(16)), batch_size=4, num_workers=2, persistent_workers=True
    )
    for _ in loader:
        pass
    workers = list(loader._iterator._workers)
    assert all(w.is_alive() for w in workers)

    _shutdown_dataloader_workers(loader, None)

    assert loader._iterator is None
    assert not any(w.is_alive() for w in workers)


def test_shutdown_dataloader_workers_noop_without_workers():
    loader = DataLoader(list(range(4)), batch_size=2)
    for _ in loader:
        pass
    _shutdown_dataloader_workers(loader)  # single-process iterator: nothing to stop


# With XCCL, destroy_process_group() blocks indefinitely while forked DataLoader
# workers are alive. This runs the end-of-training sequence in a real
# single-rank XCCL job and fails (instead of hanging) if teardown regresses.
_TEARDOWN_SCRIPT = textwrap.dedent(
    """
    import os, sys
    import torch, torch.distributed as dist
    from torch.utils.data import DataLoader
    from instructlab.training.main_ds import _shutdown_dataloader_workers

    torch.xpu.set_device(int(os.environ["LOCAL_RANK"]))
    dist.init_process_group()
    t = torch.ones(1, device="xpu")
    dist.all_reduce(t)
    loader = DataLoader(list(range(32)), batch_size=4, num_workers=2, persistent_workers=True)
    for _ in loader:
        pass
    _shutdown_dataloader_workers(loader)
    dist.barrier()
    dist.destroy_process_group()
    print("TEARDOWN_OK", flush=True)
    """
)


@requires_xpu
def test_xccl_teardown_with_dataloader_workers(tmp_path: Path):
    script = tmp_path / "teardown.py"
    script.write_text(_TEARDOWN_SCRIPT)
    # Single rank launched directly (no torchrun) in its own session, so a hang
    # is cleaned up with one killpg: forked DataLoader workers share the group.
    env = {
        **os.environ,
        "RANK": "0",
        "LOCAL_RANK": "0",
        "WORLD_SIZE": "1",
        "MASTER_ADDR": "127.0.0.1",
        "MASTER_PORT": "29631",
    }
    with subprocess.Popen(
        [sys.executable, str(script)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=env,
        start_new_session=True,
    ) as proc:
        try:
            stdout, stderr = proc.communicate(timeout=120)
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid, signal.SIGKILL)
            proc.communicate()
            pytest.fail("destroy_process_group() hung with DataLoader workers alive")
    assert proc.returncode == 0, stdout[-2000:] + stderr[-2000:]
    assert "TEARDOWN_OK" in stdout
