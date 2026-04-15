# SPDX-License-Identifier: Apache-2.0
# Standard
from pathlib import Path
from types import SimpleNamespace
import asyncio
import threading
import time

# Third Party
import pytest
import torch

pytest.importorskip("nixl", reason="nixl package is required for nixl tests")

# First Party
from lmcache.utils import CacheEngineKey
from lmcache.v1.config import LMCacheEngineConfig
from lmcache.v1.memory_management import PagedTensorMemoryAllocator
from lmcache.v1.metadata import LMCacheMetadata
from lmcache.v1.storage_backend import CreateStorageBackends
from lmcache.v1.storage_backend.nixl_storage_backend import (
    NixlDynamicStorageBackend,
    NixlStorageBackend,
    SetPresenceCache,
)


def create_key(chunk_hash: str):
    return CacheEngineKey(
        model_name="meta-llama/Llama-3.1-70B-Instruct",
        world_size=8,
        worker_id=0,
        chunk_hash=int(chunk_hash, base=16),
        dtype=torch.bfloat16,
    )


class FakeAsyncMemObj:
    def __init__(self, address: int):
        self.meta = SimpleNamespace(address=address)
        self.ref_count = 0

    def ref_count_up(self) -> None:
        self.ref_count += 1

    def ref_count_down(self) -> None:
        self.ref_count -= 1


class FakeAsyncAgent:
    mem_type = "OBJ"

    def __init__(self, states: list[str], post_async_state: str = "PENDING"):
        self.nixl_agent = self
        self._states = list(states)
        self._post_async_state = post_async_state
        self.release_handle_calls = 0
        self.release_storage_handler_calls = 0

    def create_batched_storage_handler(self, descs, page_size):
        return "reg_descs", "xfer_handler"

    def get_mem_to_storage_handle(
        self, mem_indices, storage_xfer_handler, storage_indices
    ):
        return "handle"

    def post_async(self, handle):
        return self._post_async_state

    def check_xfer_state(self, handle):
        assert self._states, "check_xfer_state called with no remaining fake states"
        return self._states.pop(0)

    def release_handle(self, handle) -> None:
        self.release_handle_calls += 1

    def release_storage_handler(self, reg_descs, xfer_handler) -> None:
        self.release_storage_handler_calls += 1


def create_fake_dynamic_backend(agent: FakeAsyncAgent) -> NixlDynamicStorageBackend:
    backend = object.__new__(NixlDynamicStorageBackend)
    backend.loop = asyncio.new_event_loop()
    backend.key_lock = threading.RLock()
    backend.progress_lock = threading.RLock()
    backend.progress_set = set()
    backend.pending_put_futures_lock = threading.RLock()
    backend.pending_put_futures = set()
    backend.async_mode = True
    backend.enable_presence_cache = True
    backend.hit_counter = 0
    backend.total_counter = 0
    backend.key_presence_cache = SetPresenceCache()
    backend.memory_allocator = SimpleNamespace(align_bytes=4096)
    backend.agent = agent
    return backend


def start_background_loop(
    backend: NixlDynamicStorageBackend,
) -> tuple[asyncio.AbstractEventLoop, threading.Thread]:
    loop = asyncio.new_event_loop()
    thread = threading.Thread(target=loop.run_forever)
    thread.start()
    backend.loop = loop
    return loop, thread


def stop_background_loop(
    loop: asyncio.AbstractEventLoop,
    thread: threading.Thread,
) -> None:
    if loop.is_running():
        loop.call_soon_threadsafe(loop.stop)
    if thread.is_alive():
        thread.join()
    loop.close()


def wait_for_async_put_completion(
    mem_obj: FakeAsyncMemObj,
    agent: FakeAsyncAgent,
    timeout_s: float = 1.0,
) -> None:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        if mem_obj.ref_count == 0 and agent.release_handle_calls == 1:
            return
        time.sleep(0.01)
    raise AssertionError("async put did not complete before timeout")


def run(config: LMCacheEngineConfig, shape, dtype):
    BACKEND_NAME = "NixlStorageBackend"
    keys = []
    objs = []
    keys.append(
        create_key("e3229141e680fb413d2c5d3ebb416c4ad300d381e309fc9e417757b91406c157")
    )
    keys.append(
        create_key("e3229141e680fb413d2c5d3ebb416c4ad300d381e309fc9e417757b91406d268")
    )
    keys.append(
        create_key("e3229141e680fb413d2c5d3ebb416c4ad300d381e309fc9e417757b91406e379")
    )
    bad_key = create_key("deadbeefdeadbeef")

    thread_loop = None
    thread = None
    try:
        thread_loop = asyncio.new_event_loop()
        thread = threading.Thread(target=thread_loop.run_forever)
        thread.start()

        metadata = LMCacheMetadata(
            model_name="Llama-3.1-70B-Instruct",
            world_size=1,
            local_world_size=1,
            worker_id=0,
            local_worker_id=0,
            kv_dtype=dtype,
            kv_shape=shape,
        )

        backends = CreateStorageBackends(
            config,
            metadata,
            thread_loop,
            dst_device=config.nixl_buffer_device,  # Pass the device directly
        )
        assert len(backends) == 2  # NixlStorageBackend + LocalCPUBackend
        assert BACKEND_NAME in backends

        nixl_backend = backends[BACKEND_NAME]
        assert isinstance(nixl_backend, NixlStorageBackend)
        assert isinstance(nixl_backend.memory_allocator, PagedTensorMemoryAllocator)
        assert nixl_backend is not None
        assert nixl_backend.memory_allocator is not None

        for key in keys:
            assert not nixl_backend.contains(key, False)
            assert not nixl_backend.exists_in_put_tasks(key)

            obj = nixl_backend.memory_allocator.allocate(shape=shape, dtype=dtype)
            assert obj is not None
            assert obj.tensor is not None
            objs.append(obj)

        # small tensor changes for data validation
        objs[0].tensor[100, 200] = 1e-3
        objs[0].tensor[200, 100] = 1e-4

        objs[1].tensor[300, 400] = 1e-2
        objs[1].tensor[400, 300] = 1e-5

        objs[2].tensor[300, 400] = 3e-2
        objs[2].tensor[400, 300] = 4e-5

        # Insert first 2 keys
        first_keys = keys[0:2]
        first_objs = objs[0:2]
        nixl_backend.batched_submit_put_task(first_keys, first_objs)

        for key, obj in zip(first_keys, first_objs, strict=False):
            returned_memory_obj = nixl_backend.get_blocking(key)
            assert returned_memory_obj is not None
            assert returned_memory_obj.get_size() == obj.get_size()
            assert returned_memory_obj.get_shape() == obj.get_shape()
            assert returned_memory_obj.get_dtype() == obj.get_dtype()
            assert returned_memory_obj.metadata.address != obj.metadata.address
            assert torch.equal(returned_memory_obj.tensor, obj.tensor)

        obj_list = asyncio.run(
            nixl_backend.batched_get_non_blocking(lookup_id="test", keys=first_keys)
        )

        for i, obj in enumerate(first_objs):
            returned_memory_obj = obj_list[i]
            assert returned_memory_obj is not None
            assert returned_memory_obj.get_size() == obj.get_size()
            assert returned_memory_obj.get_shape() == obj.get_shape()
            assert returned_memory_obj.get_dtype() == obj.get_dtype()
            assert returned_memory_obj.metadata.address != obj.metadata.address
            assert torch.equal(returned_memory_obj.tensor, obj.tensor)

        def test_eviction(new_idx, old_idx):
            nixl_backend.batched_submit_put_task([keys[new_idx]], [objs[new_idx]])

            obj = nixl_backend.get_blocking(keys[new_idx])
            assert obj is not None
            assert obj.tensor is not None
            assert torch.equal(obj.tensor, objs[new_idx].tensor)

            obj = nixl_backend.get_blocking(keys[old_idx])
            assert obj is None

        ######## Test bad key lookup #########
        obj = nixl_backend.get_blocking(bad_key)
        assert obj is None

        ######## Test eviction #########
        obj = nixl_backend.get_blocking(keys[0])
        assert obj is not None

        # At this point, key 0 & key 1 are cached. Key 1 is LRU key.
        # Submitting key 2 should evict key 1.

        test_eviction(new_idx=2, old_idx=1)

        ######## Test pin #########
        val = nixl_backend.pin(keys[2])
        assert val is True

        obj = nixl_backend.get_blocking(keys[0])
        assert obj is not None

        # At this point, key 0 & key 2 are cached.
        # Key 2 is LRU key, but is pinned.
        # Submitting key 1 should evict key 0.

        test_eviction(new_idx=1, old_idx=0)

        ######## Test unpin #########
        val = nixl_backend.unpin(keys[2])
        assert val is True

        obj = nixl_backend.get_blocking(keys[1])
        assert obj is not None

        # At this point, key 1 & key 2 are cached.
        # Key 2 is LRU key, and is now unpinned.
        # Submitting key 0 should evict key 2.

        test_eviction(new_idx=0, old_idx=2)

        for backend in backends.values():
            backend.close()

    except Exception:
        raise
    finally:
        if thread_loop and thread_loop.is_running():
            thread_loop.call_soon_threadsafe(thread_loop.stop)
        if thread and thread.is_alive():
            thread.join()


@pytest.mark.no_shared_allocator
def test_nixl_dynamic_async_put_success_clears_progress():
    key = create_key("1")
    mem_obj = FakeAsyncMemObj(address=11)
    agent = FakeAsyncAgent(states=["DONE"])
    backend = create_fake_dynamic_backend(agent)
    loop, thread = start_background_loop(backend)

    try:
        backend.batched_submit_put_task([key], [mem_obj])
        wait_for_async_put_completion(mem_obj, agent)
    finally:
        stop_background_loop(loop, thread)

    assert mem_obj.ref_count == 0
    assert not backend.exists_in_put_tasks(key)
    assert backend._cache_contains(key.chunk_hash)
    assert agent.release_handle_calls == 1
    assert agent.release_storage_handler_calls == 1


@pytest.mark.no_shared_allocator
def test_nixl_dynamic_async_put_failure_clears_progress_and_refcount():
    key = create_key("2")
    mem_obj = FakeAsyncMemObj(address=22)
    agent = FakeAsyncAgent(states=["ERR"])
    backend = create_fake_dynamic_backend(agent)
    loop, thread = start_background_loop(backend)

    try:
        backend.batched_submit_put_task([key], [mem_obj])
        wait_for_async_put_completion(mem_obj, agent)
    finally:
        stop_background_loop(loop, thread)

    assert mem_obj.ref_count == 0
    assert not backend.exists_in_put_tasks(key)
    assert not backend._cache_contains(key.chunk_hash)
    assert agent.release_handle_calls == 1
    assert agent.release_storage_handler_calls == 1


@pytest.mark.no_shared_allocator
@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA")
def test_nixl_gds_mt_cuda_backend():
    BASE_DIR = Path(__file__).parent
    config = LMCacheEngineConfig.from_file(BASE_DIR / "data/nixl.yaml")

    dtype = torch.bfloat16
    shape = [2048, 2048]

    config.nixl_buffer_device = "cuda:0"  # Use explicit device
    config.extra_config["nixl_backend"] = "GDS_MT"
    config.extra_config["enable_cuda"] = True

    run(config, shape, dtype)


@pytest.mark.no_shared_allocator
def test_nixl_gds_mt_cpu_backend():
    BASE_DIR = Path(__file__).parent
    config = LMCacheEngineConfig.from_file(BASE_DIR / "data/nixl.yaml")

    dtype = torch.bfloat16
    shape = [2048, 2048]

    config.nixl_buffer_device = "cpu"
    config.extra_config["nixl_backend"] = "GDS_MT"
    config.extra_config["enable_cuda"] = False

    run(config, shape, dtype)


@pytest.mark.no_shared_allocator
@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA")
def test_nixl_gds_cuda_backend():
    BASE_DIR = Path(__file__).parent
    config = LMCacheEngineConfig.from_file(BASE_DIR / "data/nixl.yaml")

    dtype = torch.bfloat16
    shape = [2048, 2048]

    config.nixl_buffer_device = "cuda:0"  # Use explicit device
    config.extra_config["nixl_backend"] = "GDS"
    config.extra_config["enable_cuda"] = True

    run(config, shape, dtype)


@pytest.mark.no_shared_allocator
def test_nixl_gds_cpu_backend():
    BASE_DIR = Path(__file__).parent
    config = LMCacheEngineConfig.from_file(BASE_DIR / "data/nixl.yaml")

    dtype = torch.bfloat16
    shape = [2048, 2048]

    config.nixl_buffer_device = "cpu"
    config.extra_config["nixl_backend"] = "GDS"
    config.extra_config["enable_cuda"] = False

    run(config, shape, dtype)


@pytest.mark.no_shared_allocator
def test_nixl_posix_backend():
    BASE_DIR = Path(__file__).parent
    config = LMCacheEngineConfig.from_file(BASE_DIR / "data/nixl.yaml")

    dtype = torch.bfloat16
    shape = [2048, 2048]

    config.nixl_buffer_device = "cpu"
    config.extra_config["nixl_backend"] = "POSIX"
    config.extra_config["enable_cuda"] = False

    run(config, shape, dtype)
