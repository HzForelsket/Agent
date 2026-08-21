# Copyright (c) Microsoft. All rights reserved.

# pyright: reportUnknownVariableType=false
# pyright: reportUnknownMemberType=false
# pyright: reportUnknownArgumentType=false

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Mapping, Sequence, Type, cast

import hydra
import ray
from verl.trainer.ppo.reward import load_reward_manager
from verl.trainer.ppo.utils import create_rl_sampler, need_critic, need_reference_policy

from agentlightning.adapter import TraceAdapter
from agentlightning.llm_proxy import LLMProxy
from agentlightning.store.base import LightningStore
from agentlightning.types import Dataset

from .dataset import AgentDataset, LoadedDataset

if TYPE_CHECKING:
    from .daemon import AgentModeDaemon
    from .model_download import ModelMaterialization
    from .trainer import AgentLightningTrainer

__all__ = [
    "NPUResourceTopology",
    "configure_accelerator",
    "configure_npu_resources",
    "invocation_directory",
    "main",
    "prepare_model_for_accelerator",
    "run_ppo",
    "TaskRunner",
]


@dataclass(frozen=True)
class NPUResourceTopology:
    """Resolved homogeneous NPU topology used by VERL and vLLM."""

    nnodes: int
    devices_per_node: int
    world_size: int
    rollout_tensor_parallel_size: int
    rollout_replicas: int


def configure_accelerator(config: Any) -> str:
    """Auto-select CUDA or Ascend NPU before Ray resources are created."""
    from verl.utils.device import auto_set_device

    auto_set_device(config)
    return str(config.trainer.device)


def configure_npu_resources(config: Any, ray_nodes: Sequence[Mapping[str, Any]]) -> NPUResourceTopology:
    """Use every NPU visible to Ray and validate VERL's rollout topology."""
    devices_by_node: list[int] = []
    for node in ray_nodes:
        if not node.get("Alive", False):
            continue
        resources = node.get("Resources", {})
        if not isinstance(resources, Mapping):
            raise RuntimeError("Ray 节点的 Resources 字段无效，无法自动配置 NPU。")
        raw_count = resources.get("NPU", 0)
        try:
            count = float(raw_count)
        except (TypeError, ValueError) as exc:
            raise RuntimeError(f"Ray 报告了无效的 NPU 数量：{raw_count!r}") from exc
        if count <= 0:
            continue
        if not count.is_integer():
            raise RuntimeError(f"Ray 报告了非整数 NPU 数量：{count}")
        devices_by_node.append(int(count))

    if not devices_by_node:
        raise RuntimeError(
            "已选择 NPU，但 Ray 没有发现 NPU 资源。请在启动 Ray 前检查 "
            "ASCEND_RT_VISIBLE_DEVICES、torch-npu、CANN 和设备权限。"
        )
    unique_counts = set(devices_by_node)
    if len(unique_counts) != 1:
        raise RuntimeError(
            "VERL 0.9 的 n_gpus_per_node 只支持同构节点，但 Ray 发现每节点 NPU 数量为 "
            f"{devices_by_node}。请让所有训练节点暴露相同数量的 NPU。"
        )

    devices_per_node = devices_by_node[0]
    nnodes = len(devices_by_node)
    world_size = devices_per_node * nnodes
    tensor_parallel_size = int(config.actor_rollout_ref.rollout.tensor_model_parallel_size)
    if tensor_parallel_size <= 0:
        raise ValueError(f"rollout tensor_model_parallel_size 必须为正整数，当前为 {tensor_parallel_size}。")
    if devices_per_node % tensor_parallel_size:
        raise ValueError(
            f"每节点有 {devices_per_node} 张 NPU，不能被 rollout tensor_model_parallel_size="
            f"{tensor_parallel_size} 整除。"
        )

    config.trainer.n_gpus_per_node = devices_per_node
    config.trainer.nnodes = nnodes
    topology = NPUResourceTopology(
        nnodes=nnodes,
        devices_per_node=devices_per_node,
        world_size=world_size,
        rollout_tensor_parallel_size=tensor_parallel_size,
        rollout_replicas=world_size // tensor_parallel_size,
    )
    print(
        "NPU 资源自动适配："
        f"nodes={topology.nnodes}, devices_per_node={topology.devices_per_node}, "
        f"FSDP world_size={topology.world_size}, rollout TP={topology.rollout_tensor_parallel_size}, "
        f"rollout replicas={topology.rollout_replicas}",
        flush=True,
    )
    return topology


def invocation_directory() -> Path:
    """Return the directory from which the user started the command."""
    from hydra.core.hydra_config import HydraConfig

    if HydraConfig.initialized():
        return Path(str(HydraConfig.get().runtime.cwd)).resolve()
    return Path.cwd().resolve()


def prepare_model_for_accelerator(
    config: Any,
    backend: str,
    download_root: Path | None = None,
) -> list[ModelMaterialization]:
    """Materialize remote model references before NPU Ray workers are created."""
    download_config = config.agentlightning.get("npu_model_download", {})
    if backend != "npu" or not download_config.get("enabled", True):
        return []

    from .model_download import materialize_npu_model_config

    materializations = materialize_npu_model_config(
        config,
        download_root or invocation_directory(),
        local_files_only=download_config.get("local_files_only", False),
    )
    for item in materializations:
        print(f"NPU 模型已就绪：{item.model_ref} -> {item.local_path}", flush=True)
    return materializations


@hydra.main(config_path="pkg://agentlightning/verl", config_name="config", version_base=None)
def main(config: Any):
    from .daemon import AgentModeDaemon
    from .trainer import AgentLightningTrainer

    run_ppo(
        config,
        train_dataset=None,
        val_dataset=None,
        store=None,
        llm_proxy=None,
        adapter=None,
        trainer_cls=AgentLightningTrainer,
        daemon_cls=AgentModeDaemon,
    )


def run_ppo(
    config: Any,
    train_dataset: Dataset[Any] | None,
    val_dataset: Dataset[Any] | None,
    store: LightningStore | None,
    llm_proxy: LLMProxy | None,
    adapter: TraceAdapter[Any] | None,
    trainer_cls: Type[AgentLightningTrainer],
    daemon_cls: Type[AgentModeDaemon],
) -> None:
    backend = configure_accelerator(config)
    prepare_model_for_accelerator(config, backend)
    if not ray.is_initialized():
        from omegaconf import OmegaConf

        ray_init_kwargs = cast(
            dict[str, Any], OmegaConf.to_container(config.ray_kwargs.get("ray_init", {}), resolve=True)
        )
        ray.init(**ray_init_kwargs)
    if backend == "npu":
        configure_npu_resources(config, cast(list[dict[str, Any]], ray.nodes()))

    runner = TaskRunner.remote()
    ray.get(
        runner.run.remote(  # type: ignore
            config=config,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            store=store,
            llm_proxy=llm_proxy,
            adapter=adapter,
            trainer_cls=trainer_cls,
            daemon_cls=daemon_cls,
        )
    )


@ray.remote(num_cpus=1)
class TaskRunner:
    """Build Agent Lightning on VERL 0.9's unified model-engine workers."""

    def run(
        self,
        config: Any,
        train_dataset: Dataset[Any] | None,
        val_dataset: Dataset[Any] | None,
        store: LightningStore | None,
        llm_proxy: LLMProxy | None,
        adapter: TraceAdapter[Any] | None,
        trainer_cls: Type[AgentLightningTrainer],
        daemon_cls: Type[AgentModeDaemon],
    ):
        from pprint import pprint

        from omegaconf import OmegaConf
        from verl.single_controller.ray import RayWorkerGroup, ResourcePoolManager
        from verl.trainer.ppo.utils import Role
        from verl.utils.config import omega_conf_to_dataclass
        from verl.workers.config import HFModelConfig
        from verl.workers.engine_workers import ActorRolloutRefWorker, TrainingWorker

        pprint(OmegaConf.to_container(config, resolve=True))
        OmegaConf.resolve(config)

        model_config: HFModelConfig = omega_conf_to_dataclass(config.actor_rollout_ref.model)
        tokenizer = model_config.tokenizer
        processor = model_config.processor

        if config.actor_rollout_ref.rollout.name == "vllm":
            from .async_server import install_vllm_server_patch

            install_vllm_server_patch()

        strategy = config.actor_rollout_ref.actor.strategy
        if config.agentlightning.prefix_grouper.enabled:
            if strategy not in {"fsdp", "fsdp2"}:
                raise ValueError("PrefixGrouper requires the FSDP or FSDP2 actor strategy.")
            if config.actor_rollout_ref.model.get("use_remove_padding", False):
                raise ValueError("PrefixGrouper requires actor_rollout_ref.model.use_remove_padding=false.")
            from .prefix_grouper import PrefixGrouperActorRolloutRefWorker

            actor_rollout_cls = PrefixGrouperActorRolloutRefWorker
        else:
            actor_rollout_cls = ActorRolloutRefWorker

        lora_rank = config.actor_rollout_ref.model.get("lora", {}).get("rank", 0)
        if lora_rank <= 0:
            lora_rank = config.actor_rollout_ref.model.get("lora_rank", 0)
        ref_in_actor = lora_rank > 0 or config.actor_rollout_ref.model.get("lora_adapter_path") is not None
        actor_role = Role.ActorRolloutRef if need_reference_policy(config) and not ref_in_actor else Role.ActorRollout

        role_worker_mapping: dict[Any, Any] = {actor_role: ray.remote(actor_rollout_cls)}
        mapping: dict[Any, str] = {actor_role: "global_pool"}
        if need_critic(config):
            role_worker_mapping[Role.Critic] = ray.remote(TrainingWorker)
            mapping[Role.Critic] = "global_pool"
        if config.reward.reward_model.enable:
            mapping[Role.RewardModel] = "global_pool"

        resource_pool_spec = {
            "global_pool": [config.trainer.n_gpus_per_node] * config.trainer.nnodes,
        }
        resource_pool_manager = ResourcePoolManager(
            resource_pool_spec=resource_pool_spec, mapping=cast(dict[int, str], mapping)
        )

        reward_kwargs = config.reward.reward_model.get("reward_kwargs", {})
        reward_fn = load_reward_manager(config, tokenizer, num_examine=0, **reward_kwargs)
        val_reward_fn = load_reward_manager(config, tokenizer, num_examine=1, **reward_kwargs)

        from verl.utils.dataset.rl_dataset import collate_fn

        if train_dataset is None:
            train_dataset = AgentDataset(
                data_files=config.data.train_files,
                tokenizer=tokenizer,
                processor=processor,
                config=config.data,
            )
        else:
            train_dataset = LoadedDataset(train_dataset)

        if val_dataset is None:
            val_dataset = AgentDataset(
                data_files=config.data.val_files,
                tokenizer=tokenizer,
                processor=processor,
                config=config.data,
            )
        else:
            val_dataset = LoadedDataset(val_dataset)

        train_sampler = create_rl_sampler(config.data, train_dataset)
        trainer = trainer_cls(
            config=config,
            tokenizer=tokenizer,
            processor=processor,
            role_worker_mapping=role_worker_mapping,
            resource_pool_manager=resource_pool_manager,
            ray_worker_group_cls=RayWorkerGroup,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            collate_fn=collate_fn,
            train_sampler=train_sampler,
            store=store,
            llm_proxy=llm_proxy,
            adapter=adapter,
            daemon_cls=daemon_cls,
            reward_fn=reward_fn,
            val_reward_fn=val_reward_fn,
        )
        trainer.init_workers()
        trainer.fit()


if __name__ == "__main__":
    main()
