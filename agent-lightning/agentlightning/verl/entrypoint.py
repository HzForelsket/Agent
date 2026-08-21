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
    rollout_data_parallel_size: int
    rollout_pipeline_parallel_size: int
    rollout_world_size: int
    rollout_replicas: int
    reward_rollout_world_size: int | None
    reward_rollout_replicas: int | None


def _config_value(config: Any, name: str, default: Any = None) -> Any:
    if config is None:
        return default
    if isinstance(config, Mapping):
        return config.get(name, default)
    getter = getattr(config, "get", None)
    if callable(getter):
        return getter(name, default)
    return getattr(config, name, default)


def _positive_parallel_size(config: Any, name: str, *, label: str, default: int = 1) -> int:
    value = int(_config_value(config, name, default))
    if value <= 0:
        raise ValueError(f"{label}.{name} 必须为正整数，当前为 {value}。")
    return value


def _rollout_parallelism(rollout: Any, *, label: str) -> tuple[int, int, int, int]:
    tensor_parallel_size = _positive_parallel_size(
        rollout, "tensor_model_parallel_size", label=label, default=1
    )
    data_parallel_size = _positive_parallel_size(rollout, "data_parallel_size", label=label, default=1)
    pipeline_parallel_size = _positive_parallel_size(
        rollout, "pipeline_model_parallel_size", label=label, default=1
    )

    disaggregation = _config_value(rollout, "disaggregation")
    if bool(_config_value(disaggregation, "enabled", False)):
        prefill_replicas = _positive_parallel_size(
            disaggregation, "prefill_replicas", label=f"{label}.disaggregation", default=1
        )
        decode_replicas = _positive_parallel_size(
            disaggregation, "decode_replicas", label=f"{label}.disaggregation", default=1
        )
        raw_decode_tp = _config_value(disaggregation, "decode_tensor_model_parallel_size")
        decode_parallel_size = (
            tensor_parallel_size
            if raw_decode_tp is None
            else int(raw_decode_tp)
        )
        if decode_parallel_size <= 0:
            raise ValueError(
                f"{label}.disaggregation.decode_tensor_model_parallel_size 必须为正整数，"
                f"当前为 {decode_parallel_size}。"
            )
        replica_world_size = (
            tensor_parallel_size * prefill_replicas + decode_parallel_size * decode_replicas
        ) * data_parallel_size * pipeline_parallel_size
    else:
        replica_world_size = tensor_parallel_size * data_parallel_size * pipeline_parallel_size
    return tensor_parallel_size, data_parallel_size, pipeline_parallel_size, replica_world_size


def _validate_rollout_layout(
    *,
    label: str,
    replica_world_size: int,
    devices_per_node: int,
    world_size: int,
) -> int:
    if world_size % replica_world_size:
        raise ValueError(
            f"NPU world_size={world_size} 不能被 {label} 每副本占卡数 {replica_world_size} 整除；"
            "该配置会让部分 NPU rank 不参与 rollout。"
        )
    if replica_world_size <= devices_per_node:
        if devices_per_node % replica_world_size:
            raise ValueError(
                f"每节点有 {devices_per_node} 张 NPU，不能完整容纳整数个 {label} 副本"
                f"（每副本 {replica_world_size} 张）；该配置会把单个副本错误切到两个节点。"
            )
    elif replica_world_size % devices_per_node:
        raise ValueError(
            f"{label} 每副本需要 {replica_world_size} 张 NPU，必须占用整数个完整节点；"
            f"当前每节点有 {devices_per_node} 张。"
        )
    return world_size // replica_world_size


def _configure_npu_fsdp(config: Any) -> None:
    actor = config.actor_rollout_ref.actor
    if str(actor.strategy) != "fsdp":
        raise ValueError("Ascend 多卡路径要求 actor_rollout_ref.actor.strategy=fsdp。")
    actor_fsdp = _config_value(actor, "fsdp_config")
    if actor_fsdp is not None:
        actor_fsdp.fsdp_size = -1

    critic = _config_value(config, "critic")
    critic_enabled = bool(_config_value(critic, "enable", False))
    try:
        critic_enabled = bool(need_critic(config))
    except (AttributeError, KeyError, TypeError):
        pass
    if critic_enabled:
        if str(_config_value(critic, "strategy")) != "fsdp":
            raise ValueError("Ascend 多卡路径要求 critic.strategy=fsdp。")
        critic_fsdp = _config_value(critic, "fsdp")
        if critic_fsdp is not None:
            critic_fsdp.fsdp_size = -1

    checkpoint_engine = _config_value(config.actor_rollout_ref.rollout, "checkpoint_engine")
    checkpoint_backend = str(_config_value(checkpoint_engine, "backend", "naive"))
    if checkpoint_backend != "naive":
        raise ValueError(
            "Ascend 全 rank 权重同步要求 actor_rollout_ref.rollout.checkpoint_engine.backend=naive；"
            f"当前 {checkpoint_backend} 后端会形成单发送端。"
        )


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
    rollout = config.actor_rollout_ref.rollout
    tensor_parallel_size, data_parallel_size, pipeline_parallel_size, rollout_world_size = (
        _rollout_parallelism(rollout, label="actor_rollout_ref.rollout")
    )
    rollout_replicas = _validate_rollout_layout(
        label="actor_rollout_ref.rollout",
        replica_world_size=rollout_world_size,
        devices_per_node=devices_per_node,
        world_size=world_size,
    )

    reward_rollout_world_size: int | None = None
    reward_rollout_replicas: int | None = None
    reward = _config_value(config, "reward")
    reward_model = _config_value(reward, "reward_model")
    if bool(_config_value(reward_model, "enable", False)):
        reward_rollout = _config_value(reward_model, "rollout")
        _, _, _, reward_rollout_world_size = _rollout_parallelism(
            reward_rollout, label="reward.reward_model.rollout"
        )
        reward_rollout_replicas = _validate_rollout_layout(
            label="reward.reward_model.rollout",
            replica_world_size=reward_rollout_world_size,
            devices_per_node=devices_per_node,
            world_size=world_size,
        )
        reward_model.n_gpus_per_node = devices_per_node
        reward_model.nnodes = nnodes

    config.trainer.n_gpus_per_node = devices_per_node
    config.trainer.nnodes = nnodes
    _configure_npu_fsdp(config)
    topology = NPUResourceTopology(
        nnodes=nnodes,
        devices_per_node=devices_per_node,
        world_size=world_size,
        rollout_tensor_parallel_size=tensor_parallel_size,
        rollout_data_parallel_size=data_parallel_size,
        rollout_pipeline_parallel_size=pipeline_parallel_size,
        rollout_world_size=rollout_world_size,
        rollout_replicas=rollout_replicas,
        reward_rollout_world_size=reward_rollout_world_size,
        reward_rollout_replicas=reward_rollout_replicas,
    )
    print(
        "NPU 资源自动适配："
        f"nodes={topology.nnodes}, devices_per_node={topology.devices_per_node}, "
        f"FSDP world_size={topology.world_size}, rollout TP={topology.rollout_tensor_parallel_size}, "
        f"DP={topology.rollout_data_parallel_size}, PP={topology.rollout_pipeline_parallel_size}, "
        f"devices_per_replica={topology.rollout_world_size}, "
        f"rollout replicas={topology.rollout_replicas}",
        flush=True,
    )
    if topology.reward_rollout_world_size is not None:
        print(
            "NPU reward model 资源自动适配："
            f"devices_per_replica={topology.reward_rollout_world_size}, "
            f"rollout replicas={topology.reward_rollout_replicas}",
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
        if str(config.trainer.device) == "npu":
            if strategy != "fsdp":
                raise ValueError("Ascend 多卡路径要求 actor_rollout_ref.actor.strategy=fsdp。")
            from .npu_fsdp_loader import (
                NPUShardedLoadActorRolloutRefWorker,
                NPUShardedLoadTrainingWorker,
            )

            actor_rollout_cls = NPUShardedLoadActorRolloutRefWorker
            critic_worker_cls = NPUShardedLoadTrainingWorker
        else:
            actor_rollout_cls = ActorRolloutRefWorker
            critic_worker_cls = TrainingWorker

        if config.agentlightning.prefix_grouper.enabled:
            if strategy not in {"fsdp", "fsdp2"}:
                raise ValueError("PrefixGrouper requires the FSDP or FSDP2 actor strategy.")
            if config.actor_rollout_ref.model.get("use_remove_padding", False):
                raise ValueError("PrefixGrouper requires actor_rollout_ref.model.use_remove_padding=false.")
            from .prefix_grouper import PrefixGrouperActorRolloutRefWorker

            actor_rollout_cls = PrefixGrouperActorRolloutRefWorker

        lora_rank = config.actor_rollout_ref.model.get("lora", {}).get("rank", 0)
        if lora_rank <= 0:
            lora_rank = config.actor_rollout_ref.model.get("lora_rank", 0)
        ref_in_actor = lora_rank > 0 or config.actor_rollout_ref.model.get("lora_adapter_path") is not None
        actor_role = Role.ActorRolloutRef if need_reference_policy(config) and not ref_in_actor else Role.ActorRollout

        role_worker_mapping: dict[Any, Any] = {actor_role: ray.remote(actor_rollout_cls)}
        mapping: dict[Any, str] = {actor_role: "global_pool"}
        if need_critic(config):
            role_worker_mapping[Role.Critic] = ray.remote(critic_worker_cls)
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
