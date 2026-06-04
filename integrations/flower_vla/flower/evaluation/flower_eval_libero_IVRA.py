"""LIBERO evaluation entrypoint for FLOWER VLA with IVRA enabled."""

from __future__ import annotations

import gc
import logging
import os
import sys
import time
from pathlib import Path

import cv2
import hydra
import numpy as np
import torch
import wandb
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

sys.path.insert(0, Path(__file__).absolute().parents[2].as_posix())

from libero.libero import benchmark, get_libero_path
from libero.libero.benchmark import get_benchmark
from libero.libero.envs import OffScreenRenderEnv
from libero.lifelong.utils import get_task_embs
from flower.evaluation.forward_IVRA import patch_encode_observations
from flower.evaluation.multistep_sequences import get_sequences
from flower.evaluation.utils import get_default_mode_and_env

logger = logging.getLogger(__name__)


def _safe_get_obs(env):
    """Read an observation across common LIBERO/gym wrapper variants."""
    if hasattr(env, "get_observation") and callable(env.get_observation):
        return env.get_observation()
    for name in ("get_obs", "_get_obs", "get_observations"):
        method = getattr(env, name, None)
        if callable(method):
            return method()
    for attr in ("last_obs", "obs", "observation"):
        if hasattr(env, attr):
            return getattr(env, attr)

    out = env.reset()
    if isinstance(out, tuple) and len(out) == 2:
        return out[0]
    return out


def _torch_device(device):
    if str(device) == "cpu":
        return torch.device("cpu")
    return torch.device(f"cuda:{device}")


def get_log_dir(log_dir):
    root = Path(log_dir) if log_dir is not None else Path(__file__).parents[3] / "evaluation"
    if not root.exists() and log_dir is None:
        root = Path("/tmp/evaluation")
    run_dir = root / "logs" / time.strftime("%Y-%m-%d_%H-%M-%S")
    run_dir.mkdir(parents=True, exist_ok=False)
    print(f"logging to {run_dir}")
    return run_dir


class EvaluateLibero:
    def __init__(
        self,
        model,
        transforms,
        log_dir,
        benchmark_name,
        num_sequences,
        max_steps,
        num_videos,
        n_eval,
        task_embedding_format,
        device,
    ):
        self.model = model
        self.transforms = transforms
        self.log_dir = Path(log_dir)
        self.device = _torch_device(device)
        self.task_order = 0
        self.bddl_folder = get_libero_path("bddl_files")
        self.task_embedding_format = task_embedding_format
        self.benchmark_name = benchmark_name
        self.benchmark_instance = benchmark.get_benchmark_dict()[benchmark_name]()
        self.num_tasks = self.benchmark_instance.get_num_tasks()
        self.task_names = self.benchmark_instance.get_task_names()
        self.benchmark = get_benchmark(benchmark_name)(self.task_order)
        self.n_eval = n_eval
        self.num_sequences = num_sequences
        self.max_steps = max_steps
        self.num_videos = num_videos
        self.img_h = 224
        self.img_w = 224
        self.cfg = self.create_cfg_for_libero(task_embedding_format)

        descriptions = [self.benchmark_instance.get_task(i).language for i in range(self.num_tasks)]
        self.benchmark_instance.set_task_embs(get_task_embs(self.cfg, descriptions))
        self.all_tasks = list(range(self.benchmark_instance.n_tasks))

    def setup(self) -> None:
        if self.benchmark is None:
            self.benchmark = get_benchmark(self.benchmark_name)(get_sequences(self.num_sequences))

    def start(self) -> None:
        successes = self.evaluate_policy(self.model, store_video=self.num_videos)
        avg_success = sum(successes) / len(successes)

        metrics = {"eval_lh/avg_seq_len": avg_success}
        metrics.update({f"eval_lh/sr_{name}": success for name, success in zip(self.task_names, successes)})
        if wandb.run is not None:
            wandb.log(metrics)

        logger.info("eval_lh/avg_seq_len success rate %.4f", avg_success)
        for task_name, success in zip(self.task_names, successes):
            logger.info("eval_lh/sr_%s success %.4f", task_name, success)
            print(f"Task {task_name} success rate: {success:.4f}")

    def evaluate_policy(self, model, store_video=False):
        successes = []
        for idx in self.all_tasks:
            task_name = self.task_names[idx]
            task = self.benchmark_instance.get_task(idx)
            task_emb = self.benchmark_instance.task_embs[idx]
            task_str = f"k{self.all_tasks[-1]}_p{idx}"
            logger.info("starting to evaluate: %s", task_name)
            successes.append(self.evaluate_task(model, task, task_emb, task_str, idx, store_video=store_video))
        return successes

    def evaluate_task(self, model, task, task_emb, task_str, idx, sim_states=None, store_video=0):
        env_args = {
            "bddl_file_name": os.path.join(self.bddl_folder, task.problem_folder, task.bddl_file),
            "camera_heights": self.img_h,
            "camera_widths": self.img_w,
        }

        env = None
        for attempt in range(5):
            try:
                env = OffScreenRenderEnv(**env_args)
                break
            except Exception:
                if attempt == 4:
                    raise
                time.sleep(5)

        try:
            initial_states = self.benchmark_instance.get_task_init_states(idx)
        except Exception as exc:
            logger.warning("Could not load LIBERO initial states for task %s: %s", idx, exc)
            initial_states = None

        num_success = 0
        try:
            for episode_idx in tqdm(range(self.n_eval), desc="Evaluating"):
                video_writer = None
                video_frames = []
                if episode_idx < store_video:
                    video_path = self.log_dir / f"rollout_{task_str}_nmp_{episode_idx}.mp4"
                    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                    video_writer = cv2.VideoWriter(str(video_path), fourcc, 20.0, (self.img_w, self.img_h))

                env.reset()
                model.reset()
                done = False
                steps = 0

                if initial_states is not None and episode_idx < len(initial_states):
                    try:
                        obs = env.set_init_state(initial_states[episode_idx])
                    except Exception as exc:
                        logger.warning("Failed to set LIBERO initial state: %s", exc)
                        obs = _safe_get_obs(env)
                else:
                    obs = _safe_get_obs(env)

                for _ in range(5):
                    obs, _, _, _ = env.step(np.zeros(7))

                if task_str and sim_states is not None:
                    sim_states[episode_idx].append(env.get_sim_state())

                while steps < self.max_steps:
                    steps += 1
                    data, goal = self.process_env_obs(obs, task_emb, task.language)
                    actions = model.step(data, goal).cpu().numpy()
                    obs, _reward, done, _info = env.step(actions)

                    if video_writer is not None:
                        video_frames.append(obs["agentview_image"])
                    if done:
                        break

                if video_writer is not None:
                    for frame in video_frames:
                        video_writer.write(frame)
                    video_writer.release()

                num_success += int(done)
        finally:
            if env is not None:
                env.close()
            gc.collect()

        return num_success / self.n_eval

    @staticmethod
    def create_cfg_for_libero(task_embedding_format):
        cfg = DictConfig({"task_embedding_format": task_embedding_format, "data": {"max_word_len": 25}})
        cfg.policy = OmegaConf.create()
        cfg.policy.language_encoder = OmegaConf.create()
        cfg.policy.language_encoder.network_kwargs = OmegaConf.create()
        return cfg

    @staticmethod
    def translate_obs_space(obs_space):
        rgb_obs = {}
        if "agentview_image" in obs_space:
            rgb_obs["rgb_static"] = obs_space["agentview_image"]
        if "robot0_eye_in_hand_image" in obs_space:
            rgb_obs["rgb_gripper"] = obs_space["robot0_eye_in_hand_image"]

        translated = {"rgb_obs": rgb_obs, "depth_obs": {}}
        if "robot0_joint_pos" in obs_space:
            translated["robot_obs"] = obs_space["robot0_joint_pos"]
        if "robot0_gripper_qpos" in obs_space:
            translated["gripper_states"] = obs_space["robot0_gripper_qpos"]
        return translated

    def _get_transforms_for_eval(self):
        if hasattr(self.transforms, "get"):
            return self.transforms.get("val", self.transforms)
        return self.transforms

    def apply_transforms(self, data):
        transforms_to_use = self._get_transforms_for_eval()
        fallback_keys = {
            "rgb_static": ("rgb", "agentview", "static", "agentview_rgb"),
            "rgb_gripper": ("gripper", "eye_in_hand", "hand", "eye_in_hand_rgb"),
        }

        for key, image in data["rgb_obs"].items():
            if len(image.shape) == 3:
                image = np.expand_dims(image, axis=0)
            tensor = torch.from_numpy(image).byte().permute(0, 3, 1, 2)

            transform_key = key
            if hasattr(transforms_to_use, "keys") and transform_key not in transforms_to_use:
                transform_key = next((alt for alt in fallback_keys.get(key, ()) if alt in transforms_to_use), None)

            if transform_key is not None and transform_key in transforms_to_use:
                for transform in transforms_to_use[transform_key]:
                    tensor = transform(tensor)
            else:
                tensor = tensor.float() / 255.0
            data["rgb_obs"][key] = tensor.unsqueeze(0).to(self.device)

        for key in ("robot_obs", "gripper_states"):
            if key in data and not isinstance(data[key], torch.Tensor):
                data[key] = torch.tensor(data[key], dtype=torch.float32).unsqueeze(0).to(self.device)
        return data

    def process_env_obs(self, env_obs, lang_embed, lang_text=None):
        data = self.apply_transforms(self.translate_obs_space(env_obs))
        return data, {"lang_text": lang_text, "lang": lang_embed}


@hydra.main(config_path="../../conf", config_name="eval_libero")
def main(cfg):
    model, _, dm, _ = get_default_mode_and_env(
        cfg.train_folder,
        cfg.dataset_path,
        cfg.checkpoint,
        env=42,
        lang_embeddings=None,
        eval_cfg_overwrite=cfg.eval_cfg_overwrite,
        device_id=cfg.device,
        prep_dm_and_deps=False,
    )
    patch_encode_observations(model)

    device = _torch_device(cfg.device)
    model = model.to(device)
    model.eval()

    log_dir = get_log_dir(cfg.log_dir)
    transforms = hydra.utils.instantiate(dm.transforms)
    eval_libero = EvaluateLibero(
        model=model,
        transforms=transforms,
        log_dir=log_dir,
        benchmark_name=cfg.benchmark_name,
        num_sequences=cfg.num_sequences,
        num_videos=cfg.num_videos,
        max_steps=cfg.max_steps,
        n_eval=cfg.n_eval,
        task_embedding_format=cfg.task_embedding_format,
        device=cfg.device,
    )

    run = None
    if cfg.log_wandb:
        (log_dir / "wandb").mkdir(exist_ok=False)
        run = wandb.init(
            project="ivra_flower_libero_eval",
            entity=cfg.wandb_entity,
            config=OmegaConf.to_object(cfg),
        )

    eval_libero.setup()
    eval_libero.start()

    if run is not None:
        run.finish()


if __name__ == "__main__":
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    main()
