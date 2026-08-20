# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import os
import shutil
from typing import Optional

import torch
import transformers
from torch.utils.data import Dataset, Sampler
from transformers.trainer import (
    ALL_LAYERNORM_LAYERS,
    TRAINER_STATE_NAME,
    TrainerState,
    get_last_checkpoint,
    get_parameter_names,
    is_sagemaker_mp_enabled,
)
import numpy as np 
torch.serialization.add_safe_globals([
np.core.multiarray._reconstruct,
np.ndarray,
np.dtype,
np.dtypes.UInt32DType
])

class BaseSampler(Sampler):
    """Sampler for dataset, which enables `set_epoch` for Dataset.
    `set_epoch` will be called by huggingface Trainer at the end of each epoch.
    `shuffle` is also supported for training set shuffling
    """

    def __init__(self, data_source: Dataset, shuffle: bool = False, seed: int = 0):
        self.data_source = data_source
        self.shuffle = shuffle
        self.seed = seed
        self.epoch = 0

    def __iter__(self):
        if self.shuffle:
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)
            # must not add rank here, or randomization will be different for each rank
            return iter(torch.randperm(len(self.data_source), generator=g).tolist())
        return iter(range(len(self.data_source)))

    def set_epoch(self, epoch):
        self.epoch = epoch
        if hasattr(self.data_source, "set_epoch"):
            # this is important for dataset
            self.data_source.set_epoch(epoch)

    def __len__(self):
        return len(self.data_source)


class S3CompatCheckpointStaging:
    """Mixin: GR00T_S3_COMPAT=1 checkpoint staging for S3-mounted output dirs.

    gpu26/AWS /s3ckpt (mountpoint-s3) rejects safetensors writes, renames and
    chmod/utime, so HF Trainer checkpointing fails there. With GR00T_S3_COMPAT=1
    the checkpoint is written to a local disk (GR00T_CKPT_STAGE_DIR, e.g. node
    /scratch) and then data-only-copied to the real output dir. Flag off/unset →
    stock behavior, bit-identical. Mix in BEFORE transformers.Trainer.
    """

    def _save_checkpoint(self, model, trial):
        if os.environ.get("GR00T_S3_COMPAT") != "1":
            return super()._save_checkpoint(model, trial)
        stage_root = os.environ.get("GR00T_CKPT_STAGE_DIR") or os.path.join(
            "/scratch",
            os.environ.get("USER", "user"),
            f"ckpt_stage_{os.environ.get('SLURM_JOB_ID', str(os.getpid()))}",
        )
        real_out = self.args.output_dir
        ckpt_name = f"checkpoint-{self.state.global_step}"
        os.makedirs(stage_root, exist_ok=True)
        try:
            self.args.output_dir = stage_root
            result = super()._save_checkpoint(model, trial)
        finally:
            self.args.output_dir = real_out
        src = os.path.join(stage_root, ckpt_name)
        dst = os.path.join(real_out, ckpt_name)
        # all ranks write into the staging dir (rng_state etc.) — wait for them
        # before rank 0 copies the tree out
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.barrier()
        if self.args.should_save:
            self._copy_tree_data_only(src, dst)
            shutil.rmtree(src)  # staging is scratch-only; the S3 copy is canonical
        return result

    @staticmethod
    def _copy_tree_data_only(src, dst):
        # copytree minus metadata: S3 mounts (gpu26 /s3ckpt) reject chmod/utime, so
        # shutil.copy2/copystat — and therefore stock copytree — die with EPERM
        # there even though the file data itself copies fine.
        for root, _, files in os.walk(src):
            rel = os.path.relpath(root, src)
            troot = dst if rel == "." else os.path.join(dst, rel)
            os.makedirs(troot, exist_ok=True)
            for f in files:
                shutil.copyfile(os.path.join(root, f), os.path.join(troot, f))


class DualBrainTrainer(S3CompatCheckpointStaging, transformers.Trainer):
    def __init__(self, **kwargs):
        self.compute_dtype = kwargs.pop("compute_dtype")
        super().__init__(**kwargs)

    def _get_train_sampler(self):
        return BaseSampler(self.train_dataset, shuffle=True, seed=self.args.seed)

    def _get_eval_sampler(self, eval_dataset):
        return BaseSampler(eval_dataset, shuffle=False)

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        outputs = model(inputs)
        loss = outputs["loss"]
        #print("loss:", loss, flush=True)
        return (loss, outputs) if return_outputs else loss
    
    
    def log(self, logs: dict[str, float], start_time=None) -> None:
        """Override to add gamma_output statistics to wandb"""
        
        # Extract gamma_output from model if it exists
        #try:
        if hasattr(self.model, 'module'):
            # DDP wrapped model
            action_head = self.model.module.action_head
        else:
            # Single GPU model
            action_head = self.model.action_head
        
        # Check if gamma_output exists
        if hasattr(action_head, 'gamma_output'):
            gamma_output = action_head.gamma_output  # (trm_hidden_size,)
            
            # Add statistics to logs
            with torch.no_grad():
                gamma_sigmoid = torch.sigmoid(gamma_output)
                logs['trm/gamma_output_mean'] = gamma_sigmoid.mean().item()
                logs['trm/gamma_output_std'] = gamma_sigmoid.std().item()
                logs['trm/gamma_output_min'] = gamma_sigmoid.min().item()
                logs['trm/gamma_output_max'] = gamma_sigmoid.max().item()
        
        
        if hasattr(action_head, 'flare_loss') and action_head.flare_loss is not None:
            #print(f"Flare loss: {action_head.flare_loss.item()}", flush=True)
            if torch.is_tensor(action_head.flare_loss):
                logs['loss/flare_loss'] = action_head.flare_loss.item()
            else:
                logs['loss/flare_loss'] = action_head.flare_loss

        if hasattr(action_head, 'trm_reasoning_loss') and action_head.trm_reasoning_loss is not None:
            #print(f"TRM reasoning loss: {action_head.trm_reasoning_loss.item()}", flush=True)
            if torch.is_tensor(action_head.trm_reasoning_loss):
                logs['loss/trm_reasoning_loss'] = action_head.trm_reasoning_loss.item()
            else:
                logs['loss/trm_reasoning_loss'] = action_head.trm_reasoning_loss

        if hasattr(action_head, 'trm_action_loss') and action_head.trm_action_loss is not None:
            #print(f"TRM action loss: {action_head.trm_action_loss.item()}", flush=True)
            if torch.is_tensor(action_head.trm_action_loss):
                logs['loss/trm_action_loss'] = action_head.trm_action_loss.item()
            else:
                logs['loss/trm_action_loss'] = action_head.trm_action_loss

        if hasattr(action_head, 'action_loss') and action_head.action_loss is not None:
            #print(f"Action loss: {action_head.action_loss.item()}", flush=True)
            if torch.is_tensor(action_head.action_loss):
                logs['loss/action_loss'] = action_head.action_loss.item()
            else:
                logs['loss/action_loss'] = action_head.action_loss

        if hasattr(action_head, 'discrete_action_loss') and action_head.discrete_action_loss is not None:
            #print(f"Action loss: {action_head.discrete_action_loss.item()}", flush=True)
            if torch.is_tensor(action_head.discrete_action_loss):
                logs['loss/discrete_action_loss'] = action_head.discrete_action_loss.item()
            else:
                logs['loss/discrete_action_loss'] = action_head.discrete_action_loss
        if hasattr(action_head, 'vggt_depth_loss') and action_head.vggt_depth_loss is not None:
            #print(f"VGGT depth loss: {action_head.vggt_depth_loss.item()}", flush=True)
            if torch.is_tensor(action_head.vggt_depth_loss):
                logs['loss/vggt_depth_loss'] = action_head.vggt_depth_loss.item()
            else:
                logs['loss/vggt_depth_loss'] = action_head.vggt_depth_loss
        if hasattr(action_head, 'vggt_world_points_loss') and action_head.vggt_world_points_loss is not None:
            #print(f"VGGT world points loss: {action_head.vggt_world_points_loss.item()}", flush=True)
            if torch.is_tensor(action_head.vggt_world_points_loss):
                logs['loss/vggt_world_points_loss'] = action_head.vggt_world_points_loss.item()
            else:
                logs['loss/vggt_world_points_loss'] = action_head.vggt_world_points_loss
        
        if hasattr(action_head, 'maetok_recon_loss') and action_head.maetok_recon_loss is not None:
            #print(f"VGGT world points loss: {action_head.vggt_world_points_loss.item()}", flush=True)
            if torch.is_tensor(action_head.maetok_recon_loss):
                logs['loss/maetok_recon_loss'] = action_head.maetok_recon_loss.item()
            else:
                logs['loss/maetok_recon_loss'] = action_head.maetok_recon_loss
                
        if hasattr(action_head, 'maetok_action_loss') and action_head.maetok_action_loss is not None:
            #print(f"VGGT world points loss: {action_head.vggt_world_points_loss.item()}", flush=True)
            if torch.is_tensor(action_head.maetok_action_loss):
                logs['loss/maetok_action_loss'] = action_head.maetok_action_loss.item()
            else:
                logs['loss/maetok_action_loss'] = action_head.maetok_action_loss
        #except Exception as e:
            # Silently skip if there's any error accessing gamma_output
            #pass
        
        # Call parent's log method (automatically sends to wandb)
        super().log(logs, start_time)


    def create_optimizer(self):
        """
        Setup the optimizer.

        We provide a reasonable default that works well. If you want to use something else, you can pass a tuple in the
        Trainer's init through `optimizers`, or subclass and override this method in a subclass.
        """
        if is_sagemaker_mp_enabled():
            return super().create_optimizer()

        opt_model = self.model

        if self.optimizer is None:
            decay_parameters = get_parameter_names(opt_model, ALL_LAYERNORM_LAYERS)
            decay_parameters = [name for name in decay_parameters if "bias" not in name]
            optimizer_grouped_parameters = [
                {
                    "params": [
                        p
                        for n, p in opt_model.named_parameters()
                        if (n in decay_parameters and p.requires_grad)
                    ],
                    "weight_decay": self.args.weight_decay,
                },
                {
                    "params": [
                        p
                        for n, p in opt_model.named_parameters()
                        if (n not in decay_parameters and p.requires_grad)
                    ],
                    "weight_decay": 0.0,
                },
            ]

            optimizer_cls, optimizer_kwargs = transformers.Trainer.get_optimizer_cls_and_kwargs(
                self.args
            )
            self.optimizer = optimizer_cls(optimizer_grouped_parameters, **optimizer_kwargs)

        return self.optimizer

    def save_model(self, output_dir: Optional[str], _internal_call: bool):
        ## save tuned model separately
        if self.is_deepspeed_enabled:
            state_dict = self.accelerator.get_state_dict(self.deepspeed)
        else:
            state_dict = self.model.state_dict()

        if self.args.should_save:
            return self.model.save_pretrained(output_dir, state_dict=state_dict)

    def train(
        self,
        resume_from_checkpoint=None,
        trial=None,
        ignore_keys_for_eval=None,
        **kwargs,
    ):
        """Correctly set self.state from checkpoint so get_train_dataloader can read from it."""
        if resume_from_checkpoint is False:
            resume_from_checkpoint = None

        if isinstance(resume_from_checkpoint, bool) and resume_from_checkpoint:
            # get_last_checkpoint() os.listdir()s the dir, so a --resume run whose
            # output_dir does not exist yet (first run of a stage) would raise
            # FileNotFoundError instead of falling through to "from scratch".
            resume_from_checkpoint = (
                get_last_checkpoint(self.args.output_dir)
                if os.path.isdir(self.args.output_dir)
                else None
            )
            if resume_from_checkpoint is None:
                print(f"No valid checkpoint found in output directory ({self.args.output_dir})")
                print(f"Continuing training from scratch")
                resume_from_checkpoint = None
                #raise ValueError(
                #   f"No valid checkpoint found in output directory ({self.args.output_dir})"
                #)

        if resume_from_checkpoint is not None:
            # In case of repeating the find_executable_batch_size, set `self._train_batch_size` properly
            self.state = TrainerState.load_from_json(
                os.path.join(resume_from_checkpoint, TRAINER_STATE_NAME)
            )
        return super().train(resume_from_checkpoint, trial, ignore_keys_for_eval, **kwargs)

    def _load_optimizer_and_scheduler(self, checkpoint):
        """Override to skip optimizer loading if file doesn't exist (for analysis mode)."""
        try:
            # Try to load optimizer and scheduler normally
            super()._load_optimizer_and_scheduler(checkpoint)
        except FileNotFoundError as e:
            # If optimizer.pt or scheduler.pt is missing, just skip
            print(f"⚠️  Optimizer/Scheduler file not found: {e}")
            print(f"⚠️  Skipping optimizer and scheduler loading (OK for analysis mode).")
        except Exception as e:
            # Catch any other errors related to optimizer loading
            print(f"⚠️  Error loading optimizer/scheduler: {e}")
            print(f"⚠️  Skipping optimizer and scheduler loading (OK for analysis mode).")
