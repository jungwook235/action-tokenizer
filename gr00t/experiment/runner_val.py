"""
TrainRunnerWithVal: eval_dataset을 지원하는 TrainRunner 서브클래스.
"""

import torch
from transformers import TrainingArguments

from gr00t.data.dataset import LeRobotMixtureDataset, LeRobotSingleDataset
from gr00t.experiment.runner import TrainRunner
from gr00t.experiment.trainer import DualBrainTrainer
from gr00t.model.gr00t_n1 import GR00T_N1_5
from gr00t.model.transforms import DefaultDataCollator
from gr00t.utils.experiment import CheckpointFormatCallback


class TrainRunnerWithVal(TrainRunner):
    """
    eval_dataset을 받아 validation을 수행할 수 있는 TrainRunner 서브클래스.

    Args:
        eval_dataset: validation에 사용할 dataset. None이면 기존 TrainRunner와 동일하게 동작.
    """

    def __init__(
        self,
        model: GR00T_N1_5,
        training_args: TrainingArguments,
        train_dataset: LeRobotSingleDataset | LeRobotMixtureDataset,
        resume_from_checkpoint: bool = False,
        eval_dataset: LeRobotSingleDataset | None = None,
    ):
        # create_trainer 오버라이드에서 참조하므로 super().__init__ 이전에 설정
        self._eval_dataset = eval_dataset
        super().__init__(
            model=model,
            training_args=training_args,
            train_dataset=train_dataset,
            resume_from_checkpoint=resume_from_checkpoint,
        )

    def create_trainer(
        self,
        model,
        training_args,
        train_dataset,
        data_collator,
        compute_dtype,
        global_batch_size=None,
    ):
        if global_batch_size is not None:
            bs = training_args.per_device_train_batch_size
            num_gpus = torch.cuda.device_count()
            grad_acc = max(1, global_batch_size // (bs * num_gpus))
            training_args.gradient_accumulation_steps = grad_acc
            print(f"Set global batch size to {global_batch_size}, grad accumulation steps to {grad_acc}")

        trainer = DualBrainTrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=self._eval_dataset,
            data_collator=data_collator,
            compute_dtype=compute_dtype,
        )

        run_name = training_args.run_name
        ckpt_format_callback = CheckpointFormatCallback(
            run_name=run_name, exp_cfg_dir=self.exp_cfg_dir
        )
        trainer.add_callback(ckpt_format_callback)

        train_dl_len = len(trainer.get_train_dataloader())
        eval_info = (
            f"eval dataset length: {len(self._eval_dataset)}\n"
            if self._eval_dataset is not None
            else "eval dataset: None\n"
        )
        print(
            f"train dataloader length: {train_dl_len}\n"
            f"train dataset length: {len(trainer.train_dataset)}\n"
            + eval_info
            + f"GPU memory before training: {torch.cuda.memory_allocated() / 1024 / 1024 / 1024:.2f} GB",
            flush=True,
        )
        return trainer
