import os
import torch
import click
import lightning as L

from loguru import logger
from pathlib import Path
from typing import Dict
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.callbacks import ModelCheckpoint

import rootutils
root = rootutils.setup_root(__file__, dotenv=True, pythonpath=True, cwd=False)

from dataset.semantic import Text2SemanticDataModule
from modules.gpt import (
    Text2SemanticDecoder,
    WarmupCosineLRSchedule,
    ScaledAdam,
)
from utils.io import load_yaml_config, get_newest_ckpt

class Text2SemanticLightningModule(L.LightningModule):
    def __init__(self, config, output_dir, is_train=True, ckpt_path=None):
        super().__init__()
        self.config = config
        self.top_k = 3
        self.model = Text2SemanticDecoder(config=config, top_k=self.top_k)
        if ckpt_path and is_train:
            self.model.load_state_dict(torch.load(ckpt_path, map_location="cpu")["weight"])
            
        if is_train:
            self.automatic_optimization = False
            self.save_hyperparameters()
            self.eval_dir = output_dir / "eval"
            self.eval_dir.mkdir(parents=True, exist_ok=True)

    def training_step(self, batch: Dict, batch_idx: int):
        opt = self.optimizers()
        scheduler = self.lr_schedulers()
        forward=self.model.forward if self.config["train"].get("if_dpo",False)==True else self.model.forward_old
        loss, acc = forward(
            batch["phoneme_ids"],
            batch["phoneme_ids_len"],
            batch["semantic_ids"],
            batch["semantic_ids_len"],
            batch["bert_feature"],
        )
        self.manual_backward(loss)
        if batch_idx > 0 and batch_idx % 4 == 0:
            opt.step()
            opt.zero_grad()
            scheduler.step()

        self.log(
            "total_loss",
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )
        self.log(
            "lr",
            scheduler.get_last_lr()[0],
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )
        self.log(
            f"top_{self.top_k}_acc",
            acc,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )

    # def validation_step(self, batch: Dict, batch_idx: int):
    #     return
    
    def configure_optimizers(self):
        model_parameters = self.model.parameters()
        parameters_names = []
        parameters_names.append(
            [name_param_pair[0] for name_param_pair in self.model.named_parameters()]
        )
        lm_opt = ScaledAdam(
            model_parameters,
            lr=0.01,
            betas=(0.9, 0.95),
            clipping_scale=2.0,
            parameters_names=parameters_names,
            show_dominant_parameters=False,
            clipping_update_period=1000,
        )

        return {
            "optimizer": lm_opt,
            "lr_scheduler": {
                "scheduler": WarmupCosineLRSchedule(
                    lm_opt,
                    init_lr=self.config["optimizer"]["lr_init"],
                    peak_lr=self.config["optimizer"]["lr"],
                    end_lr=self.config["optimizer"]["lr_end"],
                    warmup_steps=self.config["optimizer"]["warmup_steps"],
                    total_steps=self.config["optimizer"]["decay_steps"],
                )
            },
        }


@click.command()
@click.option("--config", type=str, default='config/s1longer-v2.yaml')
def main(config):
    config = load_yaml_config(config)

    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    ckpt_dir = output_dir / "ckpt"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    torch.set_float32_matmul_precision('high')
    L.seed_everything(config["train"]["seed"], workers=True)
    checkponint_callback = ModelCheckpoint(
        filename='{epoch}-{top_3_acc:.2f}',
        every_n_epochs=config["train"]["save_every_n_epoch"],
        save_top_k=-1,
        monitor='top_3_acc',
        mode='max',
        # save_weights_only=True,
        auto_insert_metric_name=False,
        dirpath=ckpt_dir,
        )
    os.environ["MASTER_ADDR"]="localhost"
    trainer: L.Trainer = L.Trainer(
        max_epochs=config["train"]["epochs"],
        accelerator="gpu",
        devices=[2,3],
        benchmark=False,
        fast_dev_run=False,
        strategy='auto',
        precision=config["train"]["precision"],
        logger=TensorBoardLogger(name=output_dir.stem, save_dir=output_dir),
        num_sanity_val_steps=0,
        callbacks=[checkponint_callback],
        use_distributed_sampler=False,
        limit_val_batches=0,
    )

    model: Text2SemanticLightningModule = Text2SemanticLightningModule(
        config, output_dir
    )

    data_module: Text2SemanticDataModule = Text2SemanticDataModule(
        config,
        train_semantic_path=config["train_semantic_path"],
        train_phoneme_path=config["train_phoneme_path"],
    )

    try:
        newest_ckpt_name = get_newest_ckpt(os.listdir(ckpt_dir))
        ckpt_path = ckpt_dir / newest_ckpt_name
    except Exception:
        ckpt_path = None

    logger.info(f"Coninute training with checkpoint path: <{ckpt_path}>")
    logger.info("Start training.")

    trainer.fit(model, data_module, ckpt_path=ckpt_path)
    logger.info("Training finish.")

if __name__ == "__main__":
    main()