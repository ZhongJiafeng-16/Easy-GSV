import torch
import click
import lightning as L

from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.callbacks import ModelCheckpoint

import rootutils
root = rootutils.setup_root(__file__, dotenv=True, pythonpath=True, cwd=False)

from typing import Dict
from pathlib import Path
from loguru import logger
from dataset.sovits import TextAudioSpeakerDataModule
from modules.sovits import (
    SynthesizerTrn, 
    MultiPeriodDiscriminator,
    generator_loss, 
    discriminator_loss, 
    feature_loss, 
    kl_loss,
    mel_spectrogram_torch,
    spec_to_mel_torch,
    slice_segments,
)
from utils.io import load_yaml_config

class SovitsLightningModule(L.LightningModule):
    def __init__(self, config, output_dir, is_train=True, ckpt_path=None):
        super().__init__()
        self.config = config

        self.net_g = SynthesizerTrn(
            config.data.filter_length // 2 + 1,
            config.train.segment_size // config.data.hop_length,
            n_speakers=config.data.n_speakers,
            **config.model,
        )
        self.net_d = MultiPeriodDiscriminator(config.model.use_spectral_norm)
        if is_train:
            self.automatic_optimization = False
            self.save_hyperparameters()
            self.eval_dir = output_dir / "eval"
            self.eval_dir.mkdir(parents=True, exist_ok=True)

    def training_step(self, batch: Dict, batch_idx: int):
        opt_g, opt_d = self.optimizers()
        scheduler_g, scheduler_d = self.lr_schedulers()

        # train discriminator
        self.toggle_optimizer(opt_d)
        (
            y_hat,
            kl_ssl,
            ids_slice,
            x_mask,
            z_mask,
            (z, z_p, m_p, logs_p, m_q, logs_q),
            stats_ssl,
        ) = self.net_g(
            ssl=batch["ssl"], 
            y=batch["spec"], 
            y_lengths=["spec_lengths"], 
            text=batch["phoneme"], 
            text_lengths=batch["phoneme_len"],
        )

        mel = spec_to_mel_torch(
            batch["spec"],
            self.config.data.filter_length,
            self.config.data.n_mel_channels,
            self.config.data.sampling_rate,
            self.config.data.mel_fmin,
            self.config.data.mel_fmax,
        )
        y_mel = slice_segments(
            mel, ids_slice, self.config.train.segment_size // self.config.data.hop_length
        )

        y_hat_mel = mel_spectrogram_torch(
                y_hat.squeeze(1),
                self.config.data.filter_length,
                self.config.data.n_mel_channels,
                self.config.data.sampling_rate,
                self.config.data.hop_length,
                self.config.data.win_length,
                self.config.data.mel_fmin,
                self.config.data.mel_fmax,
            )
        
        y = slice_segments(
                y, ids_slice * self.config.data.hop_length, self.config.train.segment_size
            )  # slice
        
        y_d_hat_r, y_d_hat_g, _, _ = self.net_d(y, y_hat.detach())
        loss_disc, losses_disc_r, losses_disc_g = discriminator_loss(y_d_hat_r, y_d_hat_g)
        loss_disc_all = loss_disc
        self.log("d_loss", loss_disc_all, prog_bar=True)
        self.manual_backward(loss_disc_all)
        opt_d.step()
        opt_d.zero_grad()
        self.untoggle_optimizer(opt_d)

        # train generator
        self.toggle_optimizer(opt_g)

        y_d_hat_r, y_d_hat_g, fmap_r, fmap_g = self.net_d(y, y_hat)
        loss_mel = torch.nn.functional.l1_loss(y_mel, y_hat_mel) * self.config.train.c_mel
        loss_kl = kl_loss(z_p, logs_q, m_p, logs_p, z_mask) * self.config.train.c_kl

        loss_fm = feature_loss(fmap_r, fmap_g)
        loss_gen, losses_gen = generator_loss(y_d_hat_g)
        loss_gen_all = loss_gen + loss_fm + loss_mel + kl_ssl * 1 + loss_kl

        self.log("g_loss", loss_gen_all, prog_bar=True)
        self.log("g_loss_fm", loss_fm)
        self.log("g_loss_mel", loss_mel)
        self.log("g_loss_kl", loss_kl)
        self.log("g_loss_ssl_kl", kl_ssl)

        self.manual_backward(loss_gen_all)
        opt_g.step()
        opt_g.zero_grad()
        self.untoggle_optimizer(opt_g)
        scheduler_g.step()
        scheduler_d.step()

    def configure_optimizers(self):
        for name, param in self.net_g.named_parameters():
            if not param.requires_grad:
                logger.debug(f"{name} not requires_grad")

        te_p = list(map(id, self.net_g.enc_p.text_embedding.parameters()))
        et_p = list(map(id, self.net_g.enc_p.encoder_text.parameters()))
        mrte_p = list(map(id, self.net_g.enc_p.mrte.parameters()))
        base_params = filter(
            lambda p: id(p) not in te_p + et_p + mrte_p and p.requires_grad,
            self.net_g.parameters(),
        )

        optim_g = torch.optim.AdamW(
            # filter(lambda p: p.requires_grad, net_g.parameters()),###默认所有层lr一致
            [
                {
                    "params": base_params, "lr": self.config.train.learning_rate,
                },
                {
                    "params": self.net_g.enc_p.text_embedding.parameters(),
                    "lr": self.config.train.learning_rate * self.config.train.text_low_lr_rate,
                },
                {
                    "params": self.net_g.enc_p.encoder_text.parameters(),
                    "lr": self.config.train.learning_rate * self.config.train.text_low_lr_rate,
                },
                {
                    "params": self.net_g.enc_p.mrte.parameters(),
                    "lr": self.config.train.learning_rate * self.config.train.text_low_lr_rate,
                },
            ],
            self.config.train.learning_rate,
            betas=self.config.train.betas,
            eps=self.config.train.eps,
        )
        optim_d = torch.optim.AdamW(
            self.net_d.parameters(),
            self.config.train.learning_rate,
            betas=self.config.train.betas,
            eps=self.config.train.eps,
        )

        scheduler_g = torch.optim.lr_scheduler.ExponentialLR(
            optim_g, gamma=self.config.train.lr_decay, last_epoch=-1
        )
        scheduler_d = torch.optim.lr_scheduler.ExponentialLR(
            optim_d, gamma=self.config.train.lr_decay, last_epoch=-1
        )

        return [optim_g, optim_d], [scheduler_g, scheduler_d]
        

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
        filename='{epoch}-{g_loss:.2f}',
        every_n_epochs=1,
        # save_weights_only=True,
        dirpath=ckpt_dir,
        )

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

    model: SovitsLightningModule = SovitsLightningModule(
        config, output_dir
    )

    data_module: TextAudioSpeakerDataModule = TextAudioSpeakerDataModule(
        config,
        train_wav2phone_file=config["train_wav2phone_file"],
        train_hubert_dir=config["train_hubert_dir"],
    )

    logger.info("Start training.")
    trainer.fit(model, data_module, ckpt_path=None)
    logger.info("Training finish.")

if __name__ == "__main__":
    main()