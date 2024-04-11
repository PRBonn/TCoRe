import os
import subprocess
from os.path import join

import click
import torch
import yaml
from easydict import EasyDict as edict
from tcore.datasets.fruits import IGGFruitDatasetModule
from tcore.models.model import TCoRe
from pytorch_lightning import Trainer
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint


@click.command()
@click.option("--w", type=str, required=False)
@click.option("--ckpt", type=str, default=None, required=False)
@click.option("--bb_cr", type=float, default=None, required=False)
@click.option("--dec_cr", type=float, default=None, required=False)
@click.option("--dec_blocks", type=int, default=None, required=False)
@click.option("--iterative", is_flag=True)
@click.option("--model_cfg_path", type=str, default="../config/model.yaml", required=False)
def main(w, ckpt, bb_cr, dec_cr, dec_blocks, iterative, model_cfg_path):
    model_cfg = edict(
        yaml.safe_load(open(join(getDir(__file__), model_cfg_path)))
    )
    backbone_cfg = edict(
        yaml.safe_load(open(join(getDir(__file__), "../config/backbone.yaml")))
    )
    decoder_cfg = edict(
        yaml.safe_load(open(join(getDir(__file__), "../config/decoder.yaml")))
    )
    cfg = edict({**model_cfg, **backbone_cfg, **decoder_cfg})
    cfg.git_commit_version = str(
        subprocess.check_output(["git", "rev-parse", "--short", "HEAD"]).strip()
    )

    if cfg.MODEL.DATASET == "FRUITS":
        data = IGGFruitDatasetModule(cfg)
    else:
        raise NotImplementedError

    if bb_cr:
        cfg.BACKBONE.CR = bb_cr
    if dec_cr:
        cfg.DECODER.CR = dec_cr
    if dec_blocks:
        cfg.DECODER.DEC_BLOCKS = dec_blocks


    if iterative:
        cfg.DECODER.ITERATIVE_TEMPLATE = True
    model = TCoRe(cfg)
    if w:
        w = torch.load(w, map_location="cpu")
        model.load_state_dict(w["state_dict"])

    tb_logger = pl_loggers.TensorBoardLogger(
        cfg.LOGDIR + cfg.EXPERIMENT.ID, default_hp_metric=False
    )

    # Callbacks
    lr_monitor = LearningRateMonitor(logging_interval="step")

    cd_ckpt = ModelCheckpoint(
        monitor="val_chamfer_distance",
        filename=cfg.EXPERIMENT.ID +
        "_epoch{epoch:02d}_cd{val_chamfer_distance:.2f}",
        auto_insert_metric_name=False,
        mode="min",
        save_last=True,
    )

    precision_ckpt = ModelCheckpoint(
        monitor="val_precision_auc",
        filename=cfg.EXPERIMENT.ID +
        "_epoch{epoch:02d}_pr{val_precision_auc:.2f}",
        auto_insert_metric_name=False,
        mode="max",
        save_last=True,
    )

    recall_ckpt = ModelCheckpoint(
        monitor="val_recall_auc",
        filename=cfg.EXPERIMENT.ID +
        "_epoch{epoch:02d}_re{val_recall_auc:.2f}",
        auto_insert_metric_name=False,
        mode="max",
        save_last=True,
    )

    fscore_ckpt = ModelCheckpoint(
        monitor="val_fscore_auc",
        filename=cfg.EXPERIMENT.ID + "_epoch{epoch:02d}_f{val_fscore_auc:.2f}",
        auto_insert_metric_name=False,
        mode="max",
        save_last=True,
    )

    trainer = Trainer(
        num_sanity_val_steps=0,
        gpus=cfg.TRAIN.N_GPUS,
        accelerator="cuda",
        logger=tb_logger,
        max_epochs=cfg.TRAIN.MAX_EPOCH,
        callbacks=[lr_monitor, fscore_ckpt,
                   precision_ckpt, recall_ckpt, cd_ckpt],
        log_every_n_steps=1,
        gradient_clip_val=0.5,
        accumulate_grad_batches=cfg.TRAIN.BATCH_ACC,
        resume_from_checkpoint=ckpt,
        check_val_every_n_epoch=10,
    )


    trainer.fit(model, data)
    trainer.test(model, dataloaders=data.test_dataloader())


def getDir(obj):
    return os.path.dirname(os.path.abspath(obj))


if __name__ == "__main__":
    main()
