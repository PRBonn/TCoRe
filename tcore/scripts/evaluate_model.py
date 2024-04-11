import os
from os.path import join

import click
import torch
import yaml
from easydict import EasyDict as edict
from pytorch_lightning import Trainer
from tcore.datasets.fruits import IGGFruitDatasetModule
from tcore.models.model import TCoRe


def getDir(obj):
    return os.path.dirname(os.path.abspath(obj))


@click.command()
@click.option("--w", type=str, required=True)
@click.option("--bb_cr", type=float, default=None, required=False)
@click.option("--dec_cr", type=float, default=None, required=False)
@click.option("--iterative", is_flag=True)
@click.option("--model_cfg_path", type=str, default="../config/model.yaml", required=False)
@click.option("--vis_pred", is_flag=True)
@click.option("--vis_int_pred", is_flag=True)
def main(w, bb_cr, dec_cr, iterative, model_cfg_path, vis_pred, vis_int_pred):
    model_cfg = edict(yaml.safe_load(open(join(getDir(__file__), model_cfg_path))))
    backbone_cfg = edict(
        yaml.safe_load(open(join(getDir(__file__), "../config/backbone.yaml")))
    )
    decoder_cfg = edict(
        yaml.safe_load(open(join(getDir(__file__), "../config/decoder.yaml")))
    )
    cfg = edict({**model_cfg, **backbone_cfg, **decoder_cfg})

    cfg.EVALUATE = True

    if cfg.MODEL.DATASET == "FRUITS":
        data = IGGFruitDatasetModule(cfg)
    else:
        raise NotImplementedError

    if bb_cr:
        cfg.BACKBONE.CR = bb_cr
    if dec_cr:
        cfg.DECODER.CR = dec_cr

    if iterative:
        cfg.DECODER.ITERATIVE_TEMPLATE = True

    if vis_pred:
        cfg.VIZ = True
    if vis_int_pred:
        cfg.VIZ_INT = True

    model = TCoRe(cfg)
    w = torch.load(w, map_location="cpu")
    model.load_state_dict(w["state_dict"], strict=False)

    cfg.UPDATE_METRICS = "True"
    trainer = Trainer(gpus=cfg.TRAIN.N_GPUS, logger=False)
    trainer.test(model, dataloaders=data.test_dataloader())


if __name__ == "__main__":
    main()
