import training.config as cfg
from models.segformer3d_variant import SegFormer3D
from training.config import *
from training.data import PanoramaDataModule
from training.net import Net
from training.utils import set_global_seed, make_warmup_poly
import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
from lightning.pytorch.loggers import TensorBoardLogger
from monai.losses import DiceCELoss
from torch.optim import AdamW

def run_training_job():
    set_global_seed(17)
    
data_module = PanoramaDataModule(
    data_dir=cfg.DATA_DIR,
    batch_size=cfg.BATCH_SIZE,
    num_workers=cfg.NUM_WORKERS,
    train_size=cfg.TRAIN_SIZE,
    val_size=cfg.VAL_SIZE,
    train_patch_size=cfg.PATCH_SIZE,
)

segformer_model = SegFormer3D(
    in_channels=cfg.IN_CHANNELS,
    num_classes=cfg.NUM_CLASSES,
    embed_dims=cfg.EMBED_DIMS,
    depths=cfg.DEPTHS,
    num_heads=cfg.NUM_HEADS,
    sr_ratios=cfg.SR_RATIOS,
    decoder_C=cfg.DECODER_C,
    aspp_rates=cfg.ASPP_RATES,
    fuse_all=cfg.FUSE_ALL,
)

optimizer = AdamW(segformer_model.parameters(), lr=cfg.LR, weight_decay=cfg.WEIGHT_DECAY)

lightning_model = Net(
    model=segformer_model,
    loss_function=DiceCELoss(to_onehot_y=True, softmax=True),
    optimizer=optimizer,
    lr_scheduler=make_warmup_poly(
        optimizer,
        warmup_steps=cfg.WARMUP_STEPS,
        total_steps=cfg.TOTAL_STEPS,
        power=cfg.POLY_POWER,
    ),
    patch_size=cfg.PATCH_SIZE,
    num_classes=cfg.NUM_CLASSES,
)

logger = TensorBoardLogger(save_dir=cfg.CHECKPOINT_DIR, name=cfg.LOG_NAME)

    checkpoint_callback = ModelCheckpoint(
        monitor="total_val_dice_score", mode="max", save_top_k=1,
        filename="best-{epoch:02d}-{total_val_dice_score:.4f}"
    )

    early_stopping = EarlyStopping(
        monitor="total_val_dice_score", mode="max", patience=cfg.PATIENCE, min_delta=cfg.MIN_DELTA
    )

    trainer = L.Trainer(
        max_steps=cfg.MAX_STEPS,
        accelerator="gpu",
        devices=1,
        logger=logger,
        callbacks=[checkpoint_callback, early_stopping],
        log_every_n_steps=10,
        default_root_dir=cfg.CHECKPOINT_DIR,
        gradient_clip_val=cfg.GRAD_CLIP,
    )

    trainer.fit(lightning_model, datamodule=data_module)

if __name__ == "__main__":
    run_training_job()
