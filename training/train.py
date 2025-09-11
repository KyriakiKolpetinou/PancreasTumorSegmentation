import training.config as cfg

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
