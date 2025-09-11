class Net(L.LightningModule):
    def __init__(self, model, loss_function, optimizer, lr_scheduler, patch_size, num_classes, sw_batch_size=2):
        super().__init__()
        self.model = model
        self.loss_function = loss_function
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.num_classes = num_classes

        self.sliding_window_inferer = SlidingWindowInferer(
            roi_size=patch_size, sw_batch_size=sw_batch_size, overlap=0.5, mode="gaussian"
        )

        self.post_pred = AsDiscrete(argmax=True, to_onehot=num_classes, dim=1)
        self.post_label = AsDiscrete(to_onehot=num_classes, dim=1)

        self.mean_dice = DiceMetric(include_background=False, reduction="mean_batch")
        self.confusion_matrix = ConfusionMatrixMetric(
            include_background=False,
            metric_name=["sensitivity", "specificity", "precision"],
            reduction="sum_batch",
        )

    def configure_optimizers(self):
        if self.lr_scheduler is not None:
            return {
                "optimizer": self.optimizer,
                "lr_scheduler": {
                    "scheduler": self.lr_scheduler,
                    "interval": "step", 
                    "frequency": 1,
                },
            }
        return self.optimizer

    def forward(self, x):
        return self.model(x)
      
    def log_images(self, batch, preds, stage="val", max_images=1, num_slices=3):
        if not hasattr(self, "logger") or self.logger is None:
           return

        images = batch["image"].detach().cpu()  # (B, 1, D, H, W)
        labels = batch["label"].detach().cpu()  # (B, 1, D, H, W)
        preds  = preds.detach().cpu()           # (B, C, D, H, W)  (logits)

        B, _, D, H, W = images.shape

    
        seed = int((self.current_epoch + 1) * 97 + (self.global_step + 1) * 17)
        rng = np.random.RandomState(seed)

    #samples
        idxs = rng.choice(B, size=min(max_images, B), replace=False)

        for i, b in enumerate(idxs):
            img = images[b, 0]              # (D, H, W)
            gt  = labels[b, 0]              # (D, H, W)

            prob = torch.softmax(preds[b], dim=0)   # (C, D, H, W)
            pred_cls = torch.argmax(prob, dim=0)    # (D, H, W)

        
            slice_indices = rng.choice(D, size=min(num_slices, D), replace=False)

        # Helper
            def normalize(x):
                x = (x - x.min()) / (x.max() - x.min() + 1e-5)
                return x

            panels = []
            for z in slice_indices:
                image_slice = normalize(img[z])     # (H,W)
                gt_slice    = gt[z]                 # (H,W)
                pr_slice    = pred_cls[z]           # (H,W)

            # RGB base
                image_rgb = image_slice.repeat(3, 1, 1)  # (3,H,W)
                gt_overlay = image_rgb.clone()
                gt_overlay[0][gt_slice == 1] = 1.0
                gt_overlay[1][gt_slice == 2] = 1.0
              
                pred_overlay = image_rgb.clone()
                pred_overlay[0][pr_slice == 1] = 1.0
                pred_overlay[2][pr_slice == 2] = 1.0

                panel = torch.cat([image_rgb, gt_overlay, pred_overlay], dim=2)
                panels.append(panel)

        
                combined = torch.cat(panels, dim=1) 

                step = self.global_step if self.global_step is not None else self.current_epoch
                self.logger.experiment.add_image(f"{stage}_example_b{b}", combined, step)


    def training_step(self, batch, batch_idx):
        x, y = batch["image"], batch["label"]
        logits, logits_aux = self.forward(x)
        loss_main = self.loss_function(logits, y)
        loss_aux = self.loss_function(logits_aux, y)
        loss = loss_main + 0.3 * loss_aux  # ← deep supervision weight #changed from 0.4 to 0.3

         # 📊 Log LR every step
        current_lr = self.optimizer.param_groups[0]["lr"]
        self.log("lr", current_lr, prog_bar=False, on_step=True, on_epoch=False)

        self.log("train_loss", loss, prog_bar=True)
        if batch_idx == 0:
            self.log_images(batch, logits, stage="train")

        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch["image"], batch["label"]
        #logits, _ = self.sliding_window_inferer(x, self.model)
        logits = self.sliding_window_inferer(x, lambda t: self.model(t)[0])
        val_loss = self.loss_function(logits, y)
        self.log("val_loss", val_loss, prog_bar=True, on_epoch=True, on_step=False)

        pred = self.post_pred(logits)
        y = self.post_label(y)
        self.mean_dice(pred, y)
        self.confusion_matrix(pred, y)
        if batch_idx == 0:
           self.log_images(batch, pred, stage="val")

    def on_validation_epoch_end(self):
        mean_dice = self.mean_dice.aggregate()
        conf_mat = self.confusion_matrix.aggregate()

        for i, dice in enumerate(mean_dice):
            self.log(f"dice_score_{i+1}", dice.item())

        self.log("total_val_dice_score", mean_dice.mean().item())

        sens, spec, prec = conf_mat
        for i, (sensitivity, specificity, precision) in enumerate(zip(sens, spec, prec)):
            self.log(f"sensitivity_{i+1}", sensitivity.item())
            self.log(f"specificity_{i+1}", specificity.item())
            self.log(f"precision_{i+1}", precision.item())

        self.mean_dice.reset()
        self.confusion_matrix.reset()
