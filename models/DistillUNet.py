import os
import torchvision.utils as vutils
import torch
import torch.nn as nn
import lightning.pytorch as pl
import time

class DistillUNet(pl.LightningModule):
    def __init__(self, teacher, student, lr,
                 factor=0.5, patience=3, min_lr=1e-6, sample_size=36):
        super().__init__()
        self.teacher = teacher.eval()
        for p in self.teacher.parameters():
            p.requires_grad = False

        self.student = student
        self.distill_loss = nn.L1Loss()
        self.lr = float(lr)

        # scheduler hyper-params
        self.factor = factor
        self.patience = patience
        self.min_lr = min_lr

        self.sample_size = sample_size

    def forward(self, x):
        return self.student(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        self.cur_device = x.device
        with torch.no_grad():
            t_out = self.teacher(x)
        s_out = self.student(x)
        loss = self.distill_loss(s_out, t_out)
        self.log("distill_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        with torch.no_grad():
            t_out = self.teacher(x)
            s_out = self.student(x)
        loss = self.distill_loss(s_out, t_out)
        self.log("full_val_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        self.cur_device = x.device
        with torch.no_grad():
            t_start_time = time.time()
            t_out = self.teacher(x)
            t_time = time.time() - t_start_time

            s_start_time = time.time()
            s_out = self.student(x)
            s_time = time.time() - s_start_time

        self.log("student_test_time_seconds", s_time, on_step=True, on_epoch=False)
        self.log("teacher_test_time_seconds", t_time, on_step=True, on_epoch=False)

        loss = self.distill_loss(s_out, t_out)
        self.log("distill_test_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        return loss
    def on_validation_end(self):
        self.sample_images()

    def on_test_end(self):
        self.sample_images(test=True)

    def sample_images(self, test=False):
        try:
            if test == True:
                data_iter = iter(self.trainer.datamodule.test_dataloader())
            else:
                data_iter = iter(self.trainer.datamodule.val_dataloader())
            all_dark = []
            all_bright = []
            device = self.cur_device

            # Keep pulling batches until we have required number
            while sum(t.shape[0] for t in all_dark) < self.sample_size:
                x_dark, x_bright = next(data_iter)
                all_dark.append(x_dark)
                all_bright.append(x_bright)

            x_dark = torch.cat(all_dark, dim=0)[:self.sample_size].to(device)
            x_bright = torch.cat(all_bright, dim=0)[:self.sample_size].to(device)

            t_recons = self.teacher(x_dark)
            s_recons = self.student(x_dark)

            # --- SAVE IMAGES ---
            if test:
                end_str = 'test'
            else:
                end_str = f'Epoch_{self.current_epoch}'

            real_dir = os.path.join(self.logger.log_dir, "Student Reconstructions")
            os.makedirs(real_dir, exist_ok=True)
            vutils.save_image(
                s_recons.data,
                os.path.join(real_dir,
                             f"recons_{self.logger.name}_" + end_str + ".png"),
                normalize=True,
                nrow=int(self.sample_size ** .5))

            real_dir = os.path.join(self.logger.log_dir, "Teacher Reconstructions")
            os.makedirs(real_dir, exist_ok=True)
            vutils.save_image(
                t_recons.data,
                os.path.join(real_dir, f"real_{self.logger.name}_" + end_str + ".png"),
                normalize=True,
                nrow=int(self.sample_size ** .5)
            )

            real_dir = os.path.join(self.logger.log_dir, "Real Long Exposure")
            os.makedirs(real_dir, exist_ok=True)
            vutils.save_image(
                x_bright.data,
                os.path.join(real_dir, f"real_{self.logger.name}_" + end_str + ".png"),
                normalize=True,
                nrow=int(self.sample_size ** .5)
            )

            real_dir = os.path.join(self.logger.log_dir, "Real Short Exposure")
            os.makedirs(real_dir, exist_ok=True)
            vutils.save_image(
                x_dark[:, :3].data,  # Extract RGB from 5 channel augmentation
                os.path.join(real_dir, f"real_{self.logger.name}_" + end_str + ".png"),
                normalize=True,
                nrow=int(self.sample_size ** .5)
            )
        except AttributeError as e:
            # print(e)
            pass

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.student.parameters(), lr=self.lr)

        # Reduce LR when distill_loss stops improving
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=self.factor,
            patience=self.patience,
            min_lr=self.min_lr,
        )

        # Lightning expects a dict when using ReduceLROnPlateau
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "full_val_loss",      # which metric to monitor
                "frequency": 1,                 # check once per epoch
                "interval": "epoch",            # step is called each epoch
            }
        }