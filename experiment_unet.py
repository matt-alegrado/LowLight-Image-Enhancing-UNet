import torch.optim as optim
import os
import torchvision.utils as vutils
import lightning.pytorch as pl
import torch
from torch.utils.data import DataLoader, Subset
import math
import random


class ExperimentUNet(pl.LightningModule):
    i = 0

    def __init__(self, model, lr_g=1e-3, lr_d=1e-3, weight_decay_g=0, weight_decay_d=0, sch_gamma_g=1, sch_gamma_d=1,
                 step_size=1, warmup_epochs=10, adv_weight=0, disc_cooldown=1, sample_size=36):
        super(ExperimentUNet, self).__init__()
        self.model = model
        self.u_net = model.u_net
        self.discriminator = model.discriminator
        self.vgg = model.vgg

        self.sample_size = sample_size  # For sampling the images to print

        self.save_hyperparameters(ignore=['model','sample_size'])
        self.automatic_optimization = False

        self._full_test_losses = []
        self._adv_test_losses = []
        self._perc_test_losses = []
        self._recon_test_losses = []
        self._disc_test_losses = []
        self._ssim_test_losses = []

        self._full_val_losses = []
        self._adv_val_losses = []
        self._perc_val_losses = []
        self._recon_val_losses = []
        self._disc_val_losses = []
        self._ssim_val_losses = []

        self._full_losses = []
        self._adv_losses = []
        self._perc_losses = []
        self._recon_losses = []
        self._disc_losses = []
        self._ssim_losses = []

    def forward(self, batch):
        return self.model(batch)

    def training_step(self, batch, batch_idx):
        x_dark, x_light = batch
        self.cur_device = x_dark.device

        opt_g, opt_d = self.optimizers()
        opt_g.zero_grad()
        opt_d.zero_grad()

        epoch = self.current_epoch
        warmup_epochs = int(self.hparams.warmup_epochs)

        # ——— Discriminator step ———
        # only run D when adv_weight > 0
        if epoch < warmup_epochs:
            adv_weight = 0.0
        else:
            # progress = (epoch - warmup_epochs) / max(1, self.trainer.max_epochs // 2 - warmup_epochs)
            progress = 1
            adv_weight = progress * self.hparams.adv_weight

        if adv_weight > 0 and epoch % int(self.hparams.disc_cooldown) == 0:  # Only train every n epochs, this trains much faster than gen
            self.toggle_optimizer(opt_d)
            # forward pass
            with torch.no_grad():
                fake = self.model(x_dark)
            real = x_light

            disc_loss = self.model.disc_loss(real, fake)
            self.manual_backward(disc_loss)
            opt_d.step()

            self.untoggle_optimizer(opt_d)
        else:
            disc_loss = 0
        self._disc_losses.append(disc_loss)
        # ——— Generator (u_net + GAN) step ———
        self.toggle_optimizer(opt_g)

        fake = self.model(x_dark)
        # gen_loss returns a tuple: (full_loss, recon_loss, kld_loss, raw_adv_loss)
        loss_dict = self.model.gen_loss(x_light, fake, adv_scaling=adv_weight)

        self.manual_backward(loss_dict['full'])
        opt_g.step()
        self.untoggle_optimizer(opt_g)

        self._full_losses.append(loss_dict['full'])
        self._adv_losses.append(loss_dict['adv'])
        self._perc_losses.append(loss_dict['perc'])
        self._recon_losses.append(loss_dict['recon'])
        self._ssim_losses.append(loss_dict['ssim'])

    def on_train_epoch_end(self):
        adv_loss = sum(self._adv_losses) / max(1, len(self._adv_losses))
        recon_loss = sum(self._recon_losses) / max(1, len(self._recon_losses))
        perc_loss = sum(self._perc_losses) / max(1, len(self._perc_losses))
        full_loss = sum(self._full_losses) / max(1, len(self._full_losses))
        disc_loss = sum(self._disc_losses) / max(1, len(self._disc_losses))
        ssim_loss = sum(self._ssim_losses) / max(1, len(self._ssim_losses))

        self._full_losses.clear()
        self._adv_losses.clear()
        self._perc_losses.clear()
        self._recon_losses.clear()
        self._disc_losses.clear()
        self._ssim_losses.clear()

        self.log('adv_loss', adv_loss, on_step=False, on_epoch=True)
        self.log('recon_loss', recon_loss, on_step=False, on_epoch=True)
        self.log('perc_loss', perc_loss, on_step=False, on_epoch=True)
        self.log('full_loss', full_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('disc_loss', disc_loss, on_step=False, on_epoch=True)
        self.log('ssim_loss', ssim_loss, on_step=False, on_epoch=True)
        current_lr = self.trainer.optimizers[0].param_groups[0]['lr']
        self.log('gen_lr', current_lr, on_step=False, on_epoch=True, prog_bar=True)

        sch_g, sch_d = self.lr_schedulers()

        sch_g.step()
        sch_d.step()

    def on_training_end(self):
        torch.save(self.model.state_dict(), "u_net_final.pth")

    def validation_step(self, batch, batch_idx):
        x_dark, x_light = batch
        self.cur_device = x_dark.device

        with torch.no_grad():
            # forward pass
            forward_out = self.model(x_dark)
            epoch = self.current_epoch
            warmup_epochs = int(self.hparams.warmup_epochs)

            if epoch >= warmup_epochs:
                disc_val_loss = self.model.disc_loss(x_light, forward_out)
                adv_scaling = 1
            else:
                disc_val_loss = 0
                adv_scaling = 0

            loss_dict = self.model.gen_loss(x_light, forward_out, adv_scaling=adv_scaling)

            self._full_val_losses.append(loss_dict['full'])
            self._adv_val_losses.append(loss_dict['adv'])
            self._perc_val_losses.append(loss_dict['perc'])
            self._recon_val_losses.append(loss_dict['recon'])
            self._ssim_val_losses.append(loss_dict['ssim'])
            self._disc_val_losses.append(disc_val_loss)

    def on_validation_epoch_end(self):
        try:
            adv_val_loss = sum(self._adv_val_losses) / max(1, len(self._adv_val_losses))
            recon_val_loss = sum(self._recon_val_losses) / max(1, len(self._recon_val_losses))
            perc_val_loss = sum(self._perc_val_losses) / max(1, len(self._perc_val_losses))
            full_val_loss = sum(self._full_val_losses) / max(1, len(self._full_val_losses))
            disc_val_loss = sum(self._disc_val_losses) / max(1, len(self._disc_val_losses))
            ssim_val_loss = sum(self._ssim_val_losses) / max(1, len(self._ssim_val_losses))

            self._full_val_losses.clear()
            self._adv_val_losses.clear()
            self._perc_val_losses.clear()
            self._recon_val_losses.clear()
            self._disc_val_losses.clear()
            self._ssim_val_losses.clear()

            self.log('recon_val_loss', recon_val_loss, on_step=False, on_epoch=True, prog_bar=True)
            self.log('adv_val_loss', adv_val_loss, on_step=False, on_epoch=True, prog_bar=True)
            self.log('perc_val_loss', perc_val_loss, on_step=False, on_epoch=True, prog_bar=True)
            self.log('full_val_loss', full_val_loss, on_step=False, on_epoch=True, prog_bar=True)
            self.log('disc_val_loss', disc_val_loss, on_step=False, on_epoch=True, prog_bar=True)
            self.log('ssim_val_loss', ssim_val_loss, on_step=False, on_epoch=True, prog_bar=True)

            # sched_g, sched_d = self.lr_schedulers()
            # sched_g.step(full_val_loss)
            # sched_d.step(full_val_loss)
        except RuntimeError:
            pass  # Probably sanity checking
    def on_validation_end(self):
        self.sample_images()

    def test_step(self, batch, batch_idx):
        x_dark, x_light = batch
        self.cur_device = x_dark.device

        with torch.no_grad():
            # forward pass
            forward_out = self.model(x_dark)

            loss_dict = self.model.gen_loss(x_light, forward_out, adv_scaling=0)

            self._full_test_losses.append(loss_dict['full'])
            self._perc_test_losses.append(loss_dict['perc'])
            self._recon_test_losses.append(loss_dict['recon'])
            self._ssim_test_losses.append(loss_dict['ssim'])

    def on_test_epoch_end(self):
        recon_test_loss = sum(self._recon_test_losses) / max(1, len(self._recon_test_losses))
        perc_test_loss = sum(self._perc_test_losses) / max(1, len(self._perc_test_losses))
        full_test_loss = sum(self._full_test_losses) / max(1, len(self._full_test_losses))
        ssim_test_loss = sum(self._ssim_test_losses) / max(1, len(self._ssim_test_losses))

        self._full_test_losses.clear()
        self._adv_test_losses.clear()
        self._perc_test_losses.clear()
        self._recon_test_losses.clear()
        self._disc_test_losses.clear()
        self._ssim_test_losses.clear()

        self.log('recon_test_loss', recon_test_loss, on_step=False, on_epoch=True)
        self.log('perc_test_loss', perc_test_loss, on_step=False, on_epoch=True)
        self.log('full_test_loss', full_test_loss, on_step=False, on_epoch=True)
        self.log('ssim_test_loss', ssim_test_loss, on_step=False, on_epoch=True)

    def on_test_end(self):
        self.sample_images(test=True)
        self.sample_end_images(test=True)

    def sample_images(self, test=False):
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

        # --- SAVE IMAGES ---
        if test:
            end_str = 'test'
        else:
            end_str = f'Epoch_{self.current_epoch}'

        real_dir = os.path.join(self.logger.log_dir, "Long Exposure Reconstructions")
        os.makedirs(real_dir, exist_ok=True)
        recons = self.model.generate(x_dark)
        vutils.save_image(
            recons.data,
            os.path.join(self.logger.log_dir,
                         "Long Exposure Reconstructions",
                         f"recons_{self.logger.name}_" + end_str + ".png"),
            normalize=True,
            nrow=int(self.sample_size ** .5))

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
            x_dark[:,:3].data,  # Extract RGB from 5 channel augmentation
            os.path.join(real_dir, f"real_{self.logger.name}" + end_str + ".png"),
            normalize=True,
            nrow=int(self.sample_size ** .5)
        )

    def sample_end_images(self, test=False):
        # 1. pick the right dataset
        dm = self.trainer.datamodule
        if test:
            dataset = dm.test_dataset  # assumes your DataModule exposes .test_dataset
            suffix = 'test_reversed'
        else:
            dataset = dm.val_dataset  # assumes your DataModule exposes .val_dataset
            suffix = f'Epoch_{self.current_epoch}'

        # 2. compute the slice of “last sample_size” indices
        total = len(dataset)
        start = max(0, total - self.sample_size)
        last_idxs = list(range(start, total))

        # 3. wrap them in a Subset + DataLoader
        subset = Subset(dataset, last_idxs)
        loader = DataLoader(subset,
                            batch_size=self.sample_size,
                            shuffle=False,
                            num_workers=dm.num_workers if hasattr(dm, 'num_workers') else 0)

        # 4. pull that single batch
        x_dark, x_bright = next(iter(loader))
        x_dark = x_dark.to(self.cur_device)
        x_bright = x_bright.to(self.cur_device)

        # 5. exactly the same saving logic as sample_images
        real_dir = os.path.join(self.logger.log_dir, "Long Exposure Reconstructions")
        os.makedirs(real_dir, exist_ok=True)
        recons = self.model.generate(x_dark)
        vutils.save_image(
            recons.data,
            os.path.join(real_dir, f"recons_{self.logger.name}_" + suffix + ".png"),
            normalize=True, nrow=int(self.sample_size ** .5)
        )

        real_dir = os.path.join(self.logger.log_dir, "Real Long Exposure")
        os.makedirs(real_dir, exist_ok=True)
        vutils.save_image(
            x_bright.data,
            os.path.join(real_dir, f"real_{self.logger.name}_" + suffix + ".png"),
            normalize=True, nrow=int(self.sample_size ** .5)
        )

        real_dir = os.path.join(self.logger.log_dir, "Real Short Exposure")
        os.makedirs(real_dir, exist_ok=True)
        vutils.save_image(
            x_dark[:, :3].data,
            os.path.join(real_dir, f"real_{self.logger.name}_" + suffix + ".png"),
            normalize=True, nrow=int(self.sample_size ** .5)
        )

    def configure_optimizers(self):
        # create optimizers using saved hyperparameters
        u_net_betas = (0.5, 0.999)
        disc_betas = (0.5, 0.999)

        opt_g = optim.RAdam(
            self.model.u_net.parameters(),
            lr=float(self.hparams.lr_g),
            weight_decay=float(self.hparams.weight_decay_g),
            betas=u_net_betas
        )
        opt_d = optim.RMSprop(
            self.model.discriminator.parameters(),
            lr=float(self.hparams.lr_d),
            weight_decay=float(self.hparams.weight_decay_d),
            # betas=disc_betas
        )
        optimizers = [opt_g, opt_d]

        # sched_g = torch.optim.lr_scheduler.ReduceLROnPlateau(
        #     opt_g,
        #     mode='min',
        #     factor=0.5,
        #     patience=5,
        #     min_lr=1e-5,
        # )
        #
        # # Discriminator scheduler: watch "val/disc_loss"
        # sched_d = torch.optim.lr_scheduler.ReduceLROnPlateau(
        #     opt_d,
        #     mode='min',
        #     factor=0.5,
        #     patience=3,
        #     min_lr=1e-6,
        # )
        #
        # # Package them as two dicts, each with its own "monitor" key
        # sched_dict_g = {
        #     "scheduler": sched_g,
        #     "monitor": "full_val_loss",
        #     "interval": "epoch",
        #     "frequency": 1,
        #     "strict": True,
        # }
        # sched_dict_d = {
        #     "scheduler": sched_d,
        #     "monitor": "adv_val_loss",
        #     "interval": "epoch",
        #     "frequency": 1,
        #     "strict": True,
        # }
        #
        # schedulers = [sched_dict_g, sched_dict_d]
        #
        # return optimizers, schedulers

        sched_g = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt_g,
            T_max=self.trainer.max_epochs,
            eta_min=1e-6,
        )
        sched_d = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt_d,
            T_max=self.trainer.max_epochs,
            eta_min=1e-6,
        )

        # return (
        #     [opt_g, opt_d],
        #     [
        #         {
        #             "scheduler": sched_g,
        #             "interval": "epoch",
        #             "frequency": 1,
        #         },
        #         {
        #             "scheduler": sched_d,
        #             "interval": "epoch",
        #             "frequency": 1,
        #         },
        #     ],
        # )
        return [opt_g, opt_d], [sched_g, sched_d]
