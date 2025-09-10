import lightning.pytorch as pl
from skimage.metrics import peak_signal_noise_ratio as sk_psnr


class PSNRCallback(pl.Callback):
    def __init__(self, datamodule, num_samples=100):
        """
        Args:
            feature:        which Inception feature to use for FID. Default "pool3".
            fid_normalize:  whether to normalize inputs to [0,1]→Inception’s expected range.
            inception_resize: if True, will resize to 299×299 inside InceptionScore.
            inception_split_factor: number of splits for InceptionScore (bigger → more stable).
        """
        super().__init__()
        self.num_samples = num_samples
        self.datamodule = datamodule

    def on_validation_epoch_end(self, trainer, pl_module):
        pl_module.eval()
        psnr_scores = []

        dl = trainer.datamodule.val_dataloader()
        device = pl_module.device

        for idx, batch in enumerate(dl):
            if idx >= self.num_samples:
                break
            x, y = batch  # x=input dark, y=target bright
            x, y = x.to(device), y.to(device)
            y_hat = pl_module(x)

            # PSNR (skimage expects numpy uint8 or float in [0,1])
            psnr = sk_psnr(
                y.cpu().permute(0,2,3,1).numpy(),
                y_hat.cpu().permute(0,2,3,1).numpy(),
                data_range=1.0
            )
            psnr_scores.append(psnr)

        avg_psnr = sum(psnr_scores) / len(psnr_scores)

        # log to TensorBoard
        trainer.logger.log_metrics({"val/PSNR": avg_psnr}, step=trainer.current_epoch)

        pl_module.train()

    def on_test_epoch_end(self, trainer, pl_module):
        pl_module.eval()
        psnr_scores = []

        dl = trainer.datamodule.test_dataloader()
        device = pl_module.device

        for idx, batch in enumerate(dl):
            if idx >= self.num_samples:
                break
            x, y = batch  # x=input dark, y=target bright
            x, y = x.to(device), y.to(device)
            y_hat = pl_module(x)

            # PSNR (skimage expects numpy uint8 or float in [0,1])
            psnr = sk_psnr(
                y.cpu().permute(0,2,3,1).numpy(),
                y_hat.cpu().permute(0,2,3,1).numpy(),
                data_range=1.0
            )
            psnr_scores.append(psnr)

        avg_psnr = sum(psnr_scores) / len(psnr_scores)

        # log to TensorBoard
        trainer.logger.log_metrics({"test/PSNR": avg_psnr}, step=trainer.current_epoch)

        pl_module.train()