from argparse import ArgumentParser
from lightning.pytorch import seed_everything
from lightning.pytorch.loggers import TensorBoardLogger
import lightning.pytorch as pl
import yaml
import torch
import os
from torchvision import transforms

from models.UNet import UNet
from models.Discriminator import Discriminator
from models.UNetGAN import UNetGAN
from utils.dataset import SIDDataset, SIDDataModule
from utils.psnr import PSNRCallback
from experiment_unet import ExperimentUNet
from utils.transform import SlightDimTransform
from models.DistillUNet import DistillUNet



if __name__ == "__main__":
    torch.set_float32_matmul_precision('medium')
    # Read cmd args, mainly to find config.yaml file
    parser = ArgumentParser(description='Model runner')
    parser.add_argument('--config', '-c',
                        dest="filename",
                        metavar='FILE',
                        help='path to the config file',
                        default='config.yaml')
    parser.add_argument('--resume', '-r',
                        type=str,
                        default=None,
                        help="Path to a .ckpt file to resume from (Lightning format)."
                        )
    parser.add_argument('--student', '-s',
                        type=str,
                        default=None,
                        help="Flag to activate student training mode."
                        )

    # Open config file
    args = parser.parse_args()
    ckpt_path = args.resume
    with open(args.filename, 'r') as file:
        try:
            config = yaml.safe_load(file)
        except yaml.YAMLError as exc:
            print(exc)

    tb_logger = TensorBoardLogger(save_dir=config['logging']['save_dir'])

    # For reproducibility
    seed_everything(config['experiment']['manual_seed'], True)

    u_net = UNet(
                hidden_dims=config['generator']['hidden_dims'],
                image_size=config['generator']['input_image_size'],
                dropout_rate=config['generator']['dropout_rate'],
                dropout_layers=config['generator']['dropout_layers'],
                gn=config['generator']['use_groupnorm'],
                in_channels=3,
                augmented_channels=2  # RGB + iso + exposure
    )
    discriminator = Discriminator(
        in_channels=3,
        hidden_dims=config['discriminator']['hidden_dims'],
        dropout_start=config['discriminator']['dropout_start'],
        dropout_p=config['discriminator']['dropout_p'],
        gn=config['discriminator']['use_groupnorm']
    )

    model = UNetGAN(
        # vae,
        u_net,
        discriminator,
        recon_weight=config['gan']['recon_weight'],
        ssim_weight=config['gan']['ssim_weight'],
        adv_weight=config['gan']['adv_weight'],
        perc_weight=config['gan']['perc_weight'],
    )
    if ckpt_path is None:
        experiment = ExperimentUNet(
            model,
            lr_g=config['generator']['lr'],
            lr_d=config['discriminator']['lr'],
            weight_decay_g=config['generator']["weight_decay"],
            weight_decay_d=config["discriminator"]["weight_decay"],
            step_size=config["experiment"]["step_size"],
            warmup_epochs=config['gan']['warmup_epochs'],
            adv_weight=config['gan']['adv_weight'],
            disc_cooldown=config['gan']['disc_cooldown']
        )
    else:
        experiment = ExperimentUNet.load_from_checkpoint(
            ckpt_path,
            model=model,
            lr_g=config['generator']['lr'],
            lr_d=config['discriminator']['lr'],
            weight_decay_g=config['generator']["weight_decay"],
            weight_decay_d=config["discriminator"]["weight_decay"],
            step_size=config["experiment"]["step_size"],
            warmup_epochs=config['gan']['warmup_epochs'],
            adv_weight=config['gan']['adv_weight'],
            disc_cooldown=config['gan']['disc_cooldown']
        )
    transform_train = transforms.Compose([
        transforms.Resize((int(config['generator']['input_image_size']),int(config['generator']['input_image_size']))),
        transforms.ToTensor(),
        transforms.RandomHorizontalFlip(p=0.5)
    ])
    transform_test = transforms.Compose([
        transforms.Resize((int(config['generator']['input_image_size']), int(config['generator']['input_image_size']))),
        transforms.ToTensor(),
    ])

    transform_debug = transforms.Compose([
        SlightDimTransform(base_size=int(config['generator']['input_image_size'])),
    ])
    train_data = SIDDataset(
        config['dataset']['data_path'],
        os.path.join(config['dataset']['data_path'], 'Sony_train_list.txt'),
        transform=transform_train,
        # transform_debug=transform_debug
    )
    val_data = SIDDataset(
        config['dataset']['data_path'],
        os.path.join(config['dataset']['data_path'], 'Sony_val_list.txt'),
        transform=transform_test,
        # transform_debug=transform_debug
    )
    test_data = SIDDataset(
        config['dataset']['data_path'],
        os.path.join(config['dataset']['data_path'], 'Sony_test_list.txt'),
        transform=transform_test,
    )
    datamodule = SIDDataModule(train_data, val_data, test_data, batch_size=config['dataset']['batch_size'],
                               num_workers=config['dataset']['num_workers'])

    ckpt_dir = os.path.join(config["logging"]["save_dir"], "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)

    version = tb_logger.version
    print(f'Currently on version {version}.')

    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        dirpath=ckpt_dir,
        filename=f"v{version}" + "-{epoch:03d}-{full_val_loss:.2f}",
        save_top_k=2,
        every_n_epochs=1,
        save_last=True,
        verbose=True,
        monitor="full_val_loss"
    )

    psnr_callback = PSNRCallback(datamodule)

    # Trainer
    trainer = pl.Trainer(
        logger=tb_logger,
        max_epochs=config["trainer"]["max_epochs"],
        accelerator="auto",
        devices=config["trainer"]["gpus"],
        callbacks=[
            checkpoint_callback,
            psnr_callback
        ],
    )

    if args.student is not None:
        student_unet = UNet(
            hidden_dims=config['student']['hidden_dims'],
            image_size=config['student']['input_image_size'],
            dropout_rate=config['student']['dropout_rate'],
            dropout_layers=config['student']['dropout_layers'],
            gn=config['generator']['use_groupnorm'],
            in_channels=3,
            augmented_channels=2
        )

        distill_exp = DistillUNet(
            teacher=experiment.model.u_net,
            student=student_unet,
            lr=config['student']['lr'],
        )

        trainer.fit(distill_exp, datamodule=datamodule)
        trainer.test(distill_exp, datamodule=datamodule)

