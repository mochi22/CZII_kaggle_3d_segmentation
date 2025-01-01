import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning.pytorch as pl
from monai.losses import TverskyLoss
from monai.metrics import DiceMetric
import timm


# encoder
class ResNetEncoder2D(nn.Module):
    def __init__(self, arch='resnet34d', pretrained=False):
        super().__init__()
        self.encoder = timm.create_model(
            model_name=arch, pretrained=pretrained, in_chans=1, num_classes=0, global_pool='', features_only=True,
        )
        
    def forward(self, x):
        return self.encoder(x)

class DepthWiseEncoder(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))

# decoder
class DecoderBlock2D1D(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels):
        super().__init__()
        self.conv2d = nn.Sequential(
            nn.Conv2d(in_channels + skip_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        self.conv1d = nn.Sequential(
            nn.Conv1d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x, skip=None):
        B, D, C, H, W = x.shape
        x = x.view(B*D, C, H, W)
        if skip is not None:
            skip = skip.view(B*D, -1, H, W)
            x = torch.cat([x, skip], dim=1)
        x = self.conv2d(x)
        x = x.view(B, D, -1, H*W)
        x = self.conv1d(x)
        x = x.view(B, D, -1, H, W)
        return x

class Decoder2D1D(nn.Module):
    def __init__(self, encoder_channels, decoder_channels):
        super().__init__()
        self.blocks = nn.ModuleList([
            DecoderBlock2D1D(in_ch, skip_ch, out_ch)
            for in_ch, skip_ch, out_ch in zip(encoder_channels[:-1], encoder_channels[1:], decoder_channels)
        ])

    def forward(self, features):
        features = features[::-1]
        x = features[0]
        skips = features[1:]
        
        for i, block in enumerate(self.blocks):
            skip = skips[i] if i < len(skips) else None
            x = block(x, skip)
        
        return x


# model
class model2d1d(pl.LightningModule):
    def __init__(
        self,
        n_channels: int = 1,
        n_classes: int = 7,
        lr: float = 1e-3,
        arch: str = 'resnet34d',
        is_train: bool = True
    ):
        super(model2d1d, self).__init__()
        self.save_hyperparameters()
        
        self.n_channels = n_channels
        self.n_classes = n_classes        
        self.encoder_2d = ResNetEncoder2D(arch=arch, pretrained=is_train)

        encoder_dim = {
            'resnet18': [64, 64, 128, 256, 512, ],
            'resnet18d': [64, 64, 128, 256, 512, ],
            'resnet34d': [64, 64, 128, 256, 512, ],
            'resnet50d': [64, 256, 512, 1024, 2048, ],
            'seresnext26d_32x4d': [64, 256, 512, 1024, 2048, ],
            'convnext_small.fb_in22k': [96, 192, 384, 768],
            'convnext_tiny.fb_in22k': [96, 192, 384, 768],
            'convnext_base.fb_in22k': [128, 256, 512, 1024],
            'tf_efficientnet_b4.ns_jft_in1k':[32, 56, 160, 448],
            'tf_efficientnet_b5.ns_jft_in1k':[40, 64, 176, 512],
            'tf_efficientnet_b6.ns_jft_in1k':[40, 72, 200, 576],
            'tf_efficientnet_b7.ns_jft_in1k':[48, 80, 224, 640],
            'pvt_v2_b1': [64, 128, 320, 512],
            'pvt_v2_b2': [64, 128, 320, 512],
            'pvt_v2_b4': [64, 128, 320, 512],
        }.get(arch, [64, 64, 128, 256, 512])
        #}.get(arch, [768])
        decoder_dim = [256, 128, 64, 32, 16]

        self.depth_encoders = nn.ModuleList([
            DepthWiseEncoder(dim, dim) for dim in encoder_dim
        ])
        
        decoder_dim = [256, 128, 64, 32]
        self.decoder = Decoder2D1D(encoder_dim[::-1], decoder_dim)
        
        self.final_conv = nn.Conv3d(decoder_dim[-1], n_classes, kernel_size=1)


        # setting for train/val
        self.loss_fn = TverskyLoss(include_background=True, to_onehot_y=True, softmax=True)
        self.metric_fn = DiceMetric(include_background=False, reduction="mean", ignore_empty=True)

        self.train_loss = 0
        self.val_loss = 0
        self.train_metric=0
        self.val_metric = 0
        self.num_train_batch = 0
        self.num_val_batch = 0

    def forward(self, x):
        B, D, H, W = x.shape
        x = x.view(B*D, 1, H, W)
        
        # 2d encode
        features_2d = self.encoder_2d(x)
        
        # 1d encode
        features_3d = []
        for feat, depth_encoder in zip(features_2d, self.depth_encoders):
            _, C, H, W = feat.shape
            feat = feat.view(B, D, C, H, W).permute(0, 2, 1, 3, 4).contiguous()
            feat = feat.view(B*C, D, H*W)
            feat = depth_encoder(feat)
            feat = feat.view(B, C, D, H, W).permute(0, 2, 1, 3, 4).contiguous()
            features_3d.append(feat)

        # decode
        decoded = self.decoder(features_3d)
        
        # 最終的な出力
        output = self.final_conv(decoded)
        
        return output

    def training_step(self, batch, batch_idx):
        x, y = batch['image'], batch['label']
        y_hat_seg = self(x)
        loss = self.loss_fn(y_hat_seg, y)
        metric = self.metric_fn(y_hat_seg, y)
        
        # ステップごとのロギング
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log('train_metric', metric, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        
        # グラディエントノルムのロギング
        grad_norm = torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
        self.log('grad_norm', grad_norm, on_step=True, on_epoch=True, prog_bar=False, logger=True)

        # 学習率のロギング
        self.log('lr', self.optimizers().param_groups[0]['lr'], on_step=True, on_epoch=True, prog_bar=False, logger=True)

        # メモリ使用量のロギング
        self.log('gpu_memory', memory.get_memory_usage_in_mb(), on_step=True, on_epoch=True, prog_bar=False, logger=True)

        self.train_loss += loss
        self.train_metric += metric 
        self.num_train_batch += 1
        torch.cuda.empty_cache()
        return loss

    def on_train_epoch_end(self):
        avg_loss = self.train_loss / self.num_train_batch
        avg_metric = self.train_metric / self.num_train_batch
        
        self.log('train_epoch_loss', avg_loss, prog_bar=True)
        self.log('train_epoch_metric', avg_metric, prog_bar=True)
        
        mlflow.log_metric("custom_train_loss", avg_loss, self.current_epoch)
        mlflow.log_metric("custom_train_metric", avg_metric, self.current_epoch)
        
        self.train_loss = 0
        self.train_metric = 0
        self.num_train_batch = 0

    def validation_step(self, batch, batch_idx):
        with torch.no_grad():
            x, y = batch['image'], batch['label']
            y_hat_seg = self(x)
            loss = self.loss_fn(y_hat_seg, y)
            metric = self.metric_fn(y_hat_seg, y)

            metric_val_outputs = [AsDiscrete(argmax=True, to_onehot=self.hparams.n_classes)(i) for i in decollate_batch(y_hat_seg)]
            metric_val_labels = [AsDiscrete(to_onehot=self.hparams.n_classes)(i) for i in decollate_batch(y)]

            self.metric_fn(y_pred=metric_val_outputs, y=metric_val_labels)
            metrics = self.metric_fn.aggregate(reduction="mean_batch")
            val_metric = torch.mean(metrics)
            
            self.log('val_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
            self.log('val_tmp_metric', metric, on_step=True, on_epoch=True, prog_bar=True, logger=True)
            self.log('val_metric', val_metric, on_step=True, on_epoch=True, prog_bar=True, logger=True)

            self.val_loss += loss
            self.val_metric += val_metric 
            self.num_val_batch += 1
        torch.cuda.empty_cache()
        return {'val_metric': val_metric}

    def on_validation_epoch_end(self):
        # metric_per_epoch = self.val_metric / self.num_val_batch
        avg_loss = self.val_loss / self.num_val_batch
        avg_metric = self.val_metric / self.num_val_batch

        self.log('val_epoch_loss', avg_loss, prog_bar=True, sync_dist=False)
        self.log('val_epoch_metric', avg_metric, prog_bar=True, sync_dist=False)
        
        mlflow.log_metric("custom_val_loss", avg_loss, self.current_epoch)
        mlflow.log_metric("custom_val_metric", avg_metric, self.current_epoch)

        self.val_loss = 0
        self.val_metric = 0
        self.num_val_batch = 0

    def configure_optimizers(self):
        # optimizer = torch.optim.AdamW(self.parameters(), lr=self.hparams.lr)
        # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10)
        # return {
        #     "optimizer": optimizer,
        #     "lr_scheduler": {
        #         "scheduler": scheduler,
        #         "monitor": "val_loss",
        #         "frequency": 1
        #     },
        # }
        return torch.optim.AdamW(self.parameters(), lr=self.hparams.lr)

