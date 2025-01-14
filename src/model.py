import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning.pytorch as pl
from monai.losses import TverskyLoss
from monai.metrics import DiceMetric
from monai.data import decollate_batch
from monai.transforms import AsDiscrete
import timm
import mlflow

from .metrics import SegmentationMetrics

## model25d
# encoder        
def encode_for_resnet(e, x, B, depth_scaling=[2,2,2,2,1]):
    def pool_in_depth(x, depth_scaling):
        bd, c, h, w = x.shape
        x1 = x.reshape(B, -1, c, h, w).permute(0, 2, 1, 3, 4)
        x1 = F.avg_pool3d(x1, kernel_size=(depth_scaling, 1, 1), stride=(depth_scaling, 1, 1), padding=0)
        x = x1.permute(0, 2, 1, 3, 4).reshape(-1, c, h, w)
        return x, x1

    encode = []
    x = e.conv1(x)
    x = e.bn1(x)
    x = e.act1(x)
    x, x1 = pool_in_depth(x, depth_scaling[0])
    encode.append(x1)
    x = F.avg_pool2d(x, kernel_size=2, stride=2)

    x = e.layer1(x)
    x, x1 = pool_in_depth(x, depth_scaling[1])
    encode.append(x1)

    x = e.layer2(x)
    x, x1 = pool_in_depth(x, depth_scaling[2])
    encode.append(x1)

    x = e.layer3(x)
    x, x1 = pool_in_depth(x, depth_scaling[3])
    encode.append(x1)

    x = e.layer4(x)
    x, x1 = pool_in_depth(x, depth_scaling[4])
    encode.append(x1)

    return encode


# def encode_for_efficientnet(e, x, B, depth_scaling=[2,2,2,2,1]):
#     def pool_in_depth(x, depth_scaling):
#         bd, c, h, w = x.shape
#         x1 = x.reshape(B, -1, c, h, w).permute(0, 2, 1, 3, 4)
#         x1 = F.avg_pool3d(x1, kernel_size=(depth_scaling, 1, 1), stride=(depth_scaling, 1, 1), padding=0)
#         x = x1.permute(0, 2, 1, 3, 4).reshape(-1, c, h, w)
#         return x, x1

#     encode = []
    
#     # EfficientNetの最初の畳み込み層
#     x = e.conv_stem(x)
#     x = e.bn1(x)
#     x = e.act1(x)
#     x, x1 = pool_in_depth(x, depth_scaling[0])
#     encode.append(x1)

#     # EfficientNetのブロック
#     for i, block in enumerate(e.blocks):
#         x = block(x)
#         if i in [1, 3, 5, 7]:  # EfficientNetの主要な特徴マップを抽出
#             x, x1 = pool_in_depth(x, depth_scaling[len(encode)])
#             encode.append(x1)

#     # 最後の畳み込み層
#     x = e.conv_head(x)
#     x = e.bn2(x)
#     x = e.act2(x)
#     x, x1 = pool_in_depth(x, depth_scaling[-1])
#     encode.append(x1)

#     return encode

# decoder
def adjust_dimensions(x, skip):
    _, _, d, h, w = x.size()
    _, _, sd, sh, sw = skip.size()
    
    # Depth adjustment
    if sd < d:
        skip = F.interpolate(skip, size=(d, sh, sw), mode='trilinear', align_corners=False)
    elif sd > d:
        x = F.interpolate(x, size=(sd, h, w), mode='trilinear', align_corners=False)
    
    # Height and width adjustment
    if sh > h or sw > w:
        diff_y = sh - h
        diff_x = sw - w
        x = F.pad(x, [diff_x // 2, diff_x - diff_x // 2,
                      diff_y // 2, diff_y - diff_y // 2])
    elif sh < h or sw < w:
        diff_y = h - sh
        diff_x = w - sw
        skip = F.pad(skip, [diff_x // 2, diff_x - diff_x // 2,
                            diff_y // 2, diff_y - diff_y // 2])
    
    return x, skip

class MyDecoderBlock3d(nn.Module):
    def __init__(
            self,
            in_channel,
            skip_channel,
            out_channel,
    ):
        super().__init__()
        #print(in_channel , skip_channel, out_channel,)
        self.conv1 = nn.Sequential(
            nn.Conv3d(in_channel + skip_channel, out_channel, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(out_channel),
            nn.ReLU(inplace=True),
        )
        self.attention1 = nn.Identity()
        self.conv2 = nn.Sequential(
            nn.Conv3d(out_channel, out_channel, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(out_channel),
            nn.ReLU(inplace=True),
        )
        self.attention2 = nn.Identity()

    def forward(self, x, skip=None, depth_scaling=2):
        x = F.interpolate(x, scale_factor=(depth_scaling,2,2), mode='nearest')
        if skip is not None:
            x, skip = adjust_dimensions(x, skip)
            x = torch.cat([x, skip], dim=1)
            x = self.attention1(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.attention2(x)
        return x

class MyUnetDecoder3d(nn.Module):
    def __init__(
            self,
            in_channel,
            skip_channel,
            out_channel,
    ):
        super().__init__()
        self.center = nn.Identity()

        i_channel = [in_channel, ] + out_channel[:-1]
        s_channel = skip_channel
        o_channel = out_channel
        block = [
            MyDecoderBlock3d(i, s, o)
            for i, s, o in zip(i_channel, s_channel, o_channel)
        ]
        self.block = nn.ModuleList(block)

    def forward(self, feature, skip, depth_scaling=[2,2,2,2,2,2]):
        d = self.center(feature)
        decode = []
        for i, block in enumerate(self.block):
            # print("&&"*10)
            # print(i, d.shape, skip[i].shape if skip[i] is not None else 'none')
            # print(block.conv1[0])
            # print('')
            # print(f"Layer {i}:")
            # print(f"  d shape: {d.shape}")
            # print(f"  skip shape: {skip[i].shape if skip[i] is not None else 'None'}")
            # print(f"  depth_scaling: {depth_scaling[i]}")

            s = skip[i]
            d = block(d, s, depth_scaling[i])
            decode.append(d)
        last = d
        return last, decode


## model2d1d
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
        self.loss_fn = TverskyLoss(include_background=True, to_onehot_y=False, softmax=True)
        self.metric_fn = DiceMetric(include_background=False, reduction="mean", ignore_empty=True)

        self.train_loss = 0
        self.val_loss = 0
        self.train_metric=0
        self.val_metric = 0
        self.num_train_batch = 0
        self.num_val_batch = 0

    def forward(self, x):
        B, D, H, W = x.shape
        x = x.view(B*D, 1, H, W) # 1 == chennel
        print("=="*10)
        print("B, D, H, W:", B, D, H, W)
        print("x:", x.shape)
        
        # 2d encode
        features_2d = self.encoder_2d(x)
        print("features_2d:", features_2d.shape)
        
        # 1d encode
        features_3d = []
        for feat, depth_encoder in zip(features_2d, self.depth_encoders):
            _, C, H, W = feat.shape
            feat = feat.view(B, D, C, H, W).permute(0, 2, 1, 3, 4).contiguous()
            feat = feat.view(B*C, D, H*W)
            feat = depth_encoder(feat)
            feat = feat.view(B, C, D, H, W).permute(0, 2, 1, 3, 4).contiguous()
            features_3d.append(feat)
        print(len(features_3d), features_3d[0].shape, features_3d[1].shape)

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
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        self.log('train_metric', metric, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        
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
            
            self.log('val_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
            self.log('val_tmp_metric', metric, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
            self.log('val_metric', val_metric, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)

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


def convert_to_multiclass(labels, num_classes):
    # labels: [batch_size, depth, H, W]
    # print("labels.shape:", labels.shape)
    # 新しい軸を追加してone-hotエンコーディングを適用
    one_hot = F.one_hot(labels.long(), num_classes=num_classes)
    # 軸の順序を変更して [batch_size, num_classes, depth, H, W] にする
    return one_hot.permute(0, 4, 1, 2, 3)#.float()

# class MulticlassDiceLoss(nn.Module):
#     def __init__(self):
#         super(MulticlassDiceLoss, self).__init__()

#     def forward(self, input, target, smooth=1e-6):
#         # input tensor of shape (N, C, H, W)
#         # target tensor of shape (N, H, W) with values 0-C-1
#         input = F.softmax(input, dim=1)
        
#         # One-hot encode the target
#         target_one_hot = F.one_hot(target, num_classes=input.shape[1]).permute(0, 3, 1, 2).float()
        
#         # Compute Dice loss for each class
#         dice = 0
#         for i in range(input.shape[1]):
#             input_i = input[:, i, :, :]
#             target_i = target_one_hot[:, i, :, :]
#             intersection = (input_i * target_i).sum()
#             dice += (2. * intersection + smooth) / (input_i.sum() + target_i.sum() + smooth)
        
#         return 1 - dice / input.shape[1]  # Average over classes


# id_to_name = {1: "apo-ferritin", 1
#               2: "beta-amylase", 0
#               3: "beta-galactosidase", 2
#               4: "ribosome", 1
#               5: "thyroglobulin", 2
#               6: "virus-like-particle"}1

class PerClassCrossEntropyLoss(nn.Module):
    def __init__(self, weight=[0,1,0,2,1,2,1], reduction='mean'):
        super(PerClassCrossEntropyLoss, self).__init__()
        self.weight = torch.tensor(weight, dtype=torch.float32) # 意味不明だが手動で設定が必要
        self.reduction = reduction

    def forward(self, predictions, targets):
        # predictions: [batch_size, classes, depth, H, W]
        # targets: [batch_size, classes, depth, H, W]

        predictions = nn.Softmax(dim=1)(predictions) # logits to probs

        assert predictions.shape == targets.shape, f"shape error!{predictions.shape , targets.shape}"
        
        batch_size, num_classes, depth, height, width = predictions.size()
        
        # Reshape tensors to [batch_size * depth * H * W, classes]
        predictions = predictions.permute(0, 2, 3, 4, 1).contiguous().view(-1, num_classes)
        targets = targets.permute(0, 2, 3, 4, 1).contiguous().view(-1, num_classes)
        
        # Compute cross entropy loss for each pixel
        # print("F.binary_cross_entropy_with_logits:", type(predictions), type(targets))
        # print("##"*10, predictions)
        # print(targets)
        pixel_losses = F.binary_cross_entropy_with_logits(predictions, targets.float(), reduction='none')
        
        # Reshape losses back to [batch_size, classes, depth, H, W]
        pixel_losses = pixel_losses.view(batch_size, depth, height, width, num_classes).permute(0, 4, 1, 2, 3)
        
        # Compute mean loss for each class
        class_losses = pixel_losses.mean(dim=(0, 2, 3, 4))
        
        if self.weight is not None:
            self.weight = self.weight.to(class_losses.device)
            class_losses = class_losses * self.weight
        
        if self.reduction == 'mean':
            return class_losses.mean()
        elif self.reduction == 'sum':
            return class_losses.sum()
        else:  # 'none'
            return class_losses

# from monai.losses import DiceLoss
# class customDiceLoss(nn.Module):
#     def __init__(self, smooth=1e-5):
#         super(DiceLoss, self).__init__()
#         self.smooth = smooth
        
#     def forward(self, pred, target):
#         pred = pred.contiguous()
#         target = target.contiguous()    
#         intersection = (pred * target).sum(dim=2).sum(dim=2)
#         loss = (1 - ((2. * intersection + self.smooth) / (pred.sum(dim=2).sum(dim=2) + target.sum(dim=2).sum(dim=2) + self.smooth)))
#         return loss.mean()
class ComboLoss(nn.Module):
    def __init__(self, alpha=0.5, smooth=1e-5, class_weights=torch.tensor([0,1,0,2,1,2,1])):
        super(ComboLoss, self).__init__()
        self.alpha = alpha
        self.bce = nn.BCEWithLogitsLoss(weight=class_weights.view(1, 7, 1, 1, 1))
        # self.ce = nn.CrossEntropyLoss()
        # self.dice = customDiceLoss(smooth=smooth)
        self.dice = DiceLoss(include_background=False, to_onehot_y=False, softmax=True)
        
    def forward(self, pred, target):
        target = target.float() # これやらないとタイプエラーになる
        # print("pred, target:", pred.shape, target.shape)
        bce_loss = self.bce(pred, target)
        dice_loss = self.dice(pred, target)
        # dice_loss = self.dice(torch.sigmoid(pred), target)
        return self.alpha * bce_loss + (1 - self.alpha) * dice_loss
        # return self.alpha * ce_loss + (1 - self.alpha) * dice_loss
        # return bec_loss

# class TverskyLoss(nn.Module):
#     def __init__(self, alpha=0.5, beta=0.5, smooth=1e-5):
#         super(TverskyLoss, self).__init__()
#         self.alpha = alpha
#         self.beta = beta
#         self.smooth = smooth

#     def forward(self, pred, target):
#         # pred: (B, C, D, H, W)
#         # target: (B, C, D, H, W) - one-hot encoded

#         # Softmax along the channel dimension
#         pred = F.softmax(pred, dim=1)

#         # print("pred:", pred.shape)
#         # print("target:", target.shape)

#         # Flatten the tensors
#         B, C = pred.shape[:2]
#         pred = pred.view(B, C, -1)
#         target = target.view(B, C, -1)

#         # Exclude background (assuming background is the first channel)
#         pred = pred[:, 1:, :]
#         target = target[:, 1:, :]

#         tp = (pred * target).sum(dim=2)
#         fp = (pred * (1 - target)).sum(dim=2)
#         fn = ((1 - pred) * target).sum(dim=2)

#         tversky = (tp + self.smooth) / (tp + self.alpha*fp + self.beta*fn + self.smooth)
#         loss = 1 - tversky.mean()

#         return loss

# class DiceLoss(nn.Module):
#     def __init__(self, smooth=1e-5):
#         super(DiceLoss, self).__init__()
#         self.smooth = smooth
        
#     def forward(self, pred, target):
#         pred = pred.contiguous()
#         target = target.contiguous()    
#         intersection = (pred * target).sum(dim=2).sum(dim=2)
#         loss = (1 - ((2. * intersection + self.smooth) / (pred.sum(dim=2).sum(dim=2) + target.sum(dim=2).sum(dim=2) + self.smooth)))
#         return loss.mean()
from monai.losses import DiceLoss

class model25d(pl.LightningModule):
    def __init__(
        self,
        n_channels: int = 1,
        n_classes: int = 7,
        lr: float = 1e-3,
        arch: str = 'resnet34d',
        is_train: bool = True
    ):
        super(model25d, self).__init__()
        self.save_hyperparameters()
        
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.arch = arch

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
        }.get(arch, [768])
        decoder_dim = [256, 128, 64, 32, 16]

        self.encoder = timm.create_model(
            model_name=arch, pretrained=is_train, in_chans=3, num_classes=0, global_pool='', features_only=True,
        )
        self.decoder = MyUnetDecoder3d(
            in_channel=encoder_dim[-1],
            skip_channel=encoder_dim[:-1][::-1]+[0],
            out_channel=decoder_dim,
        )
        self.mask = nn.Conv3d(decoder_dim[-1], n_classes, kernel_size=1)

        self.loss_fn = TverskyLoss()
        # self.loss_fn = ComboLoss()
        # self.loss_fn = nn.BCEWithLogitsLoss(weight=[0,1,0,2,1,2,1])
        # self.loss_fn = PerClassCrossEntropyLoss()
        # self.loss_fn = TverskyLoss(include_background=True, to_onehot_y=False, softmax=True)
        # self.loss_fn=DiceLoss()
        
        # self.loss_fn = DiceLoss(include_background=False, to_onehot_y=True, softmax=True)
        self.metric_fn = DiceMetric(include_background=False, reduction="mean", ignore_empty=True)

        self.train_loss = 0
        self.val_loss = 0
        self.train_metric=0
        self.val_metric = 0
        self.num_train_batch = 0
        self.num_val_batch = 0

    def forward(self, x):
        # print("x.shape:",x.shape)
        B, D, H, W = x.shape
        # print("=="*10)
        # print("B, D, H, W:", B, D, H, W)

        x = x.reshape(B*D, 1, H, W) # 1==channel
        # x = (x.float() - 0.5) / 0.5
        x = x.expand(-1, 3, -1, -1)
        # print(x.shape)

        if 'efficientnet' in self.arch:
            encode = encode_for_efficientnet(self.encoder, x, B, depth_scaling=[2,2,2,2,1])
        else:
            encode = encode_for_resnet(self.encoder, x, B, depth_scaling=[2,2,2,2,1])

        # [print(f'encode_{i}', e.shape) for i,e in enumerate(encode)]

        last, decode = self.decoder(
            feature=encode[-1], skip=encode[:-1][::-1]+[None], depth_scaling=[1,2,2,2,2]
        )
        # print(f'last', last.shape)

        logit = self.mask(last)

        # print("logits:", logit.shape) # logits: torch.Size([1, 7, 32, 512, 512])

        return logit

    def training_step(self, batch, batch_idx):
        x, y = batch['image'], batch['label']
        # y=y.type_as(x.type())
        # print("yyyyy", y.type())
        # print("torch.max(y)+1:", torch.max(y)+1, type(torch.max(y)+1))
        # y = convert_to_multiclass(y, 7) # multi classe to one hot
        y_hat_seg = self(x)
        # print("y_hat_seg, y:", y_hat_seg.shape, y.shape)
        # print("prediction:", pred)
        loss = self.loss_fn(y_hat_seg, y)
        metric = self.metric_fn(y_hat_seg, y)
        metric = torch.mean(metric)
        
        # ステップごとのロギング
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        self.log('train_metric', metric, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        
        # グラディエントノルムのロギング
        grad_norm = torch.nn.utils.clip_grad_norm_(self.parameters(),) #max_norm=1.0)
        # self.log('grad_norm', grad_norm, on_step=True, on_epoch=True, prog_bar=False, logger=True)

        # 学習率のロギング
        self.log('lr', self.optimizers().param_groups[0]['lr'], on_step=True, on_epoch=True, prog_bar=False, logger=True)

        # メモリ使用量のロギング
        self.log('gpu_memory', torch.cuda.memory_allocated() / 1024**3, on_step=True, on_epoch=True, prog_bar=False, logger=True)

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
        
        # mlflow.log_metric("custom_train_loss", avg_loss, self.current_epoch)
        # mlflow.log_metric("custom_train_metric", avg_metric, self.current_epoch)
        
        self.train_loss = 0
        self.train_metric = 0
        self.num_train_batch = 0

    def validation_step(self, batch, batch_idx):
        with torch.no_grad():
            # print(f"Current GPU memory usage: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
            x, y = batch['image'], batch['label']
            # y=y.type_as(x.type())
            # print("yyyyy", y.type())
            # print("torch.max(y)+1:", torch.max(y)+1, type(torch.max(y)+1))
            # y = convert_to_multiclass(y, 7)
            y_hat_seg = self(x)
            # print("valid y_hat_seg, y:", y_hat_seg.shape, y.shape)
            assert y_hat_seg.shape == y.shape, f"shape error!! y_hat_seg.shape:{y_hat_seg.shape} ,y.shape:{y.shape}"
            loss = self.loss_fn(y_hat_seg, y)
            # metric = self.metric_fn(y_hat_seg, y)

            # metric_val_outputs = [AsDiscrete(argmax=True, to_onehot=self.hparams.n_classes)(i) for i in decollate_batch(y_hat_seg)]
            # metric_val_labels = [AsDiscrete(to_onehot=self.hparams.n_classes)(i) for i in decollate_batch(y)]

            # self.metric_fn(y_pred=metric_val_outputs, y=metric_val_labels)
            
            self.metric_fn(y_hat_seg, y)
            metrics = self.metric_fn.aggregate(reduction="mean_batch")
            val_metric = torch.mean(metrics)
            
            self.log('val_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
            # self.log('val_tmp_metric', metric, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
            self.log('val_metric', val_metric, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)

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
        
        # mlflow.log_metric("custom_val_loss", avg_loss, self.current_epoch)
        # mlflow.log_metric("custom_val_metric", avg_metric, self.current_epoch)

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
        # return torch.optim.AdamW(self.parameters(), lr=self.hparams.lr)
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.hparams.lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10, eta_min=1e-6)
        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler,
        }





# class TverskyLoss(nn.Module):
#     def __init__(self, alpha=0.5, beta=0.5, smooth=1e-5):
#         super(TverskyLoss, self).__init__()
#         self.alpha = alpha
#         self.beta = beta
#         self.smooth = smooth

#     def forward(self, pred, target):
#         pred = torch.sigmoid(pred)
#         tp = (pred * target).sum(dim=(2,3))
#         fp = (pred * (1-target)).sum(dim=(2,3))
#         fn = ((1-pred) * target).sum(dim=(2,3))
#         tversky = (tp + self.smooth) / (tp + self.alpha*fp + self.beta*fn + self.smooth)
#         return 1 - tversky.mean()



class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        
    def forward(self, pred, target):
        bce_loss = F.binary_cross_entropy_with_logits(pred, target, reduction='none')
        pt = torch.exp(-bce_loss)
        focal_loss = self.alpha * (1-pt)**self.gamma * bce_loss
        return focal_loss.mean()






## 下記の飛行機のモデルのやつのパクリ
# https://github.com/knshnb/kaggle-contrails-3rd-place/blob/47fb3b4ac46195475bfb33dda93aec1bef1b1528/src/nn.py#L77-L124

# class Conv3dBlock(nn.Module):
#     def __init__(self, in_ch, out_ch, kernel_size, padding):
#         super().__init__()
#         self.conv = nn.Conv3d(in_ch, out_ch, kernel_size, padding=padding)
#         self.bn = nn.BatchNorm3d(out_ch)
#         self.relu = nn.ReLU(inplace=True)

#     def forward(self, x):
#         return self.relu(self.bn(self.conv(x)))
class Conv3dBlock(torch.nn.Sequential):
    def __init__(
        self, in_channels: int, out_channels: int, kernel_size: tuple[int, int, int], padding: tuple[int, int, int]
    ):
        super().__init__(
            torch.nn.Conv3d(in_channels, out_channels, kernel_size, padding=padding, padding_mode="replicate"),
            torch.nn.BatchNorm3d(out_channels),
            torch.nn.LeakyReLU(),
        )

class SegmentorMid25d(pl.LightningModule):
    def __init__(
        self,
        n_channels: int = 1,
        n_classes: int = 7,
        lr: float = 1e-3,
        arch: str = 'resnet34d',
        is_train: bool = True,
        n_frames: int = 32
    ):
        super(SegmentorMid25d, self).__init__()
        self.save_hyperparameters()
        
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.arch = arch
        self.n_frames = n_frames  # 追加

        # smp.Unetを使用してエンコーダーとデコーダーを置き換え
        self.backbone = smp.Unet(
            encoder_name=arch,
            encoder_weights='imagenet' if is_train else None,
            in_channels=n_channels,
            classes=n_classes,
            decoder_channels=[ch for ch in (256, 128, 64, 32, 16)],
            # decoder_channels=[ch * 8 for ch in (256, 128, 64, 32, 8)],
        )

        # print(backbone.encoder.out_channels[1:]) (64, 256, 512, 1024, 2048)

        # 3D畳み込みブロックを追加
        k = 3  # カーネルサイズ
        # self.conv3ds = nn.ModuleList([
        #     nn.Sequential(
        #         Conv3dBlock(ch, ch, (2, k, k), (0, k // 2, k // 2)),
        #         Conv3dBlock(ch, ch, (2, k, k), (0, k // 2, k // 2)),
        #     )
        #     for ch in self.backbone.encoder.out_channels[1:]
        # ])
        
        ## 時間方向に畳み込んでから元に戻す
        self.conv3ds = nn.ModuleList([
            nn.Sequential(
                # エンコーダー部分: 時間方向の圧縮
                Conv3dBlock(ch, ch, (3, k, k), (1, k // 2, k // 2)),
                Conv3dBlock(ch, ch, (3, k, k), (1, k // 2, k // 2)),
                # デコーダー部分: 時間方向の復元
                nn.ConvTranspose3d(ch, ch, (3, 1, 1), (1, 1, 1), (1, 0, 0)),
                nn.BatchNorm3d(ch),
                nn.ReLU(inplace=True),
                nn.ConvTranspose3d(ch, ch, (3, 1, 1), (1, 1, 1), (1, 0, 0)),
                nn.BatchNorm3d(ch),
                nn.ReLU(inplace=True)
            )
            for ch in self.backbone.encoder.out_channels[1:]
        ])

        # self.loss_fn = nn.BCEWithLogitsLoss(weight=torch.tensor([0,1,0,2,1,2,1]))
        # self.loss_fn = TverskyLoss()
        self.loss_fn = ComboLoss()
        # self.loss_fn = TverskyLoss(include_background=False, to_onehot_y=False, softmax=True)
        self.metric_fn = DiceMetric(include_background=False, reduction="mean", ignore_empty=True)

        self.train_loss = 0
        self.val_loss = 0
        self.train_metric=0
        self.val_metric = 0
        self.num_train_batch = 0
        self.num_val_batch = 0

    def _to2d(self, conv3d_block: nn.Module, feature: torch.Tensor) -> torch.Tensor:
        # print("feature.shape:", feature.shape)
        total_batch, ch, H, W = feature.shape
        feat_3d = feature.reshape(total_batch // self.n_frames, self.n_frames, ch, H, W).transpose(1, 2)
        feat_processed = conv3d_block(feat_3d) #.squeeze(2)
        feat_2d = feat_processed.transpose(1, 2).reshape(total_batch, ch, H, W)
        # Feature shape : torch.Size([32, 64, 128, 128]), feat_3d: torch.Size([1, 64, 32, 128, 128]), feat_2d: torch.Size([1, 64, 30, 128, 128])
        # Feature shape : torch.Size([32, 256, 64, 64]), feat_3d: torch.Size([1, 256, 32, 64, 64]), feat_2d: torch.Size([1, 256, 30, 64, 64])
        # Feature shape : torch.Size([32, 512, 32, 32]), feat_3d: torch.Size([1, 512, 32, 32, 32]), feat_2d: torch.Size([1, 512, 30, 32, 32])
        # Feature shape : torch.Size([32, 1024, 16, 16]), feat_3d: torch.Size([1, 1024, 32, 16, 16]), feat_2d: torch.Size([1, 1024, 30, 16, 16])
        # Feature shape : torch.Size([32, 2048, 8, 8]), feat_3d: torch.Size([1, 2048, 32, 8, 8]), feat_2d: torch.Size([1, 2048, 30, 8, 8])
        # print(f"Feature shape : {feature.shape}, feat_3d: {feat_3d.shape}, feat_2d: {feat_2d.shape}")
        ## 時間方向で畳み込みアンド展開後
        # Feature shape : torch.Size([32, 64, 128, 128]), feat_3d: torch.Size([1, 64, 32, 128, 128]), feat_2d: torch.Size([32, 64, 128, 128])
        # Feature shape : torch.Size([32, 256, 64, 64]), feat_3d: torch.Size([1, 256, 32, 64, 64]), feat_2d: torch.Size([32, 256, 64, 64])
        # Feature shape : torch.Size([32, 512, 32, 32]), feat_3d: torch.Size([1, 512, 32, 32, 32]), feat_2d: torch.Size([32, 512, 32, 32])
        # Feature shape : torch.Size([32, 1024, 16, 16]), feat_3d: torch.Size([1, 1024, 32, 16, 16]), feat_2d: torch.Size([32, 1024, 16, 16])
        # Feature shape : torch.Size([32, 2048, 8, 8]), feat_3d: torch.Size([1, 2048, 32, 8, 8]), feat_2d: torch.Size([32, 2048, 8, 8])
        return feat_2d

    def forward(self, x):
        B, D, H, W = x.shape
        x = x.reshape(B * D, 1, H, W) # 1はchannel

        features = self.backbone.encoder(x)
        # print("++"*10, len(features)) # 6
        features[1:] = [self._to2d(conv3d, feature) for conv3d, feature in zip(self.conv3ds, features[1:])]
        # print("*features:", *features)
        # for i in range(len(features)):
        #     print("@@@:",i, features[i].shape)
        decoder_output = self.backbone.decoder(*features)

        logit = self.backbone.segmentation_head(decoder_output) # -> ([32, 7, H, W])
        # logit = logit.unsqueeze(0) #32, 7, H, W -> 1, 32, 7, H, W
        # logit = logit.permute(0, 2, 1, 3, 4) # 1, 32, 7, H, W -> 1, 7, 32, H, W
        logit = torch.reshape(logit, (B, 7, D, H, W))
        return logit

    # training_step, validation_step などの他のメソッドは基本的に変更なし
    # ただし、入力データの形状が変わる可能性があるので、それに応じて調整が必要
    def training_step(self, batch, batch_idx):
        x, y = batch['image'], batch['label']
        y_hat_seg = self(x)
        assert y_hat_seg.shape == y.shape, f"shape error in training step!! y_hat_seg.shape:{y_hat_seg.shape} ,y.shape:{y.shape}"
        loss = self.loss_fn(y_hat_seg, y)
        metric = self.metric_fn(y_hat_seg, y)
        metric = torch.mean(metric)
        
        # ステップごとのロギング
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        self.log('train_metric', metric, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        
        # グラディエントノルムのロギング
        grad_norm = torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
        self.log('grad_norm', grad_norm, on_step=True, on_epoch=True, prog_bar=False, logger=True)

        # 学習率のロギング
        self.log('lr', self.optimizers().param_groups[0]['lr'], on_step=True, on_epoch=True, prog_bar=False, logger=True)

        # メモリ使用量のロギング
        self.log('gpu_memory', torch.cuda.memory_allocated() / 1024**3, on_step=True, on_epoch=True, prog_bar=False, logger=True)

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
        self.train_loss = 0
        self.train_metric = 0
        self.num_train_batch = 0

    def validation_step(self, batch, batch_idx):
        with torch.no_grad():
            x, y = batch['image'], batch['label']
            y_hat_seg = self(x)
            # print("valid y_hat_seg, y:", y_hat_seg.shape, y.shape)
            assert y_hat_seg.shape == y.shape, f"shape error!! y_hat_seg.shape:{y_hat_seg.shape} ,y.shape:{y.shape}"
            # if y_hat_seg.requires_grad:
            #     y_hat_seg = y_hat_seg.detach()
            
            # probs = F.softmax(y_hat_seg, dim=1)
            # # 最も確率の高いクラスを取得
            # predictions = torch.argmax(probs, dim=1)
            # y_max=torch.argmax(y, dim=1)
            
            # # 各クラスの予測数をカウント
            # class_counts = torch.bincount(predictions.flatten())
            # y_class_counts = torch.bincount(y_max.flatten())
            # print("class_counts:", class_counts, "\n", y_class_counts)
            
            loss = self.loss_fn(y_hat_seg, y)

            self.metric_fn(y_hat_seg, y)
            metrics = self.metric_fn.aggregate(reduction="mean_batch")
            val_metric = torch.mean(metrics)
            
            self.log('val_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
            self.log('val_metric', val_metric, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)

            self.val_loss += loss
            self.val_metric += val_metric 
            self.num_val_batch += 1
        torch.cuda.empty_cache()
        return {'val_metric': val_metric}

    def on_validation_epoch_end(self):
        avg_loss = self.val_loss / self.num_val_batch
        avg_metric = self.val_metric / self.num_val_batch

        self.log('val_epoch_loss', avg_loss, prog_bar=True, sync_dist=False)
        self.log('val_epoch_metric', avg_metric, prog_bar=True, sync_dist=False)
        
        self.val_loss = 0
        self.val_metric = 0
        self.num_val_batch = 0


    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.hparams.lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10, eta_min=1e-6)
        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler,
        }
        # return torch.optim.AdamW(self.parameters(), lr=self.hparams.lr)



# 別の新規モデル
# import torchvision.models as models
import segmentation_models_pytorch as smp
class ProteinSegmentor25D(pl.LightningModule):
    def __init__(self, in_ch=1, out_ch=2, frames_per_group=1, lr: float = 1e-3):
        super().__init__()
        self.save_hyperparameters()

        ## loss and metrics
        # self.loss_fn = nn.CrossEntropyLoss()
        # self.loss_fn = PerClassCrossEntropyLoss()
        # self.loss_fn = nn.BCEWithLogitsLoss()
        # self.loss_fn = TverskyLoss(include_background=True, to_onehot_y=False, softmax=True)
        self.loss_fn = ComboLoss()
        # self.metric_fn = DiceMetric(include_background=False, reduction="mean", ignore_empty=True)
        self.metric_fn = SegmentationMetrics(num_classes=7)

        self.frames_per_group = frames_per_group
        
        self.backbone = smp.Unet(
            encoder_name='resnet34',
            encoder_weights=None,  # タンパク質特有の事前学習重みがあれば使用
            in_channels=in_ch * frames_per_group,
            classes=out_ch,
            decoder_channels=[256, 128, 64, 32, 16],
        )
        
        self.conv3d = nn.Sequential(
            nn.Conv3d(64, 64, kernel_size=(3,3,3), padding=(1,1,1)),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            nn.Conv3d(64, 64, kernel_size=(3,3,3), padding=(1,1,1)),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        # x shape: (batch, channels, depth, height, width)
        B, C, D, H, W = x.shape
        
        # グループ化したフレームを処理
        grouped_frames = D // self.frames_per_group
        x = x.reshape(B * grouped_frames, C * self.frames_per_group, H, W) # B*D, 1, H, W
        
        features = self.backbone.encoder(x)
        
        # 3D畳み込みを適用
        last_feature = features[-1]
        print("-="*10, last_feature.shape)
        last_feature_size = last_feature.shape[-1]
        
        # Reshape and transpose to get the correct shape for 3D convolution
        last_feature = last_feature.reshape(B, -1, grouped_frames, last_feature_size, last_feature_size)
        last_feature = last_feature.transpose(1, 2)  # Shape: [B, grouped_frames, C, H, W]
        
        # Apply 3D convolution
        print("+="*10, last_feature.shape)
        last_feature = self.conv3d(last_feature)
        
        # Reshape back to 2D for the decoder
        last_feature = last_feature.transpose(1, 2).reshape(B * grouped_frames, -1, last_feature_size, last_feature_size)
        features[-1] = last_feature
        
        decoder_output = self.backbone.decoder(*features)
        masks = self.backbone.segmentation_head(decoder_output)
        
        # 出力を元の形状に戻す
        masks = masks.reshape(B, grouped_frames, -1, H, W)
        return masks

    def training_step(self, batch, batch_idx):
        x, y = batch['image'], batch['label']
        # y=y.type_as(x.type())
        # print("yyyyy", y.type())
        # print("torch.max(y)+1:", torch.max(y)+1, type(torch.max(y)+1))
        y = convert_to_multiclass(y, 7) # multi classe to one hot
        y_hat_seg = self(x)
        # print("y_hat_seg, y:", y_hat_seg.shape, y.shape)
        loss = self.loss_fn(y_hat_seg, y)
        metric = self.metric_fn(y_hat_seg, y)
        metric = torch.mean(metric)
        
        # ステップごとのロギング
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        self.log('train_metric', metric, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        
        # グラディエントノルムのロギング
        grad_norm = torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
        self.log('grad_norm', grad_norm, on_step=True, on_epoch=True, prog_bar=False, logger=True)

        # 学習率のロギング
        self.log('lr', self.optimizers().param_groups[0]['lr'], on_step=True, on_epoch=True, prog_bar=False, logger=True)

        # メモリ使用量のロギング
        self.log('gpu_memory', torch.cuda.memory_allocated() / 1024**3, on_step=True, on_epoch=True, prog_bar=False, logger=True)

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
        
        # mlflow.log_metric("custom_train_loss", avg_loss, self.current_epoch)
        # mlflow.log_metric("custom_train_metric", avg_metric, self.current_epoch)
        
        self.train_loss = 0
        self.train_metric = 0
        self.num_train_batch = 0

    def validation_step(self, batch, batch_idx):
        with torch.no_grad():
            # print(f"Current GPU memory usage: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
            x, y = batch['image'], batch['label']
            # y=y.type_as(x.type())
            # print("yyyyy", y.type())
            # print("torch.max(y)+1:", torch.max(y)+1, type(torch.max(y)+1))
            y = convert_to_multiclass(y, 7)
            y_hat_seg = self(x)
            # print("valid y_hat_seg, y:", y_hat_seg.shape, y.shape)
            assert y_hat_seg.shape == y.shape, f"shape error!! y_hat_seg.shape:{y_hat_seg.shape} ,y.shape:{y.shape}"
            loss = self.loss_fn(y_hat_seg, y)
            # metric = self.metric_fn(y_hat_seg, y)

            # metric_val_outputs = [AsDiscrete(argmax=True, to_onehot=self.hparams.n_classes)(i) for i in decollate_batch(y_hat_seg)]
            # metric_val_labels = [AsDiscrete(to_onehot=self.hparams.n_classes)(i) for i in decollate_batch(y)]

            # self.metric_fn(y_pred=metric_val_outputs, y=metric_val_labels)
            
            self.metric_fn(y_hat_seg, y)
            metrics = self.metric_fn.aggregate(reduction="mean_batch")
            val_metric = torch.mean(metrics)
            
            self.log('val_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
            # self.log('val_tmp_metric', metric, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
            self.log('val_metric', val_metric, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)

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
        
        # mlflow.log_metric("custom_val_loss", avg_loss, self.current_epoch)
        # mlflow.log_metric("custom_val_metric", avg_metric, self.current_epoch)

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


# class Conv3dBlock(nn.Module):
#     def __init__(self, in_ch, out_ch, kernel_size, padding):
#         super().__init__()
#         self.conv = nn.Conv3d(in_ch, out_ch, kernel_size, padding=padding)
#         self.bn = nn.BatchNorm3d(out_ch)
#         self.relu = nn.ReLU(inplace=True)

#     def forward(self, x):
#         return self.relu(self.bn(self.conv(x)))

# class ProteinSegmentor3D(pl.LightningModule):
#     def __init__(self, in_channels=1, out_channels=7, n_frames=184, lr: float = 1e-3):
#         super().__init__()
#         self.save_hyperparameters()

#         self.n_frames = n_frames


#         ## loss and metrics
#         # self.loss_fn = nn.CrossEntropyLoss()
#         # self.loss_fn = PerClassCrossEntropyLoss()
#         # self.loss_fn = nn.BCEWithLogitsLoss()
#         # self.loss_fn = TverskyLoss(include_background=True, to_onehot_y=False, softmax=True)
#         self.loss_fn = ComboLoss()
#         # self.metric_fn = DiceMetric(include_background=False, reduction="mean", ignore_empty=True)
#         self.metric_fn = SegmentationMetrics(num_classes=7)

#         # ResNet encoder
#         self.encoder = models.resnet50(pretrained=True)
#         self.encoder.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)

#         # 3D convolution blocks
#         self.conv3d_blocks = nn.ModuleList([
#             Conv3dBlock(64, 64, kernel_size=3, padding=1),
#             Conv3dBlock(256, 128, kernel_size=3, padding=1),
#             Conv3dBlock(512, 256, kernel_size=3, padding=1),
#             Conv3dBlock(1024, 512, kernel_size=3, padding=1),
#             Conv3dBlock(2048, 1024, kernel_size=3, padding=1)
#         ])

#         # Decoder
#         self.decoder = nn.ModuleList([
#             self._decoder_block(1024, 512),
#             self._decoder_block(512, 256),
#             self._decoder_block(256, 128),
#             self._decoder_block(128, 64),
#             self._decoder_block(64, 32)
#         ])

#         self.final_conv = nn.Conv2d(32, out_channels, kernel_size=1)

#     def _decoder_block(self, in_channels, out_channels):
#         return nn.Sequential(
#             nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
#             nn.BatchNorm2d(out_channels),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
#             nn.BatchNorm2d(out_channels),
#             nn.ReLU(inplace=True),
#             # nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
#         )

#     def forward(self, x):
#         print("ProteinSegmentor3D input:", x.shape)
#         # x shape: (batch, channels, frames, height, width)
#         batch, channels, frames, height, width = x.shape
#         x = x.transpose(1, 2).reshape(batch * frames, channels, height, width)

#         # Encoder
#         features = []
#         x = self.encoder.conv1(x)
#         x = self.encoder.bn1(x)
#         x = self.encoder.relu(x)
#         x = self.encoder.maxpool(x)

#         for layer in [self.encoder.layer1, self.encoder.layer2, self.encoder.layer3, self.encoder.layer4]:
#             x = layer(x)
#             print("=="*10, x.shape)
#             features.append(x)


#         # 3D convolution
#         for i, conv3d in enumerate(self.conv3d_blocks):
#             # if i == 0:
#             #     shape = (batch, 64, frames) + features[i].shape[-2:]
#             # else:
#             #     shape = (batch, features[i].shape[1], frames) + features[i].shape[-2:]
#             # feature_3d = features[i].view(*shape)
#             # feature_3d = conv3d(feature_3d)
#             # features[i] = feature_3d.view(batch * frames, -1, *feature_3d.shape[-2:])
#             feature_shape = features[i].shape
#             shape = (batch, feature_shape[1], frames) + feature_shape[2:]
            
#             feature_3d = features[i].view(*shape)
#             print("=+"*10, feature_3d.shape)
#             feature_3d = conv3d(feature_3d)
#             features[i] = feature_3d.view(batch * frames, -1, *feature_3d.shape[-2:])

#         # Decoder
#         x = features[-1]
#         for i, decoder_block in enumerate(self.decoder):
#             x = decoder_block(x)
#             if i < len(features) - 1:
#                 # x = torch.cat([x, features[-i-2]], dim=1)
#                 # Resize x to match the spatial dimensions of the corresponding feature map
#                 x = F.interpolate(x, size=features[-i-2].shape[-2:], mode='bilinear', align_corners=True)
#                 x = torch.cat([x, features[-i-2]], dim=1)


#         x = self.final_conv(x)

#         # Reshape output
#         x = x.view(batch, frames, -1, x.shape[-2], x.shape[-1])
#         x = x.transpose(1, 2)

#         return x

#     def training_step(self, batch, batch_idx):
#         x, y = batch['image'], batch['label']
#         # y=y.type_as(x.type())
#         # print("yyyyy", y.type())
#         # print("torch.max(y)+1:", torch.max(y)+1, type(torch.max(y)+1))
#         y = convert_to_multiclass(y, 7) # multi classe to one hot
#         y_hat_seg = self(x)
#         # print("y_hat_seg, y:", y_hat_seg.shape, y.shape)
#         loss = self.loss_fn(y_hat_seg, y)
#         metric = self.metric_fn(y_hat_seg, y)
#         metric = torch.mean(metric)
        
#         # ステップごとのロギング
#         self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
#         self.log('train_metric', metric, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        
#         # グラディエントノルムのロギング
#         grad_norm = torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
#         self.log('grad_norm', grad_norm, on_step=True, on_epoch=True, prog_bar=False, logger=True)

#         # 学習率のロギング
#         self.log('lr', self.optimizers().param_groups[0]['lr'], on_step=True, on_epoch=True, prog_bar=False, logger=True)

#         # メモリ使用量のロギング
#         self.log('gpu_memory', torch.cuda.memory_allocated() / 1024**3, on_step=True, on_epoch=True, prog_bar=False, logger=True)

#         self.train_loss += loss
#         self.train_metric += metric 
#         self.num_train_batch += 1
#         torch.cuda.empty_cache()
#         return loss

#     def on_train_epoch_end(self):
#         avg_loss = self.train_loss / self.num_train_batch
#         avg_metric = self.train_metric / self.num_train_batch
        
#         self.log('train_epoch_loss', avg_loss, prog_bar=True)
#         self.log('train_epoch_metric', avg_metric, prog_bar=True)
        
#         # mlflow.log_metric("custom_train_loss", avg_loss, self.current_epoch)
#         # mlflow.log_metric("custom_train_metric", avg_metric, self.current_epoch)
        
#         self.train_loss = 0
#         self.train_metric = 0
#         self.num_train_batch = 0

#     def validation_step(self, batch, batch_idx):
#         with torch.no_grad():
#             # print(f"Current GPU memory usage: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
#             x, y = batch['image'], batch['label']
#             # y=y.type_as(x.type())
#             # print("yyyyy", y.type())
#             # print("torch.max(y)+1:", torch.max(y)+1, type(torch.max(y)+1))
#             y = convert_to_multiclass(y, 7)
#             y_hat_seg = self(x)
#             # print("valid y_hat_seg, y:", y_hat_seg.shape, y.shape)
#             assert y_hat_seg.shape == y.shape, f"shape error!! y_hat_seg.shape:{y_hat_seg.shape} ,y.shape:{y.shape}"
#             loss = self.loss_fn(y_hat_seg, y)
#             # metric = self.metric_fn(y_hat_seg, y)

#             # metric_val_outputs = [AsDiscrete(argmax=True, to_onehot=self.hparams.n_classes)(i) for i in decollate_batch(y_hat_seg)]
#             # metric_val_labels = [AsDiscrete(to_onehot=self.hparams.n_classes)(i) for i in decollate_batch(y)]

#             # self.metric_fn(y_pred=metric_val_outputs, y=metric_val_labels)
            
#             self.metric_fn(y_hat_seg, y)
#             metrics = self.metric_fn.aggregate(reduction="mean_batch")
#             val_metric = torch.mean(metrics)
            
#             self.log('val_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
#             # self.log('val_tmp_metric', metric, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
#             self.log('val_metric', val_metric, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)

#             self.val_loss += loss
#             self.val_metric += val_metric 
#             self.num_val_batch += 1
#         torch.cuda.empty_cache()
#         return {'val_metric': val_metric}

#     def on_validation_epoch_end(self):
#         # metric_per_epoch = self.val_metric / self.num_val_batch
#         avg_loss = self.val_loss / self.num_val_batch
#         avg_metric = self.val_metric / self.num_val_batch

#         self.log('val_epoch_loss', avg_loss, prog_bar=True, sync_dist=False)
#         self.log('val_epoch_metric', avg_metric, prog_bar=True, sync_dist=False)
        
#         # mlflow.log_metric("custom_val_loss", avg_loss, self.current_epoch)
#         # mlflow.log_metric("custom_val_metric", avg_metric, self.current_epoch)

#         self.val_loss = 0
#         self.val_metric = 0
#         self.num_val_batch = 0

#     def configure_optimizers(self):
#         # optimizer = torch.optim.AdamW(self.parameters(), lr=self.hparams.lr)
#         # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10)
#         # return {
#         #     "optimizer": optimizer,
#         #     "lr_scheduler": {
#         #         "scheduler": scheduler,
#         #         "monitor": "val_loss",
#         #         "frequency": 1
#         #     },
#         # }
#         return torch.optim.AdamW(self.parameters(), lr=self.hparams.lr)




## 2xx 3dCNN
class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.InstanceNorm3d(out_channels),  # BatchNormの代わりにInstanceNormを使用
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.InstanceNorm3d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class Down(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool3d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)

class Up(nn.Module):
    def __init__(self, in_channels, out_channels, bilinear=False):
        super().__init__()

        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels)
        else:
            self.up = nn.ConvTranspose3d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        
        # パディングの計算
        diff_z = x2.size()[2] - x1.size()[2]
        diff_y = x2.size()[3] - x1.size()[3]
        diff_x = x2.size()[4] - x1.size()[4]
        
        x1 = F.pad(x1, [diff_x // 2, diff_x - diff_x // 2,
                       diff_y // 2, diff_y - diff_y // 2,
                       diff_z // 2, diff_z - diff_z // 2])
        
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

from torchmetrics import Metric
import numpy as np
from scipy.spatial.distance import cdist

# タンパク質の定義
PROTEINS = {
    0: {"name": "background", "radius": 0},
    1: {"name": "apo-ferritin", "radius": 60, "pdb_id": "4V1W"},
    2: {"name": "beta-amylase", "radius": 65, "pdb_id": "1FA2"},
    3: {"name": "beta-galactosidase", "radius": 90, "pdb_id": "6X1Q"},
    4: {"name": "ribosome", "radius": 150, "pdb_id": "6EK0"},
    5: {"name": "thyroglobulin", "radius": 130, "pdb_id": "6SCJ"},
    6: {"name": "virus-like-particle", "radius": 135, "pdb_id": "6N4V"}
}

# サイズグループの定義
def get_size_group(radius):
    if radius < 70:
        return 'small'
    elif radius < 100:
        return 'medium'
    else:
        return 'large'

# カスタムメトリクスの定義
class CryoETMetrics(Metric):
    def __init__(self, distance_threshold=5.0, num_classes=7):
        super().__init__()
        self.distance_threshold = distance_threshold
        self.num_classes = num_classes
        # self.device="cuda"
        
        # メトリクス用のstate追加（デバイスを指定）
        self.add_state("tp", default=torch.zeros(num_classes, device=self.device), dist_reduce_fx="sum")
        self.add_state("fp", default=torch.zeros(num_classes, device=self.device), dist_reduce_fx="sum")
        self.add_state("fn", default=torch.zeros(num_classes, device=self.device), dist_reduce_fx="sum")
        self.add_state("total_distance", default=torch.tensor(0.0, device=self.device), dist_reduce_fx="sum")
        self.add_state("valid_distance_count", default=torch.tensor(0, device=self.device), dist_reduce_fx="sum")
        self.add_state("pred_labels", default=torch.tensor([], device=self.device), dist_reduce_fx="cat")
        self.add_state("true_labels", default=torch.tensor([], device=self.device), dist_reduce_fx="cat")

    def update(self, preds, targets, coordinates=None):
        # デバイスの確認と統一
        device = preds.device
        self.tp = self.tp.to(device)
        self.fp = self.fp.to(device)
        self.fn = self.fn.to(device)
        
        # 予測とターゲットの処理
        pred_labels = preds.argmax(dim=1)
        
        # ラベルの更新（デバイス上で処理）
        self.pred_labels = torch.cat([self.pred_labels.to(device), pred_labels.flatten()])
        self.true_labels = torch.cat([self.true_labels.to(device), targets.flatten()])
        
        # 座標が提供された場合の距離計算
        if coordinates is not None:
            pred_coords = self._extract_coordinates(pred_labels)
            true_coords = self._extract_coordinates(targets)
            
            if len(pred_coords) > 0 and len(true_coords) > 0:
                distances = torch.tensor(cdist(pred_coords, true_coords), device=device)
                valid_distances = distances[distances <= self.distance_threshold]
                
                if len(valid_distances) > 0:
                    self.total_distance = self.total_distance.to(device) + valid_distances.sum()
                    self.valid_distance_count = self.valid_distance_count.to(device) + len(valid_distances)

        # クラスごとのTP, FP, FNの計算
        for i in range(self.num_classes):
            pred_mask = pred_labels == i
            true_mask = targets == i
            
            self.tp[i] += torch.logical_and(pred_mask, true_mask).sum()
            self.fp[i] += torch.logical_and(pred_mask, ~true_mask).sum()
            self.fn[i] += torch.logical_and(~pred_mask, true_mask).sum()

    def compute(self):
        # 全ての計算をCPUで行う
        tp = self.tp.cpu()
        fp = self.fp.cpu()
        fn = self.fn.cpu()
        
        precision = tp / (tp + fp + 1e-7)
        recall = tp / (tp + fn + 1e-7)
        f1 = 2 * (precision * recall) / (precision + recall + 1e-7)
        
        # サイズグループごとのメトリクス
        size_groups = {'small': [], 'medium': [], 'large': []}
        for i in range(1, self.num_classes):  # 背景を除く
            group = get_size_group(PROTEINS[i]['radius'])
            size_groups[group].append(i)
        
        size_metrics = {}
        for group, indices in size_groups.items():
            if indices:
                group_tp = tp[indices].sum()
                group_fp = fp[indices].sum()
                group_fn = fn[indices].sum()
                
                group_precision = group_tp / (group_tp + group_fp + 1e-7)
                group_recall = group_tp / (group_tp + group_fn + 1e-7)
                group_f1 = 2 * (group_precision * group_recall) / (group_precision + group_recall + 1e-7)
                
                size_metrics[group] = {
                    'precision': group_precision,
                    'recall': group_recall,
                    'f1': group_f1
                }

        # 距離ベースのメトリクス
        avg_distance = torch.tensor(0.0)
        if self.valid_distance_count > 0:
            avg_distance = self.total_distance.cpu() / self.valid_distance_count.cpu()

        return {
            'class_metrics': {
                'precision': precision,
                'recall': recall,
                'f1': f1
            },
            'size_metrics': size_metrics,
            'avg_distance': avg_distance
        }


class UMC(pl.LightningModule):
    def __init__(self, n_channels=1, n_classes=13, lr=1e-3, alpha=0.7, beta=0.3):
        super(UMC, self).__init__()
        self.save_hyperparameters()

        # メトリクスの初期化
        # self.train_metrics = CryoETMetrics()
        self.val_metrics = CryoETMetrics()
        
        # エンコーダー
        self.inc = DoubleConv(n_channels, 16)
        self.down1 = Down(16, 32)
        self.down2 = Down(32, 64)
        self.down3 = Down(64, 128)
        self.down4 = Down(128, 256)

        # セグメンテーションデコーダー
        self.up1_seg = Up(256, 128)
        self.up2_seg = Up(128, 64)
        self.up3_seg = Up(64, 32)
        self.up4_seg = Up(32, 16)
        self.outc_seg = nn.Conv3d(16, n_classes, kernel_size=1)

        # デノイジングデコーダー
        self.up1_denoise = Up(256, 128)
        self.up2_denoise = Up(128, 64)
        self.up3_denoise = Up(64, 32)
        self.up4_denoise = Up(32, 16)
        self.outc_denoise = nn.Conv3d(16, n_channels, kernel_size=1)

    def forward(self, x):
        # エンコーダー
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        # セグメンテーションデコーダー
        x_seg = self.up1_seg(x5, x4)
        x_seg = self.up2_seg(x_seg, x3)
        x_seg = self.up3_seg(x_seg, x2)
        x_seg = self.up4_seg(x_seg, x1)
        logits_seg = self.outc_seg(x_seg)

        # デノイジングデコーダー
        x_denoise = self.up1_denoise(x5, x4)
        x_denoise = self.up2_denoise(x_denoise, x3)
        x_denoise = self.up3_denoise(x_denoise, x2)
        x_denoise = self.up4_denoise(x_denoise, x1)
        logits_denoise = self.outc_denoise(x_denoise)

        return logits_seg, logits_denoise

    def tversky_loss(self, pred, target):
        pred = torch.softmax(pred, dim=1)
        
        # one-hotエンコーディング
        target_oh = F.one_hot(target, num_classes=pred.shape[1])
        target_oh = target_oh.permute(0, 4, 1, 2, 3)
        
        # 真陽性、偽陽性、偽陰性の計算
        tp = torch.sum(pred * target_oh, dim=(0, 2, 3, 4))
        fp = torch.sum(pred * (1 - target_oh), dim=(0, 2, 3, 4))
        fn = torch.sum((1 - pred) * target_oh, dim=(0, 2, 3, 4))
        
        # Tversky係数の計算
        numerator = tp + 1e-7
        denominator = tp + self.hparams.alpha * fp + self.hparams.beta * fn + 1e-7
        
        loss = 1 - torch.mean(numerator / denominator)
        return loss

    def training_step(self, batch, batch_idx):
        inputs = batch['data']
        labels = batch['label']
        
        # 順伝播
        seg_outputs, denoise_outputs = self(inputs)
        
        # 損失の計算
        seg_loss = self.tversky_loss(seg_outputs, labels)
        denoise_loss = F.mse_loss(denoise_outputs, inputs)
        total_loss = seg_loss + denoise_loss
        
        # ログの記録
        self.log('train_seg_loss', seg_loss)
        self.log('train_denoise_loss', denoise_loss)
        self.log('train_total_loss', total_loss)
        
        return total_loss

    def validation_step(self, batch, batch_idx):
        inputs = batch['data']
        labels = batch['label']
        
        seg_outputs, denoise_outputs = self(inputs)
        
        seg_loss = self.tversky_loss(seg_outputs, labels)
        denoise_loss = F.mse_loss(denoise_outputs, inputs)
        total_loss = seg_loss + denoise_loss
        
        # メトリクスの計算
        pred_labels = torch.argmax(seg_outputs, dim=1)
        accuracy = (pred_labels == labels).float().mean()
        
        self.log('val_seg_loss', seg_loss)
        self.log('val_denoise_loss', denoise_loss)
        self.log('val_total_loss', total_loss)
        self.log('val_accuracy', accuracy)

        # メトリクスの更新
        self.val_metrics.update(seg_outputs, labels)
        
        return total_loss

    def on_validation_epoch_end(self):
        # メトリクスのログ
        metrics = self.val_metrics.compute()
        for class_idx in range(self.hparams.n_classes):
            protein_name = PROTEINS[class_idx]['name']
            self.log(f'val_{protein_name}_f1', metrics['class_metrics']['f1'][class_idx])
        
        for group, group_metrics in metrics['size_metrics'].items():
            self.log(f'val_{group}_f1', group_metrics['f1'])
        
        self.log('val_avg_distance', metrics['avg_distance'])

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5, verbose=True
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_total_loss"
            }
        }