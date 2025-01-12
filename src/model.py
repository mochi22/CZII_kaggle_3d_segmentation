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

        # self.loss_fn = nn.CrossEntropyLoss()
        self.loss_fn = PerClassCrossEntropyLoss()
        # self.loss_fn = TverskyLoss(include_background=True, to_onehot_y=False, softmax=True)
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
        x = (x.float() - 0.5) / 0.5
        x = x.expand(-1, 3, -1, -1)
        # print(x.shape)

        encode = encode_for_resnet(self.encoder, x, B, depth_scaling=[2,2,2,2,1])

        # [print(f'encode_{i}', e.shape) for i,e in enumerate(encode)]

        last, decode = self.decoder(
            feature=encode[-1], skip=encode[:-1][::-1]+[None], depth_scaling=[1,2,2,2,2]
        )
        # print(f'last', last.shape)

        logit = self.mask(last)

        # print("logits:", logit.shape)

        return logit

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





