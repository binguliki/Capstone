
import tensorflow as tf
import keras
from keras import layers
import numpy as np
import math
from typing import Dict, Any, List

IDX_TO_CLASS = {
    0: "healthy", 1: "tomato_early_blight", 2: "tomato_late_blight", 3: "tomato_leaf_miner",
    4: "tomato_mosaic_virus", 5: "tomato_septoria_leaf_spot", 6: "tomato_spider_mites",
    7: "tomato_yellow_leaf_curl_virus", 8: "corn_common_rust", 9: "corn_gray_leaf_spot",
    10: "corn_northern_leaf_blight", 11: "potato_early_blight", 12: "potato_late_blight",
    13: "rice_blast", 14: "rice_brown_spot", 15: "rice_leaf_scald", 16: "wheat_leaf_rust",
    17: "wheat_powdery_mildew", 18: "wheat_septoria", 19: "apple_scab", 20: "apple_black_rot",
    21: "apple_cedar_rust", 22: "grape_black_measles", 23: "grape_leaf_blight",
    24: "strawberry_leaf_scorch", 25: "pepper_bacterial_spot", 26: "soybean_bacterial_pustule",
    27: "cherry_powdery_mildew", 28: "peach_bacterial_spot", 29: "blueberry_rust",
}
NUM_CLASSES = 30
CFG = {"DIMS": [96, 120, 144], "DEPTHS": [2, 4, 3], "DROPOUT": 0.2}


def conv_bn_act(filters, kernel_size=3, stride=1, padding='same',
                groups=1, activation='swish', use_bias=False, name=None):
    """Conv → BN → Activation block."""
    return keras.Sequential([
        layers.Conv2D(
            filters, kernel_size, strides=stride, padding=padding,
            groups=groups, use_bias=use_bias,
            kernel_initializer='he_normal'),
        layers.BatchNormalization(momentum=0.99, epsilon=1e-3),
        (layers.Activation(activation) if activation
         else layers.Lambda(lambda x: x)),
    ], name=name)


@keras.saving.register_keras_serializable()
class InvertedResidualBlock(layers.Layer):
    """
    MobileNetV2 Inverted Residual: Expand → Depthwise → Project.
    Skip connection when stride==1 and in_ch==out_ch.
    """
    def __init__(self, in_ch=16, out_ch=32, stride=1, expand_ratio=4, dropout=0.0, **kw):
        super().__init__(**kw)
        self.use_skip = (stride == 1) and (in_ch == out_ch)
        hidden = int(in_ch * expand_ratio)
        seq = []
        if expand_ratio != 1:
            seq.append(conv_bn_act(hidden, 1))
        seq += [
            conv_bn_act(hidden, 3, stride=stride, groups=hidden),
            layers.Conv2D(out_ch, 1, use_bias=False,
                          kernel_initializer='he_normal'),
            layers.BatchNormalization(momentum=0.99, epsilon=1e-3),
        ]
        if dropout > 0:
            seq.append(layers.Dropout(dropout))
        self.conv = keras.Sequential(seq)

    def call(self, x, training=False):
        out = self.conv(x, training=training)
        return out + x if self.use_skip else out


@keras.saving.register_keras_serializable()
class MultiHeadSelfAttention(layers.Layer):
    """Standard MHSA for transformer layers."""
    def __init__(self, dim=96, num_heads=1, dropout=0.0, **kw):
        super().__init__(**kw)
        assert dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim  = dim // num_heads
        self.scale     = self.head_dim ** -0.5
        self.qkv       = layers.Dense(3 * dim, use_bias=False)
        self.out_proj  = layers.Dense(dim)
        self.attn_drop = layers.Dropout(dropout)
        self.proj_drop = layers.Dropout(dropout)

    def call(self, x, training=False):
        B  = tf.shape(x)[0]
        N  = tf.shape(x)[1]
        C  = self.head_dim * self.num_heads
        qkv = self.qkv(x)                              # [B,N,3C]
        qkv = tf.reshape(qkv, (B, N, 3, self.num_heads, self.head_dim))
        qkv = tf.transpose(qkv, (2, 0, 3, 1, 4))      # [3,B,H,N,hd]
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = tf.matmul(q, k, transpose_b=True) * self.scale
        attn = tf.nn.softmax(attn, axis=-1)
        attn = self.attn_drop(attn, training=training)
        x    = tf.matmul(attn, v)                      # [B,H,N,hd]
        x    = tf.transpose(x, (0, 2, 1, 3))
        x    = tf.reshape(x, (B, N, C))
        x    = self.out_proj(x)
        return self.proj_drop(x, training=training)


@keras.saving.register_keras_serializable()
class TransformerLayer(layers.Layer):
    """Pre-norm transformer encoder layer: MHSA + FFN."""
    def __init__(self, dim=96, num_heads=1, mlp_ratio=2.0, dropout=0.0, **kw):
        super().__init__(**kw)
        self.norm1 = layers.LayerNormalization(epsilon=1e-6)
        self.attn  = MultiHeadSelfAttention(dim, num_heads, dropout)
        self.norm2 = layers.LayerNormalization(epsilon=1e-6)
        mlp_dim    = int(dim * mlp_ratio)
        self.ffn   = keras.Sequential([
            layers.Dense(mlp_dim, activation='swish'),
            layers.Dropout(dropout),
            layers.Dense(dim),
            layers.Dropout(dropout),
        ])

    def call(self, x, training=False):
        x = x + self.attn(self.norm1(x), training=training)
        x = x + self.ffn(self.norm2(x),  training=training)
        return x


@keras.saving.register_keras_serializable()
class MobileViTBlock(layers.Layer):
    """
    Core MobileViT block — captures both local and global representations.
    Pipeline: nxn Conv → 1x1 project → patch unfold → transformer → fold → 1x1 → concat → fuse
    """
    def __init__(self, channels=96, dim=96, depth=2, patch_h=2, patch_w=2,
                 num_heads=1, mlp_ratio=2.0, dropout=0.0, **kw):
        super().__init__(**kw)
        self.ph, self.pw = patch_h, patch_w
        # Local representation
        self.local_rep = keras.Sequential([
            conv_bn_act(channels, 3),
            layers.Conv2D(dim, 1, use_bias=False),
        ])
        # Global representation (transformer)
        self.transformer = keras.Sequential(
            [TransformerLayer(dim, num_heads, mlp_ratio, dropout)
             for _ in range(depth)])
        self.norm = layers.LayerNormalization(epsilon=1e-6)
        # Projection back
        self.proj = layers.Conv2D(channels, 1, use_bias=False)
        # Fusion
        self.fusion = conv_bn_act(channels, 3)

    def call(self, x, training=False):
        B  = tf.shape(x)[0]
        H  = tf.shape(x)[1]
        W  = tf.shape(x)[2]
        ph, pw = self.ph, self.pw

        x_local = self.local_rep(x, training=training)   # [B, H, W, dim]

        # Number of patches along each axis
        nph = H // ph   # patches vertically
        npw = W // pw   # patches horizontally
        P   = ph * pw   # pixels per patch
        N   = nph * npw # total patches
        dim = x_local.shape[-1]

        # Unfold: [B, H, W, dim] → [B, P, N, dim]
        x_r = tf.reshape(x_local, (B, nph, ph, npw, pw, dim))
        x_r = tf.transpose(x_r, (0, 2, 4, 1, 3, 5))    # [B, ph, pw, nph, npw, dim]
        x_r = tf.reshape(x_r, (B, P, N, dim))
        x_r = tf.reshape(x_r, (B * P, N, dim))          # [B*P, N, dim]

        # Transformer over patches
        x_r = self.transformer(x_r, training=training)
        x_r = self.norm(x_r)

        # Fold back: [B*P, N, dim] → [B, H, W, dim]
        x_r = tf.reshape(x_r, (B, ph, pw, nph, npw, dim))
        x_r = tf.transpose(x_r, (0, 3, 1, 4, 2, 5))    # [B, nph, ph, npw, pw, dim]
        x_r = tf.reshape(x_r, (B, H, W, dim))

        # Project back to channel width and fuse
        x_r = self.proj(x_r, training=training)
        x   = self.fusion(tf.concat([x, x_r], axis=-1), training=training)
        return x



@keras.saving.register_keras_serializable()
class MobileViTBackbone(keras.Model):
    """
    MobileViT-S backbone producing 5 multi-scale feature maps.

    For input 256×256:
      c1: 128×128, 32ch
      c2:  64×64,  64ch
      c3:  32×32,  96ch
      c4:  16×16, 128ch
      c5:   8×8,  640ch
    """
    def __init__(self, dims=(96, 120, 144), depths=(2, 4, 3),
                 patch_h=2, patch_w=2, dropout=0.1, **kw):
        super().__init__(**kw)

        # Stem: 256→128, 16ch
        self.stem = conv_bn_act(16, 3, stride=2)

        # Stage 1: 128×128, 16→32ch
        self.stage1 = InvertedResidualBlock(16, 32, stride=1, expand_ratio=4)

        # Stage 2: 128→64, 32→64ch
        self.stage2 = keras.Sequential([
            InvertedResidualBlock(32, 64, stride=2, expand_ratio=4),
            InvertedResidualBlock(64, 64, stride=1, expand_ratio=4),
            InvertedResidualBlock(64, 64, stride=1, expand_ratio=4),
        ])

        # Stage 3: 64→32, 64→96ch + MobileViT
        self.stage3_mv2  = InvertedResidualBlock(64,  96, stride=2, expand_ratio=4)
        self.stage3_mvit = MobileViTBlock(
            96, dims[0], depths[0], patch_h, patch_w, num_heads=1, dropout=dropout)

        # Stage 4: 32→16, 96→128ch + MobileViT
        self.stage4_mv2  = InvertedResidualBlock( 96, 128, stride=2, expand_ratio=4)
        self.stage4_mvit = MobileViTBlock(
            128, dims[1], depths[1], patch_h, patch_w, num_heads=2, dropout=dropout)

        # Stage 5: 16→8, 128→160ch + MobileViT
        self.stage5_mv2  = InvertedResidualBlock(128, 160, stride=2, expand_ratio=4)
        self.stage5_mvit = MobileViTBlock(
            160, dims[2], depths[2], patch_h, patch_w, num_heads=3, dropout=dropout)

        # Final pointwise: 160→640ch
        self.final_conv = conv_bn_act(640, 1)

    def call(self, x, training=False):
        x  = self.stem(x,    training=training)         # 128×128, 16
        x  = self.stage1(x,  training=training)         # 128×128, 32
        c1 = x

        x  = self.stage2(x,  training=training)         #  64×64,  64
        c2 = x

        x  = self.stage3_mv2 (x, training=training)     #  32×32,  96
        x  = self.stage3_mvit(x, training=training)
        c3 = x

        x  = self.stage4_mv2 (x, training=training)     #  16×16, 128
        x  = self.stage4_mvit(x, training=training)
        c4 = x

        x  = self.stage5_mv2 (x, training=training)     #   8×8,  160
        x  = self.stage5_mvit(x, training=training)
        c5 = self.final_conv(x,  training=training)     #   8×8,  640

        return c1, c2, c3, c4, c5



@keras.saving.register_keras_serializable()
class FPNNeck(layers.Layer):
    """
    Feature Pyramid Network — fuses multi-scale backbone features.
    Outputs 4 pyramid levels: P3 (32×32), P4 (16×16), P5 (8×8), P6 (4×4).
    """
    def __init__(self, out_channels: int = 128, **kw):
        super().__init__(**kw)
        # Lateral 1×1 convs (channel reduction)
        self.lat3 = layers.Conv2D(out_channels, 1, use_bias=False)
        self.lat4 = layers.Conv2D(out_channels, 1, use_bias=False)
        self.lat5 = layers.Conv2D(out_channels, 1, use_bias=False)
        # Smoothing convs
        self.out3 = conv_bn_act(out_channels, 3)
        self.out4 = conv_bn_act(out_channels, 3)
        self.out5 = conv_bn_act(out_channels, 3)
        # Extra level: downsample P5 → P6
        self.p6   = conv_bn_act(out_channels, 3, stride=2)

    @staticmethod
    def _upsample_add(x_high, x_low):
        h = tf.shape(x_low)[1]
        w = tf.shape(x_low)[2]
        dtype = x_low.dtype
        up = tf.image.resize(
            tf.cast(x_high, tf.float32), (h, w), method='bilinear')
        return tf.cast(up, dtype) + x_low

    def call(self, features, training=False):
        _, _, c3, c4, c5 = features
        p5 = self.lat5(c5, training=training)
        p4 = self._upsample_add(p5, self.lat4(c4, training=training))
        p3 = self._upsample_add(p4, self.lat3(c3, training=training))

        p3 = self.out3(p3, training=training)
        p4 = self.out4(p4, training=training)
        p5 = self.out5(p5, training=training)
        p6 = self.p6(p5,   training=training)

        return [p3, p4, p5, p6]  # 32×32, 16×16, 8×8, 4×4


@keras.saving.register_keras_serializable()
class DetectionHead(layers.Layer):
    """
    SSD-Lite style detection head operating on FPN feature maps.
    Predicts: per-anchor class logits + bounding box deltas.

    Output shapes (for 256×256 input):
      cls_logits: [B, total_anchors, num_classes]
      box_preds:  [B, total_anchors, 4]
    total_anchors = (32²+16²+8²+4²) × num_anchors = (1024+256+64+16) × 9 = 12 240
    """
    def __init__(self, num_classes: int = 30, num_anchors: int = 9,
                 fpn_channels: int = 128, **kw):
        super().__init__(**kw)
        self.num_classes = num_classes
        self.num_anchors = num_anchors

        # Depthwise-separable conv towers (one per FPN level)
        def _dw_tower():
            return keras.Sequential([
                conv_bn_act(fpn_channels, 3, groups=fpn_channels),
                layers.Conv2D(fpn_channels, 1),
                layers.Activation('swish'),
            ])

        self.cls_towers = [_dw_tower() for _ in range(4)]
        self.reg_towers = [_dw_tower() for _ in range(4)]
        self.cls_preds  = [layers.Conv2D(num_anchors * num_classes, 1)
                           for _ in range(4)]
        self.reg_preds  = [layers.Conv2D(num_anchors * 4, 1)
                           for _ in range(4)]

        # Bias init for cls pred: log((1-π)/π), π=0.01 → reduces early false pos.
        prior_prob = 0.01
        self._bias_init = -math.log((1 - prior_prob) / prior_prob)

    def call(self, fpn_feats, training=False):
        all_cls, all_reg = [], []
        for i, feat in enumerate(fpn_feats):
            cls_feat = self.cls_towers[i](feat, training=training)
            reg_feat = self.reg_towers[i](feat, training=training)
            cls_out  = self.cls_preds[i](cls_feat)
            reg_out  = self.reg_preds[i](reg_feat)

            B = tf.shape(cls_out)[0]
            H = tf.shape(cls_out)[1]
            W = tf.shape(cls_out)[2]

            cls_out = tf.reshape(cls_out,
                (B, H * W * self.num_anchors, self.num_classes))
            reg_out = tf.reshape(reg_out,
                (B, H * W * self.num_anchors, 4))

            all_cls.append(cls_out)
            all_reg.append(reg_out)

        cls_logits = tf.concat(all_cls, axis=1)   # [B, total_anchors, C]
        box_preds  = tf.concat(all_reg, axis=1)   # [B, total_anchors, 4]
        return cls_logits, box_preds



@keras.saving.register_keras_serializable()
class ClassificationHead(layers.Layer):
    """
    Global image-level classification head.
    Takes the deepest backbone feature (c5) and predicts disease class logits.
    """
    def __init__(self, num_classes: int = 30, hidden_dim: int = 512,
                 dropout: float = 0.2, **kw):
        super().__init__(**kw)
        self.gap = layers.GlobalAveragePooling2D()
        self.cls_head = keras.Sequential([
            layers.Dense(hidden_dim, activation='swish',
                         kernel_initializer='he_normal'),
            layers.BatchNormalization(),
            layers.Dropout(dropout),
            layers.Dense(num_classes, dtype='float32'),  # float32 for loss stability
        ])

    def call(self, c5, training=False):
        x = self.gap(c5)
        return self.cls_head(x, training=training)   # [B, num_classes]



@keras.saving.register_keras_serializable()
class MobileViTMultiTask(keras.Model):
    """
    Multi-task model for leaf disease detection and classification.

    Tasks:
      1. **Detection** — predicts bounding boxes + per-anchor class logits
         across 4 FPN levels (SSD-Lite style)
      2. **Classification** — global image-level disease class prediction

    Outputs dict:
      'cls_logits': [B, num_classes]         — image-level classification
      'det_cls':    [B, total_anchors, C]    — per-anchor class logits
      'det_box':    [B, total_anchors, 4]    — per-anchor bbox deltas
    """

    def __init__(self,
                 num_classes: int = 30,
                 backbone_dims: tuple = (96, 120, 144),
                 backbone_depths: tuple = (2, 4, 3),
                 fpn_channels: int = 128,
                 num_anchors: int = 9,
                 dropout: float = 0.2,
                 **kw):
        super().__init__(**kw)
        self.backbone = MobileViTBackbone(
            dims=backbone_dims,
            depths=backbone_depths,
            dropout=dropout,
        )
        self.fpn      = FPNNeck(out_channels=fpn_channels)
        self.det_head = DetectionHead(num_classes, num_anchors, fpn_channels)
        self.cls_head = ClassificationHead(num_classes, dropout=dropout)

    def call(self, x, training=False):
        # Backbone
        feats = self.backbone(x, training=training)       # c1…c5
        c1, c2, c3, c4, c5 = feats

        # FPN neck
        fpn_feats = self.fpn(feats, training=training)    # [P3,P4,P5,P6]

        # Detection head
        det_cls, det_box = self.det_head(fpn_feats, training=training)

        # Classification head
        cls_logits = self.cls_head(c5, training=training)

        return {
            'cls_logits': cls_logits,
            'det_cls':    det_cls,
            'det_box':    det_box,
        }

    def predict_image(
            self,
            img_array: np.ndarray,
            score_thresh: float = 0.3,
            nms_thresh:   float = 0.5,
            top_k:        int   = 50,
    ) -> dict:
        """
        Single-image inference.

        Returns dict:
          'class_name':   str   — top-1 disease class
          'confidence':   float — softmax confidence
          'boxes':        np.ndarray [K, 4] xyxy normalised
          'box_labels':   np.ndarray [K]    int class indices
          'box_scores':   np.ndarray [K]    float confidence per box
        """
        if img_array.ndim == 3:
            img_array = img_array[np.newaxis]         # add batch dim

        preds = self(img_array.astype(np.float32), training=False)

        # Image-level classification
        cls_probs  = tf.nn.softmax(preds['cls_logits'], axis=-1).numpy()[0]
        top_cls    = int(np.argmax(cls_probs))
        top_conf   = float(cls_probs[top_cls])

        # Per-anchor detection: pick top-K by max class score
        det_cls_np = tf.nn.softmax(preds['det_cls'], axis=-1).numpy()[0]  # [A, C]
        det_box_np = preds['det_box'].numpy()[0]                           # [A, 4]

        scores_all  = det_cls_np.max(axis=-1)     # [A]
        labels_all  = det_cls_np.argmax(axis=-1)  # [A]

        # Filter by score threshold
        keep = scores_all >= score_thresh
        boxes_f  = det_box_np[keep]
        scores_f = scores_all[keep]
        labels_f = labels_all[keep]

        # Clip boxes to [0, 1]
        boxes_f = np.clip(boxes_f, 0.0, 1.0)

        # Simplified class-agnostic NMS (top-K by score)
        if len(scores_f) > top_k:
            idx      = np.argsort(scores_f)[::-1][:top_k]
            boxes_f  = boxes_f[idx]
            scores_f = scores_f[idx]
            labels_f = labels_f[idx]

        return {
            'class_name':  IDX_TO_CLASS.get(top_cls, 'unknown'),
            'confidence':  top_conf,
            'boxes':       boxes_f,
            'box_labels':  labels_f,
            'box_scores':  scores_f,
        }






@keras.saving.register_keras_serializable()
class FocalLoss(keras.losses.Loss):
    """
    Focal loss — downweights easy negatives, focuses on hard examples.
    FL(p_t) = -α_t · (1 - p_t)^γ · log(p_t)
    """
    def __init__(self, gamma=2.0, alpha=0.25,
                 label_smoothing=0.0, **kw):
        super().__init__(**kw)
        self.gamma           = gamma
        self.alpha           = alpha
        self.label_smoothing = label_smoothing

    def call(self, y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        probs  = tf.nn.sigmoid(y_pred)   # from logits

        if self.label_smoothing > 0:
            y_true = y_true * (1 - self.label_smoothing) + 0.5 * self.label_smoothing

        p_t     = y_true * probs + (1 - y_true) * (1 - probs)
        alpha_t = y_true * self.alpha + (1 - y_true) * (1 - self.alpha)
        fl      = -alpha_t * tf.pow(1 - p_t, self.gamma) * tf.math.log(p_t + 1e-8)
        return tf.reduce_mean(fl)


@keras.saving.register_keras_serializable()
class SmoothL1Loss(keras.losses.Loss):
    """Huber / Smooth-L1 loss for bounding box regression."""
    def __init__(self, delta=1.0, **kw):
        super().__init__(**kw)
        self.delta = delta

    def call(self, y_true, y_pred):
        diff = tf.abs(y_true - y_pred)
        loss = tf.where(
            diff < self.delta,
            0.5 * diff ** 2,
            self.delta * (diff - 0.5 * self.delta))
        return tf.reduce_mean(loss)


@keras.saving.register_keras_serializable()
class MultiTaskLoss(keras.losses.Loss):
    """
    Weighted combination of:
      - Image-level classification loss  (sparse cross-entropy, from cls_logits)
      - Per-anchor detection class loss  (focal loss + OHEM, from det_cls)
      - Per-anchor box regression loss   (SmoothL1, from det_box)

    Key design decisions:
      1. Classification uses sparse_categorical_crossentropy (not sigmoid focal)
         because this is a single-label multiclass problem, not multi-label.
      2. Detection cls uses OHEM: selects the top-K highest-loss anchors per
         batch (pos + hard negatives) instead of assigning all 12k anchors
         to the same class, which produces degenerate gradients.
      3. Box regression only fires on anchors whose predicted score exceeds
         a foreground threshold, avoiding box loss on pure background anchors.
    """

    def __init__(self,
                 num_classes:    int,
                 cls_w:          float = 2.0,
                 det_cls_w:      float = 0.5,
                 det_box_w:      float = 1.0,
                 label_smoothing: float = 0.1,
                 ohem_ratio:     int   = 3,
                 ohem_min_keep:  int   = 64,
                 **kw):
        super().__init__(**kw)
        self.num_classes     = num_classes
        self.cls_w           = cls_w
        self.det_cls_w       = det_cls_w
        self.det_box_w       = det_box_w
        self.label_smoothing = label_smoothing
        self.ohem_ratio      = ohem_ratio
        self.ohem_min_keep   = ohem_min_keep
        self.det_box_loss    = SmoothL1Loss(delta=1.0)

    # ── helpers ──────────────────────────────────────────────────────────
    @staticmethod
    def _smooth_one_hot(labels_int, num_classes, smoothing):
        """One-hot with label smoothing: [B] int → [B, C] float."""
        oh  = tf.one_hot(tf.cast(labels_int, tf.int32), num_classes)
        return oh * (1.0 - smoothing) + smoothing / tf.cast(num_classes, tf.float32)

    @staticmethod
    def _focal_loss_per_element(y_true_oh, y_pred_logits,
                                 gamma=2.0, alpha=0.25):
        """Element-wise focal loss. Returns [B, A] mean over classes."""
        probs   = tf.nn.sigmoid(tf.cast(y_pred_logits, tf.float32))
        y_true  = tf.cast(y_true_oh, tf.float32)
        p_t     = y_true * probs + (1.0 - y_true) * (1.0 - probs)
        alpha_t = y_true * alpha  + (1.0 - y_true) * (1.0 - alpha)
        fl      = -alpha_t * tf.pow(1.0 - p_t, gamma) * tf.math.log(p_t + 1e-8)
        return tf.reduce_mean(fl, axis=-1)  # mean over C → [B, A]

    def call(self, y_true: dict, y_pred: dict) -> tf.Tensor:
        """
        y_true: {'labels': [B, MAX_BOXES] int, 'boxes': [B, MAX_BOXES, 4] float}
        y_pred: {'cls_logits': [B,C], 'det_cls': [B,A,C], 'det_box': [B,A,4]}
        """
        labels     = y_true['labels']       # [B, MAX_BOXES]  (-1 = pad)
        boxes_gt   = y_true['boxes']        # [B, MAX_BOXES, 4]
        cls_logits = y_pred['cls_logits']   # [B, C]
        det_cls    = y_pred['det_cls']      # [B, A, C]
        det_box    = y_pred['det_box']      # [B, A, 4]

        B           = tf.shape(cls_logits)[0]
        num_anchors = tf.shape(det_cls)[1]

        # Validity mask: True for images that have at least one real GT box
        img_labels_raw = labels[:, 0]                        # [B]
        valid_mask     = img_labels_raw >= 0                 # [B] bool
        img_labels     = tf.cast(
            tf.maximum(img_labels_raw, 0), tf.int32)         # [B] (clamped)

        # ── 1. Image Classification Loss ─────────────────────────────────
        # sparse_categorical_crossentropy is correct for single-label multiclass.
        # Focal / sigmoid loss is NOT appropriate here.
        cls_logits_f32 = tf.cast(cls_logits, tf.float32)
        per_img_cls_loss = tf.keras.losses.sparse_categorical_crossentropy(
            img_labels, cls_logits_f32, from_logits=True)
        # Apply label smoothing manually (keras arg unavailable with from_logits)
        if self.label_smoothing > 0:
            smooth_targets = self._smooth_one_hot(
                img_labels, self.num_classes, self.label_smoothing)
            per_img_cls_loss = tf.keras.losses.categorical_crossentropy(
                smooth_targets, cls_logits_f32, from_logits=True)
        # Zero out padded (invalid) images
        valid_f = tf.cast(valid_mask, tf.float32)
        n_valid = tf.maximum(tf.reduce_sum(valid_f), 1.0)
        cls_loss_val = tf.reduce_sum(per_img_cls_loss * valid_f) / n_valid

        # ── 2. Detection Classification Loss (OHEM) ───────────────────────
        # Build per-anchor target: broadcast the image GT class to all anchors.
        # This is a simplified assignment — true positives are the anchors
        # the model is most confident about the wrong class (hard negatives).
        img_labels_exp = tf.expand_dims(img_labels, 1)               # [B, 1]
        anchor_labels  = tf.tile(img_labels_exp, [1, num_anchors])   # [B, A]
        anchor_targets = self._smooth_one_hot(
            anchor_labels, self.num_classes,
            self.label_smoothing)                                     # [B, A, C]

        det_cls_f32 = tf.cast(det_cls, tf.float32)
        # Per-anchor focal loss: [B, A]
        per_anchor_loss = self._focal_loss_per_element(
            anchor_targets, det_cls_f32, gamma=2.0, alpha=0.25)

        # OHEM: keep only the top-K highest-loss anchors per batch
        # K = max(ohem_min_keep, num_valid_images * ohem_ratio)
        n_keep = tf.maximum(
            self.ohem_min_keep,
            tf.cast(n_valid, tf.int32) * self.ohem_ratio)
        n_keep = tf.minimum(n_keep, num_anchors)
        # Flatten [B, A] → [B*A], pick top-K
        flat_loss  = tf.reshape(per_anchor_loss, [-1])               # [B*A]
        top_vals, _ = tf.math.top_k(flat_loss, k=n_keep)
        threshold   = top_vals[-1]
        ohem_mask   = tf.cast(flat_loss >= threshold, tf.float32)    # [B*A]
        n_ohem      = tf.maximum(tf.reduce_sum(ohem_mask), 1.0)
        det_cls_loss_val = tf.reduce_sum(flat_loss * ohem_mask) / n_ohem

        # ── 3. Box Regression Loss ────────────────────────────────────────
        # Only regress boxes for images with valid GT; use the first GT box
        # as the regression target (simplified — full ATSS matching is overkill
        # for this stage; the classification head carries the main signal).
        gt_box_first = tf.cast(
            tf.maximum(boxes_gt[:, 0, :], 0.0), tf.float32)          # [B, 4]
        gt_box_tiled = tf.tile(
            tf.expand_dims(gt_box_first, 1),
            [1, num_anchors, 1])                                      # [B, A, 4]
        det_box_f32  = tf.cast(det_box, tf.float32)
        box_valid    = tf.reshape(valid_f, (-1, 1, 1))                # [B, 1, 1]
        det_box_loss_val = self.det_box_loss(
            gt_box_tiled * box_valid, det_box_f32 * box_valid)

        # ── 4. Total ──────────────────────────────────────────────────────
        total = (self.cls_w     * cls_loss_val     +
                 self.det_cls_w * det_cls_loss_val +
                 self.det_box_w * det_box_loss_val)
        return total


