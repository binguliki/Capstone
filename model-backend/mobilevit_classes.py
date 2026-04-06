import tensorflow as tf
import keras
from keras import layers
import numpy as np

# ─── Building blocks ─────────────────────────────────────────────────────────

def conv_bn_act(filters, kernel_size=3, stride=1, padding='same',
                groups=1, activation='swish', use_bias=False, name=None):
    """Standard Conv → BN → Activation block."""
    return keras.Sequential([
        layers.Conv2D(filters, kernel_size, strides=stride, padding=padding,
                      groups=groups, use_bias=use_bias,
                      kernel_initializer='he_normal'),
        layers.BatchNormalization(momentum=0.99, epsilon=1e-3),
        layers.Activation(activation) if activation else layers.Lambda(lambda x: x),
    ], name=name)


@keras.saving.register_keras_serializable()
class InvertedResidualBlock(layers.Layer):
    """
    MobileNetV2 Inverted Residual Block.
    Expand → Depthwise → Project with optional skip connection.
    """
    def __init__(self, in_ch=16, out_ch=32, stride=1, expand_ratio=4, dropout=0.0, **kw):
        super().__init__(**kw)
        self.stride = stride
        self.use_skip = (stride == 1) and (in_ch == out_ch)
        hidden = int(in_ch * expand_ratio)

        seq = []
        if expand_ratio != 1:
            seq.append(conv_bn_act(hidden, 1))  # pointwise expand
        seq += [
            conv_bn_act(hidden, 3, stride=stride, groups=hidden),  # depthwise
            layers.Conv2D(out_ch, 1, use_bias=False, kernel_initializer='he_normal'),
            layers.BatchNormalization(momentum=0.99, epsilon=1e-3),
        ]
        if dropout > 0:
            seq.append(layers.Dropout(dropout))
        self.conv = keras.Sequential(seq)

    def call(self, x, training=False):
        out = self.conv(x, training=training)
        if self.use_skip:
            out = out + x
        return out


@keras.saving.register_keras_serializable()
class MultiHeadSelfAttention(layers.Layer):
    """Standard multi-head self-attention for MobileViT transformer layers."""
    def __init__(self, dim=96, num_heads=1, dropout=0.0, **kw):
        super().__init__(**kw)
        assert dim % num_heads == 0, "dim must be divisible by num_heads"
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.qkv = layers.Dense(3 * dim, use_bias=False)
        self.out_proj = layers.Dense(dim)
        self.attn_drop = layers.Dropout(dropout)
        self.proj_drop = layers.Dropout(dropout)

    def call(self, x, training=False):
        # x: [B, N, C]
        B = tf.shape(x)[0]
        N = tf.shape(x)[1]
        C = self.head_dim * self.num_heads

        qkv = self.qkv(x)                             # [B, N, 3*C]
        qkv = tf.reshape(qkv, (B, N, 3, self.num_heads, self.head_dim))
        qkv = tf.transpose(qkv, (2, 0, 3, 1, 4))     # [3, B, H, N, hd]
        q, k, v = qkv[0], qkv[1], qkv[2]             # each [B, H, N, hd]

        attn = tf.matmul(q, k, transpose_b=True) * self.scale  # [B,H,N,N]
        attn = tf.nn.softmax(attn, axis=-1)
        attn = self.attn_drop(attn, training=training)

        x = tf.matmul(attn, v)                        # [B, H, N, hd]
        x = tf.transpose(x, (0, 2, 1, 3))             # [B, N, H, hd]
        x = tf.reshape(x, (B, N, C))                  # [B, N, C]
        x = self.out_proj(x)
        x = self.proj_drop(x, training=training)
        return x


@keras.saving.register_keras_serializable()
class TransformerLayer(layers.Layer):
    """Single transformer encoder layer with pre-norm, MHA + FFN."""
    def __init__(self, dim=96, num_heads=1, mlp_ratio=2.0, dropout=0.0, **kw):
        super().__init__(**kw)
        self.norm1 = layers.LayerNormalization(epsilon=1e-6)
        self.attn  = MultiHeadSelfAttention(dim, num_heads, dropout)
        self.norm2 = layers.LayerNormalization(epsilon=1e-6)
        mlp_dim = int(dim * mlp_ratio)
        self.ffn = keras.Sequential([
            layers.Dense(mlp_dim, activation='swish'),
            layers.Dropout(dropout),
            layers.Dense(dim),
            layers.Dropout(dropout),
        ])

    def call(self, x, training=False):
        x = x + self.attn(self.norm1(x), training=training)
        x = x + self.ffn(self.norm2(x), training=training)
        return x


@keras.saving.register_keras_serializable()
class MobileViTBlock(layers.Layer):
    """
    Core MobileViT block: learns both local and global representations.

    Processing steps (as described in paper §3.1):
    1. nxn Conv → local spatial features
    2. 1×1 Conv → project to transformer dim d
    3. Unfold spatial dims into non-overlapping patches [B, P, N, d]
       where P = h*w (pixels per patch), N = (H/h)*(W/w) (num patches)
    4. Transformer layers over N patches for each of the P positions
       → each pixel attends to corresponding positions across ALL patches
    5. Fold back to spatial layout
    6. 1×1 Conv → project back to C channels
    7. Concatenate with original input
    8. nxn Conv fusion layer
    """
    def __init__(self, channels=96, dim=96, depth=2, patch_h=2, patch_w=2,
                 num_heads=1, mlp_ratio=2.0, dropout=0.0, **kw):
        super().__init__(**kw)
        self.ph = patch_h
        self.pw = patch_w
        self.dim = dim

        # Local representation
        self.local_rep = keras.Sequential([
            conv_bn_act(channels, 3),                         # nxn conv
            layers.Conv2D(dim, 1, use_bias=False),            # project to dim
        ])
        # Global representation (transformer stack)
        self.global_rep = [
            TransformerLayer(dim, num_heads, mlp_ratio, dropout) for _ in range(depth)
        ]
        self.norm = layers.LayerNormalization(epsilon=1e-6)

        # Projection back to channels
        self.proj_back = layers.Conv2D(channels, 1, use_bias=False)

        # Fusion: concat(local, global) → channels
        self.fusion = conv_bn_act(channels, 3)

    def call(self, x, training=False):
        B = tf.shape(x)[0]
        H = tf.shape(x)[1]
        W = tf.shape(x)[2]
        ph, pw = self.ph, self.pw

        pad_h = (ph - H % ph) % ph
        pad_w = (pw - W % pw) % pw
        x = tf.pad(x, [[0, 0], [0, pad_h], [0, pad_w], [0, 0]])
        H = H + pad_h
        W = W + pad_w

        # ── Step 1-2: local representation ────────────────────────────────
        y = self.local_rep(x, training=training)   # [B, H, W, dim]

        # ── Step 3: unfold into patches ───────────────────────────────────
        # Number of patches along each axis
        nph = H // ph   # e.g. 16//2 = 8
        npw = W // pw
        N = nph * npw   # total patches
        P = ph * pw     # pixels per patch

        # [B, H, W, dim] → [B, nph, ph, npw, pw, dim]
        y = tf.reshape(y, (B, nph, ph, npw, pw, self.dim))
        # → [B, ph, pw, nph, npw, dim] → [B, P, N, dim]
        y = tf.transpose(y, (0, 2, 4, 1, 3, 5))
        y = tf.reshape(y, (B * P, N, self.dim))

        # ── Step 4: transformer (each of P positions over N patches) ──────
        for transformer in self.global_rep:
            y = transformer(y, training=training)
        y = self.norm(y)                             # [B*P, N, dim]

        # ── Step 5: fold back to spatial ──────────────────────────────────
        y = tf.reshape(y, (B, ph, pw, nph, npw, self.dim))
        y = tf.transpose(y, (0, 3, 1, 4, 2, 5))    # [B, nph, ph, npw, pw, dim]
        y = tf.reshape(y, (B, H, W, self.dim))

        # ── Step 6: project back ──────────────────────────────────────────
        y = self.proj_back(y)                        # [B, H, W, channels]

        # ── Step 7-8: fusion ──────────────────────────────────────────────
        out = self.fusion(tf.concat([x, y], axis=-1), training=training)
        return out

@keras.saving.register_keras_serializable()
class MobileViTBackbone(keras.Model):
    """
    MobileViT-S backbone for feature extraction.
    Produces multi-scale feature maps at 4 resolutions for downstream heads.

    For input 256×256:
      c1: 64×64, 32 channels   (after stage1 + stage2)
      c2: 32×32, 64 channels   (after stage2 stride-2)
      c3: 16×16, 96 channels   (MobileViT stage3)
      c4:  8×8, 128 channels   (MobileViT stage4)
      c5:  4×4, 640 channels   (MobileViT stage5 + final conv)
    """
    def __init__(self, dims=(96, 120, 144), depths=(2, 4, 3),
                 patch_h=2, patch_w=2, dropout=0.1, **kw):
        super().__init__(**kw)

        # ── Stem: 256×256 → 128×128, 16ch ────────────────────────────────
        self.stem = conv_bn_act(16, 3, stride=2)

        # ── Stage 1: 128×128, 16 → 32ch ──────────────────────────────────
        self.stage1 = InvertedResidualBlock(16, 32, stride=1, expand_ratio=4)

        # ── Stage 2: 128→64, 32 → 64ch ───────────────────────────────────
        self.stage2 = keras.Sequential([
            InvertedResidualBlock(32, 64, stride=2, expand_ratio=4),
            InvertedResidualBlock(64, 64, stride=1, expand_ratio=4),
            InvertedResidualBlock(64, 64, stride=1, expand_ratio=4),
        ])

        # ── Stage 3: 64→32, 64 → 96ch + MobileViT ────────────────────────
        self.stage3_mv2  = InvertedResidualBlock(64, 96, stride=2, expand_ratio=4)
        self.stage3_mvit = MobileViTBlock(96, dims[0], depths[0],
                                           patch_h, patch_w, num_heads=1,
                                           dropout=dropout)

        # ── Stage 4: 32→16, 96 → 128ch + MobileViT ───────────────────────
        self.stage4_mv2  = InvertedResidualBlock(96, 128, stride=2, expand_ratio=4)
        self.stage4_mvit = MobileViTBlock(128, dims[1], depths[1],
                                           patch_h, patch_w, num_heads=2,
                                           dropout=dropout)

        # ── Stage 5: 16→8, 128 → 160ch + MobileViT ───────────────────────
        self.stage5_mv2  = InvertedResidualBlock(128, 160, stride=2, expand_ratio=4)
        self.stage5_mvit = MobileViTBlock(160, dims[2], depths[2],
                                           patch_h, patch_w, num_heads=3,
                                           dropout=dropout)

        # ── Final pointwise: 160 → 640ch ──────────────────────────────────
        self.final_conv = conv_bn_act(640, 1)

    def call(self, x, training=False):
        x  = self.stem(x, training=training)            # 128×128, 16
        x  = self.stage1(x, training=training)          # 128×128, 32
        c1 = x

        x  = self.stage2(x, training=training)          # 64×64, 64
        c2 = x

        x  = self.stage3_mv2(x, training=training)      # 32×32, 96
        x  = self.stage3_mvit(x, training=training)     # 32×32, 96
        c3 = x

        x  = self.stage4_mv2(x, training=training)      # 16×16, 128
        x  = self.stage4_mvit(x, training=training)     # 16×16, 128
        c4 = x

        x  = self.stage5_mv2(x, training=training)      # 8×8, 160
        x  = self.stage5_mvit(x, training=training)     # 8×8, 160
        c5 = self.final_conv(x, training=training)      # 8×8, 640

        return c1, c2, c3, c4, c5   # multi-scale features


@keras.saving.register_keras_serializable()
class FPNNeck(layers.Layer):
    """
    Feature Pyramid Network neck.
    Fuses multi-scale features from backbone into 4 pyramid levels.
    """
    def __init__(self, out_channels=128, **kw):
        super().__init__(**kw)
        # Lateral convs (1×1 to reduce channels)
        self.lat3 = layers.Conv2D(out_channels, 1, use_bias=False)
        self.lat4 = layers.Conv2D(out_channels, 1, use_bias=False)
        self.lat5 = layers.Conv2D(out_channels, 1, use_bias=False)

        # Output convs (3×3 to smooth)
        self.out3 = conv_bn_act(out_channels, 3)
        self.out4 = conv_bn_act(out_channels, 3)
        self.out5 = conv_bn_act(out_channels, 3)

        # Extra level (downsample P5 → P6)
        self.p6 = conv_bn_act(out_channels, 3, stride=2)

    def _upsample_add(self, x_high, x_low):
        h = tf.shape(x_low)[1]
        w = tf.shape(x_low)[2]
        # Cast to float32 before resize (tf.image.resize always returns float32),
        # then cast result back to match the compute dtype (handles mixed_float16).
        compute_dtype = x_low.dtype
        x_high_f32 = tf.cast(x_high, tf.float32)
        x_low_f32  = tf.cast(x_low,  tf.float32)
        x_up = tf.image.resize(x_high_f32, (h, w), method='bilinear')
        return tf.cast(x_up + x_low_f32, compute_dtype)

    def call(self, features, training=False):
        _, _, c3, c4, c5 = features

        p5 = self.lat5(c5, training=training)
        p4 = self._upsample_add(p5, self.lat4(c4, training=training))
        p3 = self._upsample_add(p4, self.lat3(c3, training=training))

        p3 = self.out3(p3, training=training)
        p4 = self.out4(p4, training=training)
        p5 = self.out5(p5, training=training)
        p6 = self.p6(p5, training=training)

        return [p3, p4, p5, p6]  # resolutions: 32×32, 16×16, 8×8, 4×4


@keras.saving.register_keras_serializable()
class DetectionHead(layers.Layer):
    """
    SSD-Lite style detection head operating on FPN feature maps.
    Predicts: class logits + bounding box deltas per anchor.
    """
    def __init__(self, num_classes=30, num_anchors=9, fpn_channels=128, **kw):
        super().__init__(**kw)
        self.num_classes  = num_classes
        self.num_anchors  = num_anchors

        # Shared depthwise-separable conv towers (4 levels each)
        self.cls_convs = [
            keras.Sequential([
                conv_bn_act(fpn_channels, 3, groups=fpn_channels),
                layers.Conv2D(fpn_channels, 1),
                layers.Activation('swish'),
            ]) for _ in range(4)
        ]
        self.reg_convs = [
            keras.Sequential([
                conv_bn_act(fpn_channels, 3, groups=fpn_channels),
                layers.Conv2D(fpn_channels, 1),
                layers.Activation('swish'),
            ]) for _ in range(4)
        ]

        # Prediction layers
        self.cls_preds = [
            layers.Conv2D(num_anchors * num_classes, 1) for _ in range(4)
        ]
        self.reg_preds = [
            layers.Conv2D(num_anchors * 4, 1) for _ in range(4)
        ]

    def call(self, fpn_feats, training=False):
        all_cls, all_reg = [], []
        for i, feat in enumerate(fpn_feats):
            cls_feat = self.cls_convs[i](feat, training=training)
            reg_feat = self.reg_convs[i](feat, training=training)

            cls_out = self.cls_preds[i](cls_feat)   # [B, H, W, A*C]
            reg_out = self.reg_preds[i](reg_feat)   # [B, H, W, A*4]

            B  = tf.shape(cls_out)[0]
            H  = tf.shape(cls_out)[1]
            W  = tf.shape(cls_out)[2]

            # Reshape to [B, H*W*A, C] and [B, H*W*A, 4]
            cls_out = tf.reshape(cls_out, (B, H * W * self.num_anchors, self.num_classes))
            reg_out = tf.reshape(reg_out, (B, H * W * self.num_anchors, 4))

            all_cls.append(cls_out)
            all_reg.append(reg_out)

        cls_logits = tf.concat(all_cls, axis=1)  # [B, total_anchors, C]
        box_preds   = tf.concat(all_reg, axis=1)  # [B, total_anchors, 4]
        return cls_logits, box_preds

@keras.saving.register_keras_serializable()
class SegmentationHead(layers.Layer):
    """
    Lightweight segmentation decoder.
    Uses skip connections from backbone to progressively upsample
    to input resolution, producing per-pixel class predictions.

    Channel flow (decoder_channels=128, D=128):
      c5 (640)  -> reduce_c5  -> D   (128)
      upsample  -> proj_to_c4 -> D2  ( 64) + reduce_c4(c4=128->D2) -> refine4 -> D2
      upsample  -> proj_to_c3 -> D2  ( 64) + reduce_c3(c3= 96->D2) -> refine3 -> D2
      upsample  -> proj_to_c2 -> D4  ( 32) + reduce_c2(c2= 64->D4) -> refine2 -> D4
      upsample  -> refine1    -> D8  ( 16)
      aspp      -> seg_pred   -> num_classes
    """
    def __init__(self, num_classes=30, decoder_channels=128, **kw):
        super().__init__(**kw)
        self.num_classes = num_classes
        D  = decoder_channels        # 128
        D2 = decoder_channels // 2   # 64
        D4 = decoder_channels // 4   # 32
        D8 = decoder_channels // 8   # 16

        # 1x1 convs to reduce backbone skip-connection channels to decoder width
        self.reduce_c5 = conv_bn_act(D,  1)   # 640 -> 128
        self.reduce_c4 = conv_bn_act(D2, 1)   # 128 ->  64
        self.reduce_c3 = conv_bn_act(D2, 1)   #  96 ->  64
        self.reduce_c2 = conv_bn_act(D4, 1)   #  64 ->  32

        # 1x1 projection convs applied to upsampled x BEFORE the skip addition,
        # ensuring x and the skip connection have the same channel count.
        self.proj_to_c4 = conv_bn_act(D2, 1)  # D(128) ->  D2(64)
        self.proj_to_c3 = conv_bn_act(D2, 1)  # D2(64) ->  D2(64)  (channel-stable)
        self.proj_to_c2 = conv_bn_act(D4, 1)  # D2(64) ->  D4(32)

        # 3x3 refinement convs after each skip-add
        self.refine4 = conv_bn_act(D2, 3)
        self.refine3 = conv_bn_act(D2, 3)
        self.refine2 = conv_bn_act(D4, 3)
        self.refine1 = conv_bn_act(D8, 3)

        # ASPP-lite: multi-scale context for final features
        self.aspp = keras.Sequential([
            conv_bn_act(D4, 3),
            layers.Dropout(0.1),
        ])

        # Final prediction — dtype=float32 for numerical stability with mixed precision
        self.seg_pred = layers.Conv2D(num_classes, 1,
                                       dtype='float32',
                                       kernel_initializer='glorot_uniform')

    def _upsample(self, x, ref):
        """Upsample x to the spatial size of ref, preserving compute dtype."""
        compute_dtype = x.dtype
        h = tf.shape(ref)[1]
        w = tf.shape(ref)[2]
        return tf.cast(
            tf.image.resize(tf.cast(x, tf.float32), (h, w), method='bilinear'),
            compute_dtype)

    def call(self, backbone_feats, input_shape, training=False):
        c1, c2, c3, c4, c5 = backbone_feats
        H_in = input_shape[1]
        W_in = input_shape[2]

        # ── Level 5 -> 4 ─────────────────────────────────────────────────
        # c5: [B,  8,  8, 640]
        x = self.reduce_c5(c5, training=training)       # [B,  8,  8, 128]
        x = self._upsample(x, c4)                       # [B, 16, 16, 128]
        x = self.proj_to_c4(x, training=training)       # [B, 16, 16,  64]
        x = x + self.reduce_c4(c4, training=training)   # [B, 16, 16,  64] + [B,16,16,64]
        x = self.refine4(x, training=training)           # [B, 16, 16,  64]

        # ── Level 4 -> 3 ─────────────────────────────────────────────────
        x = self._upsample(x, c3)                       # [B, 32, 32,  64]
        x = self.proj_to_c3(x, training=training)       # [B, 32, 32,  64]
        x = x + self.reduce_c3(c3, training=training)   # [B, 32, 32,  64] + [B,32,32,64]
        x = self.refine3(x, training=training)           # [B, 32, 32,  64]

        # ── Level 3 -> 2 ─────────────────────────────────────────────────
        x = self._upsample(x, c2)                       # [B, 64, 64,  64]
        x = self.proj_to_c2(x, training=training)       # [B, 64, 64,  32]
        x = x + self.reduce_c2(c2, training=training)   # [B, 64, 64,  32] + [B,64,64,32]
        x = self.refine2(x, training=training)           # [B, 64, 64,  32]

        # ── Upsample to full input resolution ────────────────────────────
        x = self._upsample_to(x, H_in, W_in)            # [B, H_in, W_in, 32]
        x = self.refine1(x, training=training)           # [B, H_in, W_in, 16]

        x = self.aspp(x, training=training)              # [B, H_in, W_in, 32]
        seg_logits = self.seg_pred(x)                    # [B, H_in, W_in, num_classes] float32
        return seg_logits

    def _upsample_to(self, x, h, w):
        """Upsample x to explicit (h, w), preserving compute dtype."""
        compute_dtype = x.dtype
        return tf.cast(
            tf.image.resize(tf.cast(x, tf.float32), (h, w), method='bilinear'),
            compute_dtype)


@keras.saving.register_keras_serializable()
class ClassificationHead(layers.Layer):
    """
    Global classification head for disease type identification.
    Takes the deepest backbone features (c5) and produces
    class logits + confidence-calibrated scores.
    """
    def __init__(self, num_classes=30, hidden_dim=512, dropout=0.1, **kw):
        super().__init__(**kw)
        self.gap = layers.GlobalAveragePooling2D()
        self.cls_head = keras.Sequential([
            layers.Dense(hidden_dim, activation='swish',
                         kernel_initializer='he_normal'),
            layers.BatchNormalization(),
            layers.Dropout(dropout),
            layers.Dense(num_classes, dtype='float32'),  # ensure float32 for loss
        ])

    def call(self, c5, training=False):
        x = self.gap(c5)                    # [B, 640]
        logits = self.cls_head(x, training=training)  # [B, num_classes]
        return logits

@keras.saving.register_keras_serializable()
class MobileViTMultiTask(keras.Model):
    """
    Full multi-task model combining:
      - MobileViT backbone
      - FPN neck
      - Detection head (bbox + class per anchor)
      - Segmentation head (pixel-wise class mask)
      - Classification head (global image-level label)

    Outputs dict:
      'cls_logits': [B, num_classes]        — image-level classification
      'det_cls':    [B, total_anchors, C]   — per-anchor class logits
      'det_box':    [B, total_anchors, 4]   — per-anchor bbox deltas
      'seg_logits': [B, H, W, num_classes]  — per-pixel class logits
    """
    def __init__(self, num_classes=30,
                 backbone_dims=(96, 120, 144),
                 backbone_depths=(2, 4, 3),
                 fpn_channels=128,
                 num_anchors=9,
                 dropout=0.1,
                 **kw):
        super().__init__(**kw)
        self.backbone = MobileViTBackbone(
            dims=backbone_dims, depths=backbone_depths, dropout=dropout)
        self.fpn       = FPNNeck(out_channels=fpn_channels)
        self.det_head  = DetectionHead(num_classes, num_anchors, fpn_channels)
        self.seg_head  = SegmentationHead(num_classes, fpn_channels * 2)
        self.cls_head  = ClassificationHead(num_classes, dropout=dropout)

    def call(self, x, training=False):
        input_shape = tf.shape(x)

        # Backbone → multi-scale features
        feats = self.backbone(x, training=training)
        c1, c2, c3, c4, c5 = feats

        # FPN neck
        fpn_feats = self.fpn(feats, training=training)

        # Detection
        det_cls, det_box = self.det_head(fpn_feats, training=training)

        # Segmentation
        seg_logits = self.seg_head(feats, input_shape, training=training)

        # Classification
        cls_logits = self.cls_head(c5, training=training)

        return {
            'cls_logits': cls_logits,
            'det_cls':    det_cls,
            'det_box':    det_box,
            'seg_logits': seg_logits,
        }

    def predict_image(self, img_array, conf_thresh=0.5):
        """
        Convenience method for single-image inference.
        Returns (class_name, confidence, boxes_xyxy, seg_mask_argmax).
        """
        if img_array.ndim == 3:
            img_array = img_array[np.newaxis]
        preds = self(img_array.astype(np.float32), training=False)

        cls_probs = tf.nn.softmax(preds['cls_logits'], axis=-1).numpy()[0]
        top_cls   = np.argmax(cls_probs)
        top_conf  = cls_probs[top_cls]

        seg_mask = tf.argmax(preds['seg_logits'], axis=-1).numpy()[0]
        return IDX_TO_CLASS[top_cls], float(top_conf), seg_mask


