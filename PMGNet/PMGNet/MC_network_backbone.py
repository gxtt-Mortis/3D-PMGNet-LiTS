import torch
import torch.nn as nn
import torch.nn.functional as F
from monai.networks.blocks.unetr_block import UnetrBasicBlock
from monai.networks.blocks.dynunet_block import UnetOutBlock
from PMGNet.PosFuse import ProbPromptFusion
from PMGNet.mc_refine import RefineSegmentationMultiChannel as Re
from PMGNet.pmg_encoder import uxnet_conv
from monai.networks.blocks.dynunet_block import UnetBasicBlock, UnetResBlock, get_conv_layer

class Fuseblock(nn.Module):
    def __init__(self,
                 spatial_dims,
                 in_channels,
                 out_channels,
                 kernel_size,
                 upsample_kernel_size,
                 norm_name,
                 res_block: bool = False):
        super().__init__()
        upsample_stride = upsample_kernel_size
        # 转置卷积上采样 hidden
        self.transp_conv = get_conv_layer(
            spatial_dims,
            in_channels,
            out_channels,
            kernel_size=upsample_kernel_size,
            stride=upsample_stride,
            conv_only=True,
            is_transposed=True,
        )
        # 用 ProbPromptFusion 做融合
        self.fusion = ProbPromptFusion()

    def forward(self, inp, skip):
        # inp: [B, in_channels, D1, H1, W1]
        # skip: [B, skip_channels, D2, H2, W2]
        # 1) 上采样 hidden
        out = self.transp_conv(inp)  # [B, out_channels, D1*2, H1*2, W1*2]
        # 2) 对齐 skip 的空间尺寸到 out
        if skip.shape[2:] != out.shape[2:]:
            skip = F.interpolate(
                skip,
                size=out.shape[2:],       # (D_out, H_out, W_out)
                mode="trilinear",
                align_corners=False,
            )
        # 3) 融合并返回
        return self.fusion(out, skip)
class PMGNet(nn.Module):
    def __init__(
        self,
        in_chans=1,
        out_chans=13,
        depths=[2, 2, 2, 2],
        feat_size=[48, 96, 192, 384],
        hidden_size: int = 768,
        norm_name: str = "instance",
        res_block: bool = True,
        spatial_dims=3,
        mc_samples: int = 5,       # MC采样次数
        use_mc_refine: bool = True # 是否启用MC+refine
    ) -> None:
        super().__init__()
        self.in_chans = in_chans
        self.out_chans = out_chans
        self.feat_size = feat_size
        self.hidden_size = hidden_size
        self.spatial_dims = spatial_dims
        self.mc_samples = mc_samples
        self.use_mc_refine = use_mc_refine

        # Backbone (uxnet_conv)

        self.uxnet_3d = uxnet_conv(
            in_chans=in_chans,
            depths=depths,
            dims=feat_size,
            drop_path_rate=0.0,
            layer_scale_init_value=1e-6,
            out_indices=list(range(len(feat_size)))
        )

        # Encoders
        self.encoder1 = UnetrBasicBlock(spatial_dims, in_chans, feat_size[0], 3, 1, norm_name, res_block)
        self.encoder2 = UnetrBasicBlock(spatial_dims, feat_size[0], feat_size[1], 3, 1, norm_name, res_block)
        self.encoder3 = UnetrBasicBlock(spatial_dims, feat_size[1], feat_size[2], 3, 1, norm_name, res_block)
        self.encoder4 = UnetrBasicBlock(spatial_dims, feat_size[2], feat_size[3], 3, 1, norm_name, res_block)
        self.encoder5 = UnetrBasicBlock(spatial_dims, feat_size[3], hidden_size, 3, 1, norm_name, res_block)

        # Decoder

        self.decoder5 = Fuseblock(spatial_dims, hidden_size, feat_size[3], 3, 2, norm_name, res_block)
        self.decoder4 = Fuseblock(spatial_dims, feat_size[3], feat_size[2], 3, 2, norm_name, res_block)
        self.decoder3 = Fuseblock(spatial_dims, feat_size[2], feat_size[1], 3, 2, norm_name, res_block)
        self.decoder2 = Fuseblock(spatial_dims, feat_size[1], feat_size[0], 3, 2, norm_name, res_block)
        self.decoder1 = UnetrBasicBlock(spatial_dims, feat_size[0], feat_size[0], 3, 1, norm_name, res_block)

        self.out = UnetOutBlock(spatial_dims, feat_size[0], out_chans)

        # Refine 模块（只有 use_mc_refine=True 时才用）
        if self.use_mc_refine:
            self.refine1 = Re(num_channels=feat_size[0])
            self.refine2 = Re(num_channels=feat_size[1])
            self.refine3 = Re(num_channels=feat_size[2])
            self.refine4 = Re(num_channels=feat_size[3])

        self.fusion = ProbPromptFusion()

    def mc_refine_prob(self, enc_feat, refine_module):
        """
        对 enc_feat 做 MC 采样 -> softmax -> 平均 -> refine
        enc_feat: (B,C,D,H,W)
        refine_module: RefineSegmentationMultiChannel
        """
        probs = []
        for _ in range(self.mc_samples):
            p = F.softmax(enc_feat, dim=1)  # (B,C,D,H,W)
            probs.append(p)

        prob_mean = torch.stack(probs, dim=0).mean(0)  # (B,C,D,H,W)

        # 确保 refine 的输入是 (1,C,H,W,D)
        if prob_mean.dim() == 4:
            prob_mean = prob_mean.unsqueeze(0)  # (1,C,H,W,D)

        refined_prob = refine_module(prob_mean)  # (1,C,H,W,D)
        return refined_prob  # 保持5D，不 squeeze

    def forward(self, x_in):
        # Backbone features
        outs = self.uxnet_3d(x_in)

        # Encoder + skip prob
        enc1 = self.encoder1(x_in)
        if self.use_mc_refine:
            prob1 = self.mc_refine_prob(enc1, self.refine1)
        else:
            prob1 = F.softmax(enc1, dim=1)
        re_enc1 = ProbPromptFusion()(enc1, prob1)

        enc2 = self.encoder2(outs[0])
        # prob2 = F.softmax(enc2, dim=1)
        # # re_enc2 = ProbPromptFusion()(enc2, F.softmax(enc2, dim=1))
        if self.use_mc_refine:
            prob2 = self.mc_refine_prob(enc2, self.refine2)
        else:
            prob2 = F.softmax(enc2, dim=1)
        re_enc2 = ProbPromptFusion()(enc2, prob2)

        enc3 = self.encoder3(outs[1])
        if self.use_mc_refine:
            prob3 = self.mc_refine_prob(enc3, self.refine3)
        else:
            prob3 = F.softmax(enc3, dim=1)
        re_enc3 = ProbPromptFusion()(enc3, prob3)

        enc4 = self.encoder4(outs[2])
        if self.use_mc_refine:
            prob4 = self.mc_refine_prob(enc4, self.refine4)
        else:
            prob4 = F.softmax(enc4, dim=1)
        re_enc4 = ProbPromptFusion()(enc4, prob4)

        enc_hidden = self.encoder5(outs[3])

        # Decoder
        dec3 = self.decoder5(enc_hidden, re_enc4)
        dec2 = self.decoder4(dec3, re_enc3)
        dec1 = self.decoder3(dec2, re_enc2)
        dec0 = self.decoder2(dec1, re_enc1)
        out_feat = self.decoder1(dec0)

        return self.out(out_feat)
if __name__ == "__main__":
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device("cpu")
    model = PMGNet(
        in_chans=4,
        out_chans=4,
        depths=[2, 2, 2, 2],
        feat_size=[48, 96, 192, 384],
        hidden_size=768,
        spatial_dims=3,
        mc_samples=5, use_mc_refine=False
    ).to(device)
    # model.eval()
    #
    # # 启用 MC+Refine
    # model_mc = UXNET(in_chans=4, out_chans=4, mc_samples=5, use_mc_refine=True).to(device)
    #
    # # 不启用 MC+Refine，只用 softmax
    # model_plain = UXNET(in_chans=4, out_chans=4, use_mc_refine=False).to(device)

    x = torch.randn(1, 4, 64, 64, 64, device=device)

    with torch.no_grad():
        out_mc = model(x)
        # out_plain = model_plain(x)

    print("MC+Refine 输出:", out_mc.shape)
    # print("Plain softmax 输出:", out_plain.shape)