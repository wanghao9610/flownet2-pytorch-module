import torch
import torch.nn as nn
import torch.nn.init as nn_init
import torch.nn.functional as F

from . import FlowNetC, FlowNetS, FlowNetSD, FlowNetFusion
from utils.misc import tofp16, tofp32
from ops import ChannelNorm, Resample2d


class FlowNet2(nn.Module):

    def __init__(self,
                 with_bn=False,
                 fp16=False,
                 input_norm=False,
                 rgb_max=255.,
                 div_flow=20.,
                 grads=None,
                 requires_grad=False):
        super(FlowNet2, self).__init__()
        self.with_bn = with_bn
        self.div_flow = div_flow
        self.input_norm = input_norm
        self.rgb_max = rgb_max
        self.requires_grad = requires_grad
        self.grads = {} if grads is None else grads

        self.flow_warp = Resample2d()

        self.channelnorm = ChannelNorm()

        # First Block (FlowNetC)
        self.flownetc = FlowNetC(with_bn=with_bn, fp16=fp16)
        self.upsample1 = nn.Upsample(scale_factor=4, mode='bilinear')
        self.resample1 = (nn.Sequential(tofp32(), Resample2d(), tofp16())
                          if fp16 else Resample2d())

        # Block (FlowNetS1)
        self.flownets_1 = FlowNetS(with_bn=with_bn)
        self.upsample2 = nn.Upsample(scale_factor=4, mode='bilinear')
        self.resample2 = (nn.Sequential(tofp32(), Resample2d(), tofp16())
                          if fp16 else Resample2d())

        # Block (FlowNetS2)
        self.flownets_2 = FlowNetS(with_bn=with_bn)

        # Block (FlowNetSD)
        self.flownets_d = FlowNetSD(with_bn=with_bn)
        self.upsample3 = nn.Upsample(scale_factor=4, mode='nearest')
        self.upsample4 = nn.Upsample(scale_factor=4, mode='nearest')

        self.resample3 = (nn.Sequential(tofp32(), Resample2d(), tofp16())
                          if fp16 else Resample2d())
        self.resample4 = (nn.Sequential(tofp32(), Resample2d(), tofp16())
                          if fp16 else Resample2d())

        # Block (FLowNetFusion)
        self.flownetfusion = FlowNetFusion(with_bn=with_bn)

    def init_weights(self, pretrained):
        if isinstance(pretrained, str):
            logger = get_root_logger()
            load_checkpoint(self, pretrained, strict=False, logger=logger)
        else:
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    if m.bias is not None:
                        nn_init.uniform_(m.bias)
                    nn_init.xavier_uniform_(m.weight)

                if isinstance(m, nn.ConvTranspose2d):
                    if m.bias is not None:
                        nn_init.uniform_(m.bias)
                    nn_init.xavier_uniform_(m.weight)

        if not self.requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, cur_frame, pre_frame):
        B, C, H, W = cur_frame.size()
        assert cur_frame.size() == pre_frame.size(), f'the shape of two frame is not equal, {cur_frame.size()} : {pre_frame.size()}'
        # ensure input image size is multi folds of 64
        if H % 64 == 0 and W % 64 == 0:
            cur_frame = cur_frame.view(B, C, 1, H, W)
            pre_frame = pre_frame.view(B, C, 1, H, W)
        else:
            _H = H - H % 64
            _W = W - W % 64
            cur_frame = F.interpolate(cur_frame, (_H, _W), mode='bilinear')
            pre_frame = F.interpolate(pre_frame, (_H, _W), mode='bilinear')
            cur_frame = cur_frame.view(B, C, 1, _H, _W)
            pre_frame = pre_frame.view(B, C, 1, _H, _W)
        inputs = torch.cat((cur_frame, pre_frame), dim=2)
        # inputs = torch.cat((cur_frame.view(B, C, 1, H, W), pre_frame.view(B, C, 1, H, W)), dim=2)

        if self.input_norm:
            rgb_mean = inputs.contiguous().view(inputs.size()[:2] + (-1,)).mean(
                dim=-1).view(inputs.size()[:2] + (1, 1, 1,))

            x = (inputs - rgb_mean) / self.rgb_max
        else:
            x = inputs
        x1 = x[:, :, 0, :, :]
        x2 = x[:, :, 1, :, :]
        x = torch.cat((x1, x2), dim=1)

        # flownetc
        flownetc_flow2 = self.flownetc(x)[0]
        flownetc_flow = self.upsample1(flownetc_flow2 * self.div_flow)

        # warp img1 to img0; magnitude of diff between img0 and and warped_img1,
        resampled_img1 = self.resample1(x[:, 3:, :, :], flownetc_flow)
        diff_img0 = x[:, :3, :, :] - resampled_img1
        norm_diff_img0 = self.channelnorm(diff_img0)

        # concat img0, img1, img1->img0, flow, diff-mag ;
        concat1 = torch.cat(
            [x, resampled_img1, flownetc_flow / self.div_flow, norm_diff_img0],
            dim=1)

        # flownets1
        flownets1_flow2 = self.flownets_1(concat1)[0]
        flownets1_flow = self.upsample2(flownets1_flow2 * self.div_flow)

        # warp img1 to img0 using flownets1; magnitude of diff between img0 and and warped_img1
        resampled_img1 = self.resample2(x[:, 3:, :, :], flownets1_flow)
        diff_img0 = x[:, :3, :, :] - resampled_img1
        norm_diff_img0 = self.channelnorm(diff_img0)

        # concat img0, img1, img1->img0, flow, diff-mag
        concat2 = torch.cat(
            (x, resampled_img1, flownets1_flow / self.div_flow,
             norm_diff_img0),
            dim=1)

        # flownets2
        flownets2_flow2 = self.flownets_2(concat2)[0]
        flownets2_flow = self.upsample4(flownets2_flow2 * self.div_flow)
        norm_flownets2_flow = self.channelnorm(flownets2_flow)

        diff_flownets2_flow = self.resample4(x[:, 3:, :, :], flownets2_flow)
        # if not diff_flownets2_flow.volatile:
        #     diff_flownets2_flow.register_hook(
        #         save_grad(self.grads, 'diff_flownets2_flow'))

        diff_flownets2_img1 = self.channelnorm(
            (x[:, :3, :, :] - diff_flownets2_flow))
        # if not diff_flownets2_img1.volatile:
        #     diff_flownets2_img1.register_hook(
        #         save_grad(self.grads, 'diff_flownets2_img1'))

        # flownetsd
        flownetsd_flow2 = self.flownets_d(x)[0]
        flownetsd_flow = self.upsample3(flownetsd_flow2 / self.div_flow)
        norm_flownetsd_flow = self.channelnorm(flownetsd_flow)

        diff_flownetsd_flow = self.resample3(x[:, 3:, :, :], flownetsd_flow)
        # if not diff_flownetsd_flow.volatile:
        #     diff_flownetsd_flow.register_hook(
        #         save_grad(self.grads, 'diff_flownetsd_flow'))

        diff_flownetsd_img1 = self.channelnorm(
            (x[:, :3, :, :] - diff_flownetsd_flow))
        # if not diff_flownetsd_img1.volatile:
        #     diff_flownetsd_img1.register_hook(
        #         save_grad(self.grads, 'diff_flownetsd_img1'))

        # concat img1 flownetsd, flownets2, norm_flownetsd, norm_flownets2,
        # diff_flownetsd_img1, diff_flownets2_img1
        concat3 = torch.cat(
            (x[:, :3, :, :], flownetsd_flow, flownets2_flow,
             norm_flownetsd_flow, norm_flownets2_flow, diff_flownetsd_img1,
             diff_flownets2_img1),
            dim=1)
        flownetfusion_flow = self.flownetfusion(concat3)

        if H % 64 == 0 and W % 64 == 0:
            flownetfusion_flow = flownetfusion_flow
        else:
            flownetfusion_flow = F.interpolate(flownetfusion_flow, (H, W), mode='bilinear',
                                               align_corners=True)

        # if not flownetfusion_flow.volatile:
        #     flownetfusion_flow.register_hook(
        #         save_grad(self.grads, 'flownetfusion_flow'))

        return flownetfusion_flow


class FlowNet2C(FlowNetC):

    def __init__(self, with_bn=False, fp16=False, rgb_max=255., div_flow=20):
        super(FlowNet2C, self).__init__(with_bn, fp16)
        self.rgb_max = rgb_max
        self.div_flow = div_flow

    def forward(self, inputs):
        rgb_mean = inputs.contiguous().view(inputs.size()[:2] + (-1,)).mean(
            dim=-1).view(inputs.size()[:2] + (1, 1, 1,))

        x = (inputs - rgb_mean) / self.rgb_max
        x1 = x[:, :, 0, :, :]
        x2 = x[:, :, 1, :, :]

        flows = super(FlowNet2C, self).forward(x1, x2)

        if self.training:
            return flows
        else:
            return self.upsample1(flows[0] * self.div_flow)


class FlowNet2S(FlowNetS):

    def __init__(self, with_bn=False, rgb_max=255., div_flow=20):
        super(FlowNet2S, self).__init__(input_channels=6, with_bn=with_bn)
        self.rgb_max = rgb_max
        self.div_flow = div_flow

    def forward(self, inputs):
        rgb_mean = inputs.contiguous().view(inputs.size()[:2] + (-1,)).mean(
            dim=-1).view(inputs.size()[:2] + (1, 1, 1,))
        x = (inputs - rgb_mean) / self.rgb_max
        x = torch.cat((x[:, :, 0, :, :], x[:, :, 1, :, :]), dim=1)

        flows = super(FlowNet2S, self).forward(x)

        if self.training:
            return flows
        else:
            return self.upsample1(flows[0] * self.div_flow)


class FlowNet2SD(FlowNetSD):

    def __init__(self, with_bn=False, rgb_max=255., div_flow=20):
        super(FlowNet2SD, self).__init__(with_bn=with_bn)
        self.rgb_max = rgb_max
        self.div_flow = div_flow

    def forward(self, inputs):
        rgb_mean = inputs.contiguous().view(inputs.size()[:2] + (-1,)).mean(
            dim=-1).view(inputs.size()[:2] + (1, 1, 1,))
        x = (inputs - rgb_mean) / self.rgb_max
        x = torch.cat((x[:, :, 0, :, :], x[:, :, 1, :, :]), dim=1)

        flows = super(FlowNet2SD, self).forward(x)

        if self.training:
            return flows
        else:
            return self.upsample1(flows[0] * self.div_flow)


class FlowNet2CS(nn.Module):

    def __init__(self, with_bn=False, fp16=False, rgb_max=255., div_flow=20):
        super(FlowNet2CS, self).__init__()
        self.with_bn = with_bn
        self.fp16 = fp16
        self.rgb_max = rgb_max
        self.div_flow = div_flow

        self.channelnorm = ChannelNorm()

        # First Block (FlowNetC)
        self.flownetc = FlowNetC(with_bn=with_bn, fp16=fp16)
        self.upsample1 = nn.Upsample(scale_factor=4, mode='bilinear')

        self.resample1 = (nn.Sequential(tofp32(), Resample2d(), tofp16())
                          if fp16 else Resample2d())

        # Block (FlowNetS1)
        self.flownets_1 = FlowNetS(with_bn=with_bn)
        self.upsample2 = nn.Upsample(scale_factor=4, mode='bilinear')

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                if m.bias is not None:
                    nn_init.uniform_(m.bias)
                nn_init.xavier_uniform_(m.weight)

            if isinstance(m, nn.ConvTranspose2d):
                if m.bias is not None:
                    nn_init.uniform_(m.bias)
                nn_init.xavier_uniform_(m.weight)

    def forward(self, inputs):
        rgb_mean = inputs.contiguous().view(inputs.size()[:2] + (-1,)).mean(
            dim=-1).view(inputs.size()[:2] + (1, 1, 1,))

        x = (inputs - rgb_mean) / self.rgb_max
        x1 = x[:, :, 0, :, :]
        x2 = x[:, :, 1, :, :]
        x = torch.cat((x1, x2), dim=1)

        # flownetc
        flownetc_flow2 = self.flownetc(x)[0]
        flownetc_flow = self.upsample1(flownetc_flow2 * self.div_flow)

        # warp img1 to img0; magnitude of diff between img0 and and warped_img1,
        resampled_img1 = self.resample1(x[:, 3:, :, :], flownetc_flow)
        diff_img0 = x[:, :3, :, :] - resampled_img1
        norm_diff_img0 = self.channelnorm(diff_img0)

        # concat img0, img1, img1->img0, flow, diff-mag ;
        concat1 = torch.cat(
            [x, resampled_img1, flownetc_flow / self.div_flow, norm_diff_img0],
            dim=1)

        # flownets1
        flownets1_flow2 = self.flownets_1(concat1)[0]
        flownets1_flow = self.upsample2(flownets1_flow2 * self.div_flow)

        return flownets1_flow


class FlowNet2CSS(nn.Module):

    def __init__(self, with_bn=False, fp16=False, rgb_max=255., div_flow=20):
        super(FlowNet2CSS, self).__init__()
        self.with_bn = with_bn
        self.fp16 = fp16
        self.rgb_max = rgb_max
        self.div_flow = div_flow

        self.channelnorm = ChannelNorm()

        # First Block (FlowNetC)
        self.flownetc = FlowNetC(with_bn=with_bn, fp16=fp16)
        self.upsample1 = nn.Upsample(scale_factor=4, mode='bilinear')

        if fp16:
            self.resample1 = nn.Sequential(tofp32(), Resample2d(), tofp16())
        else:
            self.resample1 = Resample2d()

        # Block (FlowNetS1)
        self.flownets_1 = FlowNetS(with_bn=with_bn)
        self.upsample2 = nn.Upsample(scale_factor=4, mode='bilinear')
        if fp16:
            self.resample2 = nn.Sequential(tofp32(), Resample2d(), tofp16())
        else:
            self.resample2 = Resample2d()

        # Block (FlowNetS2)
        self.flownets_2 = FlowNetS(with_bn=with_bn)
        self.upsample3 = nn.Upsample(scale_factor=4, mode='nearest')

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                if m.bias is not None:
                    nn_init.uniform_(m.bias)
                nn_init.xavier_uniform_(m.weight)

            if isinstance(m, nn.ConvTranspose2d):
                if m.bias is not None:
                    nn_init.uniform_(m.bias)
                nn_init.xavier_uniform_(m.weight)

    def forward(self, inputs):
        rgb_mean = inputs.contiguous().view(inputs.size()[:2] + (-1,)).mean(
            dim=-1).view(inputs.size()[:2] + (1, 1, 1,))

        x = (inputs - rgb_mean) / self.rgb_max
        x1 = x[:, :, 0, :, :]
        x2 = x[:, :, 1, :, :]
        x = torch.cat((x1, x2), dim=1)

        # flownetc
        flownetc_flow2 = self.flownetc(x)[0]
        flownetc_flow = self.upsample1(flownetc_flow2 * self.div_flow)

        # warp img1 to img0; magnitude of diff between img0 and and warped_img1,
        resampled_img1 = self.resample1(x[:, 3:, :, :], flownetc_flow)
        diff_img0 = x[:, :3, :, :] - resampled_img1
        norm_diff_img0 = self.channelnorm(diff_img0)

        # concat img0, img1, img1->img0, flow, diff-mag ;
        concat1 = torch.cat(
            [x, resampled_img1, flownetc_flow / self.div_flow, norm_diff_img0],
            dim=1)

        # flownets1
        flownets1_flow2 = self.flownets_1(concat1)[0]
        flownets1_flow = self.upsample2(flownets1_flow2 * self.div_flow)

        # warp img1 to img0 using flownets1; magnitude of diff between img0 and and warped_img1
        resampled_img1 = self.resample2(x[:, 3:, :, :], flownets1_flow)
        diff_img0 = x[:, :3, :, :] - resampled_img1
        norm_diff_img0 = self.channelnorm(diff_img0)

        # concat img0, img1, img1->img0, flow, diff-mag
        concat2 = torch.cat(
            (x, resampled_img1, flownets1_flow / self.div_flow,
             norm_diff_img0),
            dim=1)

        # flownets2
        flownets2_flow2 = self.flownets_2(concat2)[0]
        flownets2_flow = self.upsample3(flownets2_flow2 * self.div_flow)

        return flownets2_flow
