import argparse
import os
import sys
import cv2
import numpy as np

import mmcv
import torch
import torch.nn as nn

this_dir = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(this_dir, '..'))

from ops.resample2d import Resample2d
from components.flownet2 import FlowNet2

MEAN = (0.485 * 255, 0.456 * 255, 0.406 * 255)
STD = (0.229 * 255, 0.224 * 255, 0.225 * 255)


def tensor2img(tensor):
    """
    Args:
        tensor(torch.Tensor): BxCxHxW
    Returns:
        array(np.array): HxWxC

    """
    img = tensor[0].detach().to("cpu").numpy()
    img = np.transpose(img, (1, 2, 0))
    img = np.ascontiguousarray(img, dtype=np.float)

    return img


class FlowModel(nn.Module):
    def __init__(self, args=None, pretrained=None, save_flow=False):
        super(FlowModel, self).__init__()
        self.args = args
        self.save_flow = save_flow
        self.flownet = FlowNet2(args, requires_grad=False)
        if pretrained is not None and type(pretrained) is str:
            ckpt = torch.load(pretrained)
            self.flownet.load_state_dict(ckpt['state_dict'])
        self.flow_warp = Resample2d()
        self.criterin_flow = nn.MSELoss(size_average=True)

    def forward(self, cur_frame, pre_frame):
        with torch.no_grad():
            flow = self.flownet(cur_frame, pre_frame)
            if self.save_flow:
                mmcv.flowwrite(tensor2img(flow), 'compressed.jpg', quantize=True, concat_axis=1)
                # mmcv.flowshow(tensor2img(flow))

        warp_cur = self.flow_warp(pre_frame, flow)
        warp_cur_img = tensor2img(warp_cur) * STD + MEAN
        # pre_frame = cv2.imread('./data/camvid/images_sequence/0006R0_f02213.png')
        # warp_cur = mmcv.flow_warp(pre_frame, tensor2img(flow))

        mmcv.imwrite(warp_cur_img, 'warped_img.jpg')
        loss = self.criterin_flow(cur_frame, warp_cur)
        return loss


def main():
    parser = argparse.ArgumentParser(description='PyTorch Semantic Segmentation')
    parser.add_argument('--st_weight', type=float, default=0.4, help='st_weight')
    os.environ["CUDA_VISIBLE_DEVICES"] = '2'
    cur_frame = cv2.imread(
        '/data/datasets/video_ss/camvid/images_sequence/0006R0_f02214.png')  # here need change to fit your local path
    pre_frame = cv2.imread('/data/datasets/video_ss/camvid/images_sequence/0006R0_f02213.png')
    input1 = torch.from_numpy(np.array((cur_frame - MEAN) / STD).transpose((2, 0, 1))).float().unsqueeze(0).cuda()
    input2 = torch.from_numpy(np.array((pre_frame - MEAN) / STD).transpose((2, 0, 1))).float().unsqueeze(0).cuda()
    flownet = FlowModel(args=None, pretrained='../init_models/FlowNet2_checkpoint.pth.tar', save_flow=True)
    flownet.cuda()
    loss = flownet(input1, input2)
    print(loss.cpu().numpy())


# if __name__ == '__main__':
main()
