import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
from kornia.geometry import warp_perspective
from src.models.resnet import resnet18
from src.models.shufflenetv2 import shufflenet_v2_x0_5
from src.models.multiview_base import MultiviewBase, cover_mean, cover_mean_std, aggregate_feat
from src.utils.image_utils import img_color_denormalize, array2heatmap
from src.utils.tensor_utils import to_tensor
from src.utils.projection import get_worldcoord_from_imgcoord_mat, project_2d_points
import matplotlib.pyplot as plt


def fill_fc_weights(layers):
    for m in layers.modules():
        if isinstance(m, nn.Conv2d):
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)


def output_head(in_dim, feat_dim, out_dim):
    if feat_dim:
        fc = nn.Sequential(nn.Conv2d(in_dim, feat_dim, 3, 1, 1), nn.ReLU(),
                           nn.Conv2d(feat_dim, out_dim, 1))
    else:
        fc = nn.Sequential(nn.Conv2d(in_dim, out_dim, 1))
    return fc


class MVDet(MultiviewBase):
    def __init__(self, dataset, arch='resnet18', aggregation='max',
                 use_bottleneck=True, hidden_dim=128, outfeat_dim=0, dropout=0.0, check_visible=False):
        super().__init__(dataset, aggregation)
        self.Rimg_shape, self.Rworld_shape = dataset.Rimg_shape, dataset.Rworld_shape
        self.Rworldgrid_from_worldcoord = dataset.Rworldgrid_from_worldcoord
        self.img_reduce = dataset.img_reduce

        if arch == 'resnet18':
            self.base = nn.Sequential(*list(resnet18(pretrained=True,
                                                     replace_stride_with_dilation=[False, True, True]).children())[:-2])
            self.base_dim = 512
        elif arch == 'shufflenet0.5':
            self.base = nn.Sequential(*list(shufflenet_v2_x0_5(pretrained=True,
                                                               replace_stride_with_dilation=[False, True, True]
                                                               ).children())[:-2])
            self.base_dim = 192
        else:
            raise Exception('architecture currently support [vgg11, resnet18]')

        if use_bottleneck:
            # https://discuss.pytorch.org/t/what-is-the-difference-between-nn-dropout2d-and-nn-dropout/108192/2
            self.bottleneck = nn.Sequential(nn.Conv2d(self.base_dim, hidden_dim, 1), nn.ReLU(), nn.Dropout2d(dropout))
            self.base_dim = hidden_dim
        else:
            self.bottleneck = nn.Identity()

        # img heads
        self.img_heatmap = output_head(self.base_dim, outfeat_dim, 1)
        self.img_offset = output_head(self.base_dim, outfeat_dim, 2)
        self.img_wh = output_head(self.base_dim, outfeat_dim, 2)
        self.img_id_feat = nn.Sequential(
            nn.Conv2d(self.base_dim, self.base_dim, 3, padding=1),
            nn.InstanceNorm2d(self.base_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.base_dim, self.base_dim // 2, 1, padding=0)
        )
        self.img_id_head = output_head(self.base_dim // 2, outfeat_dim, len(dataset.pid_dict))

        # world feat
        self.world_feat = nn.Sequential(nn.Conv2d(self.base_dim, hidden_dim, 3, padding=1), nn.ReLU(),
                                        nn.Conv2d(hidden_dim, hidden_dim, 3, padding=2, dilation=2), nn.ReLU(),
                                        nn.Conv2d(hidden_dim, hidden_dim, 3, padding=4, dilation=4), nn.ReLU(), )

        # world heads
        self.world_heatmap = output_head(hidden_dim, outfeat_dim, 1)
        self.world_offset = output_head(hidden_dim, outfeat_dim, 2)
        self.world_id_feat = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1),
            nn.InstanceNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, hidden_dim // 2, 1, padding=0)
        )
        self.world_id_head = output_head(hidden_dim // 2, outfeat_dim, len(dataset.pid_dict))

        # init
        self.img_heatmap[-1].bias.data.fill_(-2.19)
        fill_fc_weights(self.img_offset)
        fill_fc_weights(self.img_wh)
        self.world_heatmap[-1].bias.data.fill_(-2.19)
        fill_fc_weights(self.world_offset)

        # filter out visible locations
        self.check_visible = check_visible
        xx, yy = np.meshgrid(np.arange(0, self.Rworld_shape[1]), np.arange(0, self.Rworld_shape[0]))
        self.unit_world_grids = torch.tensor(np.stack([xx, yy], axis=2), dtype=torch.float).flatten(0, 1)

        pass

    def get_feat(self, imgs, M, proj_mats, visualize=False):
        B, N, _, H, W = imgs.shape
        imgs = imgs.flatten(0, 1)

        inverse_aug_mats = torch.inverse(M.view([B * N, 3, 3]))
        # image and world feature maps from xy indexing, change them into world indexing / xy indexing (img)
        imgcoord_from_Rimggrid_mat = inverse_aug_mats @ \
                                     torch.diag(torch.tensor([self.img_reduce, self.img_reduce, 1])
                                                ).unsqueeze(0).repeat(B * N, 1, 1).float()
        # [input arg] proj_mats is worldcoord_from_imgcoord
        proj_mats = to_tensor(self.Rworldgrid_from_worldcoord)[None] @ \
                    proj_mats[:, :N].flatten(0, 1) @ \
                    imgcoord_from_Rimggrid_mat

        if visualize:
            denorm = img_color_denormalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
            proj_imgs = warp_perspective(F.interpolate(imgs, scale_factor=1 / 8), proj_mats.to(imgs.device),
                                         self.Rworld_shape).unflatten(0, [B, N])
            for cam in range(N):
                visualize_img = T.ToPILImage()(denorm(imgs.detach())[cam])
                # visualize_img.save(f'../../imgs/augimg{cam + 1}.png')
                plt.imshow(visualize_img)
                plt.show()
                visualize_img = T.ToPILImage()(denorm(proj_imgs.detach())[0, cam])
                plt.imshow(visualize_img)
                plt.show()

        imgs_feat = self.base(imgs)
        imgs_feat = self.bottleneck(imgs_feat)

        # img heads
        imgs_heatmap = self.img_heatmap(imgs_feat)
        imgs_offset = self.img_offset(imgs_feat)
        imgs_wh = self.img_wh(imgs_feat)
        imgs_id = self.img_id_head(self.img_id_feat(imgs_feat))

        # if visualize:
        #     for cam in range(N):
        #         visualize_img = array2heatmap(torch.norm(imgs_feat[cam].detach(), dim=0).cpu())
        #         # visualize_img.save(f'../../imgs/augimgfeat{cam + 1}.png')
        #         plt.imshow(visualize_img)
        #         plt.show()

        # world feat
        world_feat = warp_perspective(imgs_feat, proj_mats.to(imgs.device), self.Rworld_shape).unflatten(0, [B, N])
        if self.check_visible:
            visible_mask = project_2d_points(torch.inverse(proj_mats).to(imgs.device),
                                             self.unit_world_grids.to(imgs.device),
                                             check_visible=True)[1].view([B, N, *self.Rworld_shape])
            world_feat *= visible_mask[:, :, None]
        # if visualize:
        #     for cam in range(N):
        #         visualize_img = array2heatmap(torch.norm(world_feat[0, cam].detach(), dim=0).cpu())
        #         # visualize_img.save(f'../../imgs/projfeat{cam + 1}.png')
        #         plt.imshow(visualize_img)
        #         plt.show()

        return world_feat, (F.interpolate(imgs_heatmap, self.Rimg_shape),
                            F.interpolate(imgs_offset, self.Rimg_shape),
                            F.interpolate(imgs_wh, self.Rimg_shape),
                            F.interpolate(imgs_id, self.Rimg_shape))

    def get_output(self, world_feat, visualize=False):
        B, N, C, H, W = world_feat.shape
        world_feat = aggregate_feat(world_feat, aggregation=self.aggregation)

        # world heads
        world_feat = self.world_feat(world_feat)
        world_heatmap = self.world_heatmap(world_feat)
        world_offset = self.world_offset(world_feat)
        world_id_feat = self.world_id_feat(world_feat)
        world_id = self.world_id_head(world_id_feat)

        if visualize:
            visualize_img = array2heatmap(torch.norm(world_feat[0].detach(), dim=0).cpu())
            # visualize_img.save(f'../../imgs/worldfeatall.png')
            plt.imshow(visualize_img)
            plt.show()
            visualize_img = array2heatmap(torch.sigmoid(world_heatmap.detach())[0, 0].cpu())
            # visualize_img.save(f'../../imgs/worldres.png')
            plt.imshow(visualize_img)
            plt.show()

        return world_id_feat, (world_heatmap, world_offset, world_id)

    def get_world_heatmap(self, feat):
        B, C, H, W = feat.shape
        feat = self.world_feat(feat)
        world_heatmap = self.world_heatmap(feat)
        return world_heatmap


if __name__ == '__main__':
    from src.datasets.frameDataset import frameDataset
    from src.datasets.wildtrack import Wildtrack
    from src.datasets.multiviewx import MultiviewX
    import torchvision.transforms as T
    from torch.utils.data import DataLoader
    from src.utils.decode import ctdet_decode
    from thop import profile
    import time

    dataset = frameDataset(Wildtrack(os.path.expanduser('~/Data/Wildtrack')))
    dataloader = DataLoader(dataset, 2, False, num_workers=0)

    model = MVDet(dataset).cuda()
    (step, configs, imgs, aug_mats, proj_mats, world_gt, imgs_gt, frame) = next(iter(dataloader))
    model.train()
    feat, _ = model.get_feat(imgs.cuda(), aug_mats, proj_mats, True)
    feat_mean, feat_std = cover_mean_std(feat)
    t0 = time.time()
    for i in range(100):
        feat_mean, feat_std = cover_mean_std(feat)
    print(time.time() - t0)
    (world_heatmap, world_offset), _ = model(imgs.cuda(), aug_mats, proj_mats, True)
    xysc_train = ctdet_decode(world_heatmap, world_offset)
    # macs, params = profile(model, inputs=(imgs[:, :3].cuda(), aug_mats[:, :3].contiguous()))
    # macs, params = profile(model.select_module, inputs=(torch.randn([1, 128, 160, 250]).cuda(),
    #                                                     F.one_hot(torch.tensor([1]), num_classes=6).cuda()))
    # macs, params = profile(model, inputs=(torch.rand([1, 128, 160, 250]).cuda(),))
    pass
