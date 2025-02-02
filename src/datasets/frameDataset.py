import os
import re
import json
import time
import random
from operator import itemgetter
from PIL import Image
import matplotlib.pyplot as plt
from kornia.geometry import warp_affine, warp_perspective
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as T
from torchvision.datasets import VisionDataset
from src.parameters import *
from src.utils.projection import *
from src.utils.image_utils import draw_umich_gaussian, random_affine
from src.utils.tensor_utils import to_tensor


def get_centernet_gt(Rshape, x_s, y_s, v_s, w_s=None, h_s=None, reduce=4, top_k=100, kernel_size=4):
    H, W = Rshape
    heatmap = np.zeros([1, H, W], dtype=np.float32)
    reg_mask = np.zeros([top_k], dtype=bool)
    idx = np.zeros([top_k], dtype=np.int64)
    pid = np.zeros([top_k], dtype=np.int64)
    offset = np.zeros([top_k, 2], dtype=np.float32)
    wh = np.zeros([top_k, 2], dtype=np.float32)

    for k in range(len(v_s)):
        ct = np.array([x_s[k] / reduce, y_s[k] / reduce], dtype=np.float32)
        if 0 <= ct[0] < W and 0 <= ct[1] < H:
            ct_int = ct.astype(np.int32)
            draw_umich_gaussian(heatmap[0], ct_int, kernel_size / reduce)
            reg_mask[k] = 1
            idx[k] = ct_int[1] * W + ct_int[0]
            pid[k] = v_s[k]
            offset[k] = ct - ct_int
            if w_s is not None and h_s is not None:
                wh[k] = [w_s[k] / reduce, h_s[k] / reduce]
            # plt.imshow(heatmap[0])
            # plt.show()

    ret = {'heatmap': torch.from_numpy(heatmap), 'reg_mask': torch.from_numpy(reg_mask), 'idx': torch.from_numpy(idx),
           'pid': torch.from_numpy(pid), 'offset': torch.from_numpy(offset)}
    if w_s is not None and h_s is not None:
        ret.update({'wh': torch.from_numpy(wh)})
    return ret


def read_pom(root):
    bbox_by_pos_cam = {}
    cam_pos_pattern = re.compile(r'(\d+) (\d+)')
    cam_pos_bbox_pattern = re.compile(r'(\d+) (\d+) ([-\d]+) ([-\d]+) (\d+) (\d+)')
    with open(os.path.join(root, 'rectangles.pom'), 'r') as fp:
        for line in fp:
            if 'RECTANGLE' in line:
                cam, pos = map(int, cam_pos_pattern.search(line).groups())
                if pos not in bbox_by_pos_cam:
                    bbox_by_pos_cam[pos] = {}
                if 'notvisible' in line:
                    bbox_by_pos_cam[pos][cam] = None
                else:
                    cam, pos, left, top, right, bottom = map(int, cam_pos_bbox_pattern.search(line).groups())
                    bbox_by_pos_cam[pos][cam] = [max(left, 0), max(top, 0),
                                                 min(right, 1920 - 1), min(bottom, 1080 - 1)]
    return bbox_by_pos_cam


class frameDataset(VisionDataset):
    def __init__(self, base, split='train', reID=False, world_reduce=4, trans_img_shape=(720, 1280),
                 world_kernel_size=10, img_kernel_size=10,
                 split_ratio=(0.8, 0.1, 0.1), top_k=100, force_download=True, augmentation='',
                 interactive=False, tracking_scene_len=60, seed=None):
        super().__init__(base.root)

        self.base = base
        self.num_cam, self.num_frame = base.num_cam, base.num_frame
        # world (grid) reduce: on top of the 2.5cm grid
        self.reID, self.top_k = reID, top_k
        # reduce = input/output
        self.world_reduce = world_reduce
        self.img_shape, self.worldgrid_shape = base.img_shape, base.worldgrid_shape  # H,W; N_row,N_col

        self.img_reduce = self.img_shape[0] // (trans_img_shape[0] // 8)
        self.world_kernel_size, self.img_kernel_size = world_kernel_size, img_kernel_size
        self.augmentation = augmentation
        self.transform = T.Compose([T.ToTensor(),
                                    T.Resize(trans_img_shape, antialias=True),
                                    T.ColorJitter(0.4, 0.4, 0.4),
                                    T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
                                    ]) if 'color' in self.augmentation else \
            T.Compose([T.ToTensor(), T.Resize(trans_img_shape, antialias=True),
                       T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
        self.interactive = interactive

        # Two entries below are used to denote number of frames in a scene
        # Only used when doing reID task.
        self.tracking_scene_len = tracking_scene_len
        self.curr_scene_len = 0

        self.Rworld_shape = list(map(lambda x: x // self.world_reduce, self.worldgrid_shape))
        self.Rimg_shape = np.ceil(np.array(trans_img_shape) // 8).astype(int).tolist()

        # world_grid (projected feature map) <- coordinate translation -> world_coord <- camera matrix -> image coord
        self.Rworldgrid_from_worldcoord = np.linalg.inv(self.base.worldcoord_from_worldgrid_mat @
                                                        np.diag([self.world_reduce, self.world_reduce, 1]))

        # split = ('train', 'val', 'test'), split_ratio=(0.8, 0.1, 0.1)
        split_ratio = tuple(sum(split_ratio[:i + 1]) for i in range(len(split_ratio)))
        # assert split_ratio[-1] == 1
        self.split = split
        if split == 'train':
            frame_range = range(0, int(self.num_frame * split_ratio[0]))
        elif split == 'val':
            frame_range = range(int(self.num_frame * split_ratio[0]), int(self.num_frame * split_ratio[1]))
        elif split == 'trainval':
            frame_range = range(0, int(self.num_frame * split_ratio[1]))
        elif split == 'test':
            # Only in the case of doing reID, we allow more frames for testing
            # to provide more variety of sequences.
            if split_ratio[-1] > 1:
                if reID:
                    frame_range = range(int(self.num_frame * split_ratio[1]),
                                        int(self.num_frame * split_ratio[2]))
                    # Also need to update the self.num_frame attribute!!!
                    self.num_frame = int(self.num_frame * split_ratio[2])
                else:
                    raise ValueError('split_ratio[-1] > 1 is only allowed when doing reID task.')
            else:
                frame_range = range(int(self.num_frame * split_ratio[1]), self.num_frame)
        else:
            raise Exception

        # gt in mot format for evaluation
        self.gt_fname = f'{self.root}/gt.txt'
        if self.base.__name__ == 'CarlaX':
            self.z = self.base.z

            # In CarlaX, the pedestrian IDs are just the blueprint IDs and it is fixed for all scenes
            self.pid_dict = {i: i for i in range(len(self.base.env.pedestrian_bps))}

            # generate same pedestrian layout for the same frame
            if seed is not None:
                # random_generator = random.Random(seed)
                np_random_generator = np.random.default_rng(seed)
                self.fixed_seeds = np_random_generator.choice(65536, size=self.num_frame, replace=False)[frame_range]
            else:
                self.fixed_seeds = [None] * len(frame_range)
            self.frames = list(frame_range)
            self.world_gt = {}
            # self.gt_array = np.array([]).reshape([0, 3])
            self.config_dim = base.env.config_dim
            self.action_names = base.env.action_names if interactive else None
        else:
            self.z = 0
            # get camera matrices
            self.proj_mats = torch.stack([get_worldcoord_from_imgcoord_mat(self.base.intrinsic_matrices[cam],
                                                                           self.base.extrinsic_matrices[cam],
                                                                           self.z / self.base.worldcoord_unit)
                                          for cam in range(self.num_cam)])

            self.img_fpaths = self.get_image_fpaths(frame_range)
            self.world_gt, self.imgs_gt, self.pid_dict, self.frames = self.get_gt_targets(
                split if split == 'trainval' else f'{split} \t', frame_range)
            if not os.path.exists(self.gt_fname) or force_download:
                og_gt = [[] for _ in range(self.num_cam)]
                for fname in sorted(os.listdir(os.path.join(self.root, 'annotations_positions'))):
                    frame = int(fname.split('.')[0])
                    with open(os.path.join(self.root, 'annotations_positions', fname)) as json_file:
                        all_pedestrians = json.load(json_file)
                    for pedestrian in all_pedestrians:
                        def is_in_cam(cam, grid_x, grid_y):
                            visible = not (pedestrian['views'][cam]['xmin'] == -1 and
                                           pedestrian['views'][cam]['xmax'] == -1 and
                                           pedestrian['views'][cam]['ymin'] == -1 and
                                           pedestrian['views'][cam]['ymax'] == -1)
                            in_view = (pedestrian['views'][cam]['xmin'] > 0 and
                                       pedestrian['views'][cam]['xmax'] < 1920 and
                                       pedestrian['views'][cam]['ymin'] > 0 and
                                       pedestrian['views'][cam]['ymax'] < 1080)

                            # Rgrid_x, Rgrid_y = grid_x // self.world_reduce, grid_y // self.world_reduce
                            # in_map = Rgrid_x < self.Rworld_shape[0] and Rgrid_y < self.Rworld_shape[1]
                            return visible and in_view

                        grid = self.base.get_worldgrid_from_pos(pedestrian['positionID']).squeeze()
                        grid_x, grid_y = grid[0], grid[1]
                        for cam in range(self.num_cam):
                            if is_in_cam(cam, grid_x, grid_y):
                                og_gt[cam].append(np.array([frame, grid_x, grid_y]))
                og_gt = [np.stack(og_gt[cam], axis=0) for cam in range(self.num_cam)]
                np.savetxt(self.gt_fname, np.unique(np.concatenate(og_gt, axis=0), axis=0), '%d')
                for cam in range(self.num_cam):
                    np.savetxt(f'{self.gt_fname}.{cam}', og_gt[cam], '%d')
            # self.gt_array = np.loadtxt(self.gt_fname)
            self.config_dim = None
            self.action_names = None
        pass

    def get_image_fpaths(self, frame_range):
        img_fpaths = {cam: {} for cam in range(self.num_cam)}
        for camera_folder in sorted(os.listdir(os.path.join(self.root, 'Image_subsets'))):
            cam = int(camera_folder[-1]) - 1
            if cam >= self.num_cam:
                continue
            for fname in sorted(os.listdir(os.path.join(self.root, 'Image_subsets', camera_folder))):
                frame = int(fname.split('.')[0])
                if frame in frame_range:
                    img_fpaths[cam][frame] = os.path.join(self.root, 'Image_subsets', camera_folder, fname)
        return img_fpaths

    def get_gt_targets(self, split, frame_range):
        num_world_bbox, num_imgs_bbox = 0, 0
        world_gt = {}
        imgs_gt = {}
        pid_dict = {}
        frames = []
        for fname in sorted(os.listdir(os.path.join(self.root, 'annotations_positions'))):
            frame = int(fname.split('.')[0])
            if frame in frame_range:
                frames.append(frame)
                with open(os.path.join(self.root, 'annotations_positions', fname)) as json_file:
                    all_pedestrians = json.load(json_file)
                world_pts, world_pids = [], []
                img_bboxs, img_pids = [[] for _ in range(self.num_cam)], [[] for _ in range(self.num_cam)]
                for pedestrian in all_pedestrians:
                    grid = self.base.get_worldgrid_from_pos(pedestrian['positionID']).squeeze()
                    grid_x, grid_y = grid[0], grid[1]
                    if pedestrian['personID'] not in pid_dict:
                        pid_dict[pedestrian['personID']] = len(pid_dict)
                    num_world_bbox += 1
                    world_pts.append((grid_x, grid_y))
                    world_pids.append(pid_dict[pedestrian['personID']])
                    for cam in range(self.num_cam):
                        if itemgetter('xmin', 'ymin', 'xmax', 'ymax')(pedestrian['views'][cam]) != (-1, -1, -1, -1):
                            img_bboxs[cam].append(itemgetter('xmin', 'ymin', 'xmax', 'ymax')
                                                  (pedestrian['views'][cam]))
                            img_pids[cam].append(pid_dict[pedestrian['personID']])
                            num_imgs_bbox += 1
                world_gt[frame] = (np.array(world_pts), np.array(world_pids))
                imgs_gt[frame] = {}
                for cam in range(self.num_cam):
                    # x1y1x2y2
                    imgs_gt[frame][cam] = (np.array(img_bboxs[cam]), np.array(img_pids[cam]))

        print(f'{split}:\t pid: {len(pid_dict)}, frame: {len(frames)}, '
              f'world bbox: {num_world_bbox / len(frames):.1f}, '
              f'imgs bbox per cam: {num_imgs_bbox / len(frames) / self.num_cam:.1f}')
        return world_gt, imgs_gt, pid_dict, frames

    def get_carla_gt_targets(self, all_pedestrians):
        # NOTE: for re-ID task setup, we use the pedestrian blueprint id as the identity
        world_pts, world_lwh, world_pids = [], [], []
        img_bboxs, img_pids = [[] for _ in range(self.num_cam)], [[] for _ in range(self.num_cam)]
        for pedestrian in all_pedestrians:
            # world_pts.append([pedestrian["x"], pedestrian["y"], pedestrian["z"]])
            world_pts.append(self.base.get_worldgrid_from_worldcoord(
                np.array([pedestrian["x"], pedestrian["y"]])[None]).squeeze())
            world_lwh.append([pedestrian["l"], pedestrian["w"], pedestrian["h"]])
            world_pids.append(pedestrian["bp_id"])
            for cam in range(self.num_cam):
                if itemgetter('xmin', 'ymin', 'xmax', 'ymax')(pedestrian['views'][cam]) != (-1, -1, -1, -1):
                    img_bboxs[cam].append(itemgetter('xmin', 'ymin', 'xmax', 'ymax')(pedestrian['views'][cam]))
                    img_pids[cam].append(pedestrian["bp_id"])
        for cam in range(self.num_cam):
            img_bboxs[cam], img_pids[cam] = np.array(img_bboxs[cam]), np.array(img_pids[cam])
        return np.array(world_pts), np.array(world_lwh), np.array(world_pids), img_bboxs, img_pids

    def get_world_imgs_trans(self, intrinsic, extrinsic):
        device = intrinsic.device if isinstance(intrinsic, torch.Tensor) else 'cpu'
        Rworldgrid_from_worldcoord = to_tensor(self.Rworldgrid_from_worldcoord, dtype=torch.float, device=device)
        # z in meters by default
        # projection matrices: img feat -> world feat
        worldcoord_from_imgcoord = get_worldcoord_from_imgcoord_mat(intrinsic, extrinsic,
                                                                    self.z / self.base.worldcoord_unit)
        # worldgrid(xy)_from_img(xy)
        Rworldgrid_from_imgcoord = Rworldgrid_from_worldcoord @ worldcoord_from_imgcoord
        return Rworldgrid_from_imgcoord

    def __getitem__(self, index=None, visualize=False, use_depth=False):
        if self.base.__name__ == 'CarlaX':
            # Conditions for a reset:
            # 1. When not doing testing (doing training), random frames for detection/reID is fine, just randomly reset.
            # 2. When not doing reID, random frames won't bother detection, so it's fine.
            # 3. When testing reID, for the first time we need to reset the environment.
            need_reset = (self.split != 'test' or not self.reID or 
                            self.curr_scene_len % self.tracking_scene_len == 0)

            if index is not None:  # reset (remove) all pedestrians, return DEFAULT images and info!
                # This will be called when enumerating through the dataloader/dataset
                if need_reset:
                    observation, info = self.base.env.reset(seed=self.fixed_seeds[index])
                    # When reset the scene, we resent the counter
                    self.curr_scene_len = 0
                self.cur_frame = self.frames[index]

                if self.interactive:
                    # A bit hacky, but we need to reset the step counter to ensure that the masks 
                    # are always correct.
                    self.base.env.step_counter = 0
            else:
                assert self.interactive

            # 1. not self.interactive -> case for fixed camera config tasks (det/reID)
            # 2. index is None -> interactive mode, the second time calling __getitem__
            #             (the first time is for resetting the environment)

            # only spawn pedestrian after interactive steps (index==None)
            # This check enforces no pedestrians are spawned when we decide camera configs
            if (not self.interactive or index is None):
                use_depth = ('train' in self.split) and use_depth
                # only when resetting the environment, we need to spawn the pedestrians and render
                # otherwise, we render continuous frames
                if need_reset:
                    # render the first frame, need to spawn
                    observation, info = self.base.env.spawn_and_render(use_depth, is_train='train' in self.split)
                else:
                    # render the next frame, no need to spawn
                    observation, info = self.base.env.render_and_update_pedestrian_gts(use_depth)

                # either way, we get a new image (frame)
                self.curr_scene_len += 1
            else:
                # NOTE: This will be called when `interactive`, `reID`, `test` happens. 
                #       When enumerate through the dataset, we have index but we don't have to reset environment.
                #       Some local variables are not initialized and would lead to an error.
                observation = {
                    "images": self.base.env.default_imgs,
                    "camera_configs": {cam: self.base.env.encode_camera_cfg(self.base.env.camera_configs[cam])
                                       for cam in range(self.num_cam)},
                    "step": self.base.env.step_counter
                }
                info = {"pedestrian_gts": [],
                        "depths": {},
                        "camera_intrinsics": self.base.env.camera_intrinsics,
                        "camera_extrinsics": self.base.env.camera_extrinsics}  # Set any additional information

            # get camera matrices
            self.proj_mats = torch.stack([get_worldcoord_from_imgcoord_mat(self.base.env.camera_intrinsics[cam],
                                                                           self.base.env.camera_extrinsics[cam],
                                                                           self.z / self.base.worldcoord_unit)
                                          for cam in range(self.num_cam)])

            imgs = observation["images"]
            configs = observation["camera_configs"]
            step_counter = observation["step"] if index is not None else None  # init/finish => return all cameras
            depths = info["depths"]
            world_pts, world_lwh, world_pids, img_bboxs, img_pids = self.get_carla_gt_targets(info["pedestrian_gts"])
            world_pts = world_pts.reshape([-1, 2]) if world_pts.size == 0 else world_pts

            # record world gt
            self.world_gt[self.cur_frame] = (world_pts, world_pids)
        else:
            assert index is not None
            self.cur_frame = frame = self.frames[index]
            imgs = {cam: np.array(Image.open(self.img_fpaths[cam][frame]).convert('RGB'))
                    for cam in range(self.num_cam)}
            configs = None
            step_counter = None
            depths = {}
            img_bboxs, img_pids = zip(*self.imgs_gt[frame].values())
            world_pts, world_pids = self.world_gt[frame]
        return self.prepare_gt(imgs, step_counter, configs, world_pts, world_pids, img_bboxs, img_pids, depths,
                               visualize)

    def get_gt_array(self, frames=None, reID=False):
        # self.reID -> Whether enables reID mode in the dataset
        # reID -> Whether to return the reID gt array
        # Force to off reID mode if reID was not enabled in the dataset
        reID = reID and self.reID

        if self.base.__name__ == 'CarlaX':
            gt_array = []
            for frame in self.world_gt.keys() if frames is None else frames:
                frame_res = np.concatenate([frame * np.ones([len(self.world_gt[frame][0]), 1]),
                                                self.world_gt[frame][0]], axis=1)
                # Put the pedestrian blueprint id as the identity into the 2nd column
                if reID:
                    pids = self.world_gt[frame][1].reshape(-1, 1)
                    frame_res = np.concatenate([frame_res[:,:1], pids, frame_res[:,1:]], axis=1)
                gt_array.append(frame_res)
            gt_array = np.concatenate(gt_array, axis=0)
        else:
            gt_array = np.loadtxt(self.gt_fname)
            if frames is not None:
                gt_array = gt_array[np.isin(gt_array[:, 0], frames)]
        return gt_array

    def step(self, action, visualize=False):
        observation, reward, done, info = self.base.env.step(action)

        # get camera matrices
        self.proj_mats = torch.stack([get_worldcoord_from_imgcoord_mat(self.base.env.camera_intrinsics[cam],
                                                                       self.base.env.camera_extrinsics[cam],
                                                                       self.z / self.base.worldcoord_unit)
                                      for cam in range(self.num_cam)])

        imgs = observation["images"]
        configs = observation["camera_configs"]
        step_counter = observation["step"]
        depths = {}
        world_pts, world_lwh, world_pids, img_bboxs, img_pids = self.get_carla_gt_targets(info["pedestrian_gts"])
        world_pts = world_pts.reshape([-1, 2]) if world_pts.size == 0 else world_pts
        return self.prepare_gt(imgs, step_counter, configs, world_pts, world_pids, img_bboxs, img_pids, depths,
                               visualize), done

    def prepare_gt(self, imgs, step_counter, configs, world_pts, world_pids, img_bboxs, img_pids, depths,
                   visualize=False):
        def plt_visualize():
            import cv2
            from matplotlib.patches import Circle
            fig, ax = plt.subplots(1)
            # ax.imshow(img)
            # for i in range(len(img_x_s)):
            #     x, y = img_x_s[i], img_y_s[i]
            #     if x > 0 and y > 0:
            #         ax.add_patch(Circle((x, y), 10))
            # plt.show()
            # plt.imshow(img_gt['heatmap'][0].numpy())
            # plt.show()
            img0 = img.copy()
            for bbox in cam_img_bboxs:
                bbox = tuple(int(pt) for pt in bbox)
                cv2.rectangle(img0, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)
            plt.imshow(img0)
            plt.show()

        # world gt
        world_gt = get_centernet_gt(self.Rworld_shape, world_pts[:, 0], world_pts[:, 1], world_pids,
                                    reduce=self.world_reduce, top_k=self.top_k, kernel_size=self.world_kernel_size)
        if self.interactive:
            configs = to_tensor(list(configs.values()))
            if step_counter is not None:
                # IMPORTANT: mask out future steps
                padding_mask = torch.arange(self.num_cam) > step_counter - 1
                configs[padding_mask] = CONFIGS_PADDING_VALUE
                if step_counter != 0:
                    # return only one camera
                    cam = step_counter - 1
                    imgs = {cam: imgs[cam]}
                    configs = configs[[cam]]
            else:
                # finish => return all camera views
                step_counter = self.num_cam
        else:
            step_counter = self.num_cam
            configs = torch.zeros([self.num_cam, 0])

        aug_imgs, aug_imgs_gt, aug_mats, aug_masks = [], [], [], []
        for cam, img in imgs.items():
            cam_img_bboxs, cam_img_pids = img_bboxs[cam], img_pids[cam]
            if len(cam_img_bboxs.shape) == 1:
                cam_img_bboxs = cam_img_bboxs.reshape([-1, 4])
            if 'affine' in self.augmentation:
                img, cam_img_bboxs, cam_img_pids, M = random_affine(img, cam_img_bboxs, cam_img_pids)
            else:
                M = np.eye(3)
            aug_imgs.append(self.transform(img))
            aug_mats.append(torch.from_numpy(M).float())
            img_x_s, img_y_s = (cam_img_bboxs[:, 0] + cam_img_bboxs[:, 2]) / 2, cam_img_bboxs[:, 3]
            img_w_s, img_h_s = (cam_img_bboxs[:, 2] - cam_img_bboxs[:, 0]), (cam_img_bboxs[:, 3] - cam_img_bboxs[:, 1])

            img_gt = get_centernet_gt(self.Rimg_shape, img_x_s, img_y_s, cam_img_pids, img_w_s, img_h_s,
                                      reduce=self.img_reduce, top_k=self.top_k, kernel_size=self.img_kernel_size)
            aug_imgs_gt.append(img_gt)
            # TODO: check the difference between different dataloader iteration
            if visualize:
                plt_visualize()
                # if len(depths):
                #     plt.imshow(depths[cam])
                #     plt.show()

        aug_imgs = torch.stack(aug_imgs)
        aug_mats = torch.stack(aug_mats)
        aug_imgs_gt = {key: torch.stack([img_gt[key] for img_gt in aug_imgs_gt]) for key in aug_imgs_gt[0]}
        if len(depths):
            assert 'affine' not in self.augmentation
            aug_imgs_gt['depth'] = to_tensor(np.array(list(depths.values()))[:, None])
        return (step_counter, configs, aug_imgs, aug_mats, self.proj_mats[list(imgs.keys())],
                world_gt, aug_imgs_gt, self.cur_frame)

    def __len__(self):
        return len(self.frames)


if __name__ == '__main__':
    from torch.utils.data import DataLoader
    from src.datasets.wildtrack import Wildtrack
    from src.datasets.multiviewx import MultiviewX
    from src.datasets.carlax import CarlaX
    import random
    import json

    seed = 1
    # random.seed(seed)
    # np.random.seed(seed)
    # torch.manual_seed(seed)
    # torch.cuda.manual_seed(seed)
    # torch.cuda.manual_seed_all(seed)

    # dataset = frameDataset(Wildtrack(os.path.expanduser('~/Data/Wildtrack')), augmentation='affine+color')
    # dataset = frameDataset(MultiviewX(os.path.expanduser('~/Data/MultiviewX')), force_download=True)

    with open('cfg/RL/town04building.cfg', "r") as fp:
        dataset_config = json.load(fp)
    dataset = frameDataset(CarlaX(dataset_config, port=2300, tm_port=8300, euler2vec='yaw', use_depth=True),
                           interactive=True, seed=seed)
    # min_dist = np.inf
    # for world_gt in dataset.world_gt.values():
    #     x, y = world_gt[0][:, 0], world_gt[0][:, 1]
    #     if x.size and y.size:
    #         xy_dists = ((x - x[:, None]) ** 2 + (y - y[:, None]) ** 2) ** 0.5
    #         np.fill_diagonal(xy_dists, np.inf)
    #         min_dist = min(min_dist, np.min(xy_dists))
    #         pass
    dataloader = DataLoader(dataset, 2, True, num_workers=0)
    t0 = time.time()
    people_cnt = []
    # _ = next(iter(dataloader))
    for i in range(20):
        _ = dataset.__getitem__(i % len(dataset), visualize=True)
        if dataset.base.__name__ == 'CarlaX' and dataset.interactive:
            done = False
            while not done:
                _, done = dataset.step(np.random.randn(len(dataset.action_names)), visualize=True)
            _ = dataset.__getitem__(visualize=True)
        print(_[5]['reg_mask'].sum().item())
        people_cnt.append(_[5]['reg_mask'].sum().item())
    print(np.mean(people_cnt))

    print(time.time() - t0)
    pass
    if False:
        dataset = frameDataset(MultiviewX(os.path.expanduser('~/Data/MultiviewX')), split='train')
        import matplotlib.pyplot as plt
        from src.utils.projection import get_worldcoord_from_imagecoord

        world_grid_maps = []
        xx, yy = np.meshgrid(np.arange(0, 1920, 20), np.arange(0, 1080, 20))
        H, W = xx.shape
        image_coords = np.stack([xx, yy], axis=2).reshape([-1, 2])
        for cam in range(dataset.num_cam):
            world_coords = get_worldcoord_from_imagecoord(image_coords,
                                                          dataset.base.intrinsic_matrices[cam],
                                                          dataset.base.extrinsic_matrices[cam])
            world_grids = dataset.base.get_worldgrid_from_worldcoord(world_coords).reshape([H, W, 2])
            world_grid_map = np.zeros(dataset.worldgrid_shape)
            for i in range(H):
                for j in range(W):
                    x, y = world_grids[i, j]
                    if x in range(dataset.worldgrid_shape[1]) and y in range(dataset.worldgrid_shape[0]):
                        world_grid_map[int(y), int(x)] += 1
            world_grid_map = world_grid_map != 0
            plt.imshow(world_grid_map)
            plt.show()
            world_grid_maps.append(world_grid_map)
            pass
        plt.imshow(np.sum(np.stack(world_grid_maps), axis=0))
        plt.show()
        pass
