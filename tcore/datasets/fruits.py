from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset

import numpy as np
import open3d as o3d
import json
import os

class IGGFruit(Dataset):

    def __init__(self,
                 data_source=None,
                 split='train',
                 sensor='realsense',
                 overfit=False,
                 precomputed_augmentation=False):
        super().__init__()

        assert sensor in ['laser', 'realsense'], 'sensor {} not recognized'

        self.data_source = data_source
        self.sensor = sensor
        self.split = split
        self.overfit = overfit

        with open(self.data_source + '/split.json', 'r') as file:
            self.splits = json.load(file)

        self.fruit = 'SweetPepper' if 'p' in self.splits['train'][0] else 'Strawberry'

        self.fruit_list = self.get_file_paths()
        self.precomputed_augmentation = precomputed_augmentation
        if self.precomputed_augmentation and self.split == 'train' and self.sensor == 'laser':
            self.fruit_list += self.get_augmentation_file_path()

    def get_augmentation_file_path(self):
        aug_list = []
        aug_data_source = self.data_source + self.precomputed_augmentation
        for fid in self.splits[self.split]:
            for aug_idx in range(10):
                aug_list.append(os.path.join(aug_data_source, fid + '_0' + str(aug_idx), 'laser/fruit.ply'))
        return aug_list

    def get_file_paths(self):
        self.fragment_step = 10
        fruit_list = []
        for fid in self.splits[self.split]:
            if self.sensor == 'realsense':
                poses = np.load(os.path.join(self.data_source, fid, 'tf/tf_allposes.npz'))['arr_0']

                depth_frames = self.absolute_file_paths(fid, 'realsense/depth/')
                rgb_frames = self.absolute_file_paths(fid, 'realsense/color/')

                for idx in range(0, len(poses) - self.fragment_step, self.fragment_step):

                    fragment_dict = {}

                    fragment_dict['pose'] = poses[idx:idx + self.fragment_step]
                    fragment_dict['d'] = depth_frames[idx:idx + self.fragment_step]
                    fragment_dict['rgb'] = rgb_frames[idx:idx + self.fragment_step]

                    fruit_list.append(fragment_dict)

            else:
                fruit_list.append(os.path.join(self.data_source, fid, 'laser/fruit.ply'))

        return fruit_list

    def get_fruit_id(self, item):
        if self.sensor == 'realsense':
            fid, _ = item['rgb'][0].split(self.sensor)
        else:
            fid, _ = item.split(self.sensor)
        return fid

    def get_pcd(self, item, fruit_id):
        if self.sensor == 'laser':
            return o3d.io.read_point_cloud(item)
        else:
            pcd = o3d.geometry.PointCloud()
            K = self.load_K(os.path.join(fruit_id, 'realsense/intrinsic.json'))
            for rgb, d, pose in zip(item['rgb'], item['d'], item['pose']):
                frame_pcd = self.pcd_from_rgbd(rgb, d, pose, K)
                pcd += frame_pcd
            return pcd

    def absolute_file_paths(self, fruit_id, directory):
        paths = [os.path.join(self.data_source, fruit_id, directory, frame) for frame in os.listdir(os.path.join(self.data_source, fruit_id, directory))]
        paths.sort()
        return paths

    def get_gt_pcd(self, fruit_id):
        return o3d.io.read_point_cloud(fruit_id + 'laser/fruit.ply')

    @staticmethod
    def pcd_from_rgbd(rgb, d, pose, K):
        rgb_frame = o3d.io.read_image(rgb)
        d_frame = np.load(d)

        extrinsic = np.eye(4)
        intrinsic = o3d.camera.PinholeCameraIntrinsic()
        intrinsic.set_intrinsics(
            height=d_frame.shape[0],
            width=d_frame.shape[1],
            fx=K[0, 0],
            fy=K[1, 1],
            cx=K[0, 2],
            cy=K[1, 2],
        )

        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(o3d.geometry.Image(rgb_frame),
                                                                  o3d.geometry.Image(d_frame),
                                                                  depth_scale=1000.0,
                                                                  depth_trunc=1.0,
                                                                  convert_rgb_to_intensity=False)

        pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, intrinsic, extrinsic)
        return pcd.transform(pose)

    @staticmethod
    def load_K(path):
        with open(path, 'r') as f:
            data = json.load(f)['intrinsic_matrix']
        k = np.reshape(data, (3, 3), order='F')
        return k

    @staticmethod
    def get_bounding_box(fruit_id):
        bbox = np.load(os.path.join(fruit_id, 'tf/bounding_box.npz'))['arr_0']
        return o3d.geometry.AxisAlignedBoundingBox(min_bound=bbox[0], max_bound=bbox[1])

    @staticmethod
    def collate(batch):
        item = {}
        for key in batch[0].keys():
            if key == 'extra':
                item[key] = {}
                for extra_key in batch[0][key].keys():
                    item[key][extra_key] = []
            else:
                item[key] = []

        for key in item.keys():
            for data in batch:
                if key == 'extra':
                    for extra_key in batch[0][key].keys():
                        item[key][extra_key].append(data[key][extra_key])
                else:
                    item[key].append(data[key])

        return item

    def __len__(self):
        if self.overfit:
            return 1
        return len(self.fruit_list)

    def __getitem__(self, index):
        fruit_id = self.get_fruit_id(self.fruit_list[index])
        pcd = self.get_pcd(self.fruit_list[index], fruit_id)

        if self.sensor == 'realsense':
            bbox = self.get_bounding_box(fruit_id)
            pcd = pcd.crop(bbox)

        gt = self.get_gt_pcd(fruit_id)

        item = {}
        item['points'] = np.asarray(pcd.points)

        item['extra'] = {}
        item['extra']['gt_points'] = np.asarray(gt.points)
        item['extra']['gt_normals'] = np.asarray(gt.normals)

        if self.sensor == 'realsense':
            item['filename'] = self.fruit_list[index]['rgb'][0]
            item['colors'] = np.asarray(pcd.colors)
        else:
            item['filename'] = self.fruit_list[index]
            item['normals'] = np.asarray(pcd.normals)

        return item

class IGGFruitDatasetModule(LightningDataModule):

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        precomputed_aug = self.cfg[self.cfg.MODEL.DATASET].AUGMENTATION
        self.train_dataset = IGGFruit(
            data_source=cfg.PATH,
            split='train',
            sensor=self.cfg[self.cfg.MODEL.DATASET].SENSOR,
            precomputed_augmentation=precomputed_aug,
            overfit = self.cfg.MODEL.OVERFIT
            )
        self.test_dataset = IGGFruit(
            data_source=cfg.PATH,
            split='train' if self.cfg.MODEL.OVERFIT else 'test',
            sensor=self.cfg[self.cfg.MODEL.DATASET].SENSOR,
            overfit = self.cfg.MODEL.OVERFIT
            )
        self.val_dataset = IGGFruit(
            data_source=cfg.PATH,
            split='train' if self.cfg.MODEL.OVERFIT else 'val',
            sensor=self.cfg[self.cfg.MODEL.DATASET].SENSOR,
            overfit = self.cfg.MODEL.OVERFIT
            )

    def train_dataloader(self):

        self.train_loader = DataLoader(
            dataset=self.train_dataset,
            batch_size=self.cfg.TRAIN.BATCH_SIZE,
            collate_fn=self.train_dataset.collate,
            shuffle=not self.cfg.MODEL.OVERFIT,
            num_workers=self.cfg.TRAIN.NUM_WORKERS,
            pin_memory=True,
            drop_last=False,
            timeout=0,
        )
        return self.train_loader

    def val_dataloader(self):

        self.val_loader = DataLoader(
            dataset=self.val_dataset,
            batch_size=self.cfg.TRAIN.BATCH_SIZE,
            collate_fn=self.val_dataset.collate,
            shuffle=False,
            num_workers=self.cfg.TRAIN.NUM_WORKERS,
            pin_memory=True,
            drop_last=False,
            timeout=0,
        )
        return self.val_loader

    def test_dataloader(self):

        self.test_loader = DataLoader(
            dataset=self.test_dataset,
            batch_size=self.cfg.TRAIN.BATCH_SIZE,
            collate_fn=self.test_dataset.collate,
            shuffle=False,
            num_workers=self.cfg.TRAIN.NUM_WORKERS,
            pin_memory=True,
            drop_last=False,
            timeout=0,
        )
        return self.test_loader
