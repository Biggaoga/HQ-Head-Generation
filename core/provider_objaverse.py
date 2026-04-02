import os
import cv2
import random
import numpy as np
import json

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from torch.utils.data import Dataset
from kiui.cam import orbit_camera
import kiui
from core.options import Options
from core.utils import get_rays, grid_distortion, orbit_camera_jitter

IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)


class ObjaverseDataset(Dataset):

    def _warn(self):
        raise NotImplementedError('this dataset is just an example and cannot be used directly, you should modify it to your own setting! (search keyword TODO)')

    def __init__(self, opt: Options, training=True):
        
        self.opt = opt
        self.training = training

        # TODO: remove this barrier
        self._warn()

        # TODO: load the list of objects for training
        self.items = []
        with open('TODO: file containing the list', 'r') as f:
            for line in f.readlines():
                self.items.append(line.strip())

        # naive split
        if self.training:
            self.items = self.items[:-self.opt.batch_size]
        else:
            self.items = self.items[-self.opt.batch_size:]
        
        # default camera intrinsics
        self.tan_half_fov = np.tan(0.5 * np.deg2rad(self.opt.fovy))
        self.proj_matrix = torch.zeros(4, 4, dtype=torch.float32)
        self.proj_matrix[0, 0] = 1 / self.tan_half_fov
        self.proj_matrix[1, 1] = 1 / self.tan_half_fov
        self.proj_matrix[2, 2] = (self.opt.zfar + self.opt.znear) / (self.opt.zfar - self.opt.znear)
        self.proj_matrix[3, 2] = - (self.opt.zfar * self.opt.znear) / (self.opt.zfar - self.opt.znear)
        self.proj_matrix[2, 3] = 1


    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):

        uid = self.items[idx]
        results = {}

        # load num_views images
        images = []
        masks = []
        cam_poses = []
        
        vid_cnt = 0

        # TODO: choose views, based on your rendering settings
        if self.training:
            # input views are in (36, 72), other views are randomly selected
            vids = np.random.permutation(np.arange(36, 73))[:self.opt.num_input_views].tolist() + np.random.permutation(100).tolist()
        else:
            # fixed views
            vids = np.arange(36, 73, 4).tolist() + np.arange(100).tolist()
        
        for vid in vids:

            image_path = os.path.join(uid, 'rgb', f'{vid:03d}.png')
            camera_path = os.path.join(uid, 'pose', f'{vid:03d}.txt')

            try:
                # TODO: load data (modify self.client here)
                image = np.frombuffer(self.client.get(image_path), np.uint8)
                image = torch.from_numpy(cv2.imdecode(image, cv2.IMREAD_UNCHANGED).astype(np.float32) / 255) # [512, 512, 4] in [0, 1]
                c2w = [float(t) for t in self.client.get(camera_path).decode().strip().split(' ')]
                c2w = torch.tensor(c2w, dtype=torch.float32).reshape(4, 4)
            except Exception as e:
                # print(f'[WARN] dataset {uid} {vid}: {e}')
                continue
            
            # TODO: you may have a different camera system
            # blender world + opencv cam --> opengl world & cam
            c2w[1] *= -1
            c2w[[1, 2]] = c2w[[2, 1]]
            c2w[:3, 1:3] *= -1 # invert up and forward direction

            # scale up radius to fully use the [-1, 1]^3 space!
            c2w[:3, 3] *= self.opt.cam_radius / 1.5 # 1.5 is the default scale
          
            image = image.permute(2, 0, 1) # [4, 512, 512]
            mask = image[3:4] # [1, 512, 512]
            image = image[:3] * mask + (1 - mask) # [3, 512, 512], to white bg
            image = image[[2,1,0]].contiguous() # bgr to rgb

            images.append(image)
            masks.append(mask.squeeze(0))
            cam_poses.append(c2w)

            vid_cnt += 1
            if vid_cnt == self.opt.num_views:
                break

        if vid_cnt < self.opt.num_views:
            print(f'[WARN] dataset {uid}: not enough valid views, only {vid_cnt} views found!')
            n = self.opt.num_views - vid_cnt
            images = images + [images[-1]] * n
            masks = masks + [masks[-1]] * n
            cam_poses = cam_poses + [cam_poses[-1]] * n
          
        images = torch.stack(images, dim=0) # [V, C, H, W]
        masks = torch.stack(masks, dim=0) # [V, H, W]
        cam_poses = torch.stack(cam_poses, dim=0) # [V, 4, 4]

        # normalized camera feats as in paper (transform the first pose to a fixed position)
        transform = torch.tensor([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, self.opt.cam_radius], [0, 0, 0, 1]], dtype=torch.float32) @ torch.inverse(cam_poses[0])
        cam_poses = transform.unsqueeze(0) @ cam_poses  # [V, 4, 4]

        images_input = F.interpolate(images[:self.opt.num_input_views].clone(), size=(self.opt.input_size, self.opt.input_size), mode='bilinear', align_corners=False) # [V, C, H, W]
        cam_poses_input = cam_poses[:self.opt.num_input_views].clone()

        # data augmentation
        if self.training:
            # apply random grid distortion to simulate 3D inconsistency
            if random.random() < self.opt.prob_grid_distortion:
                images_input[1:] = grid_distortion(images_input[1:])
            # apply camera jittering (only to input!)
            if random.random() < self.opt.prob_cam_jitter:
                cam_poses_input[1:] = orbit_camera_jitter(cam_poses_input[1:])

        images_input = TF.normalize(images_input, IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)

        # resize render ground-truth images, range still in [0, 1]
        results['images_output'] = F.interpolate(images, size=(self.opt.output_size, self.opt.output_size), mode='bilinear', align_corners=False) # [V, C, output_size, output_size]
        results['masks_output'] = F.interpolate(masks.unsqueeze(1), size=(self.opt.output_size, self.opt.output_size), mode='bilinear', align_corners=False) # [V, 1, output_size, output_size]

        # build rays for input views
        rays_embeddings = []
        for i in range(self.opt.num_input_views):
            rays_o, rays_d = get_rays(cam_poses_input[i], self.opt.input_size, self.opt.input_size, self.opt.fovy) # [h, w, 3]
            rays_plucker = torch.cat([torch.cross(rays_o, rays_d, dim=-1), rays_d], dim=-1) # [h, w, 6]
            rays_embeddings.append(rays_plucker)

     
        rays_embeddings = torch.stack(rays_embeddings, dim=0).permute(0, 3, 1, 2).contiguous() # [V, 6, h, w]
        final_input = torch.cat([images_input, rays_embeddings], dim=1) # [V=4, 9, H, W]
        results['input'] = final_input

        # opengl to colmap camera for gaussian renderer
        cam_poses[:, :3, 1:3] *= -1 # invert up & forward direction
        
        # cameras needed by gaussian rasterizer
        cam_view = torch.inverse(cam_poses).transpose(1, 2) # [V, 4, 4]
        cam_view_proj = cam_view @ self.proj_matrix # [V, 4, 4]
        cam_pos = - cam_poses[:, :3, 3] # [V, 3]
        
        results['cam_view'] = cam_view
        results['cam_view_proj'] = cam_view_proj
        results['cam_pos'] = cam_pos

        return results
    
class GobjaverseDataset(Dataset):

    def _warn(self):
        raise NotImplementedError('this dataset is just an example and cannot be used directly, you should modify it to your own setting! (search keyword TODO)')

    def __init__(self, opt: Options, training=True):
        
        self.opt = opt
        self.training = training

        # TODO: load the list of objects for training
        self.items = []
        with open('/data/gyj/gobjaverse/gobj_merged.json', 'r') as f:
            obj_list = json.load(f)
            for obj in obj_list:
                self.items.append(os.path.join('/data/gyj/gobjaverse', obj))


        # naive split
        if self.training:
            self.items = self.items[:-self.opt.batch_size]
        else:
            self.items = self.items[-self.opt.batch_size:]
        
        # default camera intrinsics
        self.tan_half_fov = np.tan(0.5 * np.deg2rad(self.opt.fovy))
        self.proj_matrix = torch.zeros(4, 4, dtype=torch.float32)
        self.proj_matrix[0, 0] = 1 / self.tan_half_fov
        self.proj_matrix[1, 1] = 1 / self.tan_half_fov
        self.proj_matrix[2, 2] = (self.opt.zfar + self.opt.znear) / (self.opt.zfar - self.opt.znear)
        self.proj_matrix[3, 2] = - (self.opt.zfar * self.opt.znear) / (self.opt.zfar - self.opt.znear)
        self.proj_matrix[2, 3] = 1


    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):

        uid = self.items[idx]
        results = {}

        # load num_views images
        images = []
        masks = []
        cam_poses = []
        
        vid_cnt = 0

        # TODO: choose views, based on your rendering settings
        if self.training:
            # input views are in (36, 72), other views are randomly selected
            # vids = np.random.permutation(np.arange(36, 73))[:self.opt.num_input_views].tolist() + np.random.permutation(100).tolist()
            vids = np.random.permutation(np.arange(0,40)).tolist()
        else:
            # fixed views
            vids = np.arange(0, 20, 30).tolist() + np.arange(40).tolist()
        
        for vid in vids:

            image_path = os.path.join(uid, f'{vid:05d}', f'{vid:05d}.png')
            meta_path = os.path.join(uid, f'{vid:05d}', f'{vid:05d}.json')
            nd_path = os.path.join(uid, f'{vid:05d}', f'{vid:05d}_nd.exr')

            try:
                image = torch.from_numpy(cv2.imread(image_path, cv2.IMREAD_UNCHANGED).astype(np.float32) / 255) # [512, 512, 4] in [0, 1]
                
                with open(meta_path, 'r') as f:
                    meta = json.load(f)
                c2w = np.eye(4)
                c2w[:3, 0] = np.array(meta['x'])
                c2w[:3, 1] = np.array(meta['y'])
                c2w[:3, 2] = np.array(meta['z'])
                c2w[:3, 3] = np.array(meta['origin'])
                c2w = torch.tensor(c2w, dtype=torch.float32).reshape(4, 4)

            except Exception as e:
                print(f'[WARN] dataset {uid} {vid}: {e}')
                continue
            
            # TODO: you may have a different camera system
            # blender world + opencv cam --> opengl world & cam
            c2w[1] *= -1
            c2w[[1, 2]] = c2w[[2, 1]]
            c2w[:3, 1:3] *= -1 # invert up and forward direction
          
            image = image.permute(2, 0, 1) # [4, 512, 512]
            mask = image[3:4] # [1, 512, 512]
            image = image[:3] * mask + (1 - mask) # [3, 512, 512], to white bg
            image = image[[2,1,0]].contiguous() # bgr to rgb

            images.append(image)
            masks.append(mask.squeeze(0))
            cam_poses.append(c2w)

            vid_cnt += 1
            if vid_cnt == self.opt.num_views:
                break

        if vid_cnt < self.opt.num_views:
            print(f'[WARN] dataset {uid}: not enough valid views, only {vid_cnt} views found!')
            n = self.opt.num_views - vid_cnt
            images = images + [images[-1]] * n
            masks = masks + [masks[-1]] * n
            cam_poses = cam_poses + [cam_poses[-1]] * n
          
        images = torch.stack(images, dim=0) # [V, C, H, W]
        masks = torch.stack(masks, dim=0) # [V, H, W]
        cam_poses = torch.stack(cam_poses, dim=0) # [V, 4, 4]

        radius = torch.norm(cam_poses[0, :3, 3])
        cam_poses[:, :3, 3] *= self.opt.cam_radius / radius
        # normalized camera feats as in paper (transform the first pose to a fixed position)
        transform = torch.tensor([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, self.opt.cam_radius], [0, 0, 0, 1]], dtype=torch.float32) @ torch.inverse(cam_poses[0])
        cam_poses = transform.unsqueeze(0) @ cam_poses  # [V, 4, 4]

        images_input = F.interpolate(images[:self.opt.num_input_views].clone(), size=(self.opt.input_size, self.opt.input_size), mode='bilinear', align_corners=False) # [V, C, H, W]
        cam_poses_input = cam_poses[:self.opt.num_input_views].clone()

        # data augmentation
        if self.training:
            # apply random grid distortion to simulate 3D inconsistency
            if random.random() < self.opt.prob_grid_distortion:
                images_input[1:] = grid_distortion(images_input[1:])
            # apply camera jittering (only to input!)
            if random.random() < self.opt.prob_cam_jitter:
                cam_poses_input[1:] = orbit_camera_jitter(cam_poses_input[1:])

        images_input = TF.normalize(images_input, IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)

        # resize render ground-truth images, range still in [0, 1]
        results['images_output'] = F.interpolate(images, size=(self.opt.output_size, self.opt.output_size), mode='bilinear', align_corners=False) # [V, C, output_size, output_size]
        results['masks_output'] = F.interpolate(masks.unsqueeze(1), size=(self.opt.output_size, self.opt.output_size), mode='bilinear', align_corners=False) # [V, 1, output_size, output_size]

        # build rays for input views
        rays_embeddings = []
        for i in range(self.opt.num_input_views):
            rays_o, rays_d = get_rays(cam_poses_input[i], self.opt.input_size, self.opt.input_size, self.opt.fovy) # [h, w, 3]
            rays_plucker = torch.cat([torch.cross(rays_o, rays_d, dim=-1), rays_d], dim=-1) # [h, w, 6]
            rays_embeddings.append(rays_plucker)

     
        rays_embeddings = torch.stack(rays_embeddings, dim=0).permute(0, 3, 1, 2).contiguous() # [V, 6, h, w]
        final_input = torch.cat([images_input, rays_embeddings], dim=1) # [V=4, 9, H, W]
        results['input'] = final_input

        # opengl to colmap camera for gaussian renderer
        cam_poses[:, :3, 1:3] *= -1 # invert up & forward direction
        
        # cameras needed by gaussian rasterizer
        cam_view = torch.inverse(cam_poses).transpose(1, 2) # [V, 4, 4]
        cam_view_proj = cam_view @ self.proj_matrix # [V, 4, 4]
        cam_pos = - cam_poses[:, :3, 3] # [V, 3]
        
        results['cam_view'] = cam_view
        results['cam_view_proj'] = cam_view_proj
        results['cam_pos'] = cam_pos

        return results
    


class MixDataset(Dataset):

    def _warn(self):
        raise NotImplementedError('this dataset is just an example and cannot be used directly, you should modify it to your own setting! (search keyword TODO)')

    def __init__(self, opt: Options, training=True):
        
        self.opt = opt
        self.training = training

        # TODO: load the list of objects for training
        self.items = []
        with open('/data/gyj/gobjaverse/gobj_merged.json', 'r') as f:
            obj_list = json.load(f)
            for obj in obj_list:

                self.items.append({
                    'p':os.path.join('/data/gyj/gobjaverse', obj),
                    'type':'gobj',
                    })

        for sub in os.listdir("/data/gyj/Hi3D-Official/datas/OBJAVERSE-LVIS-example/images"):
            sub_path = os.path.join("/data/gyj/Hi3D-Official/datas/OBJAVERSE-LVIS-example/images", sub)
            self.items.append({
                'p':sub_path,
                'type':'metahuman',
                })
        import random
        random.shuffle(self.items)

        # naive split
        if self.training:
            self.items = self.items[:-self.opt.batch_size]
        else:
            self.items = self.items[-self.opt.batch_size:]
        
        # default camera intrinsics
        self.tan_half_fov = np.tan(0.5 * np.deg2rad(self.opt.fovy))
        self.proj_matrix = torch.zeros(4, 4, dtype=torch.float32)
        self.proj_matrix[0, 0] = 1 / self.tan_half_fov
        self.proj_matrix[1, 1] = 1 / self.tan_half_fov
        self.proj_matrix[2, 2] = (self.opt.zfar + self.opt.znear) / (self.opt.zfar - self.opt.znear)
        self.proj_matrix[3, 2] = - (self.opt.zfar * self.opt.znear) / (self.opt.zfar - self.opt.znear)
        self.proj_matrix[2, 3] = 1

                # elevations and azimuths
        self.elevations = []
        self.azimuths = [0.0, 4.04, 15.47, 33.22, 56.25, 83.5, 113.91, 146.43, 180, -146.43, -113.91, -83.5, -56.25, -33.22, -15.47, -4.04] * 6
        # self.azimuths = [0.0, 22.5, 45, 67.5, 90, 112.5, 135, 157.5, 180, 202.5, 225, 247.5, 270, 292.5, 315, 337.5] * 6
        for i in range(6):
            ele = [-10 + 10 * i] * 16
            self.elevations.extend(ele)


    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):

        uid = self.items[idx]['p']
        class_type = self.items[idx]['type']
        results = {}

        # load num_views images
        images = []
        masks = []
        cam_poses = []
        
        vid_cnt = 0

        if class_type == 'gobj':

            # TODO: choose views, based on your rendering settings
            if self.training:
                # input views are in (36, 72), other views are randomly selected
                # vids = np.random.permutation(np.arange(36, 73))[:self.opt.num_input_views].tolist() + np.random.permutation(100).tolist()
                vids = np.random.permutation(np.arange(0,40)).tolist()
            else:
                # fixed views
                vids = np.arange(0, 20, 30).tolist() + np.arange(40).tolist()
            
            for vid in vids:

                image_path = os.path.join(uid, f'{vid:05d}', f'{vid:05d}.png')
                meta_path = os.path.join(uid, f'{vid:05d}', f'{vid:05d}.json')
                nd_path = os.path.join(uid, f'{vid:05d}', f'{vid:05d}_nd.exr')

                try:
                    image = torch.from_numpy(cv2.imread(image_path, cv2.IMREAD_UNCHANGED).astype(np.float32) / 255) # [512, 512, 4] in [0, 1]
                    
                    with open(meta_path, 'r') as f:
                        meta = json.load(f)
                    c2w = np.eye(4)
                    c2w[:3, 0] = np.array(meta['x'])
                    c2w[:3, 1] = np.array(meta['y'])
                    c2w[:3, 2] = np.array(meta['z'])
                    c2w[:3, 3] = np.array(meta['origin'])
                    c2w = torch.tensor(c2w, dtype=torch.float32).reshape(4, 4)

                except Exception as e:
                    print(f'[WARN] dataset {uid} {vid}: {e}')
                    continue
                
                # TODO: you may have a different camera system
                # blender world + opencv cam --> opengl world & cam
                c2w[1] *= -1
                c2w[[1, 2]] = c2w[[2, 1]]
                c2w[:3, 1:3] *= -1 # invert up and forward direction
            
                image = image.permute(2, 0, 1) # [4, 512, 512]
                mask = image[3:4] # [1, 512, 512]
                image = image[:3] * mask + (1 - mask) # [3, 512, 512], to white bg
                image = image[[2,1,0]].contiguous() # bgr to rgb

                images.append(image)
                masks.append(mask.squeeze(0))
                cam_poses.append(c2w)

                vid_cnt += 1
                if vid_cnt == self.opt.num_views:
                    break

            if vid_cnt < self.opt.num_views:
                print(f'[WARN] dataset {uid}: not enough valid views, only {vid_cnt} views found!')
                n = self.opt.num_views - vid_cnt
                images = images + [images[-1]] * n
                masks = masks + [masks[-1]] * n
                cam_poses = cam_poses + [cam_poses[-1]] * n
            
            images = torch.stack(images, dim=0) # [V, C, H, W]
            masks = torch.stack(masks, dim=0) # [V, H, W]
            cam_poses = torch.stack(cam_poses, dim=0) # [V, 4, 4]

            radius = torch.norm(cam_poses[0, :3, 3])
            cam_poses[:, :3, 3] *= self.opt.cam_radius / radius
        else:
            # TODO: choose views, based on your rendering settings
            if self.training:
                # input views are in (36, 72), other views are randomly selected
                sequence = np.arange(96)
                mask = np.ones_like(sequence, dtype=bool)
                for i in range(6):
                    mask[16 * i + 6: 16 * i + 11] = False
                sequence = sequence[mask]
                vids =  [16, 21, 24, 27] + np.random.permutation(np.arange(16, 32))[:4].tolist() + np.random.permutation(np.arange(96)).tolist()
            else:
                # fixed views
                vids = [16, 21, 24, 27] + np.random.permutation(96).tolist()
                # vids = np.random.permutation(np.arange(0, 96))[:self.opt.num_input_views].tolist()
            
            for vid in vids:

                image_path = os.path.join(uid, f'{vid:04d}.png')

                try:
                    image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
                    image = torch.from_numpy((image).astype(np.float32) / 255) # [512, 512, 4] in [0, 1]
                    c2w = orbit_camera(-self.elevations[vid], self.azimuths[vid], radius=self.opt.cam_radius)
                    c2w = torch.tensor(c2w, dtype=torch.float32)
                except Exception as e:
                    print(f'[WARN] dataset {uid} {vid}: {e}')
                    continue
                # scale up radius to fully use the [-1, 1]^3 space!
                c2w[:3, 3] *= self.opt.cam_radius / 1.5 # 1.5 is the default scale  
            
                image = image.permute(2, 0, 1) # [4, 512, 512]
                mask = image[3:4] # [1, 512, 512]
                image = image[:3] * mask + (1 - mask) # [3, 512, 512], to white bg
                image = image[[2,1,0]].contiguous() # bgr to rgb

                images.append(image)
                masks.append(mask.squeeze(0))
                cam_poses.append(c2w)

                vid_cnt += 1
                if vid_cnt == self.opt.num_views:
                    break

            if vid_cnt < self.opt.num_views:
                print(f'[WARN] dataset {uid}: not enough valid views, only {vid_cnt} views found!')
                n = self.opt.num_views - vid_cnt
                images = images + [images[-1]] * n
                masks = masks + [masks[-1]] * n
                cam_poses = cam_poses + [cam_poses[-1]] * n
            
            images = torch.stack(images, dim=0) # [V, C, H, W]
            masks = torch.stack(masks, dim=0) # [V, H, W]
            cam_poses = torch.stack(cam_poses, dim=0) # [V, 4, 4] 

        # normalized camera feats as in paper (transform the first pose to a fixed position)
        transform = torch.tensor([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, self.opt.cam_radius], [0, 0, 0, 1]], dtype=torch.float32) @ torch.inverse(cam_poses[0])
        cam_poses = transform.unsqueeze(0) @ cam_poses  # [V, 4, 4]

        images_input = F.interpolate(images[:self.opt.num_input_views].clone(), size=(self.opt.input_size, self.opt.input_size), mode='bilinear', align_corners=False) # [V, C, H, W]
        cam_poses_input = cam_poses[:self.opt.num_input_views].clone()

        # data augmentation
        if self.training:
            # apply random grid distortion to simulate 3D inconsistency
            if random.random() < self.opt.prob_grid_distortion:
                images_input[1:] = grid_distortion(images_input[1:])
            # apply camera jittering (only to input!)
            if random.random() < self.opt.prob_cam_jitter:
                cam_poses_input[1:] = orbit_camera_jitter(cam_poses_input[1:])

        images_input = TF.normalize(images_input, IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)

        # resize render ground-truth images, range still in [0, 1]
        results['images_output'] = F.interpolate(images, size=(self.opt.output_size, self.opt.output_size), mode='bilinear', align_corners=False) # [V, C, output_size, output_size]
        results['masks_output'] = F.interpolate(masks.unsqueeze(1), size=(self.opt.output_size, self.opt.output_size), mode='bilinear', align_corners=False) # [V, 1, output_size, output_size]

        # build rays for input views
        rays_embeddings = []
        for i in range(self.opt.num_input_views):
            rays_o, rays_d = get_rays(cam_poses_input[i], self.opt.input_size, self.opt.input_size, self.opt.fovy) # [h, w, 3]
            rays_plucker = torch.cat([torch.cross(rays_o, rays_d, dim=-1), rays_d], dim=-1) # [h, w, 6]
            rays_embeddings.append(rays_plucker)

     
        rays_embeddings = torch.stack(rays_embeddings, dim=0).permute(0, 3, 1, 2).contiguous() # [V, 6, h, w]
        final_input = torch.cat([images_input, rays_embeddings], dim=1) # [V=4, 9, H, W]
        results['input'] = final_input

        # opengl to colmap camera for gaussian renderer
        cam_poses[:, :3, 1:3] *= -1 # invert up & forward direction
        
        # cameras needed by gaussian rasterizer
        cam_view = torch.inverse(cam_poses).transpose(1, 2) # [V, 4, 4]
        cam_view_proj = cam_view @ self.proj_matrix # [V, 4, 4]
        cam_pos = - cam_poses[:, :3, 3] # [V, 3]
        
        results['cam_view'] = cam_view
        results['cam_view_proj'] = cam_view_proj
        results['cam_pos'] = cam_pos

        return results
    

class MetaHumanDataset(Dataset):

    def _warn(self):
        raise NotImplementedError('this dataset is just an example and cannot be used directly, you should modify it to your own setting! (search keyword TODO)')

    def __init__(self, opt: Options, training=True):
        
        self.opt = opt
        self.training = training

        self.items = []
        # for sub in os.listdir("/data/gyj/MetaHuman/images"):
        #     sub_path = os.path.join("/data/gyj/MetaHuman/images", sub)
        #     self.items.append(sub_path)
        # for sub in os.listdir("mini_dataset6"):
        #     sub_path = os.path.join("mini_dataset6", sub)
        self.items.append(self.opt.subject)

        # naive split, select one batch data as test data
        if self.training:
            # self.items = self.items[:-self.opt.batch_size]
            pass
        else:
            self.items = self.items[-self.opt.batch_size:]
        
        # default camera intrinsics
        self.tan_half_fov = np.tan(0.5 * np.deg2rad(self.opt.fovy))
        self.proj_matrix = torch.zeros(4, 4, dtype=torch.float32)
        self.proj_matrix[0, 0] = 1 / self.tan_half_fov
        self.proj_matrix[1, 1] = 1 / self.tan_half_fov
        self.proj_matrix[2, 2] = (self.opt.zfar + self.opt.znear) / (self.opt.zfar - self.opt.znear)
        self.proj_matrix[3, 2] = - (self.opt.zfar * self.opt.znear) / (self.opt.zfar - self.opt.znear)
        self.proj_matrix[2, 3] = 1

        # elevations and azimuths
        self.elevations = []
        # self.azimuths = np.linspace(0, 360, 32,endpoint=False).tolist() * 6
        self.azimuths = [0.0, 4.04, 15.47, 33.22, 56.25, 83.5, 113.91, 146.43, 180, -146.43, -113.91, -83.5, -56.25, -33.22, -15.47, -4.04] * 6
        for i in range(6):
            ele = [-10 + 10 * i] * 16
            self.elevations.extend(ele)

        # self.elevations = [i * 360 / 16 for i in range(16)] * 6
        # self.azimuths = [(-20 + i * 10) for i in range(5) for _ in range(16)]

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):

        uid = self.items[idx]
        results = {}
        # load num_views images
        images = []
        masks = []
        cam_poses = []
        
        vid_cnt = 0

        # TODO: choose views, based on your rendering settings
        if self.training:
            vids = [16, 21, 24, 27] + np.random.permutation(np.arange(16-self.opt.bias,32-self.opt.bias)).tolist()
        else:
            # fixed views
            vids = [16, 21, 24, 27] + np.random.permutation(np.arange(16-self.opt.bias,32-self.opt.bias)).tolist()
            # vids = np.random.permutation(np.arange(0, 96))[:self.opt.num_input_views].tolist()
        
        for vid in vids:
            iddx = vid+self.opt.bias
            if iddx <= 15:
                iddx += 16
            elif iddx >= 32:
                iddx -= 16
            image_path = os.path.join(uid, "high_stage", f'{iddx:04d}.png')
            try:
                image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
                image = torch.from_numpy((image).astype(np.float32) / 255) # [512, 512, 4] in [0, 1]
                c2w = orbit_camera(-self.elevations[vid], self.azimuths[vid], radius=self.opt.cam_radius)
                # c2w = np.array(camera_list[vid]['c2w'])
                # c2w[..., 1:3] *= -1
                c2w = torch.tensor(c2w, dtype=torch.float32)
            except Exception as e:
                print(f'[WARN] dataset {uid} {vid}: {e}')
                continue
            # scale up radius to fully use the [-1, 1]^3 space!
            c2w[:3, 3] *= self.opt.cam_radius / 1.5 # 1.5 is the default scale  
          
            image = image.permute(2, 0, 1) # [4, 512, 512]
            mask = image[3:4] # [1, 512, 512]
            image = image[:3] * mask + (1 - mask) # [3, 512, 512], to white bg
            image = image[[2,1,0]].contiguous() # bgr to rgb

            images.append(image)
            masks.append(mask.squeeze(0))
            cam_poses.append(c2w)

            vid_cnt += 1
            if vid_cnt == self.opt.num_views:
                break

        if vid_cnt < self.opt.num_views:
            print(f'[WARN] dataset {uid}: not enough valid views, only {vid_cnt} views found!')
            n = self.opt.num_views - vid_cnt
            images = images + [images[-1]] * n
            masks = masks + [masks[-1]] * n
            cam_poses = cam_poses + [cam_poses[-1]] * n
          
        images = torch.stack(images, dim=0) # [V, C, H, W]
        masks = torch.stack(masks, dim=0) # [V, H, W]
        cam_poses = torch.stack(cam_poses, dim=0) # [V, 4, 4]

        # normalized camera feats as in paper (transform the first pose to a fixed position)
        transform = torch.tensor([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, self.opt.cam_radius], [0, 0, 0, 1]], dtype=torch.float32) @ torch.inverse(cam_poses[0])
        cam_poses = transform.unsqueeze(0) @ cam_poses  # [V, 4, 4]

        images_input = F.interpolate(images[:self.opt.num_input_views].clone(), size=(self.opt.input_size, self.opt.input_size), mode='bilinear', align_corners=False) # [V, C, H, W]
        cam_poses_input = cam_poses[:self.opt.num_input_views].clone()

        # data augmentation
        if self.training:
            # apply random grid distortion to simulate 3D inconsistency
            if random.random() < self.opt.prob_grid_distortion:
                images_input[1:] = grid_distortion(images_input[1:])
            # apply camera jittering (only to input!)
            if random.random() < self.opt.prob_cam_jitter:
                cam_poses_input[1:] = orbit_camera_jitter(cam_poses_input[1:])

        images_input = TF.normalize(images_input, IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)

        # resize render ground-truth images, range still in [0, 1]
        results['images_output'] = F.interpolate(images, size=(self.opt.output_size, self.opt.output_size), mode='bilinear', align_corners=False) # [V, C, output_size, output_size]
        results['masks_output'] = F.interpolate(masks.unsqueeze(1), size=(self.opt.output_size, self.opt.output_size), mode='bilinear', align_corners=False) # [V, 1, output_size, output_size]

        # build rays for input views
        rays_embeddings = []
        for i in range(self.opt.num_input_views):
            rays_o, rays_d = get_rays(cam_poses_input[i], self.opt.input_size, self.opt.input_size, self.opt.fovy) # [h, w, 3]
            rays_plucker = torch.cat([torch.cross(rays_o, rays_d, dim=-1), rays_d], dim=-1) # [h, w, 6]
            rays_embeddings.append(rays_plucker)

     
        rays_embeddings = torch.stack(rays_embeddings, dim=0).permute(0, 3, 1, 2).contiguous() # [V, 6, h, w]
        final_input = torch.cat([images_input, rays_embeddings], dim=1) # [V=4, 9, H, W]
        results['input'] = final_input

        # opengl to colmap camera for gaussian renderer
        cam_poses[:, :3, 1:3] *= -1 # invert up & forward direction
        
        # cameras needed by gaussian rasterizer
        cam_view = torch.inverse(cam_poses).transpose(1, 2) # [V, 4, 4]
        cam_view_proj = cam_view @ self.proj_matrix # [V, 4, 4]
        cam_pos = - cam_poses[:, :3, 3] # [V, 3]
        
        results['cam_view'] = cam_view
        results['cam_view_proj'] = cam_view_proj
        results['cam_pos'] = cam_pos

        return results

class HG3DDataset(Dataset):

    def _warn(self):
        raise NotImplementedError('this dataset is just an example and cannot be used directly, you should modify it to your own setting! (search keyword TODO)')

    def __init__(self, opt: Options, training=True):
        
        self.opt = opt
        self.training = training

        self.items = []
        # for sub in os.listdir("/data/gyj/MetaHuman/images"):
        #     sub_path = os.path.join("/data/gyj/MetaHuman/images", sub)
        #     self.items.append(sub_path)
        # for sub in os.listdir("mini_dataset6"):
        #     sub_path = os.path.join("mini_dataset6", sub)
        self.items.append(self.opt.subject)

        # naive split, select one batch data as test data
        if self.training:
            # self.items = self.items[:-self.opt.batch_size]
            pass
        else:
            self.items = self.items[-self.opt.batch_size:]
        
        # default camera intrinsics
        self.tan_half_fov = np.tan(0.5 * np.deg2rad(self.opt.fovy))
        self.proj_matrix = torch.zeros(4, 4, dtype=torch.float32)
        self.proj_matrix[0, 0] = 1 / self.tan_half_fov
        self.proj_matrix[1, 1] = 1 / self.tan_half_fov
        self.proj_matrix[2, 2] = (self.opt.zfar + self.opt.znear) / (self.opt.zfar - self.opt.znear)
        self.proj_matrix[3, 2] = - (self.opt.zfar * self.opt.znear) / (self.opt.zfar - self.opt.znear)
        self.proj_matrix[2, 3] = 1

        self.azimuths = [i * 360 / 16 for i in range(16)] * 5
        self.elevations = [(-20 + i * 10) for i in range(5) for _ in range(16)]

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):

        uid = self.items[idx]
        results = {}
        # load num_views images
        images = []
        masks = []
        cam_poses = []
        
        vid_cnt = 0

        # TODO: choose views, based on your rendering settings
        if self.training:
            vids = [32, 36, 40, 44] + np.random.permutation(np.arange(0,80)).tolist()
        else:
            # fixed views
            vids = [32, 36, 40, 44] + np.random.permutation(np.arange(0,80)).tolist()
            # vids = np.random.permutation(np.arange(0, 96))[:self.opt.num_input_views].tolist()
        
        for vid in vids:
            iddx = vid+self.opt.bias
            image_path = os.path.join(uid, f'{iddx:04d}.png')

            try:
                image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
                image = torch.from_numpy((image).astype(np.float32) / 255) # [512, 512, 4] in [0, 1]
                c2w = orbit_camera(-self.elevations[vid], self.azimuths[vid], radius=self.opt.cam_radius)
                # c2w = np.array(camera_list[vid]['c2w'])
                # c2w[..., 1:3] *= -1
                c2w = torch.tensor(c2w, dtype=torch.float32)
            except Exception as e:
                print(f'[WARN] dataset {uid} {vid}: {e}')
                continue
            # scale up radius to fully use the [-1, 1]^3 space!
            c2w[:3, 3] *= self.opt.cam_radius / 1.5 # 1.5 is the default scale  
          
            image = image.permute(2, 0, 1) # [4, 512, 512]
            mask = image[3:4] # [1, 512, 512]
            image = image[:3] * mask + (1 - mask) # [3, 512, 512], to white bg
            image = image[[2,1,0]].contiguous() # bgr to rgb

            images.append(image)
            masks.append(mask.squeeze(0))
            cam_poses.append(c2w)

            vid_cnt += 1
            if vid_cnt == self.opt.num_views:
                break

        if vid_cnt < self.opt.num_views:
            print(f'[WARN] dataset {uid}: not enough valid views, only {vid_cnt} views found!')
            n = self.opt.num_views - vid_cnt
            images = images + [images[-1]] * n
            masks = masks + [masks[-1]] * n
            cam_poses = cam_poses + [cam_poses[-1]] * n
          
        images = torch.stack(images, dim=0) # [V, C, H, W]
        masks = torch.stack(masks, dim=0) # [V, H, W]
        cam_poses = torch.stack(cam_poses, dim=0) # [V, 4, 4]

        # normalized camera feats as in paper (transform the first pose to a fixed position)
        transform = torch.tensor([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, self.opt.cam_radius], [0, 0, 0, 1]], dtype=torch.float32) @ torch.inverse(cam_poses[0])
        cam_poses = transform.unsqueeze(0) @ cam_poses  # [V, 4, 4]

        images_input = F.interpolate(images[:self.opt.num_input_views].clone(), size=(self.opt.input_size, self.opt.input_size), mode='bilinear', align_corners=False) # [V, C, H, W]
        cam_poses_input = cam_poses[:self.opt.num_input_views].clone()

        # data augmentation
        # if self.training:
        #     # apply random grid distortion to simulate 3D inconsistency
        #     if random.random() < self.opt.prob_grid_distortion:
        #         images_input[1:] = grid_distortion(images_input[1:])
        #     # apply camera jittering (only to input!)
        #     if random.random() < self.opt.prob_cam_jitter:
        #         cam_poses_input[1:] = orbit_camera_jitter(cam_poses_input[1:])

        images_input = TF.normalize(images_input, IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)

        # resize render ground-truth images, range still in [0, 1]
        results['images_output'] = F.interpolate(images, size=(self.opt.output_size, self.opt.output_size), mode='bilinear', align_corners=False) # [V, C, output_size, output_size]
        results['masks_output'] = F.interpolate(masks.unsqueeze(1), size=(self.opt.output_size, self.opt.output_size), mode='bilinear', align_corners=False) # [V, 1, output_size, output_size]

        # build rays for input views
        rays_embeddings = []
        for i in range(self.opt.num_input_views):
            rays_o, rays_d = get_rays(cam_poses_input[i], self.opt.input_size, self.opt.input_size, self.opt.fovy) # [h, w, 3]
            rays_plucker = torch.cat([torch.cross(rays_o, rays_d, dim=-1), rays_d], dim=-1) # [h, w, 6]
            rays_embeddings.append(rays_plucker)

     
        rays_embeddings = torch.stack(rays_embeddings, dim=0).permute(0, 3, 1, 2).contiguous() # [V, 6, h, w]
        final_input = torch.cat([images_input, rays_embeddings], dim=1) # [V=4, 9, H, W]
        results['input'] = final_input

        # opengl to colmap camera for gaussian renderer
        cam_poses[:, :3, 1:3] *= -1 # invert up & forward direction
        
        # cameras needed by gaussian rasterizer
        cam_view = torch.inverse(cam_poses).transpose(1, 2) # [V, 4, 4]
        cam_view_proj = cam_view @ self.proj_matrix # [V, 4, 4]
        cam_pos = - cam_poses[:, :3, 3] # [V, 3]
        
        results['cam_view'] = cam_view
        results['cam_view_proj'] = cam_view_proj
        results['cam_pos'] = cam_pos

        return results
    
class Portrait3DDataset(Dataset):

    def _warn(self):
        raise NotImplementedError('this dataset is just an example and cannot be used directly, you should modify it to your own setting! (search keyword TODO)')

    def __init__(self, opt: Options, training=True):
        
        self.opt = opt
        self.training = training

        self.items = []
        # for sub in os.listdir("/data/gyj/MetaHuman/images"):
        #     sub_path = os.path.join("/data/gyj/MetaHuman/images", sub)
        #     self.items.append(sub_path)
        for sub in os.listdir("mini_dataset4"):
            sub_path = os.path.join("mini_dataset4", sub)
            self.items.append(sub_path)

        # naive split, select one batch data as test data
        if self.training:
            # self.items = self.items[:-self.opt.batch_size]
            pass
        else:
            self.items = self.items[-self.opt.batch_size:]
        
        # default camera intrinsics
        self.tan_half_fov = np.tan(0.5 * np.deg2rad(self.opt.fovy))
        self.proj_matrix = torch.zeros(4, 4, dtype=torch.float32)
        self.proj_matrix[0, 0] = 1 / self.tan_half_fov
        self.proj_matrix[1, 1] = 1 / self.tan_half_fov
        self.proj_matrix[2, 2] = (self.opt.zfar + self.opt.znear) / (self.opt.zfar - self.opt.znear)
        self.proj_matrix[3, 2] = - (self.opt.zfar * self.opt.znear) / (self.opt.zfar - self.opt.znear)
        self.proj_matrix[2, 3] = 1

        # elevations and azimuths
        self.elevations = []
        # self.azimuths = np.linspace(0, 360, 32,endpoint=False).tolist() * 6
        self.azimuths = [0.0, 4.04, 15.47, 33.22, 56.25, 83.5, 113.91, 146.43, 180, -146.43, -113.91, -83.5, -56.25, -33.22, -15.47, -4.04] * 6
        for i in range(6):
            ele = [-10 + 10 * i] * 16
            self.elevations.extend(ele)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):

        uid = self.items[idx]
        results = {}

        # load num_views images
        images = []
        masks = []
        cam_poses = []
        
        vid_cnt = 0

        # TODO: choose views, based on your rendering settings
        if self.training:
            vids = [21, 16, 27, 24] + np.random.permutation(96).tolist()
        else:
            # fixed views
            vids = [21, 16, 27, 24] + np.random.permutation(96).tolist()
            # vids = np.random.permutation(np.arange(0, 96))[:self.opt.num_input_views].tolist()
        
        for vid in vids:

            image_path = os.path.join(uid, f'img_proc_fg_{vid:06d}.png')
            # print(image_path + " " +camera_path)
            try:
                image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
                image = torch.from_numpy((image).astype(np.float32) / 255) # [512, 512, 4] in [0, 1]
                # print("image success")
                c2w = orbit_camera(-self.elevations[vid], self.azimuths[vid], radius=self.opt.cam_radius)
                c2w = torch.tensor(c2w, dtype=torch.float32)
                # print(c2w)
            except Exception as e:
                print(f'[WARN] dataset {uid} {vid}: {e}')
                continue
            # scale up radius to fully use the [-1, 1]^3 space!
            c2w[:3, 3] *= self.opt.cam_radius / 1.5 # 1.5 is the default scale  
          
            image = image.permute(2, 0, 1) # [4, 512, 512]
            mask = image[3:4] # [1, 512, 512]
            image = image[:3] * mask + (1 - mask) # [3, 512, 512], to white bg
            image = image[[2,1,0]].contiguous() # bgr to rgb

            images.append(image)
            masks.append(mask.squeeze(0))
            cam_poses.append(c2w)

            vid_cnt += 1
            if vid_cnt == self.opt.num_views:
                break

        if vid_cnt < self.opt.num_views:
            print(f'[WARN] dataset {uid}: not enough valid views, only {vid_cnt} views found!')
            n = self.opt.num_views - vid_cnt
            images = images + [images[-1]] * n
            masks = masks + [masks[-1]] * n
            cam_poses = cam_poses + [cam_poses[-1]] * n
          
        images = torch.stack(images, dim=0) # [V, C, H, W]
        masks = torch.stack(masks, dim=0) # [V, H, W]
        cam_poses = torch.stack(cam_poses, dim=0) # [V, 4, 4]

        # normalized camera feats as in paper (transform the first pose to a fixed position)
        transform = torch.tensor([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, self.opt.cam_radius], [0, 0, 0, 1]], dtype=torch.float32) @ torch.inverse(cam_poses[0])
        cam_poses = transform.unsqueeze(0) @ cam_poses  # [V, 4, 4]

        images_input = F.interpolate(images[:self.opt.num_input_views].clone(), size=(self.opt.input_size, self.opt.input_size), mode='bilinear', align_corners=False) # [V, C, H, W]
        cam_poses_input = cam_poses[:self.opt.num_input_views].clone()

        # data augmentation
        if self.training:
            # apply random grid distortion to simulate 3D inconsistency
            if random.random() < self.opt.prob_grid_distortion:
                images_input[1:] = grid_distortion(images_input[1:])
            # apply camera jittering (only to input!)
            if random.random() < self.opt.prob_cam_jitter:
                cam_poses_input[1:] = orbit_camera_jitter(cam_poses_input[1:])

        images_input = TF.normalize(images_input, IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)

        # resize render ground-truth images, range still in [0, 1]
        results['images_output'] = F.interpolate(images, size=(self.opt.output_size, self.opt.output_size), mode='bilinear', align_corners=False) # [V, C, output_size, output_size]
        results['masks_output'] = F.interpolate(masks.unsqueeze(1), size=(self.opt.output_size, self.opt.output_size), mode='bilinear', align_corners=False) # [V, 1, output_size, output_size]

        # build rays for input views
        rays_embeddings = []
        for i in range(self.opt.num_input_views):
            rays_o, rays_d = get_rays(cam_poses_input[i], self.opt.input_size, self.opt.input_size, self.opt.fovy) # [h, w, 3]
            rays_plucker = torch.cat([torch.cross(rays_o, rays_d, dim=-1), rays_d], dim=-1) # [h, w, 6]
            rays_embeddings.append(rays_plucker)

     
        rays_embeddings = torch.stack(rays_embeddings, dim=0).permute(0, 3, 1, 2).contiguous() # [V, 6, h, w]
        final_input = torch.cat([images_input, rays_embeddings], dim=1) # [V=4, 9, H, W]
        results['input'] = final_input

        # opengl to colmap camera for gaussian renderer
        cam_poses[:, :3, 1:3] *= -1 # invert up & forward direction
        
        # cameras needed by gaussian rasterizer
        cam_view = torch.inverse(cam_poses).transpose(1, 2) # [V, 4, 4]
        cam_view_proj = cam_view @ self.proj_matrix # [V, 4, 4]
        cam_pos = - cam_poses[:, :3, 3] # [V, 3]
        
        results['cam_view'] = cam_view
        results['cam_view_proj'] = cam_view_proj
        results['cam_pos'] = cam_pos

        return results