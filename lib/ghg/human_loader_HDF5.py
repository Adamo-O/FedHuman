from torch.utils.data import Dataset
import numpy as np
import os
from PIL import Image
import cv2
import torch
from lib.graphics_utils import getWorld2View2, getProjectionMatrix, focal2fov
from pathlib import Path
import logging
import json
from tqdm import tqdm
import imageio
import cv2
import random
import pdb
import h5py

# def read_img(name):
#     img = np.array(Image.open(name))
#     return img

def read_img(name):
    # Read image using OpenCV in color mode.
    img = cv2.imread(name, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Image {name} not found.")
    # Convert the image from BGR to RGB to match the PIL output.
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img

class HumanDataset(Dataset):
    def __init__(self, opt, phase='train'):       
        self.opt = opt
        self.phase = phase

        # If the HDF5 file has only "train" and "val" groups, then
        # force "test"/"freeview" to read from the "val" group:
        if phase in ['test', 'freeview']:
            self.phase_in_h5 = 'val'
        else:
            self.phase_in_h5 = phase

        # Example of also setting data_root if needed:
        if self.phase == 'train':
            self.data_root = os.path.join(opt.data_root, 'train')
        elif self.phase == 'val':
            self.data_root = os.path.join(opt.data_root, 'val')
        elif self.phase in ['test','freeview']:
            self.data_root = opt.test_data_root

        self.num_total_cams = opt.num_total_cams
        # Make sure you define self.resolution if you are using it later:
        self.resolution = 1024

        # HDF5 path
        self.h5_path = os.path.join(opt.data_root, 'dataset.h5')
        self.h5_file = None  # lazy open

        # Now read the sample list from "train" or "val" only:
        import h5py
        with h5py.File(self.h5_path, 'r') as f:
            self.sample_list = sorted(list(f[self.phase_in_h5]['img'].keys()))


        # (Keep your original keys for position maps if needed, since these will be stored as datasets in HDF5.)
        self.smplx_position_map_path = 'position_map_uv_space'
        self.smplx_outer_shell_position_map_path = 'position_map_uv_space_outer_shell_%d'
        self.smplx_visibility_map_path = 'visibility_map_uv_space'
        self.smplx_outer_shell_visibility_map_path = 'visibility_map_uv_space_outer_shell_%d'


    # def load_single_view(self, sample_name, source_id, hr_img=False, require_mask=True):

    #     img_name = self.img_path % (sample_name, source_id)  # 0004_000 sample_name = subject name + angle
    #     image_hr_name = self.img_hr_path % (sample_name, source_id)
    #     mask_name = self.mask_path % (sample_name, source_id)
    #     intr_name = self.intr_path % (sample_name, source_id)
    #     extr_name = self.extr_path % (sample_name, source_id)
    #     visibility_name = self.smplx_visibility_map_path % (sample_name)

    #     intr, extr = np.load(intr_name), np.load(extr_name)
    #     mask, visibility, outer_visibility_1, outer_visibility_2, outer_visibility_3, outer_visibility_4 = None, None, None, None, None, None

    #     if hr_img:
    #         img = read_img(image_hr_name)
    #         intr[:2] *= 2
    #     else:
    #         img = read_img(img_name)

    #     if require_mask:  # input
    #         mask = read_img(mask_name)
    #         visibility = np.load(visibility_name)

    #         outer_visibility_name_1 = self.smplx_outer_shell_visibility_map_path % (1, sample_name)
    #         outer_visibility_1 = np.load(outer_visibility_name_1)

    #         outer_visibility_name_2 = self.smplx_outer_shell_visibility_map_path % (2, sample_name)
    #         outer_visibility_2 = np.load(outer_visibility_name_2)

    #         outer_visibility_name_3 = self.smplx_outer_shell_visibility_map_path % (3, sample_name)
    #         outer_visibility_3 = np.load(outer_visibility_name_3)

    #         outer_visibility_name_4 = self.smplx_outer_shell_visibility_map_path % (4, sample_name)
    #         outer_visibility_4 = np.load(outer_visibility_name_4)

    #     else:
    #         if self.phase != 'test':
    #             mask = read_img(mask_name)

    #     return img, mask, intr, extr, visibility, outer_visibility_1, outer_visibility_2, outer_visibility_3, outer_visibility_4
    
    def load_single_view(self, sample_name, source_id, hr_img=False, require_mask=True):
        self._init_h5()  # ensure the HDF5 file is open
        # Get image from the "img" group. The key naming should match how you stored them.
        img_group = self.h5_file[self.phase_in_h5]['img'][sample_name]
        img_key = f"{source_id}_hr.jpg" if hr_img else f"{source_id}.jpg"
        img = img_group[img_key][()]

        # Get intrinsics and extrinsics from the "parm" group
        parm_group = self.h5_file[self.phase_in_h5]['parm'][sample_name]
        intr = parm_group[f"{source_id}_intrinsic.npy"][()]
        extr = parm_group[f"{source_id}_extrinsic.npy"][()]
        if hr_img:
            intr[:2] *= 2

        mask = None
        visibility = None
        outer_visibility_1 = outer_visibility_2 = outer_visibility_3 = outer_visibility_4 = None
        if require_mask:
            mask = self.h5_file[self.phase_in_h5]['mask'][sample_name][f"{source_id}.png"][()]
            # Here we assume that the key for the numpy arrays is the sample name with .npy appended.
            visibility = self.h5_file[self.phase_in_h5][self.smplx_visibility_map_path][sample_name + ".npy"][()]
            outer_visibility_1 = self.h5_file[self.phase_in_h5][self.smplx_outer_shell_visibility_map_path % 1][sample_name + ".npy"][()]
            outer_visibility_2 = self.h5_file[self.phase_in_h5][self.smplx_outer_shell_visibility_map_path % 2][sample_name + ".npy"][()]
            outer_visibility_3 = self.h5_file[self.phase_in_h5][self.smplx_outer_shell_visibility_map_path % 3][sample_name + ".npy"][()]
            outer_visibility_4 = self.h5_file[self.phase_in_h5][self.smplx_outer_shell_visibility_map_path % 4][sample_name + ".npy"][()]
        else:
            if self.phase != 'test':
                mask = self.h5_file[self.phase]['mask'][sample_name][f"{source_id}.png"][()]

        return img, mask, intr, extr, visibility, outer_visibility_1, outer_visibility_2, outer_visibility_3, outer_visibility_4

    # def get_novel_view_tensor(self, sample_name, view_id):

    #     img, mask, intr, extr, _, _, _, _, _ = self.load_single_view(
    #         sample_name, view_id,
    #         hr_img=False,
    #         require_mask=False)
    #     height, width = img.shape[:2]
    #     img = torch.from_numpy(img).permute(2, 0, 1)
    #     img = img / 255.0

    #     R = np.array(extr[:3, :3], np.float32).reshape(3, 3).transpose(1, 0)
    #     T = np.array(extr[:3, 3], np.float32)

    #     FovX = focal2fov(intr[0, 0], width)
    #     FovY = focal2fov(intr[1, 1], height)
    #     projection_matrix = getProjectionMatrix(znear=self.opt.znear,
    #                                             zfar=self.opt.zfar, K=intr,
    #                                             h=height, w=width).transpose(0,1)
    #     world_view_transform = torch.tensor(
    #         getWorld2View2(R, T, np.array(self.opt.trans),
    #                        self.opt.scale)).transpose(0, 1)
    #     full_proj_transform = (world_view_transform.unsqueeze(0).bmm(
    #         projection_matrix.unsqueeze(0))).squeeze(0)
    #     camera_center = world_view_transform.inverse()[3, :3]

    #     novel_view_data = {
    #         'view_id': torch.IntTensor([view_id]),
    #         'img': img,
    #         'intr': torch.FloatTensor(intr),
    #         'extr': torch.FloatTensor(extr),
    #         'FovX': FovX,
    #         'FovY': FovY,
    #         'width': width,
    #         'height': height,
    #         'world_view_transform': world_view_transform,
    #         'full_proj_transform': full_proj_transform,
    #         'camera_center': camera_center
    #     }

    #     if self.phase != 'test':
    #         mask = torch.from_numpy(mask[..., 0:1]).permute(2, 0, 1).float()
    #         mask = mask / 255.0
    #         mask[mask < 0.5] = 0.0
    #         mask[mask >= 0.5] = 1.0
    #         novel_view_data.update({'mask': mask})
    #         # black background
    #         img = img * mask
    #         novel_view_data.update({'img': img})
    #     return novel_view_data
    
    def get_novel_view_tensor(self, sample_name, view_id):
        img, mask, intr, extr, _, _, _, _, _ = self.load_single_view(
            sample_name, view_id, hr_img=False, require_mask=False)
        height, width = img.shape[:2]
        img = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0

        R = np.array(extr[:3, :3], np.float32).reshape(3, 3).transpose(1, 0)
        T = np.array(extr[:3, 3], np.float32)

        FovX = focal2fov(intr[0, 0], width)
        FovY = focal2fov(intr[1, 1], height)
        projection_matrix = getProjectionMatrix(znear=self.opt.znear, zfar=self.opt.zfar, K=intr, h=height, w=width).transpose(0,1)
        world_view_transform = torch.tensor(getWorld2View2(R, T, np.array(self.opt.trans), self.opt.scale)).transpose(0, 1)
        full_proj_transform = (world_view_transform.unsqueeze(0).bmm(projection_matrix.unsqueeze(0))).squeeze(0)
        camera_center = world_view_transform.inverse()[3, :3]

        novel_view_data = {
            'view_id': torch.IntTensor([view_id]),
            'img': img,
            'intr': torch.FloatTensor(intr),
            'extr': torch.FloatTensor(extr),
            'FovX': FovX,
            'FovY': FovY,
            'width': width,
            'height': height,
            'world_view_transform': world_view_transform,
            'full_proj_transform': full_proj_transform,
            'camera_center': camera_center
        }

        if self.phase != 'test':
            mask = torch.from_numpy(mask[..., 0:1]).permute(2, 0, 1).float() / 255.0
            mask[mask < 0.5] = 0.0
            mask[mask >= 0.5] = 1.0
            novel_view_data.update({'mask': mask})
            img = img * mask
            novel_view_data.update({'img': img})
        return novel_view_data

    def normalize_tensor(self, tensor_input):
        min_val = torch.min(tensor_input)
        max_val = torch.max(tensor_input)

        normalized_tensor = (tensor_input - min_val) / (max_val - min_val)

        return normalized_tensor

    def get_item(self, index, novel_id=None):

        self._init_h5()
        sample_id = index % len(self.sample_list)
        sample_name = self.sample_list[sample_id]
        human = sample_name[:-4]
        angle = int(sample_name[-3:])

        # Load position map from HDF5
        pos_key = f"{human}_{self.resolution}.npy"
        pos = self.h5_file[self.phase_in_h5]['position_map_uv_space'][pos_key][()]
        pos = torch.from_numpy(pos).permute(2, 0, 1)

        # Load outer shell position maps
        outer_pos_1 = torch.from_numpy(self.h5_file[self.phase_in_h5]['position_map_uv_space_outer_shell_1'][f"{human}_1024.npy"][()]).permute(2, 0, 1)
        outer_pos_2 = torch.from_numpy(self.h5_file[self.phase_in_h5]['position_map_uv_space_outer_shell_2'][f"{human}_1024.npy"][()]).permute(2, 0, 1)
        outer_pos_3 = torch.from_numpy(self.h5_file[self.phase_in_h5]['position_map_uv_space_outer_shell_3'][f"{human}_1024.npy"][()]).permute(2, 0, 1)
        outer_pos_4 = torch.from_numpy(self.h5_file[self.phase_in_h5]['position_map_uv_space_outer_shell_4'][f"{human}_1024.npy"][()]).permute(2, 0, 1)

        ### --- sample novel views for multi-view supervision
        NUM_CAMERAS = self.num_total_cams
        camera_angles = [i * (360 / NUM_CAMERAS) for i in range(NUM_CAMERAS)]

        # Select the first camera
        first_camera = angle
        # Possible positions for the second camera (5 or 6 steps away from the first)
        second_camera_options = [(first_camera + 5) % NUM_CAMERAS,
                                 (first_camera + 6) % NUM_CAMERAS]
        second_camera = random.choice(second_camera_options)

        # Possible positions for the third camera (5 or 6 steps away from the second, not considering the first)
        third_camera_options = [(second_camera + 5) % NUM_CAMERAS,
                                (second_camera + 6) % NUM_CAMERAS]
        third_camera = random.choice(
            [option for option in third_camera_options if
             option != first_camera])

        # The selected novel views for the initial set
        novel_view = [first_camera, second_camera, third_camera]

        # find the nearest camera to a given angle
        def find_nearest_camera(target_angle):
            nearest_camera = min(range(NUM_CAMERAS), key=lambda i: abs(camera_angles[i] - target_angle))
            return nearest_camera

        # Sample input views that are between selected novel views
        midway_indices = []
        for i in range(len(novel_view)):
            current_camera = novel_view[i]
            next_camera = novel_view[(i + 1) % len(novel_view)]

            # Calculate the midpoint in terms of camera indices
            if next_camera > current_camera:
                midway_index = (current_camera + next_camera) // 2
            else:
                midway_index = ((current_camera + next_camera + NUM_CAMERAS) // 2) % NUM_CAMERAS

            midway_indices.append(midway_index)

        # Find the nearest camera to each midway point for the additional cameras
        input_view = [find_nearest_camera(camera_angles[midway_index])
                      for midway_index in midway_indices]

        # visibility_tensor is the visibility of the base SMPL surface from input cameras
        img_tensor, mask_tensor, intr_tensor, extr_tensor, visibility_tensor = [], [], [], [], []
        input_view_0 = {}
        input_view_1 = {}
        input_view_2 = {}
        views_list = [input_view_0,input_view_1,input_view_2]
        outer_visibility_1_tensor, outer_visibility_2_tensor, outer_visibility_3_tensor, outer_visibility_4_tensor = [], [], [], []
        for i in range(self.opt.num_inputs):
            input_name = human + '_' + str(input_view[i]).zfill(3)
            img, mask, intr, extr, visibility, outer_visibility_1, outer_visibility_2, outer_visibility_3, outer_visibility_4 = \
                self.load_single_view(input_name, self.opt.source_id[0],
                                      hr_img=False, require_mask=True)
            height, width = img.shape[:2]
            img = torch.from_numpy(img).permute(2, 0, 1)
            img = 2 * (img / 255.0) - 1.0
            mask = torch.from_numpy(mask).permute(2, 0, 1).float()
            mask = mask / 255.0


            mask[mask < 0.5] = 0.0
            mask[mask >= 0.5] = 1.0

            img = img * mask - (1 - mask)
            img_tensor.append(img)
            mask_tensor.append(mask)
            
            #PINONS WORK ----------
            R = np.array(extr[:3, :3], np.float32).reshape(3, 3).transpose(1, 0)
            T = np.array(extr[:3, 3], np.float32)
            
            FovX = focal2fov(intr[0, 0], width)
            FovY = focal2fov(intr[1, 1], height)
            projection_matrix = getProjectionMatrix(znear=self.opt.znear,
                                                zfar=self.opt.zfar, K=intr,
                                                h=height, w=width).transpose(0,1)
            world_view_transform = torch.tensor(
                getWorld2View2(R, T, np.array(self.opt.trans),
                            self.opt.scale)).transpose(0, 1)
            full_proj_transform = (world_view_transform.unsqueeze(0).bmm(
                projection_matrix.unsqueeze(0))).squeeze(0)
            camera_center = world_view_transform.inverse()[3, :3]

            views_list[i]['FovX'] = FovX
            views_list[i]['FovY'] = FovY
            views_list[i]['width'] = width
            views_list[i]['height'] = height
            views_list[i]['world_view_transform'] = world_view_transform
            views_list[i]['full_proj_transform'] = full_proj_transform
            views_list[i]['camera_center'] = camera_center
            
            intr = torch.FloatTensor(intr)
            extr = torch.FloatTensor(extr)
            visibility = torch.FloatTensor(visibility)
            outer_visibility_1 = torch.FloatTensor(outer_visibility_1)
            outer_visibility_2 = torch.FloatTensor(outer_visibility_2)
            outer_visibility_3 = torch.FloatTensor(outer_visibility_3)
            outer_visibility_4 = torch.FloatTensor(outer_visibility_4)

            intr_tensor.append(intr)
            extr_tensor.append(extr)
            visibility_tensor.append(visibility)
            outer_visibility_1_tensor.append(outer_visibility_1)
            outer_visibility_2_tensor.append(outer_visibility_2)
            outer_visibility_3_tensor.append(outer_visibility_3)
            outer_visibility_4_tensor.append(outer_visibility_4)

        img_tensor = torch.stack(img_tensor)
        mask_tensor = torch.stack(mask_tensor)
        intr_tensor = torch.stack(intr_tensor)
        extr_tensor = torch.stack(extr_tensor)
        visibility_tensor = torch.stack(visibility_tensor)
        outer_visibility_1_tensor = torch.stack(outer_visibility_1_tensor)
        outer_visibility_2_tensor = torch.stack(outer_visibility_2_tensor)
        outer_visibility_3_tensor = torch.stack(outer_visibility_3_tensor)
        outer_visibility_4_tensor = torch.stack(outer_visibility_4_tensor)
        
        

        input_view = {'img': img_tensor, 'mask': mask_tensor,
                      'intr': intr_tensor, 'extr': extr_tensor,
                      'visibility': visibility_tensor,
                      'outer_visibility_1': outer_visibility_1_tensor,
                      'outer_visibility_2': outer_visibility_2_tensor,
                      'outer_visibility_3': outer_visibility_3_tensor,
                      'outer_visibility_4': outer_visibility_4_tensor,
                      'input_view_0': input_view_0,
                      'input_view_1': input_view_1,
                      'input_view_2': input_view_2
                      }
        dict_tensor = {'name': sample_name, 'pos': pos,
                       'outer_pos_1': outer_pos_1,
                       'outer_pos_2': outer_pos_2,
                       'outer_pos_3': outer_pos_3,
                       'outer_pos_4': outer_pos_4,
                       'input_view': input_view}

        ###--- multi-view supervision
        if novel_id:
            # subangle
            novel_id = np.random.choice(novel_id, size=3, replace=True)
            sample_name_0 = human + '_' + str(novel_view[0]).zfill(3)
            sample_name_1 = human + '_' + str(novel_view[1]).zfill(3)
            sample_name_2 = human + '_' + str(novel_view[2]).zfill(3)

            dict_tensor.update({
                'novel_view_0': self.get_novel_view_tensor(sample_name_0,
                                                           novel_id[0]),
                'novel_view_1': self.get_novel_view_tensor(sample_name_1,
                                                           novel_id[1]),
                'novel_view_2': self.get_novel_view_tensor(sample_name_2,
                                                           novel_id[2])
            })

        return dict_tensor

    # def get_test_item(self, index, source_id=None):

    #     sample_id = index % len(self.sample_list)
    #     sample_name = self.sample_list[sample_id]
    #     split = sample_name.split('_')
    #     human = split[0]

    #     # load position map
    #     pos_name = self.smplx_position_map_path % (human, self.resolution)
    #     pos = np.load(pos_name)
    #     pos = torch.from_numpy(pos).permute(2, 0, 1)

    #     # load position map of outer shell (+1cm)
    #     outer_pos_1_name = self.smplx_outer_shell_position_map_path % (1, human)
    #     outer_pos_1 = np.load(outer_pos_1_name)
    #     outer_pos_1 = torch.from_numpy(outer_pos_1).permute(2, 0, 1)

    #     # load position map of outer shell (+2cm)
    #     outer_pos_2_name = self.smplx_outer_shell_position_map_path % (2, human)
    #     outer_pos_2 = np.load(outer_pos_2_name)
    #     outer_pos_2 = torch.from_numpy(outer_pos_2).permute(2, 0, 1)

    #     # load position map of outer shell (+3cm)
    #     outer_pos_3_name = self.smplx_outer_shell_position_map_path % (3, human)
    #     outer_pos_3 = np.load(outer_pos_3_name)
    #     outer_pos_3 = torch.from_numpy(outer_pos_3).permute(2, 0, 1)

    #     # load position map of outer shell (+4cm)
    #     outer_pos_4_name = self.smplx_outer_shell_position_map_path % (4, human)
    #     outer_pos_4 = np.load(outer_pos_4_name)
    #     outer_pos_4 = torch.from_numpy(outer_pos_4).permute(2, 0, 1)

    #     input_view = [0, 6, 11]
    #     # if you want to try random inputs, uncomment the following line:
    #     # input_view = np.random.randint(16,size=3)

    #     # visibility_tensor is the visibility of the base SMPL surface from input cameras
    #     input_view_0 = {}
    #     input_view_1 = {}
    #     input_view_2 = {}
    #     views_list = [input_view_0,input_view_1,input_view_2]
    #     img_tensor, mask_tensor, intr_tensor, extr_tensor, visibility_tensor = [], [], [], [], []
    #     outer_visibility_1_tensor, outer_visibility_2_tensor, outer_visibility_3_tensor, outer_visibility_4_tensor = [], [], [], []
    #     for i in range(self.opt.num_inputs):
    #         input_name = human + '_' + str(input_view[i]).zfill(3)
    #         img, mask, intr, extr, visibility, outer_visibility_1, outer_visibility_2, outer_visibility_3, outer_visibility_4 = \
    #             self.load_single_view(input_name, self.opt.source_id[0],
    #                                   hr_img=False, require_mask=True)
    #         height, width = img.shape[:2]
    #         dilate_kernel = np.ones((3, 3), np.uint8)
    #         image_dilated = cv2.dilate(img, dilate_kernel)
    #         mask_dilated = cv2.dilate(mask, dilate_kernel)


    #         mask_eroded = mask
    #         boundary_mask = mask_dilated & (~(mask_eroded))

    #         boundary_dilated_rgb = (boundary_mask / 255.0) * image_dilated + (1 - boundary_mask / 255.0) * img
    #         boundary_dilated_rgb = boundary_dilated_rgb.astype(np.float32)

    #         img = torch.from_numpy(boundary_dilated_rgb).permute(2, 0, 1)
    #         img = 2 * (img / 255.0) - 1.0


    #         mask_dilated = torch.from_numpy(mask_dilated).permute(2, 0, 1).float()
    #         mask_dilated = mask_dilated / 255.0

    #         mask = torch.from_numpy(mask).permute(2, 0, 1).float()
    #         mask = mask / 255.0


    #         mask[mask < 0.5] = 0.0
    #         mask[mask >= 0.5] = 1.0

    #         # set white background
    #         #img = img * mask_dilated + (1 - mask_dilated)
    #         # set black background
    #         img = img * mask_dilated - (1 - mask_dilated)

    #         img_tensor.append(img)
    #         mask_tensor.append(mask)
            
    #          #PINONS WORK ----------
    #         R = np.array(extr[:3, :3], np.float32).reshape(3, 3).transpose(1, 0)
    #         T = np.array(extr[:3, 3], np.float32)
            
    #         FovX = focal2fov(intr[0, 0], width)
    #         FovY = focal2fov(intr[1, 1], height)
    #         projection_matrix = getProjectionMatrix(znear=self.opt.znear,
    #                                             zfar=self.opt.zfar, K=intr,
    #                                             h=height, w=width).transpose(0,1)
    #         world_view_transform = torch.tensor(
    #             getWorld2View2(R, T, np.array(self.opt.trans),
    #                         self.opt.scale)).transpose(0, 1)
    #         full_proj_transform = (world_view_transform.unsqueeze(0).bmm(
    #             projection_matrix.unsqueeze(0))).squeeze(0)
    #         camera_center = world_view_transform.inverse()[3, :3]

    #         views_list[i]['FovX'] = torch.tensor([FovX])
    #         views_list[i]['FovY'] =  torch.tensor([FovY])
    #         views_list[i]['width'] = torch.tensor([width])
    #         views_list[i]['height'] = torch.tensor([height])
    #         views_list[i]['world_view_transform'] = world_view_transform
    #         views_list[i]['full_proj_transform'] = full_proj_transform
    #         views_list[i]['camera_center'] = camera_center

    #         intr = torch.FloatTensor(intr)
    #         extr = torch.FloatTensor(extr)
    #         visibility = torch.FloatTensor(visibility)
    #         outer_visibility_1 = torch.FloatTensor(outer_visibility_1)
    #         outer_visibility_2 = torch.FloatTensor(outer_visibility_2)
    #         outer_visibility_3 = torch.FloatTensor(outer_visibility_3)
    #         outer_visibility_4 = torch.FloatTensor(outer_visibility_4)

    #         intr_tensor.append(intr)
    #         extr_tensor.append(extr)
    #         visibility_tensor.append(visibility)
    #         outer_visibility_1_tensor.append(outer_visibility_1)
    #         outer_visibility_2_tensor.append(outer_visibility_2)
    #         outer_visibility_3_tensor.append(outer_visibility_3)
    #         outer_visibility_4_tensor.append(outer_visibility_4)

    #     img_tensor = torch.stack(img_tensor)
    #     mask_tensor = torch.stack(mask_tensor)
    #     intr_tensor = torch.stack(intr_tensor)
    #     extr_tensor = torch.stack(extr_tensor)
    #     visibility_tensor = torch.stack(visibility_tensor)
    #     outer_visibility_1_tensor = torch.stack(outer_visibility_1_tensor)
    #     outer_visibility_2_tensor = torch.stack(outer_visibility_2_tensor)
    #     outer_visibility_3_tensor = torch.stack(outer_visibility_3_tensor)
    #     outer_visibility_4_tensor = torch.stack(outer_visibility_4_tensor)

    #     input_view = {'img': img_tensor, 'mask': mask_tensor,
    #                   'intr': intr_tensor, 'extr': extr_tensor,
    #                   'visibility': visibility_tensor,
    #                   'outer_visibility_1': outer_visibility_1_tensor,
    #                   'outer_visibility_2': outer_visibility_2_tensor,
    #                   'outer_visibility_3': outer_visibility_3_tensor,
    #                   'outer_visibility_4': outer_visibility_4_tensor,
    #                   'input_view_0': input_view_0,
    #                   'input_view_1': input_view_1,
    #                   'input_view_2': input_view_2,
    #                   }
    #     dict_tensor = {'name': sample_name, 'pos': pos,
    #                    'outer_pos_1': outer_pos_1,
    #                    'outer_pos_2': outer_pos_2,
    #                    'outer_pos_3': outer_pos_3,
    #                    'outer_pos_4': outer_pos_4,                   
    #                    'input_view': input_view}

    #     # TODO
    #     img_len = 1024
    #     novel_dict = {
    #         'height': torch.IntTensor([img_len]),
    #         'width': torch.IntTensor([img_len])
    #     }

    #     dict_tensor.update({
    #         'novel_view': novel_dict
    #     })
    #     return dict_tensor
    
    def get_test_item(self, index, source_id=None):
        self._init_h5()  # Ensure the HDF5 file is open
        sample_id = index % len(self.sample_list)
        sample_name = self.sample_list[sample_id]
        split = sample_name.split('_')
        human = split[0]

        # --- Load position maps from HDF5 ---
        pos_key = f"{human}_{self.resolution}.npy"
        pos = self.h5_file[self.phase_in_h5]['position_map_uv_space'][pos_key][()]
        pos = torch.from_numpy(pos).permute(2, 0, 1)

        outer_pos_1 = torch.from_numpy(
            self.h5_file[self.phase_in_h5]['position_map_uv_space_outer_shell_1'][f"{human}_1024.npy"][()]
        ).permute(2, 0, 1)
        outer_pos_2 = torch.from_numpy(
            self.h5_file[self.phase_in_h5]['position_map_uv_space_outer_shell_2'][f"{human}_1024.npy"][()]
        ).permute(2, 0, 1)
        outer_pos_3 = torch.from_numpy(
            self.h5_file[self.phase_in_h5]['position_map_uv_space_outer_shell_3'][f"{human}_1024.npy"][()]
        ).permute(2, 0, 1)
        outer_pos_4 = torch.from_numpy(
            self.h5_file[self.phase_in_h5]['position_map_uv_space_outer_shell_4'][f"{human}_1024.npy"][()]
        ).permute(2, 0, 1)

        # --- Load input view data using the new HDF5 loader ---
        input_view = [0, 6, 11]  # or randomize if needed
        input_view_0 = {}
        input_view_1 = {}
        input_view_2 = {}
        views_list = [input_view_0, input_view_1, input_view_2]
        img_tensor, mask_tensor, intr_tensor, extr_tensor, visibility_tensor = [], [], [], [], []
        outer_visibility_1_tensor, outer_visibility_2_tensor, outer_visibility_3_tensor, outer_visibility_4_tensor = [], [], [], []
        
        for i in range(self.opt.num_inputs):
            input_name = human + '_' + str(input_view[i]).zfill(3)
            img, mask, intr, extr, visibility, ov1, ov2, ov3, ov4 = \
                self.load_single_view(input_name, self.opt.source_id[0], hr_img=False, require_mask=True)
            
            height, width = img.shape[:2]
            
            # Process boundary: dilate mask and blend
            dilate_kernel = np.ones((3, 3), np.uint8)
            image_dilated = cv2.dilate(img, dilate_kernel)
            mask_dilated = cv2.dilate(mask, dilate_kernel)
            mask_eroded = mask
            boundary_mask = mask_dilated & (~mask_eroded)
            boundary_dilated_rgb = ((boundary_mask / 255.0) * image_dilated +
                                    (1 - boundary_mask / 255.0) * img).astype(np.float32)
            
            img_tensor.append(torch.from_numpy(boundary_dilated_rgb).permute(2, 0, 1).float() * 2 - 1)
            mask_tensor.append(torch.from_numpy(mask).permute(2, 0, 1).float() / 255.0)
            
            R = np.array(extr[:3, :3], np.float32).reshape(3, 3).transpose(1, 0)
            T = np.array(extr[:3, 3], np.float32)
            FovX = focal2fov(intr[0, 0], width)
            FovY = focal2fov(intr[1, 1], height)
            projection_matrix = getProjectionMatrix(znear=self.opt.znear,
                                                    zfar=self.opt.zfar, K=intr,
                                                    h=height, w=width).transpose(0, 1)
            world_view_transform = torch.tensor(getWorld2View2(R, T, np.array(self.opt.trans), self.opt.scale)).transpose(0, 1)
            full_proj_transform = (world_view_transform.unsqueeze(0).bmm(projection_matrix.unsqueeze(0))).squeeze(0)
            camera_center = world_view_transform.inverse()[3, :3]
            
            views_list[i]['FovX'] = torch.tensor([FovX])
            views_list[i]['FovY'] = torch.tensor([FovY])
            views_list[i]['width'] = torch.tensor([width])
            views_list[i]['height'] = torch.tensor([height])
            views_list[i]['world_view_transform'] = world_view_transform
            views_list[i]['full_proj_transform'] = full_proj_transform
            views_list[i]['camera_center'] = camera_center
            
            intr_tensor.append(torch.FloatTensor(intr))
            extr_tensor.append(torch.FloatTensor(extr))
            visibility_tensor.append(torch.FloatTensor(visibility))
            outer_visibility_1_tensor.append(torch.FloatTensor(ov1))
            outer_visibility_2_tensor.append(torch.FloatTensor(ov2))
            outer_visibility_3_tensor.append(torch.FloatTensor(ov3))
            outer_visibility_4_tensor.append(torch.FloatTensor(ov4))
        
        img_tensor = torch.stack(img_tensor)
        mask_tensor = torch.stack(mask_tensor)
        intr_tensor = torch.stack(intr_tensor)
        extr_tensor = torch.stack(extr_tensor)
        visibility_tensor = torch.stack(visibility_tensor)
        outer_visibility_1_tensor = torch.stack(outer_visibility_1_tensor)
        outer_visibility_2_tensor = torch.stack(outer_visibility_2_tensor)
        outer_visibility_3_tensor = torch.stack(outer_visibility_3_tensor)
        outer_visibility_4_tensor = torch.stack(outer_visibility_4_tensor)
        
        input_view_dict = {
            'img': img_tensor,
            'mask': mask_tensor,
            'intr': intr_tensor,
            'extr': extr_tensor,
            'visibility': visibility_tensor,
            'outer_visibility_1': outer_visibility_1_tensor,
            'outer_visibility_2': outer_visibility_2_tensor,
            'outer_visibility_3': outer_visibility_3_tensor,
            'outer_visibility_4': outer_visibility_4_tensor,
            'input_view_0': input_view_0,
            'input_view_1': input_view_1,
            'input_view_2': input_view_2
        }
        
        dict_tensor = {
            'name': sample_name,
            'pos': pos,
            'outer_pos_1': outer_pos_1,
            'outer_pos_2': outer_pos_2,
            'outer_pos_3': outer_pos_3,
            'outer_pos_4': outer_pos_4,
            'input_view': input_view_dict
        }
        
        # Multi-view supervision: add novel view dimensions
        img_len = 1024
        novel_dict = {
            'height': torch.IntTensor([img_len]),
            'width': torch.IntTensor([img_len])
        }
        dict_tensor.update({'novel_view': novel_dict})
        
        return dict_tensor

    def get_item_debug(self, index):
        return self.__getitem__(index)

    def __getitem__(self, index):
        if self.phase == 'train':
            return self.get_item(index, novel_id=self.opt.train_novel_id)
        elif self.phase == 'val':
            return self.get_item(index, novel_id=self.opt.val_novel_id)

    def __len__(self):
        self.train_boost = 50
        self.val_boost = 200
        if self.phase == 'train':
            return len(self.sample_list) * self.train_boost
        elif self.phase == 'val':
            return len(self.sample_list) * self.val_boost
        else:
            return len(self.sample_list)
        
    def _init_h5(self):
        """Lazy opens the HDF5 file. Each worker will call this on first access."""
        if self.h5_file is None:
            self.h5_file = h5py.File(self.h5_path, 'r')
            
            

# class CachedHumanDataset(HumanDataset):
#     def __init__(self, opt, phase='train'):
#         super().__init__(opt, phase)
#         self.cache = {}  # Cache to store already loaded samples

#     def __getitem__(self, index):
#         if index in self.cache:
#             return self.cache[index]
#         sample = super().__getitem__(index)
#         self.cache[index] = sample
#         return sample
    
    