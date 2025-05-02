from __future__ import print_function, division
import copy 
import argparse
import logging
import numpy as np
import cv2
import os
from pathlib import Path
from tqdm import tqdm

from lib.ghg.human_loader import HumanDataset
from lib.ghg.network_train_nightly_ver import GaussianRegressor
from config.default_config import HumanConfig as config
from lib.ghg.utils import get_eval_calib
from lib.ghg.GaussianRender import pts2render,features2render
from lib.loss import l1_loss, ssim, psnr, LPIPSLoss
from lib.utils import print_memory_usage

from PIL import Image
import torch
import warnings
import torch.optim as optim
warnings.filterwarnings("ignore", category=UserWarning)
import learn2learn as l2l

class HumanRender:
    def __init__(self, cfg_file, phase):
        self.cfg = cfg_file
        self.cfg.defrost()
        self.cfg.comet = False
        self.cfg.defrost()

        self.bs = self.cfg.batch_size
        self.generator_dict = None
        self.dataset_name = 'THuman2.0'
        self.model = GaussianRegressor(self.cfg, with_gs_render=True)
        self.dataset = HumanDataset(self.cfg.dataset, phase=phase)
    
        self.model.cuda()
        
        if self.cfg.restore_ckpt and self.cfg.generator_ckpt:
            self.load_ckpt(self.cfg.restore_ckpt, self.cfg.generator_ckpt,strict=True)
            
        if self.cfg.eval_TTA:
            print("Using TTA")
            self.model.train()
            self.maml = l2l.algorithms.MAML(self.model, lr=0.005, first_order=True,allow_nograd=True)
            self.lpips_loss = LPIPSLoss()
        else:   
            self.model.eval()
        
        
        
        
    def infer_static(self, view_select, novel_view_nums, bg_color):
        total_samples = len(os.listdir(os.path.join(self.cfg.dataset.test_data_root, 'img')))
        
        subangle_map={0:0, 1:2, 2:3, 3:4}  # Maps output indices to camera angles
        desired_cameras = [3, 8, 13]  # Only process these camera angles
        
        for itemno in tqdm(range(total_samples)):
            # for quantitative evaluation            
            item = self.dataset.get_test_item(itemno, source_id=view_select)
            
            
            camera_angle = int(item['name'][-3:])# Extract the camera angle from the sample name.
            # (Assuming the file name ends with a 3-digit camera angle, e.g., "subject_003")            
            
            # Only process if the camera angle is one of 3, 8, or 11.
            if camera_angle not in desired_cameras:
                continue  # Skip this sample to save computation.

            data = self.fetch_data(item) 
            
            if self.cfg.eval_TTA:
                fmodel = self.maml.clone()        
                fmodel.train()
                # Auxilary Branch ->      
                # ------------------------------------------------
                #   INNER LOOP: K adaptation steps on L_aux
                # ------------------------------------------------
                for i in range(5):                   
                    # Forward pass through the functional model for the auxiliary task
                    aux_output = fmodel(data)
                    
                    # Compute the reconstruction loss over the three input views
                    recon_loss = 0.0
                    for idx,input_view in enumerate(['input_view_0', 'input_view_1', 'input_view_2']):
                        recon_image = aux_output['input_view'][input_view]['recon_image'].float()                        
                        gt_input = aux_output['input_view']['img'].squeeze(0)[idx].cuda().detach()
                        
                        loss_l1 = 1.0 * l1_loss(recon_image, gt_input)
                        loss_ssim = 1.0 * (1.0 - ssim(recon_image, gt_input))              
                        loss_lpips = 1.0 *  self.lpips_loss(recon_image, gt_input,scale=False) 
                        
                        recon_loss += loss_l1 + loss_ssim + loss_lpips
                        
                    recon_loss = recon_loss / 3.0
                    # Update the functional model's parameters using the differentiable optimizer
                    fmodel.adapt(recon_loss)  
                    
                     

            with torch.no_grad():                
                if self.cfg.eval_TTA:                    
                    fmodel.eval()
                    data = fmodel(data)
                else:
                    data = self.model(data)
                    
                # for i in range(novel_view_nums):
                i = 0    
                subangle = subangle_map[i]
                data_i = get_eval_calib(data, self.cfg.dataset, data['name'],subangle)

                if bg_color == 'black':
                    data_i = pts2render(data_i, bg_color=[0,0,0],phase='test')
                elif bg_color == 'white':
                    raise NotImplementedError("White background is not implemented")
                    data_i = pts2render(data_i,bg_color=[1,1,1],phase='test')

                render_novel = self.tensor2np(data_i['novel_view']['img_pred'])
                cv2.imwrite(self.cfg.test_out_path + '/%s_novel%s.jpg' % (data_i['name'], str(i).zfill(2)), render_novel)

    def tensor2np(self, img_tensor):
        img_np = img_tensor.permute(0, 2, 3, 1)[0].detach().cpu().numpy()
        img_np = img_np * 255
        img_np = img_np[:, :, ::-1].astype(np.uint8)
        return img_np

    def fetch_data(self, data):

        for key in data.keys():
            if key in ['pos','outer_pos','outer_pos_1','outer_pos_2','outer_pos_3','outer_pos_4']: # position map
                data[key] = data[key].cuda().unsqueeze(0)
            elif key in ['input_view']:
               for sub_key in data[key].keys():                    
                    if sub_key.startswith("input_view_"):
                        # print(sub_key)
                        for items in data[key][sub_key].keys():
                            data[key][sub_key][items] = data[key][sub_key][items].cuda()
                    else:
                        data[key][sub_key] = data[key][sub_key].cuda().unsqueeze(0)


        
        return data

    def load_ckpt(self, regressor_path, inpaintor_path,strict=False):

        assert os.path.exists(regressor_path)
        logging.info(f"Loading checkpoint from {regressor_path} ...")
        ckpt = torch.load(regressor_path, map_location='cuda')

        missing_keys, unexpected_keys = self.model.load_state_dict(ckpt['network'], strict=strict)
        # generator_state_dict = torch.load(inpaintor_path,map_location='cuda')
        # generator_prefix = 'generator.'
        # generator_specific_dict = {generator_prefix + k: v for k, v in
        #                            generator_state_dict.items()}
        # missing_keys, unexpected_keys = self.model.load_state_dict(generator_specific_dict, strict=False)

    def print_gpu_memory(self,msg=""):
        allocated = torch.cuda.memory_allocated() / (1024 ** 2)
        reserved = torch.cuda.memory_reserved() / (1024 ** 2)
        print(f"{msg} - Allocated: {allocated:.1f} MB, Reserved: {reserved:.1f} MB")



if __name__ == '__main__':

    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s')
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_data_root', type=str, required=True)
    parser.add_argument('--regressor_path', type=str, required=True) # Gaussian Regressor
    parser.add_argument('--inpaintor_path', type=str, required=True) # Inpaint Net

    parser.add_argument('--novel_view_nums', type=int, default=4)
    parser.add_argument('--bg_color', type=str, default='black')

    arg = parser.parse_args()

    cfg = config()
    cfg_for_train = os.path.join('./config', 'config.yaml')
    cfg.load(cfg_for_train)
    cfg = cfg.get_cfg()

    cfg.defrost()
    cfg.batch_size = 1
    cfg.dataset.test_data_root = arg.test_data_root
    cfg.dataset.use_processed_data = False
    cfg.restore_ckpt = arg.regressor_path
    cfg.generator_ckpt = arg.inpaintor_path
    if cfg.raft.mixed_precision:
        os.environ['AUTOCAST'] = '1'


    exp_name = cfg.eval_name
    print("saving in ",exp_name)
    '''
    Makes sure inpainting is turned on 
    Auxilary output and GT is in range [-1,1]
    '''
    cfg.test_out_path = os.path.join('./outputs/eval',exp_name)


    Path(cfg.test_out_path).mkdir(exist_ok=True, parents=True)
    cfg.freeze()

    render = HumanRender(cfg, phase='test')

    render.infer_static(view_select=[0, 1], novel_view_nums=arg.novel_view_nums, bg_color=arg.bg_color)
    print("Saved in ",exp_name)