from __future__ import print_function, division
import logging

from comet_ml import start

EXP_NAME = "DEBUG"


from fed_entrypoint import load_data
from lib.comet_logger import CometLogger

from torch.cuda.amp import autocast 
import torch.nn as nn

import numpy as np
import cv2
import os
from pathlib import Path
from tqdm import tqdm
from datetime import datetime
# import lpips
from lib.ghg.human_loader import HumanDataset
from lib.ghg.network_train_nightly_ver import GaussianRegressor
from config.default_config import HumanConfig as config
from lib.train_recorder import Logger, file_backup
from lib.ghg.GaussianRender import pts2render, input2Render
from lib.loss import l1_loss, ssim, psnr, LPIPSLoss
from lib.utils import save_image , bytes_to_human, print_memory_usage, annotate_image_with_psnr
from torch.utils.data import Subset

import torch
import torch.optim as optim
from torch.cuda.amp import GradScaler
from torch.utils.data import DataLoader
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
import os
import torch
import numpy as np
import imageio
import random

class Trainer:
    def __init__(self, cfg_file: config, fed=False, experiment=None):
        self.cfg = cfg_file       
        
        self.best_psnr = 0.0
        self.best_lpips = float('inf')  # lower is better for LPIPS       
        
        if self.cfg.comet:
            if experiment is not None:
                self.experiment = experiment
            else:
                self.experiment = start(
                    api_key=os.environ.get("COMET_API_KEY"),
                    project_name="fed-ghg",
                    workspace="pinon"
                )
                self.experiment.set_name(EXP_NAME)
        
        N = self.cfg.subset_size
        #DATASET
        self.model = GaussianRegressor(self.cfg, with_gs_render=True)
        
        if not fed:
            if self.cfg.personalized:
                self.train_loader, self.val_loader = load_data(self.cfg, self.experiment, 0, 1, self.cfg.batch_size)
            elif self.cfg.subset:
                # self.subset_train = Subset(self.train_set, list(range(N)))
                random.seed(23)            
                subset_indices = random.sample(range(len(self.train_set)), N)           
                self.train_set = HumanDataset(self.cfg.dataset, phase='train')
                self.val_set = HumanDataset(self.cfg.dataset, phase='val')

                self.subset_train = Subset(self.train_set, subset_indices)
                self.train_loader = DataLoader(self.subset_train, batch_size=self.cfg.batch_size, shuffle=True,
                                        num_workers=6, pin_memory=True,persistent_workers=True,prefetch_factor=2)
                self.val_loader = DataLoader(self.subset_train, batch_size=1, shuffle=True,
                                        num_workers=4, pin_memory=True, prefetch_factor=2)
                self.len_val = N
            else:
                self.train_set = HumanDataset(self.cfg.dataset, phase='train')
                self.val_set = HumanDataset(self.cfg.dataset, phase='val')

                self.train_loader = DataLoader(self.train_set, batch_size=self.cfg.batch_size, shuffle=True,
                                        num_workers=6, pin_memory=True,persistent_workers=True,prefetch_factor=2)
                self.val_loader = DataLoader(self.val_set, batch_size=1, shuffle=True,
                                        num_workers=4, pin_memory=True, prefetch_factor=2)
                self.len_val = int(len(self.val_loader) / self.val_set.val_boost)  # real length of val set
            
            self.train_iterator = iter(self.train_loader)     
            self.val_iterator = iter(self.val_loader)
            
            self.generator_dict = None
        
        self.model.cuda()    
        self.model.train()
        
        all_params = list(filter(lambda p: p.requires_grad, self.model.parameters()))        
        self.optimizer = optim.AdamW(
            filter(lambda p: p.requires_grad, all_params),
            lr=self.cfg.lr,
            weight_decay=self.cfg.wdecay, eps=1e-8)
        self.scheduler = optim.lr_scheduler.OneCycleLR(self.optimizer, self.cfg.lr, self.cfg.num_steps + 100,
                                                       pct_start=0.01, cycle_momentum=False, anneal_strategy='linear')

        self.total_steps = 0
        
        if self.cfg.raft.mixed_precision:
            self.scaler = GradScaler(enabled=self.cfg.raft.mixed_precision)
        
            
            
        self.foreground_loss = nn.BCELoss()        
        self.lpips_loss = LPIPSLoss()
        # self.lpips_loss_fn = lpips.LPIPS(net='vgg').cuda()

        if self.cfg.restore_ckpt:
            self.load_ckpt(self.cfg.restore_ckpt,self.cfg.generator_ckpt,load_optimizer=False)
        
    def train(self):   
        psnr_list = []
        lpips_list = []
        progress_bar = tqdm(range(self.total_steps, self.cfg.num_steps))
        for _ in progress_bar:           
           
            self.optimizer.zero_grad()
            data = self.fetch_data(phase='train')
            # # print_memory_usage("After train ")
            
            eval = self.total_steps and self.total_steps % self.cfg.record.eval_freq == 0
            with torch.cuda.amp.autocast(enabled=self.cfg.raft.mixed_precision):
                data = self.model(data, eval=eval)
            
            
            # Gaussian Render            
            data = pts2render(data, bg_color=self.cfg.dataset.bg_color)
            # # print_memory_usage("Post Rendering")      
            
            # Multi-view Supervision
            primary_loss = 0
            psnr_value = 0.0
            lpips_value = 0.0
            for idx, novel_view in enumerate(['novel_view_0','novel_view_1','novel_view_2']):
                render_novel = data[novel_view]['img_pred']            
                gt_novel = data[novel_view]['img'].cuda()              
                render_fg = data[novel_view]['alpha_pred']
                gt_fg = data[novel_view]['mask'].cuda()
                                  
                
                psnr_value += psnr(render_novel, gt_novel).mean().double()
                lpips_value = self.lpips_loss(render_novel, gt_novel,scale=True)
                fg_loss = self.foreground_loss(render_fg, gt_fg)
                
                primary_loss += 1.0 * l1_loss(render_novel, gt_novel) + \
                                1.0 * (1.0 - ssim(render_novel, gt_novel)) + \
                                0.02 * fg_loss + \
                                1.0 * lpips_value 
                                
                # if eval and self.cfg.comet:
                #     with torch.no_grad():
                #         render_np = render_novel[0].detach().squeeze().permute(1, 2, 0).cpu().numpy()
                #         gt_np = gt_novel[0].detach().permute(1, 2, 0).cpu().numpy()          
                #         combined_image = np.concatenate((gt_np, render_np), axis=1)
                #         annotated_img = annotate_image_with_psnr(combined_image, psnr_value, fg_loss)
                #         self.experiment.log_image(annotated_img, name=f"Training_Novel_{idx}", step=self.total_steps)

            primary_loss = primary_loss / 3 
            lpips_value = lpips_value / 3.0
            psnr_value = psnr_value / 3.0
            lpips_list.append(lpips_value.item())
            psnr_list.append(psnr_value.item())
            
            
            
              
            
            
            # Combine the losses using the computed weight.
            loss = primary_loss
            
            # Update the tqdm progress bar with current losses
            progress_bar.set_postfix({              
                
                'loss': loss.item()
                
            })
            
            # print( self.total_steps,self.total_steps % self.cfg.record.eval_freq)
            if self.cfg.comet:
                if eval:
                    train_psnr = np.round(np.mean(np.array(psnr_list)), 4)
                    train_lpips = np.round(np.mean(np.array(lpips_list)), 4)
                    
                    self.experiment.log_metric("PSNR", train_psnr, step=self.total_steps)
                    self.experiment.log_metric("LPIPS", train_lpips, step=self.total_steps)
                    
                    
                    self.experiment.log_metric("total_loss", loss.item(), step=self.total_steps)
                    
                    
                    # ---- Check and update best metrics ----
                    if train_psnr > self.best_psnr:
                        self.best_psnr = train_psnr
                        # Save the model weights as best for PSNR
                        self.save_ckpt(save_path=Path(f'{self.cfg.record.ckpt_path}/model_best_psnr.pth'), show_log=True)
                    if train_lpips < self.best_lpips:
                        self.best_lpips = train_lpips
                        # Save the model weights as best for LPIPS
                        self.save_ckpt(save_path=Path(f'{self.cfg.record.ckpt_path}/model_best_lpips.pth'), show_log=True)
                    
                    psnr_list.clear()
                    lpips_list.clear()
            
            if self.total_steps and self.total_steps % self.cfg.record.loss_freq == 0:
                self.save_ckpt(save_path=Path('%s/%s_latest.pth' % (cfg.record.ckpt_path, cfg.name)), show_log=False)

            if self.total_steps and self.total_steps % 10000 == 0:
                self.save_ckpt(save_path=Path('%s/%s_%s.pth' % (
                    cfg.record.ckpt_path, cfg.name, str(self.total_steps))),
                    show_log=False)
            
            if self.cfg.raft.mixed_precision:   
                self.scaler.scale(loss).backward()           
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.scaler.step(self.optimizer)
                self.scheduler.step()
                self.scaler.update()
            else:
                loss.backward()
                self.optimizer.step()
                self.scheduler.step()
            
            if eval:
                self.model.eval()
                self.run_eval()
                self.model.train()

            self.total_steps += 1

        print("FINISHED TRAINING")
        self.save_ckpt(save_path=Path('%s/%s_final.pth' % (cfg.record.ckpt_path, cfg.name)))

    def run_eval(self):

        logging.info(f"Doing validation ...")
        torch.cuda.empty_cache()
        psnr_list = []
        lpips_list = []
        fg_list = []        
        
        
        for eval_idx in range(25):

            data = self.fetch_data(phase='val')
            
            with torch.no_grad():

                data = self.model(data)
                data = pts2render(data, bg_color=self.cfg.dataset.bg_color)
                

                psnr_value = 0.0
                lpips_value = 0.0
                fg_value = 0.0                
                for idx, novel_view in enumerate(['novel_view_0', 'novel_view_1', 'novel_view_2']):
                    render_novel = data[novel_view]['img_pred']
                    gt_novel = data[novel_view]['img'].cuda()
                    render_fg = data[novel_view]['alpha_pred']
                    gt_fg = data[novel_view]['mask'].cuda()
                
                    tmp_psnr = psnr(render_novel, gt_novel).mean().double()  
                    tmp_lpips = self.lpips_loss(render_novel, gt_novel,scale=True)
                    psnr_value += tmp_psnr
                    lpips_value += tmp_lpips
                    # foreground loss
                    tmp_fg = self.foreground_loss(render_fg, gt_fg)
                    fg_value += tmp_fg
                
                    render_novel = render_novel[0].detach().permute(1, 2, 0).cpu().numpy()
                    gt_novel = gt_novel[0].detach().permute(1, 2, 0).cpu().numpy()
                    combined_image = np.concatenate((gt_novel, render_novel), axis=1)
                    annotated_img = annotate_image_with_psnr(combined_image, tmp_psnr,tmp_lpips)
                    # feature = data['input_view_1']['feat_pred'].detach().permute(1, 2, 0).cpu().numpy()
                    # self.experiment.log_image(feature, name=f"Validation_Shared_Feature_{self.total_steps}_{idx}", step=self.total_steps)
                    if self.cfg.comet:
                        self.experiment.log_image(annotated_img, name=f"Novel_Validation_{self.total_steps}_{idx}", step=self.total_steps)
                
                                
                
                lpips_value = lpips_value / 3.0
                psnr_value = psnr_value / 3.0                
                fg_value = fg_value / 3.0
            
                
                
                psnr_list.append(psnr_value.item())
                lpips_list.append(lpips_value.item())
                fg_list.append(fg_value.item())
                
                
                

                if eval_idx == 0:
                    nv_idx = 0
                    for novel_view in ['novel_view_0', 'novel_view_1', 'novel_view_2']:

                        subject_name = data['name'][0]
                        gt = data[novel_view]['img']
                        gt = gt[0].detach().permute(1, 2, 0).cpu().numpy()
                        gt = 255 * gt
                        gt = gt.astype(np.uint8)
                        gt_name = '%s/iter_%s_gt_view_%s_%s.jpg' % (cfg.record.show_path, str(self.total_steps).zfill(7),str(nv_idx),subject_name)
                        imageio.imsave(gt_name, gt)

                        pred = data[novel_view]['img_pred']
                        pred = pred[0].detach().permute(1, 2,0).cpu().numpy()
                        pred = 255 * pred
                        pred = pred.astype(np.uint8)
                        pred_name = '%s/iter_%s_pred_view_%s_%s.jpg' % (cfg.record.show_path, str(self.total_steps).zfill(7),str(nv_idx),subject_name)
                        imageio.imsave(pred_name, pred)


                        gt_mask = data[novel_view]['mask']
                        gt_mask = gt_mask[0].detach().permute(1, 2, 0).cpu().numpy()
                        gt_mask = 255 * gt_mask
                        gt_mask = gt_mask.astype(np.uint8)
                        gt_mask_name = '%s/iter_%s_gt_mask_view_%s_%s.jpg' % (
                        cfg.record.show_path, str(self.total_steps).zfill(7),str(nv_idx),subject_name)
                        imageio.imsave(gt_mask_name, gt_mask[...,0])

                        pred_mask = data[novel_view]['alpha_pred']
                        pred_mask = pred_mask[0].detach().permute(1, 2, 0).cpu().numpy()
                        pred_mask = 255 * pred_mask
                        pred_mask = pred_mask.astype(np.uint8)
                        pred_mask_name = '%s/iter_%s_pred_mask_view_%s_%s.jpg' % (
                        cfg.record.show_path, str(self.total_steps).zfill(7),
                        str(nv_idx),subject_name)
                        imageio.imsave(pred_mask_name, pred_mask[...,0])

                        nv_idx = nv_idx + 1


                    input_views = data['input_view']['img'][0].detach().permute(0,2,3,1).cpu().numpy()
                    input_views = 0.5*(input_views + 1)
                    input_views = 255*input_views
                    input_views = input_views.astype(np.uint8)
                    for input_idx in range(input_views.shape[0]):
                        input_name = '%s/iter_%s_input_%d_%s.jpg' % (cfg.record.show_path, str(self.total_steps).zfill(7),input_idx,subject_name)
                        imageio.imsave(input_name, input_views[input_idx])

                    for out_shell_name in ['in_shell','out_shell_1','out_shell_2','out_shell_3','out_shell_4']:
                        out_shell_uvmap = data[out_shell_name]['rgb_maps'][0].detach().permute(1, 2, 0).cpu().numpy()
                        out_shell_uvmap = 0.5 * (out_shell_uvmap + 1)
                        out_shell_uvmap = 255 * out_shell_uvmap
                        out_shell_uvmap = out_shell_uvmap.astype(np.uint8)
                        out_shell_uvmap_name = '%s/iter_%s_%s_uvmap_%s.jpg' % (
                        cfg.record.show_path, str(self.total_steps).zfill(7),out_shell_name,subject_name)
                        imageio.imsave(out_shell_uvmap_name, out_shell_uvmap)

                    inpaint_input = data['in_shell']['inpaint_input'][0].detach().permute(1, 2, 0).cpu().numpy()
                    inpaint_input = 0.5 * (inpaint_input + 1)
                    inpaint_input = 255 * inpaint_input
                    inpaint_input = inpaint_input.astype(np.uint8)
                    inpaint_input_name = '%s/iter_%s_inpaint_input_%s.jpg' % (cfg.record.show_path, str(self.total_steps).zfill(7),subject_name)
                    imageio.imsave(inpaint_input_name, inpaint_input)

                    inpaint_mask = data['in_shell']['inpaint_mask'][0].detach().permute(1, 2, 0).cpu().numpy()
                    inpaint_mask = 255 * inpaint_mask
                    inpaint_mask = inpaint_mask.astype(np.uint8)
                    inpaint_mask_name = '%s/iter_%s_inpaint_mask_%s.jpg' % (
                    cfg.record.show_path, str(self.total_steps).zfill(7),subject_name)
                    imageio.imsave(inpaint_mask_name, inpaint_mask[...,0])


        val_psnr = np.round(np.mean(np.array(psnr_list)), 4)
        val_fg = np.round(np.mean(np.array(fg_list)), 4)        
        val_lpips = np.round(np.mean(np.array(lpips_list)), 4)

        # print(type(val_aux),type(val_fg),type(val_psnr))
        if self.cfg.comet:
            self.experiment.log_metric("LPIPS Validation", val_lpips, step=self.total_steps)            
            self.experiment.log_metric("Foreground Loss Validation", val_fg, step=self.total_steps)
            self.experiment.log_metric("PSNR Validation", val_psnr, step=self.total_steps)
        

        # self.logger.write_dict( {'val_psnr': val_psnr, 'val_fg': val_fg},write_step=self.total_steps)
        

    def fetch_data(self, phase):
        if phase == 'train':
            try:
                data = next(self.train_iterator)
            except:
                self.train_iterator = iter(self.train_loader)
                data = next(self.train_iterator)
        elif phase == 'val':
            try:
                data = next(self.val_iterator)
            except:
                self.val_iterator = iter(self.val_loader)
                data = next(self.val_iterator)

        for key in data.keys():
            if key in ['pos','outer_pos','outer_pos_1','outer_pos_2','outer_pos_3','outer_pos_4']:
                data[key] = data[key].cuda()
            elif key in ['input_view','novel_view_0','novel_view_1','novel_view_2']:
                for sub_key in data[key].keys():                    
                    if sub_key.startswith("input_view_"):
                        # print(sub_key)
                        for items in data[key][sub_key].keys():
                            data[key][sub_key][items] = data[key][sub_key][items].cuda()
                    else:
                        data[key][sub_key] = data[key][sub_key].cuda()
                        
                    
            # if key in ['input_view_0','input_view_1','input_view_2']:
            #         print(type(data[key]['FovX']))

        return data

    def load_ckpt(self, load_path,inpaintor_path, load_optimizer=False, strict=False):
        assert os.path.exists(load_path)
        logging.info(f"Loading checkpoint from {load_path} ...")
        ckpt = torch.load(load_path, map_location='cuda')
        self.model.load_state_dict(ckpt['network'], strict=strict)
        logging.info(f"Parameter loading done")
        
        # generator_state_dict = torch.load(inpaintor_path,map_location='cuda')
        # generator_prefix = 'generator.'
        # generator_specific_dict = {generator_prefix + k: v for k, v in
        #                            generator_state_dict.items()}
        # missing_keys, unexpected_keys = self.model.load_state_dict(generator_specific_dict, strict=False)
       
        
        if load_optimizer:
            self.total_steps = ckpt['total_steps'] + 1
            # self.logger.total_steps = self.total_steps
            self.optimizer.load_state_dict(ckpt['optimizer'])
            self.scheduler.load_state_dict(ckpt['scheduler'])
            
            logging.info(f"Optimizer loading done")

    def save_ckpt(self, save_path, show_log=True):
        if show_log:
            logging.info(f"Save checkpoint to {save_path} ...")
        torch.save({
            'total_steps': self.total_steps,
            'network': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict()
        }, save_path)


if __name__ == '__main__':

    # logging.basicConfig(level=logging.INFO,
    #                     format='%(asctime)s %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s')

    cfg = config()
    cfg.load("config/config.yaml")
    cfg = cfg.get_cfg()
    cfg.defrost() 
  
    if cfg.debug:
        cfg.exp_name = "DEBUG-"+cfg.name
    else:
        cfg.exp_name = cfg.name
    EXP_NAME = cfg.exp_name
        
    print("Data will be saved in ",cfg.exp_name)
    cfg.record.ckpt_path = "experiments/%s/ckpt" % cfg.exp_name
    cfg.record.show_path = "experiments/%s/show" % cfg.exp_name
    cfg.record.logs_path = "experiments/%s/logs" % cfg.exp_name
    cfg.record.file_path = "experiments/%s/file" % cfg.exp_name
    cfg.freeze()

    for path in [cfg.record.ckpt_path, cfg.record.show_path, cfg.record.logs_path, cfg.record.file_path]:
        Path(path).mkdir(exist_ok=True, parents=True)

    file_backup(cfg.record.file_path, cfg, train_script=os.path.basename(__file__))

    torch.manual_seed(1314)
    np.random.seed(1314)

    trainer = Trainer(cfg)
    trainer.train()
