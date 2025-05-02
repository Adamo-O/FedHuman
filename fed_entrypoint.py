from pathlib import Path
import os
import json

from comet_ml import CometExperiment
import imageio
import numpy as np
from tqdm import tqdm
from collections import OrderedDict
import random

from datasets import Dataset
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

from flwr_datasets.partitioner import DirichletPartitioner
from yacs.config import CfgNode

from lib.ghg.GaussianRender import pts2render
from lib.ghg.human_loader import HumanDataset
from lib.loss import l1_loss, psnr, ssim
from lib.utils import annotate_image_with_psnr, print_memory_usage
import typing
if typing.TYPE_CHECKING:
  import train_nightly_ver

def get_weights(model: nn.Module):
  return [val.cpu().numpy() for _, val in model.state_dict().items()]

def set_weights(model: nn.Module, parameters):
  params_dict = zip(model.state_dict().keys(), parameters)
  state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
  model.load_state_dict(state_dict, strict=True)
  
# Cached datasets and partitioned indices
train_set = None
val_set = None
# Cache for client partitions - will store {client_id: (train_indices, val_indices)}
client_partition_cache = {}

def load_data(cfg: CfgNode, comet_exp: CometExperiment, partition_id: int, num_partitions: int, batch_size: int):
  global train_set
  global val_set
  global client_partition_cache
  
  # Only load datasets once, then just use them in future cases
  if train_set is None or val_set is None:
    # Load the person mapping file
    person_mapping_path = os.path.join('datasets', 'THuman', 'personMapping.json')
    with open(person_mapping_path, 'r') as f:
      person_mapping = json.load(f)
    
    # Extract all person IDs from the mapping and sort them for deterministic ordering
    all_person_ids = sorted([int(pid) for pid in person_mapping.keys()])
    
    # Filter person IDs to ensure they have both training and validation data
    valid_person_ids = []
    for pid in all_person_ids:
      pid_str = str(pid)
      if pid_str in person_mapping:
        # Check if person has both training and validation data
        has_train_data = len(person_mapping[pid_str].get('train_ids', [])) > 0
        has_val_data = len(person_mapping[pid_str].get('val_ids', [])) > 0
        if has_train_data and has_val_data:
          valid_person_ids.append(pid)
    
    print(f"Found {len(valid_person_ids)} people with both training and validation data out of {len(all_person_ids)} total people")
    
    # Determine the person subset size
    person_subset_size = getattr(cfg, 'person_subset_size', len(valid_person_ids))
    if person_subset_size > len(valid_person_ids):
      print(f"WARNING: person_subset_size ({person_subset_size}) is larger than the number of valid people ({len(valid_person_ids)})")
      person_subset_size = len(valid_person_ids)
    
    # Take the first person_subset_size people instead of random sampling
    selected_person_ids = [str(pid) for pid in valid_person_ids[:person_subset_size]]
    print(f"Selected {len(selected_person_ids)} people: {selected_person_ids}")
    
    comet_exp.log_parameters({
      "person_subset_size": person_subset_size,
      "selected_person_ids": selected_person_ids,
      "total_people": len(all_person_ids),
      "valid_people": len(valid_person_ids)
    })
    
    # Load THuman dataset with the selected person subset
    print("Loading full datasets (first time)...")
    train_set = HumanDataset(cfg.dataset, phase='train', person_subset=selected_person_ids, personalized=cfg.personalized)
    val_set = HumanDataset(cfg.dataset, phase='val', person_subset=selected_person_ids, personalized=cfg.personalized)

    # Apply subsetting if enabled and if it should be done at the dataset level
    if cfg.subset and getattr(cfg, 'subset_at_dataset_level', True):
      N = cfg.subset_size
      random.seed(42)
      subset_train_indices = random.sample(range(len(train_set)), N)
      subset_val_indices = random.sample(range(len(val_set)), N)
      
      train_set = Subset(train_set, subset_train_indices).dataset
      val_set = Subset(val_set, subset_val_indices).dataset
      print(f"Applied dataset-level subsetting: {N} samples per dataset")
  
  # Check if we've already partitioned this client's data
  if partition_id in client_partition_cache:
    print(f"Using cached partition for client {partition_id}")
    train_indices, val_indices = client_partition_cache[partition_id]
  else:
    # Create paths for the dataset
    print(f"Processing train and val datasets to create paths for client {partition_id}")
    
    # Partitioning logic - either personalized, IID, or non-IID
    if cfg.personalized:
      train_indices, val_indices = create_person_based_partition(
        train_set, val_set, partition_id, num_partitions)
    elif cfg.fl.iid:
      train_indices, val_indices = create_iid_partition(
        train_set, val_set, partition_id, num_partitions)
    else:
      train_indices, val_indices = create_dirichlet_partition(
        train_set, val_set, partition_id, num_partitions, cfg.fl.alpha)
    
    # Apply subsetting if enabled and if it should be done at the partition level
    if cfg.subset and not getattr(cfg, 'subset_at_dataset_level', True):
      N = cfg.subset_size
      random.seed(42)
      if len(train_indices) > N:
        train_indices = random.sample(train_indices, N)
      if len(val_indices) > N:
        val_indices = random.sample(val_indices, N)
    
    print(f"Client {partition_id}: {len(train_indices)} training samples, {len(val_indices)} validation samples")
    
    comet_exp.log_parameters({
      "train_indices": train_indices,
      "val_indices": val_indices,
    })
    
    # Store the generated indices in cache
    client_partition_cache[partition_id] = (train_indices, val_indices)
  
  train_subset = Subset(train_set, train_indices)
  val_subset = Subset(val_set, val_indices)
  
  train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, 
                          #  num_workers=6, pin_memory=True, prefetch_factor=2, 
                          #  persistent_workers=False)  
                          num_workers=8, pin_memory=True, prefetch_factor=3, 
                                                   persistent_workers=True)  
  val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=True, 
                        #  num_workers=4, pin_memory=True, prefetch_factor=2, 
                        #  persistent_workers=False)  
                        num_workers=6, pin_memory=True, prefetch_factor=3, 
                                                 persistent_workers=True)  
  # print_memory_usage("After dataloader creation")
  return train_loader, val_loader

# Helper functions for different partitioning strategies
def create_person_based_partition(train_set, val_set, partition_id, num_partitions):
  print(f"Creating person-based partition for client {partition_id}")
  
  # Load the person mapping file
  person_mapping_path = os.path.join('datasets', 'THuman', 'personMapping.json')
  with open(person_mapping_path, 'r') as f:
    person_mapping = json.load(f)
  
  # Get the selected person subset from the dataset
  selected_people = train_set.person_subset
  if selected_people is None:
    raise ValueError("Dataset must be initialized with a person_subset for person-based partitioning")
  
  # Calculate people per client
  people_per_client = max(1, len(selected_people) // num_partitions)
  remainder = len(selected_people) % num_partitions
  
  # Calculate start/end indices for this client's persons
  start_idx = partition_id * people_per_client + min(partition_id, remainder)
  end_idx = (partition_id + 1) * people_per_client + min(partition_id + 1, remainder)
  
  # Get person IDs for this client
  client_people = sorted(selected_people[start_idx:end_idx])
  print(f"Client {partition_id} assigned people: {client_people}")
  
  # Create a mapping from sample names to indices for both datasets
  train_indices = []
  val_indices = []
  
  # Process training set indices
  for person_id in client_people:
    if person_id in person_mapping:
      # Get all subject IDs for this person from the mapping
      person_subjects = person_mapping[person_id]['train_ids']
      
      # Filter subjects that are in the training set
      valid_subjects = [s for s in person_subjects if s in train_set.sample_id_list]
      
      # Create sample names for each valid subject and angle
      for subject_id in valid_subjects:
        # Assuming 16 angles per subject (0-15)
        for angle in range(16):
          sample_name = f"{subject_id}_{angle:03d}"
          # Find the index of this sample in the dataset
          try:
            idx = train_set.sample_list.index(sample_name)
            train_indices.append(idx)
          except ValueError:
            continue
  
  # Process validation set indices
  for person_id in client_people:
    if person_id in person_mapping:
      # Get all subject IDs for this person from the mapping
      person_subjects = person_mapping[person_id]['val_ids']
      
      # Filter subjects that are in the validation set
      valid_subjects = [s for s in person_subjects if s in val_set.sample_id_list]
      
      # Create sample names for each valid subject and angle
      for subject_id in valid_subjects:
        # Assuming 16 angles per subject (0-15)
        for angle in range(16):
          sample_name = f"{subject_id}_{angle:03d}"
          # Find the index of this sample in the dataset
          try:
            idx = val_set.sample_list.index(sample_name)
            val_indices.append(idx)
          except ValueError:
            continue
            
  return train_indices, val_indices

def create_iid_partition(train_set, val_set, partition_id, num_partitions):
  print(f"Creating IID partition for client {partition_id}")
  random.seed(42)
  
  # Get dataset sizes
  num_train_samples = len(train_set)
  num_val_samples = len(val_set)
  
  # Randomly select indices for training and validation
  train_all_indices = list(range(num_train_samples))
  val_all_indices = list(range(num_val_samples))
  
  random.shuffle(train_all_indices)
  random.shuffle(val_all_indices)
  
  # Partition sizes
  train_partition_size = num_train_samples // num_partitions
  val_partition_size = num_val_samples // num_partitions
  
  # Indices for each partition
  train_start_idx = partition_id * train_partition_size
  train_end_idx = (partition_id + 1) * train_partition_size if partition_id < num_partitions - 1 else num_train_samples
  
  val_start_idx = partition_id * val_partition_size
  val_end_idx = (partition_id + 1) * val_partition_size if partition_id < num_partitions - 1 else num_val_samples
  
  # Indices
  train_indices = train_all_indices[train_start_idx:train_end_idx]
  val_indices = val_all_indices[val_start_idx:val_end_idx]
  
  return train_indices, val_indices

def create_dirichlet_partition(train_set, val_set, partition_id, num_partitions, alpha):
  print(f"Creating Dirichlet partition (α={alpha}) for client {partition_id}")
  
  # Create paths for the dataset
  # TODO: check before running this since it breaks if we're not using a subset
  train_paths = []
  for i in range(len(train_set)):
      sample_data = train_set[i]
      name = sample_data['name']
      subject_id = name.split('_')[0] if '_' in name else name
      
      train_paths.append({
          "name": name,
          "subject_id": subject_id,
          "idx": i
      })
  
  val_paths = []
  for i in range(len(val_set)):
      sample_data = val_set[i]
      name = sample_data['name']
      subject_id = name.split('_')[0] if '_' in name else name
      
      val_paths.append({
          "name": name,
          "subject_id": subject_id,
          "idx": i
      })
  
  # Create Hugging Face datasets with just the paths
  train_path_set = Dataset.from_list(train_paths)
  val_path_set = Dataset.from_list(val_paths)

  # print_memory_usage("After creating path datasets")

  # Get partitioners
  train_partitioner = DirichletPartitioner(
      num_partitions=num_partitions, 
      alpha=alpha,
      partition_by="subject_id",  # Specify the column to partition by
      min_partition_size=10,
      self_balancing=True,
      shuffle=True,
      seed=42
  )
  val_partitioner = DirichletPartitioner(
      num_partitions=num_partitions, 
      alpha=alpha,
      partition_by="subject_id",  # Specify the column to partition by
      min_partition_size=10,
      self_balancing=True,
      shuffle=True,
      seed=42
  )
  
  # Set partitioner datasets 
  train_partitioner.dataset = train_path_set
  val_partitioner.dataset = val_path_set
  
  # Get partition with partition_id 
  train_path_partition = train_partitioner.load_partition(partition_id)
  val_path_partition = val_partitioner.load_partition(partition_id)
  
  # Extract indices from the partitioned datasets
  train_indices = [item['idx'] for item in train_path_partition]
  val_indices = [item['idx'] for item in val_path_partition]
  
  return train_indices, val_indices

val_set_server = None
def load_data_server(cfg: CfgNode):
  global val_set_server
  # print_memory_usage("Before server dataloader creation")
  
  if val_set_server is None:
    # Load the person mapping file to get the same filtered person IDs
    person_mapping_path = os.path.join('datasets', 'THuman', 'personMapping.json')
    with open(person_mapping_path, 'r') as f:
      person_mapping = json.load(f)
    
    # Extract all person IDs from the mapping and sort them for deterministic ordering
    all_person_ids = sorted([int(pid) for pid in person_mapping.keys()])
    
    # Filter person IDs to ensure they have both training and validation data
    valid_person_ids = []
    for pid in all_person_ids:
      pid_str = str(pid)
      if pid_str in person_mapping:
        # Check if person has both training and validation data
        has_train_data = len(person_mapping[pid_str].get('train_ids', [])) > 0
        has_val_data = len(person_mapping[pid_str].get('val_ids', [])) > 0
        if has_train_data and has_val_data:
          valid_person_ids.append(pid)
    
    # Determine the person subset size
    person_subset_size = getattr(cfg, 'person_subset_size', len(valid_person_ids))
    if person_subset_size > len(valid_person_ids):
      person_subset_size = len(valid_person_ids)
    
    # Take the first person_subset_size people
    selected_person_ids = [str(pid) for pid in valid_person_ids[:person_subset_size]]
    
    # Load validation dataset with the selected person subset
    val_set_server = HumanDataset(cfg.dataset, phase='val', person_subset=selected_person_ids, personalized=cfg.personalized)
    
  val_loader = DataLoader(val_set_server, batch_size=cfg.batch_size, shuffle=True, num_workers=4, pin_memory=True, prefetch_factor=2)
  
  # print_memory_usage("After server dataloader creation")
  return val_loader
  
def train(self, trainer: 'train_nightly_ver.Trainer', cfg: CfgNode, model: nn.Module, train_loader: DataLoader, client_id, local_steps: int):
  device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
  model.to(device)
  model.train()
  
  total_steps = self.state.metric_records['progress']['total_steps']
  if total_steps is None:
      total_steps = 0
  
  psnr_list = []
  lpips_list = []
  
  progress_bar = tqdm(range(local_steps))
  for _ in progress_bar:
    total_steps = self.state.metric_records['progress']['total_steps']
    trainer.optimizer.zero_grad()
    data = fetch_data(train_loader)

    eval = total_steps % cfg.fl.client_eval_freq == 0
    with torch.cuda.amp.autocast(enabled=cfg.raft.mixed_precision):
        data = model(data, eval=eval)
        
    #  Gaussian Render
    data = pts2render(data, bg_color=cfg.dataset.bg_color)
    # data = input2render(data, bg_color=cfg.dataset.bg_color)

    # Multi-view Supervision
    primary_loss = 0
    psnr_value = 0.0
    lpips_value = 0.0
    for idx,novel_view in enumerate(['novel_view_0','novel_view_1','novel_view_2']):
        render_novel = data[novel_view]['img_pred']
        gt_novel = data[novel_view]['img'].cuda()      
        render_fg = data[novel_view]['alpha_pred']
        gt_fg = data[novel_view]['mask'].cuda()
        
        # Compute metrics
        tmp_psnr = psnr(render_novel, gt_novel).mean().double()
        psnr_value += tmp_psnr
        lpips_value += trainer.lpips_loss(render_novel, gt_novel, scale=True)
        tmp_fg = trainer.foreground_loss(render_fg, gt_fg)
        
        # Compute loss
        primary_loss += 1.0 * l1_loss(render_novel, gt_novel) + \
                        1.0 * (1.0 - ssim(render_novel, gt_novel)) + \
                        0.02 * tmp_fg + \
                        1.0 * lpips_value
              
        if eval and cfg.comet:
          with torch.no_grad():
            render_np = render_novel[0].detach().squeeze().permute(1, 2, 0).cpu().numpy()
            gt_np = gt_novel[0].detach().permute(1, 2, 0).cpu().numpy()          
            combined_image = np.concatenate((gt_np, render_np), axis=1)
            annotated_img = annotate_image_with_psnr(combined_image, tmp_psnr, tmp_fg)
            trainer.experiment.log_image(annotated_img, name=f"Training_Novel_C{client_id}_{idx}", step=total_steps)

    primary_loss = primary_loss / 3
    lpips_value = lpips_value / 3.0
    psnr_value = psnr_value / 3.0
    
    lpips_list.append(lpips_value.item())
    psnr_list.append(psnr_value.item())
    
    loss = primary_loss
    
    progress_bar.set_postfix({
      "loss": loss.item(),
    })
                
    if cfg.comet and eval:
      train_psnr = np.round(np.mean(np.array(psnr_list)), 4)
      train_lpips = np.round(np.mean(np.array(lpips_list)), 4)
      
      trainer.experiment.log_metrics({
        "PSNR": train_psnr,
        "LPIPS": train_lpips,
        "total_loss": loss.item()
      }, step=total_steps)
      
      
      # ---- Check and update best metrics ----
      # TODO do this in aggregation instead
      if train_psnr > self.best_psnr:
        self.best_psnr = train_psnr
        trainer.save_ckpt(save_path=Path(f'{cfg.record.ckpt_path}/model_best_psnr.pth'), show_log=True)
      if train_lpips < self.best_lpips:
        self.best_lpips = train_lpips
        trainer.save_ckpt(save_path=Path(f'{cfg.record.ckpt_path}/model_best_lpips.pth'), show_log=True)
      
      psnr_list.clear()
      lpips_list.clear()

        
    # Save checkpoint at loss_freq
    # TODO check if works
    if total_steps and total_steps % cfg.record.loss_freq == 0:
        trainer.save_ckpt(save_path=Path('%s/%s_latest.pth' % (cfg.record.ckpt_path, cfg.name)), show_log=False)

    if total_steps and total_steps % 10000 == 0:
      trainer.save_ckpt(save_path=Path('%s/%s_%s.pth' % (cfg.record.ckpt_path, cfg.name, str(total_steps))), show_log=False)
    
    # Backward pass with mixed precision
    if cfg.raft.mixed_precision:
      trainer.scaler.scale(loss).backward()
      trainer.scaler.unscale_(trainer.optimizer)
      torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

      trainer.scaler.step(trainer.optimizer)
      trainer.scheduler.step()
      trainer.scaler.update()
      
    else:
      loss.backward()
      # torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
      trainer.optimizer.step()
      trainer.scheduler.step()

    # TODO maybe do this controlled via flower config instead
    # if eval:
    #     model.eval()
    #     print(f"[Client train {client_id}] Validation")
    #     test(trainer, cfg, model, train_loader, client_id=client_id, round_id=total_steps)
    #     # trainer.run_eval()
    #     model.train()

    if cfg.comet:
      trainer.experiment.log_metric("Total steps", total_steps, step=total_steps)

    self.state.metric_records['progress']['total_steps'] += 1
    # total_steps += 1

  # print_memory_usage("After training")
  print(f"[Client {client_id}] training done")
  # trainer.save_ckpt(save_path=Path('%s/%s_final.pth' % (cfg.record.ckpt_path, cfg.name)))

  return loss.item()
  
def test(trainer: 'train_nightly_ver.Trainer', cfg: CfgNode, model: nn.Module, val_loader: DataLoader, client_id=None, round_id: int=None):
  print(f"Doing validation ...")
  torch.cuda.empty_cache()
  
  if round_id is None:
      round_id = trainer.total_steps
  
  psnr_list = []
  lpips_list = []
  fg_list = []

  # Create a progress bar for validation
  eval_progress = tqdm(range(25), desc="Validation")
  for eval_idx in eval_progress:
      data = fetch_data(val_loader)

      with torch.no_grad():

          data = model(data)
          data = pts2render(data, bg_color=cfg.dataset.bg_color)

          psnr_value = 0.0
          lpips_value = 0.0
          fg_value = 0.0
          for idx, novel_view in enumerate(['novel_view_0', 'novel_view_1', 'novel_view_2']):
            render_novel = data[novel_view]['img_pred']
            gt_novel = data[novel_view]['img'].cuda()

            render_fg = data[novel_view]['alpha_pred']
            gt_fg = data[novel_view]['mask'].cuda()

            tmp_psnr = psnr(render_novel, gt_novel).mean().double()
            tmp_lpips = trainer.lpips_loss(render_novel, gt_novel,scale=True)
            psnr_value += tmp_psnr
            lpips_value += tmp_lpips

            # foreground loss
            tmp_fg = trainer.foreground_loss(render_fg, gt_fg)
            fg_value += tmp_fg
            
            render_novel = render_novel[0].detach().permute(1, 2, 0).cpu().numpy()
            gt_novel = gt_novel[0].detach().permute(1, 2, 0).cpu().numpy()
            combined_image = np.concatenate((gt_novel, render_novel), axis=1)
            annotated_img = annotate_image_with_psnr(combined_image, tmp_psnr, tmp_fg)
            
            if cfg.comet:
              trainer.experiment.log_image(annotated_img, name=f"Novel_Validation{f'_C#{client_id}' if client_id else ''}_{trainer.total_steps}_{idx}", step=round_id)

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
                  gt_name = '%s/iter_%s_gt_view_%s_%s.jpg' % (cfg.record.show_path, str(round_id).zfill(7),str(nv_idx),subject_name)
                  imageio.imsave(gt_name, gt)

                  pred = data[novel_view]['img_pred']
                  pred = pred[0].detach().permute(1, 2,0).cpu().numpy()
                  pred = 255 * pred
                  pred = pred.astype(np.uint8)
                  pred_name = '%s/iter_%s_pred_view_%s_%s.jpg' % (cfg.record.show_path, str(round_id).zfill(7),str(nv_idx),subject_name)
                  imageio.imsave(pred_name, pred)


                  gt_mask = data[novel_view]['mask']
                  gt_mask = gt_mask[0].detach().permute(1, 2, 0).cpu().numpy()
                  gt_mask = 255 * gt_mask
                  gt_mask = gt_mask.astype(np.uint8)
                  gt_mask_name = '%s/iter_%s_gt_mask_view_%s_%s.jpg' % (
                  cfg.record.show_path, str(round_id).zfill(7),str(nv_idx),subject_name)
                  imageio.imsave(gt_mask_name, gt_mask[...,0])

                  pred_mask = data[novel_view]['alpha_pred']
                  pred_mask = pred_mask[0].detach().permute(1, 2, 0).cpu().numpy()
                  pred_mask = 255 * pred_mask
                  pred_mask = pred_mask.astype(np.uint8)
                  pred_mask_name = '%s/iter_%s_pred_mask_view_%s_%s.jpg' % (
                  cfg.record.show_path, str(round_id).zfill(7),
                  str(nv_idx),subject_name)
                  imageio.imsave(pred_mask_name, pred_mask[...,0])

                  nv_idx = nv_idx + 1

              input_views = data['input_view']['img'][0].detach().permute(0,2,3,1).cpu().numpy()
              input_views = 0.5*(input_views + 1)
              input_views = 255*input_views
              input_views = input_views.astype(np.uint8)
              for input_idx in range(input_views.shape[0]):
                  input_name = '%s/iter_%s_input_%d_%s.jpg' % (cfg.record.show_path, str(round_id).zfill(7),input_idx,subject_name)
                  imageio.imsave(input_name, input_views[input_idx])

              for out_shell_name in ['in_shell','out_shell_1','out_shell_2','out_shell_3','out_shell_4']:
                  out_shell_uvmap = data[out_shell_name]['rgb_maps'][0].detach().permute(1, 2, 0).cpu().numpy()
                  out_shell_uvmap = 0.5 * (out_shell_uvmap + 1)
                  out_shell_uvmap = 255 * out_shell_uvmap
                  out_shell_uvmap = out_shell_uvmap.astype(np.uint8)
                  out_shell_uvmap_name = '%s/iter_%s_%s_uvmap_%s.jpg' % (
                  cfg.record.show_path, str(round_id).zfill(7),out_shell_name,subject_name)
                  imageio.imsave(out_shell_uvmap_name, out_shell_uvmap)

              inpaint_input = data['in_shell']['inpaint_input'][0].detach().permute(1, 2, 0).cpu().numpy()
              inpaint_input = 0.5 * (inpaint_input + 1)
              inpaint_input = 255 * inpaint_input
              inpaint_input = inpaint_input.astype(np.uint8)
              inpaint_input_name = '%s/iter_%s_inpaint_input_%s.jpg' % (cfg.record.show_path, str(round_id).zfill(7),subject_name)
              imageio.imsave(inpaint_input_name, inpaint_input)

              inpaint_mask = data['in_shell']['inpaint_mask'][0].detach().permute(1, 2, 0).cpu().numpy()
              inpaint_mask = 255 * inpaint_mask
              inpaint_mask = inpaint_mask.astype(np.uint8)
              inpaint_mask_name = '%s/iter_%s_inpaint_mask_%s.jpg' % (
              cfg.record.show_path, str(round_id).zfill(7),subject_name)
              imageio.imsave(inpaint_mask_name, inpaint_mask[...,0])

  val_psnr = np.round(np.mean(np.array(psnr_list)), 4)
  val_fg = np.round(np.mean(np.array(fg_list)), 4)
  val_lpips = np.round(np.mean(np.array(lpips_list)), 4)

  print(f"Validation Metrics ({round_id}): psnr {val_psnr}, fg {val_fg}, lpips {val_lpips}")

  if cfg.comet:
    if client_id is not None:
      trainer.experiment.log_metrics({ f"PSNR Validation Client {client_id}": val_psnr, f"Foreground Loss Validation Client {client_id}": val_fg, f"LPIPS Validation Client {client_id}": val_lpips }, step=round_id)
    else:
      trainer.experiment.log_metrics({ f"PSNR Validation": val_psnr, f"Foreground Loss Validation": val_fg, "LPIPS Validation": val_lpips }, step=round_id)
  # torch.cuda.empty_cache()
  
  return val_psnr, val_fg, val_lpips
  
def fetch_data(dataloader):
    # Initialize iterator if it doesn't exist or is None
    if not hasattr(dataloader, '_iterator') or dataloader._iterator is None:
        dataloader._iterator = iter(dataloader)
    
    try:
        data = next(dataloader._iterator)
    except StopIteration:
        # Reset iterator when we reach the end of the dataset
        dataloader._iterator = iter(dataloader)
        data = next(dataloader._iterator)
            
    for key in data.keys():
        if key in ['pos','outer_pos','outer_pos_1','outer_pos_2','outer_pos_3','outer_pos_4']:
            data[key] = data[key].cuda()
        elif key in ['input_view','novel_view_0','novel_view_1','novel_view_2']:
            for sub_key in data[key].keys():
                if sub_key.startswith("input_view_"):
                    for items in data[key][sub_key].keys():
                        data[key][sub_key][items] = data[key][sub_key][items].cuda()
                else:
                    data[key][sub_key] = data[key][sub_key].cuda()

    return data
