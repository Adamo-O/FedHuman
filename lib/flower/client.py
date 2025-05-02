from pathlib import Path
import torch
from torch.utils.data import DataLoader
import numpy as np
from fed_entrypoint import get_weights, load_data, set_weights, test, train
from lib.flower.comet_logger import comet_exp
from train_nightly_ver import Trainer
from fed_config import cfg

from flwr.client import ClientApp, NumPyClient
from flwr.common import Context

from typing import Tuple, Dict
from flwr.common import NDArrays, Scalar, MetricRecord, ArrayRecord, Array, ConfigRecord

class FlowerClient(NumPyClient):
  def __init__(self, cfg, context: Context, trainer: Trainer, trainloader: DataLoader, valloader: DataLoader, client_id):
    self.cfg = cfg
    self.state = (
      context.state
    )
    
    # Initialize metric records in client state
    if 'progress' not in self.state.metric_records:
      self.state.metric_records['progress'] = MetricRecord({
        "total_steps": 0,
      })
      
    # Set the experiment name and log parameters on first instance
    if 'config' not in self.state.config_records:
      self.state.config_records['config'] = ConfigRecord()
      config = self.state.config_records['config']
      config['exp_name'] = f"C{client_id}_{cfg.fl.exp_name}"
      comet_exp.set_name(config['exp_name'])
      
      num_partitions = context.node_config['num-partitions']
      comet_exp.log_parameters({"client_id": client_id, "num_partitions": num_partitions})
      
    self.trainer: Trainer = trainer
    self.trainloader = trainloader
    self.valloader = valloader
    self.client_id = client_id
    self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    self.trainer.model.to(self.device)
    self.total_steps = 0
    self.best_psnr = 0.0
    self.best_lpips = float("inf")
    
  def get_parameters(self):
    return [val.cpu().numpy() for _, val in self.trainer.model.state_dict().items()]
  
  def set_parameters(self, parameters):
    params_dict = zip(self.trainer.model.state_dict().keys(), parameters)
    state_dict = {k: torch.Tensor(v) for k, v in params_dict}
    self.trainer.model.load_state_dict(state_dict, strict=True)
    
  def fit(self, parameters, cfg) -> Tuple[NDArrays, int, Dict[str, Scalar]]:
    """Train model locally."""
    print(f"[Client {self.client_id}] Before train")
    
    # Set the global model parameters from the server
    set_weights(self.trainer.model, parameters)
    
    # For personalized federated learning, we need to:
    # 1. Load the client's personalized weights if they exist
    # 2. Train on the client's data
    # 3. Save the updated personalized weights
    if self.cfg.personalized:
      # Try to load personalized weights
      self._load_weights(load_optimizer=True)
      
      # Train on client's data
      loss = train(self, self.trainer, self.cfg, self.trainer.model, self.trainloader, self.client_id, self.cfg.fl.num_local_steps)
      
      # Save the updated personalized weights
      self._save_weights()
      
      # Return the updated weights to the server
      # Note: In personalized FL, we still return the global model parameters
      # The personalization happens locally and is maintained separately
      print(f"[Client {self.client_id}] After train loss: {loss}")
      return get_weights(self.trainer.model), len(self.trainloader.dataset), {"train_loss": loss}
    else:
      # Standard federated learning without personalization
      loss = train(self, self.trainer, self.cfg, self.trainer.model, self.trainloader, self.client_id, self.cfg.fl.num_local_steps)
      print(f"[Client {self.client_id}] After train loss: {loss}")
      return get_weights(self.trainer.model), len(self.trainloader.dataset), {"train_loss": loss}
    
  def evaluate(self, parameters, config) -> Tuple[float, int, Dict[str, Scalar]]:
    """Evaluate model on validation set."""
    print(f"[Client {self.client_id}] Before eval")
    
    # Set the global model parameters from the server
    set_weights(self.trainer.model, parameters)
    
    # For personalized evaluation, load the client's personalized weights
    if self.cfg.personalized:
      self._load_weights(load_optimizer=False)
      print(f"[Client {self.client_id}] Using personalized weights for evaluation")
    
    # Evaluate on the client's validation data
    psnr, fg, lpips = test(self.trainer, self.cfg, self.trainer.model, self.valloader, client_id=self.client_id, round_id=self.state.metric_records['progress'].total_steps)
    
    print(f"[Client {self.client_id}] After eval PSNR: {psnr} FG: {fg} LPIPS: {lpips}")
    
    # Return metrics with client ID for personalized tracking
    metrics = {"PSNR": psnr, "Foreground Loss": fg, "LPIPS": lpips}
    if self.cfg.personalized:
      metrics["client_id"] = self.client_id
    
    return fg, len(self.valloader.dataset), metrics
  
  def _load_selected_modules(self, state_dict, load_optimizer=False, strict=False):
    """Load selected modules from state dict."""
    # Get the current model state dict
    current_state_dict = self.trainer.model.state_dict()
    
    # Create a new state dict with selective loading
    new_state_dict = {}
    for key, val in current_state_dict.items():
      # Determine which module this parameter belongs to and check if it should be personalized
      should_personalize = False
      if key.startswith('img_encoder'):
        should_personalize = self.cfg.fl.personalize_img_encoder
      elif key.startswith('dino_encoder'):
        should_personalize = self.cfg.fl.personalize_dino_encoder
      elif key.startswith('gs_parm_regressor'):
        should_personalize = self.cfg.fl.personalize_gs_parm_regressor
      
      # If the module is marked for personalization and we have personalized weights, use them
      if should_personalize and key in state_dict:
        # print(f"Using personalized weights for parameter: {key}")
        new_state_dict[key] = state_dict[key]
      else:
        # Otherwise, keep the current weights (which are from the server)
        new_state_dict[key] = val
    
    # Load the mixed state dict
    self.trainer.model.load_state_dict(new_state_dict, strict=strict)
    print(f"Module-specific parameter loading from state done")
  
  def _load_weights(self, load_optimizer=False, strict=False):
    """Load weights from previous round."""
    
    if self.cfg.personalized or self.cfg.fl.module_personalization:
      # First try to load from client state
      personalized_key = f'network_params_C{self.client_id}'
      if personalized_key in self.state.array_records:
        print(f"Loading personalized weights from client state for client {self.client_id}")
        network_params = self.state.array_records[personalized_key]
        
        state_dict = {}
        for key, val in network_params.items():
          state_dict[key] = torch.from_numpy(val.numpy())
        
        if self.cfg.fl.module_personalization:
          self._load_selected_modules(state_dict, load_optimizer, strict)
        else:
          # If not using module personalization, load all personalized weights
          self.trainer.model.load_state_dict(state_dict, strict=strict)
          print(f"Personalized parameter loading from state done")
        return
      
      # If not in state, try to load from file
      regressor_path = Path(f'{self.cfg.record.ckpt_path}/{self.cfg.name}_C{self.client_id}_latest.pth')
      
      # If client-specific weights don't exist yet, use the global model
      if not regressor_path.exists():
        print(f"No personalized weights found for client {self.client_id}, using global model")
        return
      
      print(f"Loading personalized checkpoint from {regressor_path} ...")
      ckpt = torch.load(regressor_path, map_location='cuda')
      
      # Load the network weights
      if self.cfg.fl.module_personalization:
        self._load_selected_modules(ckpt['network'], load_optimizer, strict)
      else:
        # If not using module personalization, load all personalized weights
        self.trainer.model.load_state_dict(ckpt['network'], strict=strict)
        print(f"Personalized parameter loading done")
      
      # Load optimizer and scheduler if requested
      if load_optimizer:
        if 'total_steps' in ckpt:
          self.total_steps = ckpt['total_steps'] + 1
        
        if 'optimizer' in ckpt:
          self.trainer.optimizer.load_state_dict(ckpt['optimizer'])
        if 'scheduler' in ckpt:
          self.trainer.scheduler.load_state_dict(ckpt['scheduler'])
        print(f"Optimizer loading done")
    else:
      # First try to load from client state
      if 'network_params' in self.state.array_records:
        print(f"Loading global weights from client state")
        network_params = self.state.array_records['network_params']
        
        state_dict = {}
        for key, val in network_params.items():
          state_dict[key] = torch.from_numpy(val.numpy())
        
        self.trainer.model.load_state_dict(state_dict, strict=strict)
        print(f"Global parameter loading from state done")
        return
      
      # If not in state, try to load from file
      regressor_path = Path(f'{self.cfg.record.ckpt_path}/{self.cfg.name}_latest.pth')
      
      # If server weights don't exist, use the global model
      if not regressor_path.exists():
        print(f"No server weights found, using global model")
        return
      
      print(f"Loading server checkpoint from {regressor_path} ...")
      ckpt = torch.load(regressor_path, map_location='cuda')
      
      # Load the network weights
      self.trainer.model.load_state_dict(ckpt['network'], strict=strict)
      
      # Load optimizer and scheduler if requested
      if load_optimizer:
        if 'total_steps' in ckpt:
          self.total_steps = ckpt['total_steps'] + 1
        
        if 'optimizer' in ckpt:
          self.trainer.optimizer.load_state_dict(ckpt['optimizer'])
        if 'scheduler' in ckpt:
          self.trainer.scheduler.load_state_dict(ckpt['scheduler'])
        print(f"Optimizer loading done")
      
      print(f"Server parameter loading done")
    
  def _save_weights(self, show_log=True):
    """Save weights after training for next round."""
    
    if self.cfg.personalized:
      # Personalized -> save to client-based path
      save_path = Path(f'{self.cfg.record.ckpt_path}/{self.cfg.name}_C{self.client_id}_latest.pth')
    else:
      # Non-personalized -> save to server-based path
      save_path = Path(f'{self.cfg.record.ckpt_path}/{self.cfg.name}_latest.pth')
      
    if show_log:
      print(f"Save checkpoint to {save_path} ...")
    torch.save({
      'total_steps': self.total_steps,
      'network': self.trainer.model.state_dict(),
      'optimizer': self.trainer.optimizer.state_dict(),
      'scheduler': self.trainer.scheduler.state_dict()
    }, save_path)
    
    # Save personalized weights to client state
    param_record = ArrayRecord()
    for key, val in self.trainer.model.state_dict().items():
      param_record[key] = Array.from_numpy_ndarray(val.detach().cpu().numpy())
      
    if self.cfg.personalized:
      # Store in client state with a personalized key
      self.state.array_records[f'network_params_C{self.client_id}'] = param_record
      print(f"Saved personalized weights to client state for client {self.client_id}")
    else:
      # For non-personalized, save to the standard key
      self.state.array_records['network_params'] = param_record
      print(f"Saved global weights to client state")
  
def client_fn(context: Context):
  partition_id = context.node_config['partition-id']
  num_partitions = context.node_config['num-partitions']
  
  # logging.info(f"Initializing client {partition_id}/{num_partitions}")
  print(f"Initializing client {partition_id}/{num_partitions}")
  trainer = Trainer(cfg, fed=True, experiment=comet_exp)
  
  trainloader, valloader = load_data(cfg, comet_exp, partition_id, num_partitions, cfg.batch_size)
  return FlowerClient(cfg, context, trainer, trainloader, valloader, partition_id).to_client()
    
app = ClientApp(client_fn)