from pathlib import Path
from typing import Dict, Optional, Tuple
import torch
from yacs.config import CfgNode

from flwr.server.strategy import FedAvg
from flwr.common import parameters_to_ndarrays, Scalar, Parameters
from fed_entrypoint import set_weights
from lib.flower.comet_logger import comet_exp, log_metrics
from train_nightly_ver import Trainer


class CustomFedAvg(FedAvg):
  """
  Just like FedAvg, but with Comet logging.
  """
  
  def __init__(self, cfg: CfgNode, *args, **kwargs):
    super().__init__(*args, **kwargs)
    
    self.ckpt_path = cfg.record.ckpt_path
    self.comet = cfg.comet
    self.cfg = cfg
    self.best_psnr = 0.0 # Higher is better
    self.best_lpips = float("inf") # Lower is better

  def _update_best_metrics(self, round, metrics, parameters):
    """Update best metrics and save model if needed."""
    psnr = metrics["PSNR"]
    lpips = metrics["LPIPS"]

    # If at least one better metric, save model with new parameters (avoids re-initializing trainer)
    if psnr > self.best_psnr or lpips < self.best_lpips:
      param_arrays = parameters_to_ndarrays(parameters)
      
      # Prepare model with current parameters
      trainer = Trainer(self.cfg, fed=True, experiment=comet_exp)
      set_weights(trainer.model, param_arrays)

      # New best PSNR
      if psnr > self.best_psnr:
        self.best_psnr = psnr
        file_name = f"{self.ckpt_path}/model_best_psnr.pth"
        torch.save({
          'total_steps': round,
          'network': trainer.model.state_dict(),
          'optimizer': trainer.optimizer.state_dict(),
          'scheduler': trainer.scheduler.state_dict()
        }, Path(file_name))
        
      # New best LPIPS
      if lpips < self.best_lpips:
        self.best_lpips = lpips
        file_name = f"{self.ckpt_path}/model_best_lpips.pth"
        torch.save({
          'total_steps': round,
          'network': trainer.model.state_dict(),
          'optimizer': trainer.optimizer.state_dict(),
          'scheduler': trainer.scheduler.state_dict()
        }, Path(file_name))

  def evaluate(self, server_round, parameters):
    """Run centralized evaluation if callback was passed to strategy init."""
    loss, metrics = super().evaluate(server_round, parameters)

    round_id = server_round * self.cfg.fl.num_local_steps
    if round_id % self.cfg.fl.server_eval_freq == 0:
      print(f"[Server] Updating best metrics at round {round_id}")
      
      # Update best metrics and save model if needed
      self._update_best_metrics(round_id, metrics, parameters)

      # Log metrics to Comet
      if self.comet:
        log_metrics(metrics, step=round_id, prefix="centralized")
      
    return loss, metrics

  def aggregate_evaluate(self, server_round, results, failures):
    """Aggregate results from federated evaluation."""
    loss, metrics = super().aggregate_evaluate(server_round, results, failures)

    print(f"Aggregate results: {results}")

    round_id = server_round * self.cfg.fl.num_local_steps
    # Log metrics to Comet
    if self.comet:
      log_metrics(metrics, step=round_id, prefix="aggregate")
      
    print(f"AFTER aggregate_evaluate from server: {server_round}")
    return loss, metrics