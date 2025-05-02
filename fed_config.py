import logging
import os
from pathlib import Path
from datetime import datetime
from lib.train_recorder import file_backup
from config.default_config import HumanFLConfig as config
import torch
import numpy as np

def load_config():
  logging.basicConfig(level=logging.INFO,
                      format='%(asctime)s %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s')

  cfg = config()
  cfg.load("config/config_fl.yaml")
  cfg = cfg.get_cfg()

  cfg.defrost()
  dt = datetime.today()
  cfg.exp_name = '%s_%s%s' % (cfg.name, str(dt.month).zfill(2), str(dt.day).zfill(2))
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
  
  return cfg

cfg = load_config()