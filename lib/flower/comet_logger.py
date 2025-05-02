
# Logging
import os
from typing import Any, Dict
from comet_ml import start
from comet_ml.integration.pytorch import log_model as log_model_comet

from fed_config import load_config

# Stop comet from exporting conda environment
os.environ['COMET_DISABLE_AUTO_LOGGING'] = '1'
os.environ['COMET_AUTO_LOG_ENV_DETAILS'] = '0'
os.environ['COMET_DISABLE_ENV_INFO'] = '1'
os.environ['RAY_DEDUP_LOGS'] = '0'

# Initialize Comet experiment
comet_exp = start(
  api_key=os.environ['COMET_API_KEY'],
  project_name="fed-ghg",
  workspace="pinon",
)
comet_exp.disable_mp()

cfg = load_config()
comet_exp.log_parameters(cfg.fl)

def log_metrics(metrics: Dict[str, Any], prefix: str = None, step: int = None):
  comet_exp.log_metrics(metrics, prefix=prefix, step=step)
  
def log_model(model, model_name: str, metadata: str = None):
  log_model_comet(comet_exp, model, model_name=model_name, metadata=metadata)

def log_image(image, name: str = None, metadata: str = None, step: int = None):
  comet_exp.log_image(image, name=name, metadata=metadata, step=step)