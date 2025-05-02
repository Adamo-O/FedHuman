"""FedHumanTTA: A Flower / PyTorch app."""

from typing import List, Tuple, Dict
from flwr.common import Context, Metrics, ndarrays_to_parameters
from flwr.server import ServerApp, ServerAppComponents, ServerConfig
from numpy import ndarray, number

from torch.utils.data import DataLoader
from fed_entrypoint import get_weights, load_data_server, set_weights, test
from lib.flower.comet_logger import comet_exp
from fed_config import cfg
from lib.flower.strategy import CustomFedAvg
from lib.flower.fedadam import CustomFedAdam
from train_nightly_ver import Trainer

def generate_evaluate_fn(num_local_steps: int, valloader: DataLoader, trainer: Trainer):
    """Generate centralized evaluation function."""
    
    def evaluate(server_round: int, parameters: list[ndarray], _: Dict[str, number]):
        """Evaluate global model on centralized test set."""
        
        round_id = server_round * num_local_steps
        if round_id % trainer.cfg.fl.server_eval_freq == 0:
            print(f"[Server] Evaluating global model at round {round_id}")
        
            set_weights(trainer.model, parameters)
            trainer.model.cuda()
            psnr, fg, lpips = test(trainer, cfg, trainer.model, valloader, round_id=round_id)
            print(f"[Server] Evaluation results: PSNR: {psnr}, FG Loss: {fg}, LPIPS: {lpips}")
            
            return fg, {"PSNR": psnr, "Foreground Loss": fg, "LPIPS": lpips}
        
        print(f"[Server] Skipping evaluation at round {round_id}")
        return 0.0, {"PSNR": 0.0, "Foreground Loss": 0.0, "LPIPS": 100.0}
    
    return evaluate

# Define metric aggregation function
def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    """Aggregate Foreground Loss, PSNR, LPIPS using weighted average."""

    total_examples = sum(num_examples for num_examples, _ in metrics)

    # Weighted sum for PSNR, FG Loss, LPIPS
    weighted_psnr = sum(num_examples * m["PSNR"] for num_examples, m in metrics)
    weighted_fg_loss = sum(num_examples * m["Foreground Loss"] for num_examples, m in metrics)
    weighted_lpips = sum(num_examples * m["LPIPS"] for num_examples, m in metrics)

    # Compute weighted averages
    avg_psnr = weighted_psnr / total_examples
    avg_fg_loss = weighted_fg_loss / total_examples
    avg_lpips = weighted_lpips / total_examples

    return {
        "PSNR": avg_psnr,
        "Foreground Loss": avg_fg_loss,
        "LPIPS": avg_lpips,
    }

def server_fn(context: Context):
    # Read from config
    fraction_fit = context.run_config["fraction-fit"]
    fraction_evaluate = context.run_config["fraction-evaluate"]
    min_available_clients = context.run_config["min-available-clients"]
    min_evaluate_clients = context.run_config["min-evaluate-clients"]
    min_fit_clients = context.run_config["min-fit-clients"]

    # Get number of rounds from total steps / client local steps
    num_rounds = cfg.num_steps // cfg.fl.num_local_steps

    # Initialize model parameters
    comet_exp.set_name(cfg.fl.exp_name)
    comet_exp.log_parameters({
        "fraction_fit": fraction_fit,
        "fraction_evaluate": fraction_evaluate,
        "min_available_clients": min_available_clients,
        "min_evaluate_clients": min_evaluate_clients,
        "min_fit_clients": min_fit_clients,
        "num_rounds": num_rounds,
    })
    trainer = Trainer(cfg, fed=True, experiment=comet_exp)
    ndarrays = get_weights(trainer.model)
    parameters = ndarrays_to_parameters(ndarrays)

    # Get val loader for server (no partitioning necessary)
    valloader = load_data_server(cfg)

    # Define strategy based on configuration
    if cfg.fl.optimizer == "fedavg":
        strategy = CustomFedAvg(
            cfg,
            fraction_fit=fraction_fit,
            fraction_evaluate=fraction_evaluate,
            min_evaluate_clients=min_evaluate_clients,
            min_available_clients=min_available_clients,
            min_fit_clients=min_fit_clients,
            evaluate_metrics_aggregation_fn=weighted_average,
            evaluate_fn=generate_evaluate_fn(cfg.fl.num_local_steps, valloader, trainer),
            initial_parameters=parameters,
        )
    elif cfg.fl.optimizer == "fedadam":
        strategy = CustomFedAdam(
            cfg,
            fraction_fit=fraction_fit,
            fraction_evaluate=fraction_evaluate,
            min_evaluate_clients=min_evaluate_clients,
            min_available_clients=min_available_clients,
            min_fit_clients=min_fit_clients,
            evaluate_metrics_aggregation_fn=weighted_average,
            evaluate_fn=generate_evaluate_fn(cfg.fl.num_local_steps, valloader, trainer),
            initial_parameters=parameters,
            eta=cfg.fl.fedadam_eta,
            eta_l=cfg.fl.fedadam_eta_l,
            beta_1=cfg.fl.fedadam_beta1,
            beta_2=cfg.fl.fedadam_beta2,
            tau=cfg.fl.fedadam_tau,
        )
    else:
        raise ValueError(f"Unknown optimizer: {cfg.fl.optimizer}")
        
    config = ServerConfig(num_rounds=num_rounds)

    return ServerAppComponents(strategy=strategy, config=config)

# Create ServerApp
app = ServerApp(server_fn=server_fn)
