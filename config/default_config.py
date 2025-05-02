from yacs.config import CfgNode as CN


class HumanConfig:
    def __init__(self):
        self.cfg = CN()
        self.cfg.name = ''
        self.cfg.eval_name = ''
        self.cfg.debug = True
        self.cfg.comet = False
        self.cfg.eval_TTA = False
        self.cfg.subset_size = 1
        self.cfg.subset = True
        self.cfg.stage1_ckpt = None
        self.cfg.restore_ckpt = None
        self.cfg.generator_ckpt = None
        self.cfg.finetune_ckpt = None
        self.cfg.finetune_generator = None
        self.cfg.fix_scale_min = None
        self.cfg.num_out_scaffold = None
        self.cfg.lr = 0.0
        self.cfg.wdecay = 0.0
        self.cfg.batch_size = 0
        self.cfg.num_steps = 0
        self.cfg.personalized = True
        self.cfg.person_subset_size = 4
        self.cfg.subset_at_dataset_level = False

        self.cfg.dataset = CN()
        self.cfg.dataset.source_id = None
        self.cfg.dataset.train_novel_id = None
        self.cfg.dataset.val_novel_id = None
        self.cfg.dataset.use_hr_img = None
        self.cfg.dataset.use_processed_data = None
        self.cfg.dataset.num_inputs = None
        self.cfg.dataset.num_total_cams = None
        self.cfg.dataset.test_input_view = None
        self.cfg.dataset.data_root = ''

        # gsussian render settings
        self.cfg.dataset.bg_color = [0, 0, 0]
        self.cfg.dataset.zfar = 100.0
        self.cfg.dataset.znear = 0.01
        self.cfg.dataset.trans = [0.0, 0.0, 0.0]
        self.cfg.dataset.scale = 1.0

        self.cfg.raft = CN()
        self.cfg.raft.mixed_precision = None
        self.cfg.raft.train_iters = 0
        self.cfg.raft.val_iters = 0
        self.cfg.raft.corr_implementation = 'reg_cuda'  # or 'reg'
        self.cfg.raft.corr_levels = 4
        self.cfg.raft.corr_radius = 4
        self.cfg.raft.n_downsample = 3
        self.cfg.raft.n_gru_layers = 1
        self.cfg.raft.slow_fast_gru = None
        self.cfg.raft.encoder_dims = [64, 96, 128]
        self.cfg.raft.hidden_dims = [128] * 3

        self.cfg.gsnet = CN()
        self.cfg.gsnet.encoder_dims = None
        self.cfg.gsnet.decoder_dims = None
        self.cfg.gsnet.parm_head_dim = None

        self.cfg.record = CN()
        self.cfg.record.ckpt_path = None
        self.cfg.record.show_path = None
        self.cfg.record.logs_path = None
        self.cfg.record.file_path = None
        self.cfg.record.loss_freq = 0
        self.cfg.record.eval_freq = 0

    def get_cfg(self):
        return self.cfg.clone()

    def load(self, config_file):
        self.cfg.defrost()
        self.cfg.merge_from_file(config_file)
        self.cfg.freeze()

class HumanFLConfig(HumanConfig):
    """Extended HumanConfig class with FL config params"""
    def __init__(self):
        super().__init__()
        
        self.cfg.fl = CN()
        self.cfg.fl.num_clients = 1
        self.cfg.fl.num_local_steps = 1
        self.cfg.fl.epochs = 10
        self.cfg.fl.steps_per_round = 1
        self.cfg.fl.iid = True
        self.cfg.fl.alpha = 0.5
        self.cfg.fl.train_test_split = 0.8
        self.cfg.fl.exp_name = 'DEFAULT'
        self.cfg.fl.min_evaluate_clients = 1
        self.cfg.fl.min_fit_clients = 1
        self.cfg.fl.min_available_clients = 1
        self.cfg.fl.client_eval_freq = 1000
        self.cfg.fl.server_eval_freq = 500
        
        # Module personalization settings
        self.cfg.fl.module_personalization = False  # Enable/disable module personalization
        self.cfg.fl.personalize_img_encoder = False  # Whether to use personalized weights for img_encoder
        self.cfg.fl.personalize_dino_encoder = False  # Whether to use personalized weights for dino_encoder
        self.cfg.fl.personalize_gs_parm_regressor = False  # Whether to use personalized weights for gs_parm_regressor
        
        # Federated learning optimizer settings
        self.cfg.fl.optimizer = "fedavg"  # Options: "fedavg", "fedadam"
        
        # FedAdam specific parameters
        self.cfg.fl.fedadam_eta = 0.0001  # Learning rate
        self.cfg.fl.fedadam_eta_l = 0.0001  # Server Learning rate
        self.cfg.fl.fedadam_beta1 = 0.9  # First moment decay
        self.cfg.fl.fedadam_beta2 = 0.99  # Second moment decay
        self.cfg.fl.fedadam_tau = 0.001  # Adaptivity parameter