# pylint: disable=line-too-long
r"""Default configs for X-ViT on structural variant classification using coverage pileups.

"""

import ml_collections

_TRAIN_SIZE = 31_316 * 24


def get_config():
  """Returns the X-ViT experiment configuration for SV classification."""
  config = ml_collections.ConfigDict()
  config.experiment_name = 'npsvit'

  # Dataset.
  config.dataset_name = 'pileup_coverage_simulated'
  config.data_dtype_str = 'float32'

  config.dataset_configs = ml_collections.ConfigDict()
  config.dataset_configs.train_path = '/storage/mlinderman/projects/sv/npsv2-experiments/training/freeze3.sv.alt.passing.training.hg38.DEL.images/**/+pileup.snv_input=True,generator=single_depth_phaseread,pileup.discrete_mapq=True,pileup.render_snv=True,simulation.augment=False,simulation.chrom_norm_covg=True,simulation.replicates=5/images.tfrecords.gz'
  config.dataset_configs.eval_path = '/storage/mlinderman/projects/sv/npsv2-experiments/training/freeze3.sv.alt.passing.training.hg38.DEL.images/**/+pileup.snv_input=True,generator=single_depth_phaseread,pileup.discrete_mapq=True,pileup.render_snv=True,simulation.augment=False,simulation.chrom_norm_covg=True,simulation.replicates=5/images.tfrecords.gz'
  config.dataset_configs.test_path = '/storage/mlinderman/projects/sv/npsv2-experiments/training/freeze3.sv.alt.passing.training.hg38.DEL.images/**/+pileup.snv_input=True,generator=single_depth_phaseread,pileup.discrete_mapq=True,pileup.render_snv=True,simulation.augment=False,simulation.chrom_norm_covg=True,simulation.replicates=5/images.tfrecords.gz'
  
  # Model.
  config.model_name = 'xvit_paired'
  config.model_dtype_str = 'float32'
  config.model = ml_collections.ConfigDict()
  config.model.patches = ml_collections.ConfigDict()
  config.model.hidden_size = 768
  config.model.patches = ml_collections.ConfigDict()
  config.model.patches.size = [16, 16]
  config.model.mlp_dim = 3072
  config.model.num_layers = 12
  config.model.representation_size = None
  config.model.classifier = 'token'
  config.model.attention_dropout_rate = 0.
  config.model.dropout_rate = 0.
  config.model.transformer_encoder_configs = ml_collections.ConfigDict()
  config.model.transformer_encoder_configs.type = 'global'
  config.model.attention_fn = 'standard'
  config.model.attention_configs = ml_collections.ConfigDict()
  config.model.attention_configs.num_heads = 12

  # Training.
  config.trainer_name = 'paired_trainer'
  config.optimizer = 'adam'
  config.optimizer_configs = ml_collections.ConfigDict()
  config.optimizer_configs.beta1 = 0.9
  config.optimizer_configs.beta2 = 0.999
  config.optimizer_configs.weight_decay = 0.1
  config.explicit_weight_decay = None  # No explicit weight decay
  config.l2_decay_factor = None
  config.max_grad_norm = None
  config.label_smoothing = None
  config.num_training_epochs = 200
  config.log_eval_steps = 1000
  config.batch_size = 64  # >=256 causes RESOURCE EXHAUSTED errors.
  config.rng_seed = 42
  config.init_head_bias = None

  # Learning rate.
  steps_per_epoch = _TRAIN_SIZE // config.batch_size
  total_steps = config.num_training_epochs * steps_per_epoch
  base_lr = 3e-3
  config.lr_configs = ml_collections.ConfigDict()
  config.lr_configs.learning_rate_schedule = 'compound'
  config.lr_configs.factors = 'constant*linear_warmup*linear_decay'
  config.lr_configs.total_steps = total_steps
  config.lr_configs.end_learning_rate = 1e-5
  config.lr_configs.warmup_steps = 10_000
  config.lr_configs.base_learning_rate = base_lr

  # Logging.
  config.write_summary = True
  config.xprof = True  # Profile using xprof.
  config.checkpoint = False  # Do checkpointing.
  config.checkpoint_steps = 5000
  config.debug_train = True  # Debug mode during training.
  config.debug_eval = False  # Debug mode during eval.

  # Evaluation:
  config.global_metrics = [
      # 'truvari_recall_events',
      # 'truvari_precision_events',
      # 'truvari_recall',
      # 'truvari_precision',
      # 'gt_concordance',
      # 'nonref_concordance',
  ]

  return config


def get_hyper(hyper):
  """Defines the hyper-parameters sweeps for doing grid search."""
  return hyper.product([])