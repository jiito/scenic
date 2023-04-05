"""X-ViT model.

Todo(dehghani, yitay): Write a paper on the results of XViT.
"""

from typing import Any, Optional

import flax.linen as nn
import jax.numpy as jnp
import jax
from jax.experimental.host_callback import call
import ml_collections
import numpy as np
from absl import logging
from scenic.model_lib.base_models.multilabel_classification_model import MultiLabelClassificationModel
from scenic.model_lib.layers import attention_layers
from scenic.model_lib.layers import nn_layers
from scenic.model_lib.layers import nn_ops
from scenic.projects.fast_vit import model_utils
from scenic.projects.fast_vit.xvit import XViT
from scenic.projects.baselines import vit

class Distance(nn.Module):
  @nn.compact
  def __call__(self, x, y):
    dist = jnp.linalg.norm(x-y, axis=1)
    return dist

class PairXViTFlax(nn.Module):
  """Paired-XViT model.

  Attributes:
    num_outputs: number of classes.
    mlp_dim: Dimension of the MLP on top of attention block.
    num_layers: Number of layers.
    attention_configs: Configurations passed to the self-attention.
    attention_fn: Self-attention function used in the model.
    patches: Configuration of the patches extracted in the stem of the model.
    hidden_size: Size of the hidden dimension on the stem of the model.
    fast_vit: Configurations of the fast_vit (omnidirectional attention).
    representation_size: Size of the final representation.
    dropout_rate: Dropout rate.
    attention_dropout_rate: Dropout rate for attention heads.
    classifier:  Type of the classifier.
  """
  num_outputs: int
  mlp_dim: int
  num_layers: int
  attention_configs: ml_collections.ConfigDict
  attention_fn: ml_collections.ConfigDict
  patches: ml_collections.ConfigDict
  hidden_size: int
  transformer_encoder_configs: ml_collections.ConfigDict
  representation_size: Optional[int] = None
  dropout_rate: float = 0.1
  attention_dropout_rate: float = 0.1
  classifier: str = 'gap'

  @nn.compact
  def __call__(self,
               inputs: jnp.ndarray,
               *args,
               train: bool,
               debug: Optional[bool] = False) -> jnp.ndarray:
    """X-ViT model.

    Args:
      inputs: Input data.
      train: If it is training.
      debug: If we are running the model in the debug mode.

    Returns:
      Output of the model.
    """
    # TODO: stamp out XVIT Here
    # take in two inputs and spit out euclidean distance
    

    xvit = XViT( 
        num_outputs=self.num_outputs,
        mlp_dim=self.mlp_dim,
        num_layers=self.num_layers,
        attention_configs=self.attention_configs,
        attention_fn=self.attention_fn,
        patches=self.patches,
        hidden_size=self.hidden_size,
        transformer_encoder_configs=self.transformer_encoder_configs,
        representation_size=self.representation_size,
        dropout_rate=self.dropout_rate,
        attention_dropout_rate=self.attention_dropout_rate,
        classifier=self.classifier,
        name='xvit-query'
    )

    logging.info('Running X-ViT model.')
    logging.info('Inputs shape: %s', inputs.shape)
    
    if args:
      for arg in args:
        logging.info('Arg shape: %s', arg.shape)
  
    query = inputs 
    simulated = args[0] if args else query
    # TODO: Change to embeddings NOT outputprojection
    x = xvit(inputs, train=train, debug=debug)
    # y = xvit(simulated, train=train, debug=debug)

    x = Distance()(x,x)
    
    return x


