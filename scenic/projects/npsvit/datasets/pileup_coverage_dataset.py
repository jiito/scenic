"""Data generators for the coverage pileup images."""

import functools
from typing import Optional, Tuple

from absl import logging
from flax import jax_utils
import jax
import jax.numpy as jnp
from scenic.dataset_lib import dataset_utils
from scenic.dataset_lib import datasets
import tensorflow as tf



def _example_image(example):
    image_data = tf.io.parse_tensor(example.features.feature["image/encoded"].bytes_list.value[0], tf.uint8).numpy()
    return image_data


def _example_image_shape(example):
    return tuple(example.features.feature["image/shape"].int64_list.value)


def _example_addl_attribute(example, attr):
    return float(example.features.feature[attr].float_list.value[0])


def _example_label(example):
    return int(example.features.feature["label"].int64_list.value[0])


def _example_sim_images_shape(example):
    if "sim/images/shape" in example.features.feature:
        return tuple(example.features.feature["sim/images/shape"].int64_list.value)
    else:
        return (3, 0, None, None, None)


def _example_sim_images(example):
    image_data = tf.io.parse_tensor(
        example.features.feature["sim/images/encoded"].bytes_list.value[0], tf.uint8
    ).numpy()
    return image_data


def _extract_metadata_from_first_example(filename, pileup_image_channels=None):
    raw_example = next(
        iter(tf.data.TFRecordDataset(filenames=filename, compression_type=_filename_to_compression(filename)))
    )
    example = tf.train.Example.FromString(raw_example.numpy())

    image_shape = _example_image_shape(example)
    ac, replicates, *sim_image_shape = _example_sim_images_shape(example)
    if replicates > 0:
        assert ac == 3, "Incorrect number of genotypes in simulated data"
        assert image_shape == tuple(sim_image_shape), "Simulated and actual image shapes don't match"
    if pileup_image_channels:
        assert len(pileup_image_channels) <= image_shape[-1], "More channels requested than available"
        image_shape = image_shape[:-1] + (len(pileup_image_channels),)

    return image_shape, replicates

# def _extract_metadata_from_first_example(filename: str) -> Tuple[int]:
#   """Extracts the image shape from the first example."""
#   raw_example = next(
#       iter(
#           tf.data.TFRecordDataset(
#               filenames=filename,
#               compression_type=_filename_to_compression(filename))))
#   example = tf.train.Example.FromString(raw_example.numpy())

#   return tuple(example.features.feature['image/shape'].int64_list.value)


def _filename_to_compression(filename: str) -> Optional[str]:
  return 'GZIP' if tf.strings.regex_full_match(filename, '.*.gz') else None
  


def create_coverage_based_dataset(
    filenames: str,
    with_label: bool = True,
    with_simulation: bool = True
) -> tf.data.Dataset:
  """Creates a coverage based pileup dataset from a filepath.

  Args:
    filenames: The data directory/pattern containing data files.
    with_label: whether to load the labels or not.

  Returns:
    tf.data.Dataset
  """
  dataset_files = tf.io.matching_files(filenames)

  # Extract image shape from the first example
  shape, num_replicates = _extract_metadata_from_first_example(dataset_files[0])

  proto_features = {
      'variant/encoded': tf.io.FixedLenFeature(shape=(), dtype=tf.string),
      'image/encoded': tf.io.FixedLenFeature(shape=(), dtype=tf.string),
      'image/shape': tf.io.FixedLenFeature(shape=(len(shape),), dtype=tf.int64),
  }
  if with_label:
    proto_features['label'] = tf.io.FixedLenFeature(shape=1, dtype=tf.int64)
  if with_simulation and num_replicates > 0:
    proto_features.update(
      {
        "sim/images/shape": tf.io.FixedLenFeature(shape=(len(shape) + 2,), dtype=tf.int64),
        "sim/images/encoded": tf.io.FixedLenFeature(shape=(), dtype=tf.string),
      }
    )

  def _process_input(proto_string):
    """Helper function for input function that parses a serialized example."""

    parsed_features = tf.io.parse_single_example(
        serialized=proto_string, features=proto_features)
    logging.info(parsed_features)

    features = {
      'variant/encoded': parsed_features['variant/encoded'],
        'image': tf.io.parse_tensor(parsed_features['image/encoded'], tf.uint8),
    }

    features['image'].set_shape(shape)

    if with_simulation:
      features["sim/images"] = tf.io.parse_tensor(parsed_features["sim/images/encoded"], tf.uint8)
      logging.info("With simulation (shape): %s", features["sim/images"].shape)


    # logging.info("With simulation: %s", features["sim/images"])
    # TODO: Use if selecting certain pileup channels
    # if pileup_image_channels:
    #     features["image"] = tf.gather(features["image"], indices=list(pileup_image_channels), axis=-1)
    #     if with_simulations:
    #         features["sim/images"] = tf.gather(features["sim/images"], indices=list(pileup_image_channels), axis=-1)

    if with_label:
      return features, parsed_features['label']
    else:
      return features, None

  compression = _filename_to_compression(dataset_files[0])

  logging.info('Loading TFRecords as bytes.')
  dataset = tf.data.Dataset.from_tensor_slices(dataset_files)

  # pylint: disable=g-long-lambda
  # interleave parallelizes the data loading step by interleaving the I/O
  # operation to read the file. It speeds up the I/O step.
  dataset = dataset.interleave(
      lambda filename: tf.data.TFRecordDataset(
          filename,
          compression_type=compression,
      ).map(
          _process_input,
          num_parallel_calls=tf.data.experimental.AUTOTUNE,
      ),
      cycle_length=len(dataset_files),
  )

  # map the sim/images to the real image 


  # dataset.flat_map(flatten_sim_images )
  # logging.info("Dataset shape after Flattened sim images: %s", dataset.shape)

  return dataset

def flatten_sim_images(features, label) -> tf.data.Dataset:
  logging.info("Flattening sim images: %s, %s", features, label)
  # unpack the simulated images 
 


  # return the new dataset
  return sim_images








def preprocess(features, label):
  """Preprocessing code specific to ViT models."""
  label_tensor = tf.cast(tf.squeeze(label, [-1]), tf.int32)
  logging.info("Calling preprocess function")
  logging.info("Label tensor: %s", label_tensor)

  support = tf.reshape(features["sim/images"], [15, 100, 300, 9])

  qeury = tf.tile(tf.expand_dims(features["image"], axis=0), [15, 1, 1, 1])

  label_oh = tf.one_hot(label_tensor, 3)

  label_oh = tf.reshape(tf.repeat(label_oh, repeats=5, axis=0),[15])

  logging.info("Label one hot: %s", label_oh)

  new = {
      "support": support,
      "query": qeury,
    'label': label_oh
  }
  return tf.data.Dataset.from_tensor_slices(new)

  # return {
  #   # TODO: modify to use support and query images (batch["inputs"])
  #     'inputs': {
  #         'support': features['sim/image'],
  #         'query': features['image'],
  #     },  # Resize pileups to make side length divisible by 4.
  #     'label': tf.one_hot(label_tensor, 3)
  # }


def build_dataset(dataset_fn,
                  batch_size=None,
                  shuffle_buffer_size=256,
                  seed=None,
                  strategy=None,
                  **dataset_kwargs):
  """Dataset builder that takes care of strategy, batching and shuffling.

  Args:
    dataset_fn: function; A function that loads the dataset.
    batch_size: int; Size of the batch.
    shuffle_buffer_size: int; Size of the buffer for used for shuffling.
    seed: int; Random seed used for shuffling.
    strategy: TF strategy that handles the distribution policy.
    **dataset_kwargs: dict; Arguments passed to TFDS.

  Returns:
    Dataset.
  """

  def _dataset_fn(input_context=None):
    """Dataset function."""
    replica_batch_size = batch_size
    if input_context:
      replica_batch_size = input_context.get_per_replica_batch_size(batch_size)
    ds = dataset_fn(**dataset_kwargs)
    split = dataset_kwargs.get('split')
    if split == 'train':
      # First repeat then shuffle, then batch.
      ds = ds.repeat()
      local_seed = seed  # Seed for this machine.
      if local_seed is not None and input_context:
        local_seed += input_context.input_pipeline_id
      ds = ds.shuffle(shuffle_buffer_size, seed=local_seed)
      ds = ds.batch(replica_batch_size, drop_remainder=True)
    else:  # Test and validation.
      # First batch then repeat.
      ds = ds.batch(replica_batch_size, drop_remainder=False)
      ds = ds.repeat()
    options = tf.data.Options()
    options.experimental_optimization.parallel_batch = True
    ds = ds.with_options(options)
    return ds.prefetch(tf.data.experimental.AUTOTUNE)

  if strategy:
    ds = strategy.experimental_distribute_datasets_from_function(_dataset_fn)
  else:
    ds = _dataset_fn()

  return ds

def get_dataset_name(dataset_path: Optional[str] = None):
  """Extract dataset name for eval_iter in xmanager measurements.

  Parent directory of the dataset files is used as its name.
  Args:
    dataset_path: Path to the dataset files.

  Returns:
    Dataset name.
  """
  return 'test' if not dataset_path else dataset_path.split('/')[-2]

print("building pileup dataset")
@datasets.add_dataset('pileup_coverage_simulated')
def get_dataset(*,
                batch_size,
                eval_batch_size,
                num_shards,
                dtype_str='float32',
                shuffle_seed=0,
                rng=None,
                prefetch_buffer_size=2,
                dataset_configs=None,
                dataset_service_address: Optional[str] = None):
  """Returns generators for the pileup window train, validation, and test set.

  Args:
    batch_size: int; Determines the train batch size.
    eval_batch_size: int; Determines the evaluation batch size.
    num_shards: int;  Number of shards --> batch shape: [num_shards, bs, ...].
    dtype_str: Data type of the image (e.g. 'float32').
    shuffle_seed: int; Seed for shuffling the training data.
    rng: JAX rng key, which can be used for augmentation, shuffling, etc.
    prefetch_buffer_size: int; Buffer size for the prefetch.
    dataset_configs: dict; Dataset specific configurations.
    dataset_service_address: If set, will distribute the training dataset using
      the given tf.data service at the given address.

  Returns:
    A dataset_utils.Dataset() which includes a train_iter, a valid_iter,
      a test_iter, and a dict of meta_data.
  """
  del rng

  def build_pileup_window_dataset(split):
    """dataset_fn called by data.build_dataset(**kwargs)."""

    if split == 'train':
      if dataset_configs.train_path:
        path = dataset_configs.train_path
      else:
        # Add path to your training data here:
        path = ''
    elif split == 'valid':
      if dataset_configs.eval_path:
        path = dataset_configs.eval_path
      else:
        # Add path to your validation data here:
        path = ''
    elif split == 'test':
      if dataset_configs.test_path:
        path = dataset_configs.test_path
      else:
        # Add path to your test data here:
        path = ''
    else:
      raise ValueError('Invalid split value.')

    if not path:
      raise ValueError('No path provide. Please modify the path variable to '
                       'hardcode the %s path.' %split)
    dataset = create_coverage_based_dataset(filenames=path, with_label=True, with_simulation=True)

    # Creating a Dataset that includes only 1/num_shards of data so the data is
    # splitted between different hosts.
    num_hosts, host_id = jax.process_count(), jax.process_index()
    dataset = dataset.shard(num_shards=num_hosts, index=host_id)

    # dataset = dataset.map(
    #     preprocess, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    dataset = dataset.flat_map(
        preprocess)
    logging.info("Dataset after preprocessing: %s", dataset)
    return dataset

  # Use different seed for each host.
  if shuffle_seed is None:
    local_seed = None
  else:
    data_seed = 0
    local_seed = data_seed + jax.process_index()

  train_dataset = build_dataset(
      dataset_fn=build_pileup_window_dataset,
      batch_size=batch_size,
      seed=local_seed,
      split='train',
      strategy=None)

  if dataset_service_address:
    if shuffle_seed is not None:
      raise ValueError('Using dataset service with a random seed causes each '
                       'worker to produce exactly the same data. Add '
                       'config.shuffle_seed = None to your config if you '
                       'want to run with dataset service.')
    logging.info('Using the tf.data service at %s', dataset_service_address)
    train_dataset = dataset_utils.distribute(train_dataset,
                                             dataset_service_address)

  eval_dataset = build_dataset(
      dataset_fn=build_pileup_window_dataset,
      split='valid',
      batch_size=eval_batch_size,
      strategy=None)

  test_dataset = build_dataset(
      dataset_fn=build_pileup_window_dataset,
      split='test',
      batch_size=eval_batch_size,
      strategy=None)

  shard_batches = functools.partial(dataset_utils.shard, n_devices=num_shards)
  maybe_pad_batches_train = functools.partial(
      dataset_utils.maybe_pad_batch, train=True, batch_size=batch_size,inputs_key='support')
  maybe_pad_batches_train = functools.partial(
      dataset_utils.maybe_pad_batch, train=True, batch_size=batch_size,inputs_key='query')
  maybe_pad_batches_eval = functools.partial(
      dataset_utils.maybe_pad_batch, train=False, batch_size=eval_batch_size,inputs_key='support')
  maybe_pad_batches_eval = functools.partial(
      dataset_utils.maybe_pad_batch, train=False, batch_size=eval_batch_size,inputs_key='query')

  train_iter = iter(train_dataset)
  train_iter = map(dataset_utils.tf_to_numpy, train_iter)
  train_iter = map(maybe_pad_batches_train, train_iter)
  train_iter = map(shard_batches, train_iter)
  train_iter = jax_utils.prefetch_to_device(train_iter, prefetch_buffer_size)

  valid_iter = iter(eval_dataset)
  valid_iter = map(dataset_utils.tf_to_numpy, valid_iter)
  valid_iter = map(maybe_pad_batches_eval, valid_iter)
  valid_iter = map(shard_batches, valid_iter)
  valid_iter = jax_utils.prefetch_to_device(valid_iter, prefetch_buffer_size)

  test_iter = iter(test_dataset)
  test_iter = map(dataset_utils.tf_to_numpy, test_iter)
  test_iter = map(maybe_pad_batches_eval, test_iter)
  test_iter = map(shard_batches, test_iter)
  test_iter = jax_utils.prefetch_to_device(test_iter, prefetch_buffer_size)

  num_classes = 3
  image_size = 256

  # TODO: Change shape to list 
  # Batch (-1) is determined at runtime.
  input_shape = [-1, 100, 300, 9]

  meta_data = {
      'num_classes': num_classes,
      'input_shape': input_shape,
      'num_train_examples': 31000 * 24,
      'num_eval_examples': 31000 * 6,
      'num_test_examples': 31000,
      'test_name': get_dataset_name(dataset_configs.test_path),
      'eval_name': get_dataset_name(dataset_configs.eval_path),
      'input_dtype': getattr(jnp, dtype_str),
      'target_is_onehot': True,
  }

  return dataset_utils.Dataset(train_iter, valid_iter, test_iter, meta_data)
