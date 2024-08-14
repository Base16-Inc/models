# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""A function to build a DetectionModel from configuration."""

import functools
import sys

from absl import logging

from object_detection.builders import image_resizer_builder
from object_detection.builders import losses_builder
from object_detection.builders import post_processing_builder
from object_detection.meta_architectures import center_net_meta_arch
from object_detection.protos import losses_pb2
from object_detection.protos import model_pb2
from object_detection.utils import label_map_util

# pylint: disable=g-import-not-at-top
from object_detection.models import center_net_hourglass_feature_extractor
from object_detection.models import center_net_mobilenet_v2_feature_extractor
from object_detection.models import center_net_mobilenet_v2_fpn_feature_extractor
from object_detection.models import center_net_resnet_feature_extractor

CENTER_NET_EXTRACTOR_FUNCTION_MAP = {
    'resnet_v2_50':
        center_net_resnet_feature_extractor.resnet_v2_50,
    'resnet_v2_101':
        center_net_resnet_feature_extractor.resnet_v2_101,
    'hourglass_10':
        center_net_hourglass_feature_extractor.hourglass_10,
    'hourglass_20':
        center_net_hourglass_feature_extractor.hourglass_20,
    'hourglass_32':
        center_net_hourglass_feature_extractor.hourglass_32,
    'hourglass_52':
        center_net_hourglass_feature_extractor.hourglass_52,
    'hourglass_104':
        center_net_hourglass_feature_extractor.hourglass_104,
    'mobilenet_v2':
        center_net_mobilenet_v2_feature_extractor.mobilenet_v2,
    'mobilenet_v2_fpn':
        center_net_mobilenet_v2_fpn_feature_extractor.mobilenet_v2_fpn,
    'mobilenet_v2_fpn_sep_conv':
        center_net_mobilenet_v2_fpn_feature_extractor.mobilenet_v2_fpn,
}

FEATURE_EXTRACTOR_MAPS = [
    CENTER_NET_EXTRACTOR_FUNCTION_MAP,
]

def _check_feature_extractor_exists(feature_extractor_type):
  feature_extractors = set().union(*FEATURE_EXTRACTOR_MAPS)
  if feature_extractor_type not in feature_extractors:
    raise ValueError('{} is not supported. See `model_builder.py` for features '
                     'extractors compatible with different versions of '
                     'Tensorflow'.format(feature_extractor_type))

# The class ID in the groundtruth/model architecture is usually 0-based while
# the ID in the label map is 1-based. The offset is used to convert between the
# the two.
CLASS_ID_OFFSET = 1
KEYPOINT_STD_DEV_DEFAULT = 1.0

def keypoint_proto_to_params(kp_config, keypoint_map_dict):
  """Converts CenterNet.KeypointEstimation proto to parameter namedtuple."""
  label_map_item = keypoint_map_dict[kp_config.keypoint_class_name]

  classification_loss, localization_loss, _, _, _, _, _ = (
      losses_builder.build(kp_config.loss))

  keypoint_indices = [
      keypoint.id for keypoint in label_map_item.keypoints
  ]
  keypoint_labels = [
      keypoint.label for keypoint in label_map_item.keypoints
  ]
  keypoint_std_dev_dict = {
      label: KEYPOINT_STD_DEV_DEFAULT for label in keypoint_labels
  }
  if kp_config.keypoint_label_to_std:
    for label, value in kp_config.keypoint_label_to_std.items():
      keypoint_std_dev_dict[label] = value
  keypoint_std_dev = [keypoint_std_dev_dict[label] for label in keypoint_labels]
  if kp_config.HasField('heatmap_head_params'):
    heatmap_head_num_filters = list(kp_config.heatmap_head_params.num_filters)
    heatmap_head_kernel_sizes = list(kp_config.heatmap_head_params.kernel_sizes)
  else:
    heatmap_head_num_filters = [256]
    heatmap_head_kernel_sizes = [3]
  if kp_config.HasField('offset_head_params'):
    offset_head_num_filters = list(kp_config.offset_head_params.num_filters)
    offset_head_kernel_sizes = list(kp_config.offset_head_params.kernel_sizes)
  else:
    offset_head_num_filters = [256]
    offset_head_kernel_sizes = [3]
  if kp_config.HasField('regress_head_params'):
    regress_head_num_filters = list(kp_config.regress_head_params.num_filters)
    regress_head_kernel_sizes = list(
        kp_config.regress_head_params.kernel_sizes)
  else:
    regress_head_num_filters = [256]
    regress_head_kernel_sizes = [3]
  return center_net_meta_arch.KeypointEstimationParams(
      task_name=kp_config.task_name,
      class_id=label_map_item.id - CLASS_ID_OFFSET,
      keypoint_indices=keypoint_indices,
      classification_loss=classification_loss,
      localization_loss=localization_loss,
      keypoint_labels=keypoint_labels,
      keypoint_std_dev=keypoint_std_dev,
      task_loss_weight=kp_config.task_loss_weight,
      keypoint_regression_loss_weight=kp_config.keypoint_regression_loss_weight,
      keypoint_heatmap_loss_weight=kp_config.keypoint_heatmap_loss_weight,
      keypoint_offset_loss_weight=kp_config.keypoint_offset_loss_weight,
      heatmap_bias_init=kp_config.heatmap_bias_init,
      keypoint_candidate_score_threshold=(
          kp_config.keypoint_candidate_score_threshold),
      num_candidates_per_keypoint=kp_config.num_candidates_per_keypoint,
      peak_max_pool_kernel_size=kp_config.peak_max_pool_kernel_size,
      unmatched_keypoint_score=kp_config.unmatched_keypoint_score,
      box_scale=kp_config.box_scale,
      candidate_search_scale=kp_config.candidate_search_scale,
      candidate_ranking_mode=kp_config.candidate_ranking_mode,
      offset_peak_radius=kp_config.offset_peak_radius,
      per_keypoint_offset=kp_config.per_keypoint_offset,
      predict_depth=kp_config.predict_depth,
      per_keypoint_depth=kp_config.per_keypoint_depth,
      keypoint_depth_loss_weight=kp_config.keypoint_depth_loss_weight,
      score_distance_offset=kp_config.score_distance_offset,
      clip_out_of_frame_keypoints=kp_config.clip_out_of_frame_keypoints,
      rescore_instances=kp_config.rescore_instances,
      heatmap_head_num_filters=heatmap_head_num_filters,
      heatmap_head_kernel_sizes=heatmap_head_kernel_sizes,
      offset_head_num_filters=offset_head_num_filters,
      offset_head_kernel_sizes=offset_head_kernel_sizes,
      regress_head_num_filters=regress_head_num_filters,
      regress_head_kernel_sizes=regress_head_kernel_sizes,
      score_distance_multiplier=kp_config.score_distance_multiplier,
      std_dev_multiplier=kp_config.std_dev_multiplier,
      rescoring_threshold=kp_config.rescoring_threshold,
      gaussian_denom_ratio=kp_config.gaussian_denom_ratio,
      argmax_postprocessing=kp_config.argmax_postprocessing)


def object_detection_proto_to_params(od_config):
  """Converts CenterNet.ObjectDetection proto to parameter namedtuple."""
  loss = losses_pb2.Loss()
  # Add dummy classification loss to avoid the loss_builder throwing error.
  # TODO(yuhuic): update the loss builder to take the classification loss
  # directly.
  loss.classification_loss.weighted_sigmoid.CopyFrom(
      losses_pb2.WeightedSigmoidClassificationLoss())
  loss.localization_loss.CopyFrom(od_config.localization_loss)
  _, localization_loss, _, _, _, _, _ = (losses_builder.build(loss))
  if od_config.HasField('scale_head_params'):
    scale_head_num_filters = list(od_config.scale_head_params.num_filters)
    scale_head_kernel_sizes = list(od_config.scale_head_params.kernel_sizes)
  else:
    scale_head_num_filters = [256]
    scale_head_kernel_sizes = [3]
  if od_config.HasField('offset_head_params'):
    offset_head_num_filters = list(od_config.offset_head_params.num_filters)
    offset_head_kernel_sizes = list(od_config.offset_head_params.kernel_sizes)
  else:
    offset_head_num_filters = [256]
    offset_head_kernel_sizes = [3]
  return center_net_meta_arch.ObjectDetectionParams(
      localization_loss=localization_loss,
      scale_loss_weight=od_config.scale_loss_weight,
      offset_loss_weight=od_config.offset_loss_weight,
      task_loss_weight=od_config.task_loss_weight,
      scale_head_num_filters=scale_head_num_filters,
      scale_head_kernel_sizes=scale_head_kernel_sizes,
      offset_head_num_filters=offset_head_num_filters,
      offset_head_kernel_sizes=offset_head_kernel_sizes)


def object_center_proto_to_params(oc_config):
  """Converts CenterNet.ObjectCenter proto to parameter namedtuple."""
  loss = losses_pb2.Loss()
  # Add dummy localization loss to avoid the loss_builder throwing error.
  # TODO(yuhuic): update the loss builder to take the localization loss
  # directly.
  loss.localization_loss.weighted_l2.CopyFrom(
      losses_pb2.WeightedL2LocalizationLoss())
  loss.classification_loss.CopyFrom(oc_config.classification_loss)
  classification_loss, _, _, _, _, _, _ = (losses_builder.build(loss))
  keypoint_weights_for_center = []
  if oc_config.keypoint_weights_for_center:
    keypoint_weights_for_center = list(oc_config.keypoint_weights_for_center)

  if oc_config.HasField('center_head_params'):
    center_head_num_filters = list(oc_config.center_head_params.num_filters)
    center_head_kernel_sizes = list(oc_config.center_head_params.kernel_sizes)
  else:
    center_head_num_filters = [256]
    center_head_kernel_sizes = [3]
  return center_net_meta_arch.ObjectCenterParams(
      classification_loss=classification_loss,
      object_center_loss_weight=oc_config.object_center_loss_weight,
      heatmap_bias_init=oc_config.heatmap_bias_init,
      min_box_overlap_iou=oc_config.min_box_overlap_iou,
      max_box_predictions=oc_config.max_box_predictions,
      use_labeled_classes=oc_config.use_labeled_classes,
      keypoint_weights_for_center=keypoint_weights_for_center,
      center_head_num_filters=center_head_num_filters,
      center_head_kernel_sizes=center_head_kernel_sizes,
      peak_max_pool_kernel_size=oc_config.peak_max_pool_kernel_size)


def mask_proto_to_params(mask_config):
  """Converts CenterNet.MaskEstimation proto to parameter namedtuple."""
  loss = losses_pb2.Loss()
  # Add dummy localization loss to avoid the loss_builder throwing error.
  loss.localization_loss.weighted_l2.CopyFrom(
      losses_pb2.WeightedL2LocalizationLoss())
  loss.classification_loss.CopyFrom(mask_config.classification_loss)
  classification_loss, _, _, _, _, _, _ = (losses_builder.build(loss))
  if mask_config.HasField('mask_head_params'):
    mask_head_num_filters = list(mask_config.mask_head_params.num_filters)
    mask_head_kernel_sizes = list(mask_config.mask_head_params.kernel_sizes)
  else:
    mask_head_num_filters = [256]
    mask_head_kernel_sizes = [3]
  return center_net_meta_arch.MaskParams(
      classification_loss=classification_loss,
      task_loss_weight=mask_config.task_loss_weight,
      mask_height=mask_config.mask_height,
      mask_width=mask_config.mask_width,
      score_threshold=mask_config.score_threshold,
      heatmap_bias_init=mask_config.heatmap_bias_init,
      mask_head_num_filters=mask_head_num_filters,
      mask_head_kernel_sizes=mask_head_kernel_sizes)


def densepose_proto_to_params(densepose_config):
  """Converts CenterNet.DensePoseEstimation proto to parameter namedtuple."""
  classification_loss, localization_loss, _, _, _, _, _ = (
      losses_builder.build(densepose_config.loss))
  return center_net_meta_arch.DensePoseParams(
      class_id=densepose_config.class_id,
      classification_loss=classification_loss,
      localization_loss=localization_loss,
      part_loss_weight=densepose_config.part_loss_weight,
      coordinate_loss_weight=densepose_config.coordinate_loss_weight,
      num_parts=densepose_config.num_parts,
      task_loss_weight=densepose_config.task_loss_weight,
      upsample_to_input_res=densepose_config.upsample_to_input_res,
      heatmap_bias_init=densepose_config.heatmap_bias_init)


def tracking_proto_to_params(tracking_config):
  """Converts CenterNet.TrackEstimation proto to parameter namedtuple."""
  loss = losses_pb2.Loss()
  # Add dummy localization loss to avoid the loss_builder throwing error.
  # TODO(yuhuic): update the loss builder to take the localization loss
  # directly.
  loss.localization_loss.weighted_l2.CopyFrom(
      losses_pb2.WeightedL2LocalizationLoss())
  loss.classification_loss.CopyFrom(tracking_config.classification_loss)
  classification_loss, _, _, _, _, _, _ = losses_builder.build(loss)
  return center_net_meta_arch.TrackParams(
      num_track_ids=tracking_config.num_track_ids,
      reid_embed_size=tracking_config.reid_embed_size,
      classification_loss=classification_loss,
      num_fc_layers=tracking_config.num_fc_layers,
      task_loss_weight=tracking_config.task_loss_weight)


def temporal_offset_proto_to_params(temporal_offset_config):
  """Converts CenterNet.TemporalOffsetEstimation proto to param-tuple."""
  loss = losses_pb2.Loss()
  # Add dummy classification loss to avoid the loss_builder throwing error.
  # TODO(yuhuic): update the loss builder to take the classification loss
  # directly.
  loss.classification_loss.weighted_sigmoid.CopyFrom(
      losses_pb2.WeightedSigmoidClassificationLoss())
  loss.localization_loss.CopyFrom(temporal_offset_config.localization_loss)
  _, localization_loss, _, _, _, _, _ = losses_builder.build(loss)
  return center_net_meta_arch.TemporalOffsetParams(
      localization_loss=localization_loss,
      task_loss_weight=temporal_offset_config.task_loss_weight)


def _build_center_net_model(center_net_config, is_training, add_summaries):
  """Build a CenterNet detection model.

  Args:
    center_net_config: A CenterNet proto object with model configuration.
    is_training: True if this model is being built for training purposes.
    add_summaries: Whether to add tf summaries in the model.

  Returns:
    CenterNetMetaArch based on the config.

  """

  image_resizer_fn = image_resizer_builder.build(
      center_net_config.image_resizer)
  _check_feature_extractor_exists(center_net_config.feature_extractor.type)
  feature_extractor = _build_center_net_feature_extractor(
      center_net_config.feature_extractor, is_training)
  object_center_params = object_center_proto_to_params(
      center_net_config.object_center_params)

  object_detection_params = None
  if center_net_config.HasField('object_detection_task'):
    object_detection_params = object_detection_proto_to_params(
        center_net_config.object_detection_task)

  if center_net_config.HasField('deepmac_mask_estimation'):
    logging.error(('Building experimental DeepMAC meta-arch.'
                  ' Some features may be omitted.'))
    return
    # deepmac_params = deepmac_meta_arch.deepmac_proto_to_params(
    #     center_net_config.deepmac_mask_estimation)
    # return deepmac_meta_arch.DeepMACMetaArch(
    #     is_training=is_training,
    #     add_summaries=add_summaries,
    #     num_classes=center_net_config.num_classes,
    #     feature_extractor=feature_extractor,
    #     image_resizer_fn=image_resizer_fn,
    #     object_center_params=object_center_params,
    #     object_detection_params=object_detection_params,
    #     deepmac_params=deepmac_params)

  keypoint_params_dict = None
  if center_net_config.keypoint_estimation_task:
    label_map_proto = label_map_util.load_labelmap(
        center_net_config.keypoint_label_map_path)
    keypoint_map_dict = {
        item.name: item for item in label_map_proto.item if item.keypoints
    }
    keypoint_params_dict = {}
    keypoint_class_id_set = set()
    all_keypoint_indices = []
    for task in center_net_config.keypoint_estimation_task:
      kp_params = keypoint_proto_to_params(task, keypoint_map_dict)
      keypoint_params_dict[task.task_name] = kp_params
      all_keypoint_indices.extend(kp_params.keypoint_indices)
      if kp_params.class_id in keypoint_class_id_set:
        raise ValueError(('Multiple keypoint tasks map to the same class id is '
                          'not allowed: %d' % kp_params.class_id))
      else:
        keypoint_class_id_set.add(kp_params.class_id)
    if len(all_keypoint_indices) > len(set(all_keypoint_indices)):
      raise ValueError('Some keypoint indices are used more than once.')

  mask_params = None
  if center_net_config.HasField('mask_estimation_task'):
    mask_params = mask_proto_to_params(center_net_config.mask_estimation_task)

  densepose_params = None
  if center_net_config.HasField('densepose_estimation_task'):
    densepose_params = densepose_proto_to_params(
        center_net_config.densepose_estimation_task)

  track_params = None
  if center_net_config.HasField('track_estimation_task'):
    track_params = tracking_proto_to_params(
        center_net_config.track_estimation_task)

  temporal_offset_params = None
  if center_net_config.HasField('temporal_offset_task'):
    temporal_offset_params = temporal_offset_proto_to_params(
        center_net_config.temporal_offset_task)
  non_max_suppression_fn = None
  if center_net_config.HasField('post_processing'):
    non_max_suppression_fn, _ = post_processing_builder.build(
        center_net_config.post_processing)

  return center_net_meta_arch.CenterNetMetaArch(
      is_training=is_training,
      add_summaries=add_summaries,
      num_classes=center_net_config.num_classes,
      feature_extractor=feature_extractor,
      image_resizer_fn=image_resizer_fn,
      object_center_params=object_center_params,
      object_detection_params=object_detection_params,
      keypoint_params_dict=keypoint_params_dict,
      mask_params=mask_params,
      densepose_params=densepose_params,
      track_params=track_params,
      temporal_offset_params=temporal_offset_params,
      use_depthwise=center_net_config.use_depthwise,
      compute_heatmap_sparse=center_net_config.compute_heatmap_sparse,
      non_max_suppression_fn=non_max_suppression_fn)


def _build_center_net_feature_extractor(feature_extractor_config, is_training):
  """Build a CenterNet feature extractor from the given config."""

  if feature_extractor_config.type not in CENTER_NET_EXTRACTOR_FUNCTION_MAP:
    raise ValueError('\'{}\' is not a known CenterNet feature extractor type'
                     .format(feature_extractor_config.type))
  # For backwards compatibility:
  use_separable_conv = (
      feature_extractor_config.use_separable_conv or
      feature_extractor_config.type == 'mobilenet_v2_fpn_sep_conv')
  kwargs = {
      'channel_means':
          list(feature_extractor_config.channel_means),
      'channel_stds':
          list(feature_extractor_config.channel_stds),
      'bgr_ordering':
          feature_extractor_config.bgr_ordering,
      'depth_multiplier':
          feature_extractor_config.depth_multiplier,
      'use_separable_conv':
          use_separable_conv,
      'upsampling_interpolation':
          feature_extractor_config.upsampling_interpolation,
  }


  return CENTER_NET_EXTRACTOR_FUNCTION_MAP[feature_extractor_config.type](
      **kwargs)


META_ARCH_BUILDER_MAP = {
    'center_net': _build_center_net_model
}


def build(model_config, is_training, add_summaries=True):
  """Builds a DetectionModel based on the model config.

  Args:
    model_config: A model.proto object containing the config for the desired
      DetectionModel.
    is_training: True if this model is being built for training purposes.
    add_summaries: Whether to add tensorflow summaries in the model graph.
  Returns:
    DetectionModel based on the config.

  Raises:
    ValueError: On invalid meta architecture or model.
  """
  if not isinstance(model_config, model_pb2.DetectionModel):
    raise ValueError('model_config not of type model_pb2.DetectionModel.')

  meta_architecture = model_config.WhichOneof('model')

  if meta_architecture not in META_ARCH_BUILDER_MAP:
    raise ValueError('Unknown meta architecture: {}'.format(meta_architecture))
  else:
    build_func = META_ARCH_BUILDER_MAP[meta_architecture]
    return build_func(getattr(model_config, meta_architecture), is_training,
                      add_summaries)
