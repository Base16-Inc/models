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

"""Base target assigner module.

The job of a TargetAssigner is, for a given set of anchors (bounding boxes) and
groundtruth detections (bounding boxes), to assign classification and regression
targets to each anchor as well as weights to each anchor (specifying, e.g.,
which anchors should not contribute to training loss).

It assigns classification/regression targets by performing the following steps:
1) Computing pairwise similarity between anchors and groundtruth boxes using a
  provided RegionSimilarity Calculator
2) Computing a matching based on the similarity matrix using a provided Matcher
3) Assigning regression targets based on the matching and a provided BoxCoder
4) Assigning classification targets based on the matching and groundtruth labels

Note that TargetAssigners only operate on detections from a single
image at a time, so any logic for applying a TargetAssigner to multiple
images must be handled externally.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from six.moves import range
from six.moves import zip
import tensorflow.compat.v1 as tf
import tensorflow.compat.v2 as tf2

from object_detection.core import box_coder
from object_detection.core import box_list
from object_detection.core import box_list_ops
from object_detection.core import keypoint_ops
from object_detection.core import matcher as mat
from object_detection.core import region_similarity_calculator as sim_calc
from object_detection.core import standard_fields as fields
from object_detection.utils import shape_utils
from object_detection.utils import target_assigner_utils as ta_utils

ResizeMethod = tf2.image.ResizeMethod

_DEFAULT_KEYPOINT_OFFSET_STD_DEV = 1.0


class TargetAssigner(object):
  """Target assigner to compute classification and regression targets."""

  def __init__(self,
               similarity_calc,
               matcher,
               box_coder_instance,
               negative_class_weight=1.0):
    """Construct Object Detection Target Assigner.

    Args:
      similarity_calc: a RegionSimilarityCalculator
      matcher: an object_detection.core.Matcher used to match groundtruth to
        anchors.
      box_coder_instance: an object_detection.core.BoxCoder used to encode
        matching groundtruth boxes with respect to anchors.
      negative_class_weight: classification weight to be associated to negative
        anchors (default: 1.0). The weight must be in [0., 1.].

    Raises:
      ValueError: if similarity_calc is not a RegionSimilarityCalculator or
        if matcher is not a Matcher or if box_coder is not a BoxCoder
    """
    if not isinstance(similarity_calc, sim_calc.RegionSimilarityCalculator):
      raise ValueError('similarity_calc must be a RegionSimilarityCalculator')
    if not isinstance(matcher, mat.Matcher):
      raise ValueError('matcher must be a Matcher')
    if not isinstance(box_coder_instance, box_coder.BoxCoder):
      raise ValueError('box_coder must be a BoxCoder')
    self._similarity_calc = similarity_calc
    self._matcher = matcher
    self._box_coder = box_coder_instance
    self._negative_class_weight = negative_class_weight

  @property
  def box_coder(self):
    return self._box_coder

  # TODO(rathodv): move labels, scores, and weights to groundtruth_boxes fields.
  def assign(self,
             anchors,
             groundtruth_boxes,
             groundtruth_labels=None,
             unmatched_class_label=None,
             groundtruth_weights=None):
    """Assign classification and regression targets to each anchor.

    For a given set of anchors and groundtruth detections, match anchors
    to groundtruth_boxes and assign classification and regression targets to
    each anchor as well as weights based on the resulting match (specifying,
    e.g., which anchors should not contribute to training loss).

    Anchors that are not matched to anything are given a classification target
    of self._unmatched_cls_target which can be specified via the constructor.

    Args:
      anchors: a BoxList representing N anchors
      groundtruth_boxes: a BoxList representing M groundtruth boxes
      groundtruth_labels:  a tensor of shape [M, d_1, ... d_k]
        with labels for each of the ground_truth boxes. The subshape
        [d_1, ... d_k] can be empty (corresponding to scalar inputs).  When set
        to None, groundtruth_labels assumes a binary problem where all
        ground_truth boxes get a positive label (of 1).
      unmatched_class_label: a float32 tensor with shape [d_1, d_2, ..., d_k]
        which is consistent with the classification target for each
        anchor (and can be empty for scalar targets).  This shape must thus be
        compatible with the groundtruth labels that are passed to the "assign"
        function (which have shape [num_gt_boxes, d_1, d_2, ..., d_k]).
        If set to None, unmatched_cls_target is set to be [0] for each anchor.
      groundtruth_weights: a float tensor of shape [M] indicating the weight to
        assign to all anchors match to a particular groundtruth box. The weights
        must be in [0., 1.]. If None, all weights are set to 1. Generally no
        groundtruth boxes with zero weight match to any anchors as matchers are
        aware of groundtruth weights. Additionally, `cls_weights` and
        `reg_weights` are calculated using groundtruth weights as an added
        safety.

    Returns:
      cls_targets: a float32 tensor with shape [num_anchors, d_1, d_2 ... d_k],
        where the subshape [d_1, ..., d_k] is compatible with groundtruth_labels
        which has shape [num_gt_boxes, d_1, d_2, ... d_k].
      cls_weights: a float32 tensor with shape [num_anchors, d_1, d_2 ... d_k],
        representing weights for each element in cls_targets.
      reg_targets: a float32 tensor with shape [num_anchors, box_code_dimension]
      reg_weights: a float32 tensor with shape [num_anchors]
      match: an int32 tensor of shape [num_anchors] containing result of anchor
        groundtruth matching. Each position in the tensor indicates an anchor
        and holds the following meaning:
        (1) if match[i] >= 0, anchor i is matched with groundtruth match[i].
        (2) if match[i]=-1, anchor i is marked to be background .
        (3) if match[i]=-2, anchor i is ignored since it is not background and
            does not have sufficient overlap to call it a foreground.

    Raises:
      ValueError: if anchors or groundtruth_boxes are not of type
        box_list.BoxList
    """
    if not isinstance(anchors, box_list.BoxList):
      raise ValueError('anchors must be an BoxList')
    if not isinstance(groundtruth_boxes, box_list.BoxList):
      raise ValueError('groundtruth_boxes must be an BoxList')

    if unmatched_class_label is None:
      unmatched_class_label = tf.constant([0], tf.float32)

    if groundtruth_labels is None:
      groundtruth_labels = tf.ones(tf.expand_dims(groundtruth_boxes.num_boxes(),
                                                  0))
      groundtruth_labels = tf.expand_dims(groundtruth_labels, -1)

    unmatched_shape_assert = shape_utils.assert_shape_equal(
        shape_utils.combined_static_and_dynamic_shape(groundtruth_labels)[1:],
        shape_utils.combined_static_and_dynamic_shape(unmatched_class_label))
    labels_and_box_shapes_assert = shape_utils.assert_shape_equal(
        shape_utils.combined_static_and_dynamic_shape(
            groundtruth_labels)[:1],
        shape_utils.combined_static_and_dynamic_shape(
            groundtruth_boxes.get())[:1])

    if groundtruth_weights is None:
      num_gt_boxes = groundtruth_boxes.num_boxes_static()
      if not num_gt_boxes:
        num_gt_boxes = groundtruth_boxes.num_boxes()
      groundtruth_weights = tf.ones([num_gt_boxes], dtype=tf.float32)

    # set scores on the gt boxes
    scores = 1 - groundtruth_labels[:, 0]
    groundtruth_boxes.add_field(fields.BoxListFields.scores, scores)

    with tf.control_dependencies(
        [unmatched_shape_assert, labels_and_box_shapes_assert]):
      match_quality_matrix = self._similarity_calc.compare(groundtruth_boxes,
                                                           anchors)
      match = self._matcher.match(match_quality_matrix,
                                  valid_rows=tf.greater(groundtruth_weights, 0))
      reg_targets = self._create_regression_targets(anchors,
                                                    groundtruth_boxes,
                                                    match)
      cls_targets = self._create_classification_targets(groundtruth_labels,
                                                        unmatched_class_label,
                                                        match)
      reg_weights = self._create_regression_weights(match, groundtruth_weights)

      cls_weights = self._create_classification_weights(match,
                                                        groundtruth_weights)
      # convert cls_weights from per-anchor to per-class.
      class_label_shape = tf.shape(cls_targets)[1:]
      weights_shape = tf.shape(cls_weights)
      weights_multiple = tf.concat(
          [tf.ones_like(weights_shape), class_label_shape],
          axis=0)
      for _ in range(len(cls_targets.get_shape()[1:])):
        cls_weights = tf.expand_dims(cls_weights, -1)
      cls_weights = tf.tile(cls_weights, weights_multiple)

    num_anchors = anchors.num_boxes_static()
    if num_anchors is not None:
      reg_targets = self._reset_target_shape(reg_targets, num_anchors)
      cls_targets = self._reset_target_shape(cls_targets, num_anchors)
      reg_weights = self._reset_target_shape(reg_weights, num_anchors)
      cls_weights = self._reset_target_shape(cls_weights, num_anchors)

    return (cls_targets, cls_weights, reg_targets, reg_weights,
            match.match_results)

  def _reset_target_shape(self, target, num_anchors):
    """Sets the static shape of the target.

    Args:
      target: the target tensor. Its first dimension will be overwritten.
      num_anchors: the number of anchors, which is used to override the target's
        first dimension.

    Returns:
      A tensor with the shape info filled in.
    """
    target_shape = target.get_shape().as_list()
    target_shape[0] = num_anchors
    target.set_shape(target_shape)
    return target

  def _create_regression_targets(self, anchors, groundtruth_boxes, match):
    """Returns a regression target for each anchor.

    Args:
      anchors: a BoxList representing N anchors
      groundtruth_boxes: a BoxList representing M groundtruth_boxes
      match: a matcher.Match object

    Returns:
      reg_targets: a float32 tensor with shape [N, box_code_dimension]
    """
    matched_gt_boxes = match.gather_based_on_match(
        groundtruth_boxes.get(),
        unmatched_value=tf.zeros(4),
        ignored_value=tf.zeros(4))
    matched_gt_boxlist = box_list.BoxList(matched_gt_boxes)
    if groundtruth_boxes.has_field(fields.BoxListFields.keypoints):
      groundtruth_keypoints = groundtruth_boxes.get_field(
          fields.BoxListFields.keypoints)
      matched_keypoints = match.gather_based_on_match(
          groundtruth_keypoints,
          unmatched_value=tf.zeros(groundtruth_keypoints.get_shape()[1:]),
          ignored_value=tf.zeros(groundtruth_keypoints.get_shape()[1:]))
      matched_gt_boxlist.add_field(fields.BoxListFields.keypoints,
                                   matched_keypoints)
    matched_reg_targets = self._box_coder.encode(matched_gt_boxlist, anchors)
    match_results_shape = shape_utils.combined_static_and_dynamic_shape(
        match.match_results)

    # Zero out the unmatched and ignored regression targets.
    unmatched_ignored_reg_targets = tf.tile(
        self._default_regression_target(), [match_results_shape[0], 1])
    matched_anchors_mask = match.matched_column_indicator()
    reg_targets = tf.where(matched_anchors_mask,
                           matched_reg_targets,
                           unmatched_ignored_reg_targets)
    return reg_targets

  def _default_regression_target(self):
    """Returns the default target for anchors to regress to.

    Default regression targets are set to zero (though in
    this implementation what these targets are set to should
    not matter as the regression weight of any box set to
    regress to the default target is zero).

    Returns:
      default_target: a float32 tensor with shape [1, box_code_dimension]
    """
    return tf.constant([self._box_coder.code_size*[0]], tf.float32)

  def _create_classification_targets(self, groundtruth_labels,
                                     unmatched_class_label, match):
    """Create classification targets for each anchor.

    Assign a classification target of for each anchor to the matching
    groundtruth label that is provided by match.  Anchors that are not matched
    to anything are given the target self._unmatched_cls_target

    Args:
      groundtruth_labels:  a tensor of shape [num_gt_boxes, d_1, ... d_k]
        with labels for each of the ground_truth boxes. The subshape
        [d_1, ... d_k] can be empty (corresponding to scalar labels).
      unmatched_class_label: a float32 tensor with shape [d_1, d_2, ..., d_k]
        which is consistent with the classification target for each
        anchor (and can be empty for scalar targets).  This shape must thus be
        compatible with the groundtruth labels that are passed to the "assign"
        function (which have shape [num_gt_boxes, d_1, d_2, ..., d_k]).
      match: a matcher.Match object that provides a matching between anchors
        and groundtruth boxes.

    Returns:
      a float32 tensor with shape [num_anchors, d_1, d_2 ... d_k], where the
      subshape [d_1, ..., d_k] is compatible with groundtruth_labels which has
      shape [num_gt_boxes, d_1, d_2, ... d_k].
    """
    return match.gather_based_on_match(
        groundtruth_labels,
        unmatched_value=unmatched_class_label,
        ignored_value=unmatched_class_label)

  def _create_regression_weights(self, match, groundtruth_weights):
    """Set regression weight for each anchor.

    Only positive anchors are set to contribute to the regression loss, so this
    method returns a weight of 1 for every positive anchor and 0 for every
    negative anchor.

    Args:
      match: a matcher.Match object that provides a matching between anchors
        and groundtruth boxes.
      groundtruth_weights: a float tensor of shape [M] indicating the weight to
        assign to all anchors match to a particular groundtruth box.

    Returns:
      a float32 tensor with shape [num_anchors] representing regression weights.
    """
    return match.gather_based_on_match(
        groundtruth_weights, ignored_value=0., unmatched_value=0.)

  def _create_classification_weights(self,
                                     match,
                                     groundtruth_weights):
    """Create classification weights for each anchor.

    Positive (matched) anchors are associated with a weight of
    positive_class_weight and negative (unmatched) anchors are associated with
    a weight of negative_class_weight. When anchors are ignored, weights are set
    to zero. By default, both positive/negative weights are set to 1.0,
    but they can be adjusted to handle class imbalance (which is almost always
    the case in object detection).

    Args:
      match: a matcher.Match object that provides a matching between anchors
        and groundtruth boxes.
      groundtruth_weights: a float tensor of shape [M] indicating the weight to
        assign to all anchors match to a particular groundtruth box.

    Returns:
      a float32 tensor with shape [num_anchors] representing classification
      weights.
    """
    return match.gather_based_on_match(
        groundtruth_weights,
        ignored_value=0.,
        unmatched_value=self._negative_class_weight)

  def get_box_coder(self):
    """Get BoxCoder of this TargetAssigner.

    Returns:
      BoxCoder object.
    """
    return self._box_coder

def _smallest_positive_root(a, b, c):
  """Returns the smallest positive root of a quadratic equation."""

  discriminant = tf.sqrt(b ** 2 - 4 * a * c)

  # TODO(vighneshb) We are currently using the slightly incorrect
  # CenterNet implementation. The commented lines implement the fixed version
  # in https://github.com/princeton-vl/CornerNet. Change the implementation
  # after verifying it has no negative impact.
  # root1 = (-b - discriminant) / (2 * a)
  # root2 = (-b + discriminant) / (2 * a)

  # return tf.where(tf.less(root1, 0), root2, root1)

  return (-b + discriminant) / (2.0)


def max_distance_for_overlap(height, width, min_iou):
  """Computes how far apart bbox corners can lie while maintaining the iou.

  Given a bounding box size, this function returns a lower bound on how far
  apart the corners of another box can lie while still maintaining the given
  IoU. The implementation is based on the `gaussian_radius` function in the
  Objects as Points github repo: https://github.com/xingyizhou/CenterNet

  Args:
    height: A 1-D float Tensor representing height of the ground truth boxes.
    width: A 1-D float Tensor representing width of the ground truth boxes.
    min_iou: A float representing the minimum IoU desired.

  Returns:
   distance: A 1-D Tensor of distances, of the same length as the input
     height and width tensors.
  """

  # Given that the detected box is displaced at a distance `d`, the exact
  # IoU value will depend on the angle at which each corner is displaced.
  # We simplify our computation by assuming that each corner is displaced by
  # a distance `d` in both x and y direction. This gives us a lower IoU than
  # what is actually realizable and ensures that any box with corners less
  # than `d` distance apart will always have an IoU greater than or equal
  # to `min_iou`

  # The following 3 cases can be worked on geometrically and come down to
  # solving a quadratic inequality. In each case, to ensure `min_iou` we use
  # the smallest positive root of the equation.

  # Case where detected box is offset from ground truth and no box completely
  # contains the other.

  distance_detection_offset = _smallest_positive_root(
      a=1, b=-(height + width),
      c=width * height * ((1 - min_iou) / (1 + min_iou))
  )

  # Case where detection is smaller than ground truth and completely contained
  # in it.
  distance_detection_in_gt = _smallest_positive_root(
      a=4, b=-2 * (height + width),
      c=(1 - min_iou) * width * height
  )

  # Case where ground truth is smaller than detection and completely contained
  # in it.
  distance_gt_in_detection = _smallest_positive_root(
      a=4 * min_iou, b=(2 * min_iou) * (width + height),
      c=(min_iou - 1) * width * height
  )

  return tf.reduce_min([distance_detection_offset,
                        distance_gt_in_detection,
                        distance_detection_in_gt], axis=0)


def get_batch_predictions_from_indices(batch_predictions, indices):
  """Gets the values of predictions in a batch at the given indices.

  The indices are expected to come from the offset targets generation functions
  in this library. The returned value is intended to be used inside a loss
  function.

  Args:
    batch_predictions: A tensor of shape [batch_size, height, width, channels]
      or [batch_size, height, width, class, channels] for class-specific
      features (e.g. keypoint joint offsets).
    indices: A tensor of shape [num_instances, 3] for single class features or
      [num_instances, 4] for multiple classes features.

  Returns:
    values: A tensor of shape [num_instances, channels] holding the predicted
      values at the given indices.
  """
  # Note, gather_nd (and its gradient scatter_nd) runs significantly slower (on
  # TPU) than gather with flattened inputs, so reshape the tensor, flatten the
  # indices, and run gather.
  shape = shape_utils.combined_static_and_dynamic_shape(batch_predictions)

  # [B, H, W, C] -> [H*W, W, 1] or [B, H, W, N, C] -> [H*W*N, W*N, N, 1]
  rev_cum_interior_indices = tf.reverse(tf.math.cumprod(shape[-2:0:-1]), [0])
  rev_cum_interior_indices = tf.concat([rev_cum_interior_indices, [1]], axis=0)

  # Compute flattened indices and gather.
  flattened_inds = tf.linalg.matmul(
      indices, rev_cum_interior_indices[:, tf.newaxis])[:, 0]
  batch_predictions_2d = tf.reshape(batch_predictions, [-1, shape[-1]])
  return tf.gather(batch_predictions_2d, flattened_inds, axis=0)


def _compute_std_dev_from_box_size(boxes_height, boxes_width, min_overlap):
  """Computes the standard deviation of the Gaussian kernel from box size.

  Args:
    boxes_height: A 1D tensor with shape [num_instances] representing the height
      of each box.
    boxes_width: A 1D tensor with shape [num_instances] representing the width
      of each box.
    min_overlap: The minimum IOU overlap that boxes need to have to not be
      penalized.

  Returns:
    A 1D tensor with shape [num_instances] representing the computed Gaussian
    sigma for each of the box.
  """
  # We are dividing by 3 so that points closer than the computed
  # distance have a >99% CDF.
  sigma = max_distance_for_overlap(boxes_height, boxes_width, min_overlap)
  sigma = (2 * tf.math.maximum(tf.math.floor(sigma), 0.0) + 1) / 6.0
  return sigma


def _preprocess_keypoints_and_weights(out_height, out_width, keypoints,
                                      class_onehot, class_weights,
                                      keypoint_weights, class_id,
                                      keypoint_indices):
  """Preprocesses the keypoints and the corresponding keypoint weights.

  This function performs several common steps to preprocess the keypoints and
  keypoint weights features, including:
    1) Select the subset of keypoints based on the keypoint indices, fill the
       keypoint NaN values with zeros and convert to absolute coordinates.
    2) Generate the weights of the keypoint using the following information:
       a. The class of the instance.
       b. The NaN value of the keypoint coordinates.
       c. The provided keypoint weights.

  Args:
    out_height: An integer or an integer tensor indicating the output height
      of the model.
    out_width: An integer or an integer tensor indicating the output width of
      the model.
    keypoints: A float tensor of shape [num_instances, num_total_keypoints, 2]
      representing the original keypoint grountruth coordinates.
    class_onehot: A float tensor of shape [num_instances, num_classes]
      containing the class targets with the 0th index assumed to map to the
      first non-background class.
    class_weights: A float tensor of shape [num_instances] containing weights
      for groundtruth instances.
    keypoint_weights: A float tensor of shape
      [num_instances, num_total_keypoints] representing the weights of each
      keypoints.
    class_id: int, the ID of the class (0-indexed) that contains the target
      keypoints to consider in this task.
    keypoint_indices: A list of integers representing the indices of the
      keypoints to be considered in this task. This is used to retrieve the
      subset of the keypoints that should be considered in this task.

  Returns:
    A tuple of two tensors:
      keypoint_absolute: A float tensor of shape
        [num_instances, num_keypoints, 2] which is the selected and updated
        keypoint coordinates.
      keypoint_weights: A float tensor of shape [num_instances, num_keypoints]
        representing the updated weight of each keypoint.
  """
  # Select the targets keypoints by their type ids and generate the mask
  # of valid elements.
  valid_mask, keypoints = ta_utils.get_valid_keypoint_mask_for_class(
      keypoint_coordinates=keypoints,
      class_id=class_id,
      class_onehot=class_onehot,
      class_weights=class_weights,
      keypoint_indices=keypoint_indices)
  # Keypoint coordinates in absolute coordinate system.
  # The shape of the tensors: [num_instances, num_keypoints, 2].
  keypoints_absolute = keypoint_ops.to_absolute_coordinates(
      keypoints, out_height, out_width)
  # Assign default weights for the keypoints.
  if keypoint_weights is None:
    keypoint_weights = tf.ones_like(keypoints[:, :, 0])
  else:
    keypoint_weights = tf.gather(
        keypoint_weights, indices=keypoint_indices, axis=1)
  keypoint_weights = keypoint_weights * valid_mask
  return keypoints_absolute, keypoint_weights


class CenterNetCenterHeatmapTargetAssigner(object):
  """Wrapper to compute the object center heatmap."""

  def __init__(self,
               stride,
               min_overlap=0.7,
               compute_heatmap_sparse=False,
               keypoint_class_id=None,
               keypoint_indices=None,
               keypoint_weights_for_center=None):
    """Initializes the target assigner.

    Args:
      stride: int, the stride of the network in output pixels.
      min_overlap: The minimum IOU overlap that boxes need to have to not be
        penalized.
      compute_heatmap_sparse: bool, indicating whether or not to use the sparse
        version of the Op that computes the heatmap. The sparse version scales
        better with number of classes, but in some cases is known to cause
        OOM error. See (b/170989061).
      keypoint_class_id: int, the ID of the class (0-indexed) that contains the
        target keypoints to consider in this task.
      keypoint_indices: A list of integers representing the indices of the
        keypoints to be considered in this task. This is used to retrieve the
        subset of the keypoints from gt_keypoints that should be considered in
        this task.
      keypoint_weights_for_center: The keypoint weights used for calculating the
        location of object center. The number of weights need to be the same as
        the number of keypoints. The object center is calculated by the weighted
        mean of the keypoint locations. If not provided, the object center is
        determined by the center of the bounding box (default behavior).
    """

    self._stride = stride
    self._min_overlap = min_overlap
    self._compute_heatmap_sparse = compute_heatmap_sparse
    self._keypoint_class_id = keypoint_class_id
    self._keypoint_indices = keypoint_indices
    self._keypoint_weights_for_center = keypoint_weights_for_center

  #!DK - READ
  def assign_center_targets_from_boxes(self,
                                       height,
                                       width,
                                       gt_boxes_list,
                                       gt_classes_list,
                                       gt_weights_list=None,
                                       maximum_normalized_coordinate=1.1):
    """Computes the object center heatmap target.

    Args:
      height: int, height of input to the model. This is used to
        determine the height of the output.
      width: int, width of the input to the model. This is used to
        determine the width of the output.
      gt_boxes_list: A list of float tensors with shape [num_boxes, 4]
        representing the groundtruth detection bounding boxes for each sample in
        the batch. The box coordinates are expected in normalized coordinates.
      gt_classes_list: A list of float tensors with shape [num_boxes,
        num_classes] representing the one-hot encoded class labels for each box
        in the gt_boxes_list.
      gt_weights_list: A list of float tensors with shape [num_boxes]
        representing the weight of each groundtruth detection box.
      maximum_normalized_coordinate: Maximum coordinate value to be considered
        as normalized, default to 1.1. This is used to check bounds during
        converting normalized coordinates to absolute coordinates.

    Returns:
      heatmap: A Tensor of size [batch_size, output_height, output_width,
        num_classes] representing the per class center heatmap. output_height
        and output_width are computed by dividing the input height and width by
        the stride specified during initialization.
    """

    out_height = tf.cast(tf.maximum(height // self._stride, 1), tf.float32)
    out_width = tf.cast(tf.maximum(width // self._stride, 1), tf.float32)
    # Compute the yx-grid to be used to generate the heatmap. Each returned
    # tensor has shape of [out_height, out_width]
    (y_grid, x_grid) = ta_utils.image_shape_to_grids(out_height, out_width)

    heatmaps = []
    if gt_weights_list is None:
      gt_weights_list = [None] * len(gt_boxes_list)
    # TODO(vighneshb) Replace the for loop with a batch version.
    for boxes, class_targets, weights in zip(gt_boxes_list, gt_classes_list,
                                             gt_weights_list):
      boxes = box_list.BoxList(boxes)
      # Convert the box coordinates to absolute output image dimension space.
      boxes = box_list_ops.to_absolute_coordinates(
          boxes,
          tf.maximum(height // self._stride, 1),
          tf.maximum(width // self._stride, 1),
          maximum_normalized_coordinate=maximum_normalized_coordinate)
      # Get the box center coordinates. Each returned tensors have the shape of
      # [num_instances]
      (y_center, x_center, boxes_height,
       boxes_width) = boxes.get_center_coordinates_and_sizes()

      # Compute the sigma from box size. The tensor shape: [num_instances].
      sigma = _compute_std_dev_from_box_size(boxes_height, boxes_width,
                                             self._min_overlap)
      # Apply the Gaussian kernel to the center coordinates. Returned heatmap
      # has shape of [out_height, out_width, num_classes]
      heatmap = ta_utils.coordinates_to_heatmap(
          y_grid=y_grid,
          x_grid=x_grid,
          y_coordinates=y_center,
          x_coordinates=x_center,
          sigma=sigma,
          channel_onehot=class_targets,
          channel_weights=weights,
          sparse=self._compute_heatmap_sparse)
      heatmaps.append(heatmap)

    # Return the stacked heatmaps over the batch.
    return tf.stack(heatmaps, axis=0)

  def assign_center_targets_from_keypoints(self,
                                           height,
                                           width,
                                           gt_classes_list,
                                           gt_keypoints_list,
                                           gt_weights_list=None,
                                           gt_keypoints_weights_list=None):
    """Computes the object center heatmap target using keypoint locations.

    Args:
      height: int, height of input to the model. This is used to
        determine the height of the output.
      width: int, width of the input to the model. This is used to
        determine the width of the output.
      gt_classes_list: A list of float tensors with shape [num_boxes,
        num_classes] representing the one-hot encoded class labels for each box
        in the gt_boxes_list.
      gt_keypoints_list: A list of float tensors with shape [num_boxes, 4]
        representing the groundtruth detection bounding boxes for each sample in
        the batch. The box coordinates are expected in normalized coordinates.
      gt_weights_list: A list of float tensors with shape [num_boxes]
        representing the weight of each groundtruth detection box.
      gt_keypoints_weights_list: [Optional] a list of 3D tf.float32 tensors of
        shape [num_instances, num_total_keypoints] representing the weights of
        each keypoints. If not provided, then all not NaN keypoints will be
        equally weighted.

    Returns:
      heatmap: A Tensor of size [batch_size, output_height, output_width,
        num_classes] representing the per class center heatmap. output_height
        and output_width are computed by dividing the input height and width by
        the stride specified during initialization.
    """
    assert (self._keypoint_weights_for_center is not None and
            self._keypoint_class_id is not None and
            self._keypoint_indices is not None)
    out_height = tf.cast(tf.maximum(height // self._stride, 1), tf.float32)
    out_width = tf.cast(tf.maximum(width // self._stride, 1), tf.float32)
    # Compute the yx-grid to be used to generate the heatmap. Each returned
    # tensor has shape of [out_height, out_width]
    (y_grid, x_grid) = ta_utils.image_shape_to_grids(out_height, out_width)

    heatmaps = []
    if gt_weights_list is None:
      gt_weights_list = [None] * len(gt_classes_list)
    if gt_keypoints_weights_list is None:
      gt_keypoints_weights_list = [None] * len(gt_keypoints_list)

    for keypoints, classes, kp_weights, weights in zip(
        gt_keypoints_list, gt_classes_list, gt_keypoints_weights_list,
        gt_weights_list):

      keypoints_absolute, kp_weights = _preprocess_keypoints_and_weights(
          out_height=out_height,
          out_width=out_width,
          keypoints=keypoints,
          class_onehot=classes,
          class_weights=weights,
          keypoint_weights=kp_weights,
          class_id=self._keypoint_class_id,
          keypoint_indices=self._keypoint_indices)
      # _, num_keypoints, _ = (
      #     shape_utils.combined_static_and_dynamic_shape(keypoints_absolute))

      # Update the keypoint weights by the specified keypoints weights.
      kp_loc_weights = tf.constant(
          self._keypoint_weights_for_center, dtype=tf.float32)
      updated_kp_weights = kp_weights * kp_loc_weights[tf.newaxis, :]

      # Obtain the sum of the weights for each instance.
      # instance_weight_sum has shape: [num_instance].
      instance_weight_sum = tf.reduce_sum(updated_kp_weights, axis=1)

      # Weight the keypoint coordinates by updated_kp_weights.
      # weighted_keypoints has shape: [num_instance, num_keypoints, 2]
      weighted_keypoints = keypoints_absolute * tf.expand_dims(
          updated_kp_weights, axis=2)

      # Compute the mean of the keypoint coordinates over the weighted
      # keypoints.
      # keypoint_mean has shape: [num_instance, 2]
      keypoint_mean = tf.math.divide(
          tf.reduce_sum(weighted_keypoints, axis=1),
          tf.expand_dims(instance_weight_sum, axis=-1))

      # Replace the NaN values (due to divided by zeros in the above operation)
      # by 0.0 where the sum of instance weight is zero.
      # keypoint_mean has shape: [num_instance, 2]
      keypoint_mean = tf.where(
          tf.stack([instance_weight_sum, instance_weight_sum], axis=1) > 0.0,
          keypoint_mean, tf.zeros_like(keypoint_mean))

      # Compute the distance from each keypoint to the mean location using
      # broadcasting and weighted by updated_kp_weights.
      # keypoint_dist has shape: [num_instance, num_keypoints]
      keypoint_mean = tf.expand_dims(keypoint_mean, axis=1)
      keypoint_dist = tf.math.sqrt(
          tf.reduce_sum(
              tf.math.square(keypoints_absolute - keypoint_mean), axis=2))
      keypoint_dist = keypoint_dist * updated_kp_weights

      # Compute the average of the distances from each keypoint to the mean
      # location and update the average value by zero when the instance weight
      # is zero.
      # avg_radius has shape: [num_instance]
      avg_radius = tf.math.divide(
          tf.reduce_sum(keypoint_dist, axis=1), instance_weight_sum)
      avg_radius = tf.where(
          instance_weight_sum > 0.0, avg_radius, tf.zeros_like(avg_radius))

      # Update the class instance weight. If the instance doesn't contain enough
      # valid keypoint values (i.e. instance_weight_sum == 0.0), then set the
      # instance weight to zero.
      # updated_class_weights has shape: [num_instance]
      updated_class_weights = tf.where(
          instance_weight_sum > 0.0, weights, tf.zeros_like(weights))

      # Compute the sigma from average distance. We use 2 * average distance to
      # to approximate the width/height of the bounding box.
      # sigma has shape: [num_instances].
      sigma = _compute_std_dev_from_box_size(2 * avg_radius, 2 * avg_radius,
                                             self._min_overlap)

      # Apply the Gaussian kernel to the center coordinates. Returned heatmap
      # has shape of [out_height, out_width, num_classes]
      heatmap = ta_utils.coordinates_to_heatmap(
          y_grid=y_grid,
          x_grid=x_grid,
          y_coordinates=keypoint_mean[:, 0, 0],
          x_coordinates=keypoint_mean[:, 0, 1],
          sigma=sigma,
          channel_onehot=classes,
          channel_weights=updated_class_weights,
          sparse=self._compute_heatmap_sparse)
      heatmaps.append(heatmap)

    # Return the stacked heatmaps over the batch.
    return tf.stack(heatmaps, axis=0)


class CenterNetBoxTargetAssigner(object):
  """Wrapper to compute target tensors for the object detection task.

  This class has methods that take as input a batch of ground truth tensors
  (in the form of a list) and return the targets required to train the object
  detection task.
  """

  def __init__(self, stride):
    """Initializes the target assigner.

    Args:
      stride: int, the stride of the network in output pixels.
    """

    self._stride = stride

  #!DK - READ
  def assign_size_and_offset_targets(self,
                                     height,
                                     width,
                                     gt_boxes_list,
                                     gt_weights_list=None):
    """Returns the box height/width and center offset targets and their indices.

    The returned values are expected to be used with predicted tensors
    of size (batch_size, height//self._stride, width//self._stride, 2). The
    predicted values at the relevant indices can be retrieved with the
    get_batch_predictions_from_indices function.

    Args:
      height: int, height of input to the model. This is used to determine the
        height of the output.
      width: int, width of the input to the model. This is used to determine the
        width of the output.
      gt_boxes_list: A list of float tensors with shape [num_boxes, 4]
        representing the groundtruth detection bounding boxes for each sample in
        the batch. The coordinates are expected in normalized coordinates.
      gt_weights_list: A list of tensors with shape [num_boxes] corresponding to
        the weight of each groundtruth detection box.

    Returns:
      batch_indices: an integer tensor of shape [num_boxes, 3] holding the
        indices inside the predicted tensor which should be penalized. The
        first column indicates the index along the batch dimension and the
        second and third columns indicate the index along the y and x
        dimensions respectively.
      batch_box_height_width: a float tensor of shape [num_boxes, 2] holding
        expected height and width of each box in the output space.
      batch_offsets: a float tensor of shape [num_boxes, 2] holding the
        expected y and x offset of each box in the output space.
      batch_weights: a float tensor of shape [num_boxes] indicating the
        weight of each prediction.
    """

    if gt_weights_list is None:
      gt_weights_list = [None] * len(gt_boxes_list)

    batch_indices = []
    batch_box_height_width = []
    batch_weights = []
    batch_offsets = []

    for i, (boxes, weights) in enumerate(zip(gt_boxes_list, gt_weights_list)):
      boxes = box_list.BoxList(boxes)
      boxes = box_list_ops.to_absolute_coordinates(
          boxes,
          tf.maximum(height // self._stride, 1),
          tf.maximum(width // self._stride, 1))
      # Get the box center coordinates. Each returned tensors have the shape of
      # [num_boxes]
      (y_center, x_center, boxes_height,
       boxes_width) = boxes.get_center_coordinates_and_sizes()
      num_boxes = tf.shape(x_center)

      # Compute the offsets and indices of the box centers. Shape:
      #   offsets: [num_boxes, 2]
      #   indices: [num_boxes, 2]
      (offsets, indices) = ta_utils.compute_floor_offsets_with_indices(
          y_source=y_center, x_source=x_center)

      # Assign ones if weights are not provided.
      if weights is None:
        weights = tf.ones(num_boxes, dtype=tf.float32)

      # Shape of [num_boxes, 1] integer tensor filled with current batch index.
      batch_index = i * tf.ones_like(indices[:, 0:1], dtype=tf.int32)
      batch_indices.append(tf.concat([batch_index, indices], axis=1))
      batch_box_height_width.append(
          tf.stack([boxes_height, boxes_width], axis=1))
      batch_weights.append(weights)
      batch_offsets.append(offsets)

    batch_indices = tf.concat(batch_indices, axis=0)
    batch_box_height_width = tf.concat(batch_box_height_width, axis=0)
    batch_weights = tf.concat(batch_weights, axis=0)
    batch_offsets = tf.concat(batch_offsets, axis=0)
    return (batch_indices, batch_box_height_width, batch_offsets, batch_weights)


# TODO(yuhuic): Update this class to handle the instance/keypoint weights.
# Currently those weights are used as "mask" to indicate whether an
# instance/keypoint should be considered or not (expecting only either 0 or 1
# value). In reality, the weights can be any value and this class should handle
# those values properly.
class CenterNetKeypointTargetAssigner(object):
  """Wrapper to compute target tensors for the CenterNet keypoint estimation.

  This class has methods that take as input a batch of groundtruth tensors
  (in the form of a list) and returns the targets required to train the
  CenterNet model for keypoint estimation. Specifically, the class methods
  expect the groundtruth in the following formats (consistent with the
  standard Object Detection API). Note that usually the groundtruth tensors are
  packed with a list which represents the batch dimension:

  gt_classes_list: [Required] a list of 2D tf.float32 one-hot
    (or k-hot) tensors of shape [num_instances, num_classes] containing the
    class targets with the 0th index assumed to map to the first non-background
    class.
  gt_keypoints_list: [Required] a list of 3D tf.float32 tensors of
    shape [num_instances, num_total_keypoints, 2] containing keypoint
    coordinates. Note that the "num_total_keypoints" should be the sum of the
    num_keypoints over all possible keypoint types, e.g. human pose, face.
    For example, if a dataset contains both 17 human pose keypoints and 5 face
    keypoints, then num_total_keypoints = 17 + 5 = 22.
    If an intance contains only a subet of keypoints (e.g. human pose keypoints
    but not face keypoints), the face keypoints will be filled with zeros.
    Also note that keypoints are assumed to be provided in normalized
    coordinates and missing keypoints should be encoded as NaN.
  gt_keypoints_weights_list: [Optional] a list 3D tf.float32 tensors of shape
    [num_instances, num_total_keypoints] representing the weights of each
    keypoints. If not provided, then all not NaN keypoints will be equally
    weighted.
  gt_boxes_list: [Optional] a list of 2D tf.float32 tensors of shape
    [num_instances, 4] containing coordinates of the groundtruth boxes.
    Groundtruth boxes are provided in [y_min, x_min, y_max, x_max] format and
    assumed to be normalized and clipped relative to the image window with
    y_min <= y_max and x_min <= x_max.
    Note that the boxes are only used to compute the center targets but are not
    considered as required output of the keypoint task. If the boxes were not
    provided, the center targets will be inferred from the keypoints
    [not implemented yet].
  gt_weights_list: [Optional] A list of 1D tf.float32 tensors of shape
    [num_instances] containing weights for groundtruth boxes. Only useful when
    gt_boxes_list is also provided.
  """

  def __init__(self,
               stride,
               class_id,
               keypoint_indices,
               keypoint_std_dev=None,
               per_keypoint_offset=False,
               peak_radius=0,
               compute_heatmap_sparse=False,
               per_keypoint_depth=False):
    """Initializes a CenterNet keypoints target assigner.

    Args:
      stride: int, the stride of the network in output pixels.
      class_id: int, the ID of the class (0-indexed) that contains the target
        keypoints to consider in this task. For example, if the task is human
        pose estimation, the class id should correspond to the "human" class.
      keypoint_indices: A list of integers representing the indices of the
        keypoints to be considered in this task. This is used to retrieve the
        subset of the keypoints from gt_keypoints that should be considered in
        this task.
      keypoint_std_dev: A list of floats represent the standard deviation of the
        Gaussian kernel used to generate the keypoint heatmap (in the unit of
        output pixels). It is to provide the flexibility of using different
        sizes of Gaussian kernel for each keypoint type. If not provided, then
        all standard deviation will be the same as the default value (10.0 in
        the output pixel space). If provided, the length of keypoint_std_dev
        needs to be the same as the length of keypoint_indices, indicating the
        standard deviation of each keypoint type.
      per_keypoint_offset: boolean, indicating whether to assign offset for
        each keypoint channel. If set False, the output offset target will have
        the shape [batch_size, out_height, out_width, 2]. If set True, the
        output offset target will have the shape [batch_size, out_height,
        out_width, 2 * num_keypoints].
      peak_radius: int, the radius (in the unit of output pixel) around heatmap
        peak to assign the offset targets.
      compute_heatmap_sparse: bool, indicating whether or not to use the sparse
        version of the Op that computes the heatmap. The sparse version scales
        better with number of keypoint types, but in some cases is known to
        cause an OOM error. See (b/170989061).
      per_keypoint_depth: A bool indicates whether the model predicts the depth
        of each keypoints in independent channels. Similar to
        per_keypoint_offset but for the keypoint depth.
    """

    self._stride = stride
    self._class_id = class_id
    self._keypoint_indices = keypoint_indices
    self._per_keypoint_offset = per_keypoint_offset
    self._per_keypoint_depth = per_keypoint_depth
    self._peak_radius = peak_radius
    self._compute_heatmap_sparse = compute_heatmap_sparse
    if keypoint_std_dev is None:
      self._keypoint_std_dev = ([_DEFAULT_KEYPOINT_OFFSET_STD_DEV] *
                                len(keypoint_indices))
    else:
      assert len(keypoint_indices) == len(keypoint_std_dev)
      self._keypoint_std_dev = keypoint_std_dev

  def assign_keypoint_heatmap_targets(self,
                                      height,
                                      width,
                                      gt_keypoints_list,
                                      gt_classes_list,
                                      gt_keypoints_weights_list=None,
                                      gt_weights_list=None,
                                      gt_boxes_list=None):
    """Returns the keypoint heatmap targets for the CenterNet model.

    Args:
      height: int, height of input to the CenterNet model. This is used to
        determine the height of the output.
      width: int, width of the input to the CenterNet model. This is used to
        determine the width of the output.
      gt_keypoints_list: A list of float tensors with shape [num_instances,
        num_total_keypoints, 2]. See class-level description for more detail.
      gt_classes_list: A list of float tensors with shape [num_instances,
        num_classes]. See class-level description for more detail.
      gt_keypoints_weights_list: A list of tensors with shape [num_instances,
        num_total_keypoints] corresponding to the weight of each keypoint.
      gt_weights_list: A list of float tensors with shape [num_instances]. See
        class-level description for more detail.
      gt_boxes_list: A list of float tensors with shape [num_instances, 4]. See
        class-level description for more detail. If provided, the keypoint
        standard deviations will be scaled based on the box sizes.

    Returns:
      heatmap: A float tensor of shape [batch_size, output_height, output_width,
        num_keypoints] representing the per keypoint type center heatmap.
        output_height and output_width are computed by dividing the input height
        and width by the stride specified during initialization. Note that the
        "num_keypoints" is defined by the length of keypoint_indices, which is
        not necessarily equal to "num_total_keypoints".
      num_instances_batch: A 2D int tensor of shape
        [batch_size, num_keypoints] representing number of instances for each
        keypoint type.
      valid_mask: A float tensor with shape [batch_size, output_height,
        output_width, num_keypoints] where all values within the regions of the
        blackout boxes are 0.0 and 1.0 else where. Note that the blackout boxes
        are per keypoint type and are blacked out if the keypoint
        visibility/weight (of the corresponding keypoint type) is zero.
    """
    out_width = tf.cast(tf.maximum(width // self._stride, 1), tf.float32)
    out_height = tf.cast(tf.maximum(height // self._stride, 1), tf.float32)
    # Compute the yx-grid to be used to generate the heatmap. Each returned
    # tensor has shape of [out_height, out_width]
    y_grid, x_grid = ta_utils.image_shape_to_grids(out_height, out_width)

    if gt_keypoints_weights_list is None:
      gt_keypoints_weights_list = [None] * len(gt_keypoints_list)
    if gt_weights_list is None:
      gt_weights_list = [None] * len(gt_classes_list)
    if gt_boxes_list is None:
      gt_boxes_list = [None] * len(gt_keypoints_list)

    heatmaps = []
    num_instances_list = []
    valid_mask_list = []
    for keypoints, classes, kp_weights, weights, boxes in zip(
        gt_keypoints_list, gt_classes_list, gt_keypoints_weights_list,
        gt_weights_list, gt_boxes_list):
      keypoints_absolute, kp_weights = _preprocess_keypoints_and_weights(
          out_height=out_height,
          out_width=out_width,
          keypoints=keypoints,
          class_onehot=classes,
          class_weights=weights,
          keypoint_weights=kp_weights,
          class_id=self._class_id,
          keypoint_indices=self._keypoint_indices)
      num_instances, num_keypoints, _ = (
          shape_utils.combined_static_and_dynamic_shape(keypoints_absolute))

      # A tensor of shape [num_instances, num_keypoints] with
      # each element representing the type dimension for each corresponding
      # keypoint:
      # [[0, 1, ..., k-1],
      #  [0, 1, ..., k-1],
      #          :
      #  [0, 1, ..., k-1]]
      keypoint_types = tf.tile(
          input=tf.expand_dims(tf.range(num_keypoints), axis=0),
          multiples=[num_instances, 1])

      # A tensor of shape [num_instances, num_keypoints] with
      # each element representing the sigma of the Gaussian kernel for each
      # keypoint.
      keypoint_std_dev = tf.tile(
          input=tf.expand_dims(tf.constant(self._keypoint_std_dev), axis=0),
          multiples=[num_instances, 1])

      # If boxes is not None, then scale the standard deviation based on the
      # size of the object bounding boxes similar to object center heatmap.
      if boxes is not None:
        boxes = box_list.BoxList(boxes)
        # Convert the box coordinates to absolute output image dimension space.
        boxes = box_list_ops.to_absolute_coordinates(
            boxes,
            tf.maximum(height // self._stride, 1),
            tf.maximum(width // self._stride, 1))
        # Get the box height and width. Each returned tensors have the shape
        # of [num_instances]
        (_, _, boxes_height,
         boxes_width) = boxes.get_center_coordinates_and_sizes()

        # Compute the sigma from box size. The tensor shape: [num_instances].
        sigma = _compute_std_dev_from_box_size(boxes_height, boxes_width, 0.7)
        keypoint_std_dev = keypoint_std_dev * tf.stack(
            [sigma] * num_keypoints, axis=1)

        # Generate the per-keypoint type valid region mask to ignore regions
        # with keypoint weights equal to zeros (e.g. visibility is 0).
        # shape of valid_mask: [out_height, out_width, num_keypoints]
        kp_weight_list = tf.unstack(kp_weights, axis=1)
        valid_mask_channel_list = []
        for kp_weight in kp_weight_list:
          blackout = kp_weight < 1e-3
          valid_mask_channel_list.append(
              ta_utils.blackout_pixel_weights_by_box_regions(
                  out_height, out_width, boxes.get(), blackout))
        valid_mask = tf.stack(valid_mask_channel_list, axis=2)
        valid_mask_list.append(valid_mask)

      # Apply the Gaussian kernel to the keypoint coordinates. Returned heatmap
      # has shape of [out_height, out_width, num_keypoints].
      heatmap = ta_utils.coordinates_to_heatmap(
          y_grid=y_grid,
          x_grid=x_grid,
          y_coordinates=tf.keras.backend.flatten(keypoints_absolute[:, :, 0]),
          x_coordinates=tf.keras.backend.flatten(keypoints_absolute[:, :, 1]),
          sigma=tf.keras.backend.flatten(keypoint_std_dev),
          channel_onehot=tf.one_hot(
              tf.keras.backend.flatten(keypoint_types), depth=num_keypoints),
          channel_weights=tf.keras.backend.flatten(kp_weights))
      num_instances_list.append(
          tf.cast(tf.reduce_sum(kp_weights, axis=0), dtype=tf.int32))
      heatmaps.append(heatmap)
    return (tf.stack(heatmaps, axis=0), tf.stack(num_instances_list, axis=0),
            tf.stack(valid_mask_list, axis=0))

  def _get_keypoint_types(self, num_instances, num_keypoints, num_neighbors):
    """Gets keypoint type index tensor.

    The function prepares the tensor of keypoint indices with shape
    [num_instances, num_keypoints, num_neighbors]. Each element represents the
    keypoint type index for each corresponding keypoint and tiled along the 3rd
    axis:
    [[0, 1, ..., num_keypoints - 1],
     [0, 1, ..., num_keypoints - 1],
             :
     [0, 1, ..., num_keypoints - 1]]

    Args:
      num_instances: int, the number of instances, used to define the 1st
        dimension.
      num_keypoints: int, the number of keypoint types, used to define the 2nd
        dimension.
      num_neighbors: int, the number of neighborhood pixels to consider for each
        keypoint, used to define the 3rd dimension.

    Returns:
      A integer tensor of shape [num_instances, num_keypoints, num_neighbors].
    """
    keypoint_types = tf.range(num_keypoints)[tf.newaxis, :, tf.newaxis]
    tiled_keypoint_types = tf.tile(keypoint_types,
                                   multiples=[num_instances, 1, num_neighbors])
    return tiled_keypoint_types

  def assign_keypoints_offset_targets(self,
                                      height,
                                      width,
                                      gt_keypoints_list,
                                      gt_classes_list,
                                      gt_keypoints_weights_list=None,
                                      gt_weights_list=None):
    """Returns the offsets and indices of the keypoints for location refinement.

    The returned values are used to refine the location of each keypoints in the
    heatmap. The predicted values at the relevant indices can be retrieved with
    the get_batch_predictions_from_indices function.

    Args:
      height: int, height of input to the CenterNet model. This is used to
        determine the height of the output.
      width: int, width of the input to the CenterNet model. This is used to
        determine the width of the output.
      gt_keypoints_list: A list of tensors with shape [num_instances,
        num_total_keypoints]. See class-level description for more detail.
      gt_classes_list: A list of tensors with shape [num_instances,
        num_classes]. See class-level description for more detail.
      gt_keypoints_weights_list: A list of tensors with shape [num_instances,
        num_total_keypoints] corresponding to the weight of each keypoint.
      gt_weights_list: A list of float tensors with shape [num_instances]. See
        class-level description for more detail.

    Returns:
      batch_indices: an integer tensor of shape [num_total_instances, 3] (or
        [num_total_instances, 4] if 'per_keypoint_offset' is set True) holding
        the indices inside the predicted tensor which should be penalized. The
        first column indicates the index along the batch dimension and the
        second and third columns indicate the index along the y and x
        dimensions respectively. The fourth column corresponds to the channel
        dimension (if 'per_keypoint_offset' is set True).
      batch_offsets: a float tensor of shape [num_total_instances, 2] holding
        the expected y and x offset of each box in the output space.
      batch_weights: a float tensor of shape [num_total_instances] indicating
        the weight of each prediction.
      Note that num_total_instances = batch_size * num_instances *
                                      num_keypoints * num_neighbors
    """

    batch_indices = []
    batch_offsets = []
    batch_weights = []

    if gt_keypoints_weights_list is None:
      gt_keypoints_weights_list = [None] * len(gt_keypoints_list)
    if gt_weights_list is None:
      gt_weights_list = [None] * len(gt_classes_list)
    for i, (keypoints, classes, kp_weights, weights) in enumerate(
        zip(gt_keypoints_list, gt_classes_list, gt_keypoints_weights_list,
            gt_weights_list)):
      keypoints_absolute, kp_weights = _preprocess_keypoints_and_weights(
          out_height=tf.maximum(height // self._stride, 1),
          out_width=tf.maximum(width // self._stride, 1),
          keypoints=keypoints,
          class_onehot=classes,
          class_weights=weights,
          keypoint_weights=kp_weights,
          class_id=self._class_id,
          keypoint_indices=self._keypoint_indices)
      num_instances, num_keypoints, _ = (
          shape_utils.combined_static_and_dynamic_shape(keypoints_absolute))

      # [num_instances * num_keypoints]
      y_source = tf.keras.backend.flatten(keypoints_absolute[:, :, 0])
      x_source = tf.keras.backend.flatten(keypoints_absolute[:, :, 1])

      # All keypoint coordinates and their neighbors:
      # [num_instance * num_keypoints, num_neighbors]
      (y_source_neighbors, x_source_neighbors,
       valid_sources) = ta_utils.get_surrounding_grids(
           tf.cast(tf.maximum(height // self._stride, 1), tf.float32),
           tf.cast(tf.maximum(width // self._stride, 1), tf.float32),
           y_source, x_source,
           self._peak_radius)
      _, num_neighbors = shape_utils.combined_static_and_dynamic_shape(
          y_source_neighbors)

      # Update the valid keypoint weights.
      # [num_instance * num_keypoints, num_neighbors]
      valid_keypoints = tf.cast(
          valid_sources, dtype=tf.float32) * tf.stack(
              [tf.keras.backend.flatten(kp_weights)] * num_neighbors, axis=-1)

      # Compute the offsets and indices of the box centers. Shape:
      #   offsets: [num_instances * num_keypoints, num_neighbors, 2]
      #   indices: [num_instances * num_keypoints, num_neighbors, 2]
      offsets, indices = ta_utils.compute_floor_offsets_with_indices(
          y_source=y_source_neighbors,
          x_source=x_source_neighbors,
          y_target=y_source,
          x_target=x_source)
      # Reshape to:
      #   offsets: [num_instances * num_keypoints * num_neighbors, 2]
      #   indices: [num_instances * num_keypoints * num_neighbors, 2]
      offsets = tf.reshape(offsets, [-1, 2])
      indices = tf.reshape(indices, [-1, 2])

      # Prepare the batch indices to be prepended.
      batch_index = tf.fill(
          [num_instances * num_keypoints * num_neighbors, 1], i)
      if self._per_keypoint_offset:
        tiled_keypoint_types = self._get_keypoint_types(
            num_instances, num_keypoints, num_neighbors)
        batch_indices.append(
            tf.concat([batch_index, indices,
                       tf.reshape(tiled_keypoint_types, [-1, 1])], axis=1))
      else:
        batch_indices.append(tf.concat([batch_index, indices], axis=1))
      batch_offsets.append(offsets)
      batch_weights.append(tf.keras.backend.flatten(valid_keypoints))

    # Concatenate the tensors in the batch in the first dimension:
    # shape: [batch_size * num_instances * num_keypoints * num_neighbors, 3] or
    # [batch_size * num_instances * num_keypoints * num_neighbors, 4] if
    # 'per_keypoint_offset' is set to True.
    batch_indices = tf.concat(batch_indices, axis=0)
    # shape: [batch_size * num_instances * num_keypoints * num_neighbors]
    batch_weights = tf.concat(batch_weights, axis=0)
    # shape: [batch_size * num_instances * num_keypoints * num_neighbors, 2]
    batch_offsets = tf.concat(batch_offsets, axis=0)
    return (batch_indices, batch_offsets, batch_weights)

  def assign_keypoints_depth_targets(self,
                                     height,
                                     width,
                                     gt_keypoints_list,
                                     gt_classes_list,
                                     gt_keypoint_depths_list,
                                     gt_keypoint_depth_weights_list,
                                     gt_keypoints_weights_list=None,
                                     gt_weights_list=None):
    """Returns the target depths of the keypoints.

    The returned values are the relative depth information of each keypoints.

    Args:
      height: int, height of input to the CenterNet model. This is used to
        determine the height of the output.
      width: int, width of the input to the CenterNet model. This is used to
        determine the width of the output.
      gt_keypoints_list: A list of tensors with shape [num_instances,
        num_total_keypoints, 2]. See class-level description for more detail.
      gt_classes_list: A list of tensors with shape [num_instances,
        num_classes]. See class-level description for more detail.
      gt_keypoint_depths_list: A list of tensors with shape [num_instances,
        num_total_keypoints] corresponding to the relative depth of the
        keypoints.
      gt_keypoint_depth_weights_list: A list of tensors with shape
        [num_instances, num_total_keypoints] corresponding to the weights of
        the relative depth.
      gt_keypoints_weights_list: A list of tensors with shape [num_instances,
        num_total_keypoints] corresponding to the weight of each keypoint.
      gt_weights_list: A list of float tensors with shape [num_instances]. See
        class-level description for more detail.

    Returns:
      batch_indices: an integer tensor of shape [num_total_instances, 3] (or
        [num_total_instances, 4] if 'per_keypoint_depth' is set True) holding
        the indices inside the predicted tensor which should be penalized. The
        first column indicates the index along the batch dimension and the
        second and third columns indicate the index along the y and x
        dimensions respectively. The fourth column corresponds to the channel
        dimension (if 'per_keypoint_offset' is set True).
      batch_depths: a float tensor of shape [num_total_instances, 1] (or
        [num_total_instances, num_keypoints] if per_keypoint_depth is set True)
        indicating the target depth of each keypoint.
      batch_weights: a float tensor of shape [num_total_instances] indicating
        the weight of each prediction.
      Note that num_total_instances = batch_size * num_instances *
                                      num_keypoints * num_neighbors
    """

    batch_indices = []
    batch_weights = []
    batch_depths = []

    if gt_keypoints_weights_list is None:
      gt_keypoints_weights_list = [None] * len(gt_keypoints_list)
    if gt_weights_list is None:
      gt_weights_list = [None] * len(gt_classes_list)
    if gt_keypoint_depths_list is None:
      gt_keypoint_depths_list = [None] * len(gt_classes_list)
    for i, (keypoints, classes, kp_weights, weights,
            keypoint_depths, keypoint_depth_weights) in enumerate(
                zip(gt_keypoints_list, gt_classes_list,
                    gt_keypoints_weights_list, gt_weights_list,
                    gt_keypoint_depths_list, gt_keypoint_depth_weights_list)):
      keypoints_absolute, kp_weights = _preprocess_keypoints_and_weights(
          out_height=tf.maximum(height // self._stride, 1),
          out_width=tf.maximum(width // self._stride, 1),
          keypoints=keypoints,
          class_onehot=classes,
          class_weights=weights,
          keypoint_weights=kp_weights,
          class_id=self._class_id,
          keypoint_indices=self._keypoint_indices)
      num_instances, num_keypoints, _ = (
          shape_utils.combined_static_and_dynamic_shape(keypoints_absolute))

      # [num_instances * num_keypoints]
      y_source = tf.keras.backend.flatten(keypoints_absolute[:, :, 0])
      x_source = tf.keras.backend.flatten(keypoints_absolute[:, :, 1])

      # All keypoint coordinates and their neighbors:
      # [num_instance * num_keypoints, num_neighbors]
      (y_source_neighbors, x_source_neighbors,
       valid_sources) = ta_utils.get_surrounding_grids(
           tf.cast(tf.maximum(height // self._stride, 1), tf.float32),
           tf.cast(tf.maximum(width // self._stride, 1), tf.float32),
           y_source, x_source,
           self._peak_radius)
      _, num_neighbors = shape_utils.combined_static_and_dynamic_shape(
          y_source_neighbors)

      # Update the valid keypoint weights.
      # [num_instance * num_keypoints, num_neighbors]
      valid_keypoints = tf.cast(
          valid_sources, dtype=tf.float32) * tf.stack(
              [tf.keras.backend.flatten(kp_weights)] * num_neighbors, axis=-1)

      # Compute the offsets and indices of the box centers. Shape:
      #   indices: [num_instances * num_keypoints, num_neighbors, 2]
      _, indices = ta_utils.compute_floor_offsets_with_indices(
          y_source=y_source_neighbors,
          x_source=x_source_neighbors,
          y_target=y_source,
          x_target=x_source)
      # Reshape to:
      #   indices: [num_instances * num_keypoints * num_neighbors, 2]
      indices = tf.reshape(indices, [-1, 2])

      # Gather the keypoint depth from corresponding keypoint indices:
      #   [num_instances, num_keypoints]
      keypoint_depths = tf.gather(
          keypoint_depths, self._keypoint_indices, axis=1)
      # Tile the depth target to surrounding pixels.
      #   [num_instances, num_keypoints, num_neighbors]
      tiled_keypoint_depths = tf.tile(
          tf.expand_dims(keypoint_depths, axis=-1),
          multiples=[1, 1, num_neighbors])

      # [num_instances, num_keypoints]
      keypoint_depth_weights = tf.gather(
          keypoint_depth_weights, self._keypoint_indices, axis=1)
      # [num_instances, num_keypoints, num_neighbors]
      keypoint_depth_weights = tf.tile(
          tf.expand_dims(keypoint_depth_weights, axis=-1),
          multiples=[1, 1, num_neighbors])
      # Update the weights of keypoint depth by the weights of the keypoints.
      # A keypoint depth target is valid only if its corresponding keypoint
      # target is also valid.
      # [num_instances, num_keypoints, num_neighbors]
      tiled_depth_weights = (
          tf.reshape(valid_keypoints,
                     [num_instances, num_keypoints, num_neighbors]) *
          keypoint_depth_weights)
      invalid_depths = tf.logical_or(
          tf.math.is_nan(tiled_depth_weights),
          tf.math.is_nan(tiled_keypoint_depths))
      # Assign zero values and weights to NaN values.
      final_keypoint_depths = tf.where(invalid_depths,
                                       tf.zeros_like(tiled_keypoint_depths),
                                       tiled_keypoint_depths)
      final_keypoint_depth_weights = tf.where(
          invalid_depths,
          tf.zeros_like(tiled_depth_weights),
          tiled_depth_weights)
      # [num_instances * num_keypoints * num_neighbors, 1]
      batch_depths.append(tf.reshape(final_keypoint_depths, [-1, 1]))

      # Prepare the batch indices to be prepended.
      batch_index = tf.fill(
          [num_instances * num_keypoints * num_neighbors, 1], i)
      if self._per_keypoint_depth:
        tiled_keypoint_types = self._get_keypoint_types(
            num_instances, num_keypoints, num_neighbors)
        batch_indices.append(
            tf.concat([batch_index, indices,
                       tf.reshape(tiled_keypoint_types, [-1, 1])], axis=1))
      else:
        batch_indices.append(tf.concat([batch_index, indices], axis=1))
      batch_weights.append(
          tf.keras.backend.flatten(final_keypoint_depth_weights))

    # Concatenate the tensors in the batch in the first dimension:
    # shape: [batch_size * num_instances * num_keypoints * num_neighbors, 3] or
    # [batch_size * num_instances * num_keypoints * num_neighbors, 4] if
    # 'per_keypoint_offset' is set to True.
    batch_indices = tf.concat(batch_indices, axis=0)
    # shape: [batch_size * num_instances * num_keypoints * num_neighbors]
    batch_weights = tf.concat(batch_weights, axis=0)
    # shape: [batch_size * num_instances * num_keypoints * num_neighbors, 1]
    batch_depths = tf.concat(batch_depths, axis=0)
    return (batch_indices, batch_depths, batch_weights)

  def assign_joint_regression_targets(self,
                                      height,
                                      width,
                                      gt_keypoints_list,
                                      gt_classes_list,
                                      gt_boxes_list=None,
                                      gt_keypoints_weights_list=None,
                                      gt_weights_list=None):
    """Returns the joint regression from center grid to keypoints.

    The joint regression is used as the grouping cue from the estimated
    keypoints to instance center. The offsets are the vectors from the floored
    object center coordinates to the keypoint coordinates.

    Args:
      height: int, height of input to the CenterNet model. This is used to
        determine the height of the output.
      width: int, width of the input to the CenterNet model. This is used to
        determine the width of the output.
      gt_keypoints_list: A list of float tensors with shape [num_instances,
        num_total_keypoints]. See class-level description for more detail.
      gt_classes_list: A list of float tensors with shape [num_instances,
        num_classes]. See class-level description for more detail.
      gt_boxes_list: A list of float tensors with shape [num_instances, 4]. See
        class-level description for more detail. If provided, then the center
        targets will be computed based on the center of the boxes.
      gt_keypoints_weights_list: A list of float tensors with shape
        [num_instances, num_total_keypoints] representing to the weight of each
        keypoint.
      gt_weights_list: A list of float tensors with shape [num_instances]. See
        class-level description for more detail.

    Returns:
      batch_indices: an integer tensor of shape [num_instances, 4] holding the
        indices inside the predicted tensor which should be penalized. The
        first column indicates the index along the batch dimension and the
        second and third columns indicate the index along the y and x
        dimensions respectively, the last dimension refers to the keypoint type
        dimension.
      batch_offsets: a float tensor of shape [num_instances, 2] holding the
        expected y and x offset of each box in the output space.
      batch_weights: a float tensor of shape [num_instances] indicating the
        weight of each prediction.
      Note that num_total_instances = batch_size * num_instances * num_keypoints

    Raises:
      NotImplementedError: currently the object center coordinates need to be
        computed from groundtruth bounding boxes. The functionality of
        generating the object center coordinates from keypoints is not
        implemented yet.
    """

    batch_indices = []
    batch_offsets = []
    batch_weights = []
    batch_size = len(gt_keypoints_list)
    if gt_keypoints_weights_list is None:
      gt_keypoints_weights_list = [None] * batch_size
    if gt_boxes_list is None:
      gt_boxes_list = [None] * batch_size
    if gt_weights_list is None:
      gt_weights_list = [None] * len(gt_classes_list)
    for i, (keypoints, classes, boxes, kp_weights, weights) in enumerate(
        zip(gt_keypoints_list, gt_classes_list,
            gt_boxes_list, gt_keypoints_weights_list, gt_weights_list)):
      keypoints_absolute, kp_weights = _preprocess_keypoints_and_weights(
          out_height=tf.maximum(height // self._stride, 1),
          out_width=tf.maximum(width // self._stride, 1),
          keypoints=keypoints,
          class_onehot=classes,
          class_weights=weights,
          keypoint_weights=kp_weights,
          class_id=self._class_id,
          keypoint_indices=self._keypoint_indices)
      num_instances, num_keypoints, _ = (
          shape_utils.combined_static_and_dynamic_shape(keypoints_absolute))

      # If boxes are provided, compute the joint center from it.
      if boxes is not None:
        # Compute joint center from boxes.
        boxes = box_list.BoxList(boxes)
        boxes = box_list_ops.to_absolute_coordinates(
            boxes,
            tf.maximum(height // self._stride, 1),
            tf.maximum(width // self._stride, 1))
        y_center, x_center, _, _ = boxes.get_center_coordinates_and_sizes()
      else:
        # TODO(yuhuic): Add the logic to generate object centers from keypoints.
        raise NotImplementedError((
            'The functionality of generating object centers from keypoints is'
            ' not implemented yet. Please provide groundtruth bounding boxes.'
        ))

      # Tile the yx center coordinates to be the same shape as keypoints.
      y_center_tiled = tf.tile(
          tf.reshape(y_center, shape=[num_instances, 1]),
          multiples=[1, num_keypoints])
      x_center_tiled = tf.tile(
          tf.reshape(x_center, shape=[num_instances, 1]),
          multiples=[1, num_keypoints])
      # [num_instance * num_keypoints, num_neighbors]
      (y_source_neighbors, x_source_neighbors,
       valid_sources) = ta_utils.get_surrounding_grids(
           tf.cast(tf.maximum(height // self._stride, 1), tf.float32),
           tf.cast(tf.maximum(width // self._stride, 1), tf.float32),
           tf.keras.backend.flatten(y_center_tiled),
           tf.keras.backend.flatten(x_center_tiled), self._peak_radius)

      _, num_neighbors = shape_utils.combined_static_and_dynamic_shape(
          y_source_neighbors)
      valid_keypoints = tf.cast(
          valid_sources, dtype=tf.float32) * tf.stack(
              [tf.keras.backend.flatten(kp_weights)] * num_neighbors, axis=-1)

      # Compute the offsets and indices of the box centers. Shape:
      #   offsets: [num_instances * num_keypoints, 2]
      #   indices: [num_instances * num_keypoints, 2]
      (offsets, indices) = ta_utils.compute_floor_offsets_with_indices(
          y_source=y_source_neighbors,
          x_source=x_source_neighbors,
          y_target=tf.keras.backend.flatten(keypoints_absolute[:, :, 0]),
          x_target=tf.keras.backend.flatten(keypoints_absolute[:, :, 1]))
      # Reshape to:
      #   offsets: [num_instances * num_keypoints * num_neighbors, 2]
      #   indices: [num_instances * num_keypoints * num_neighbors, 2]
      offsets = tf.reshape(offsets, [-1, 2])
      indices = tf.reshape(indices, [-1, 2])

      # keypoint type tensor: [num_instances, num_keypoints, num_neighbors].
      tiled_keypoint_types = self._get_keypoint_types(
          num_instances, num_keypoints, num_neighbors)

      batch_index = tf.fill(
          [num_instances * num_keypoints * num_neighbors, 1], i)
      batch_indices.append(
          tf.concat([batch_index, indices,
                     tf.reshape(tiled_keypoint_types, [-1, 1])], axis=1))
      batch_offsets.append(offsets)
      batch_weights.append(tf.keras.backend.flatten(valid_keypoints))

    # Concatenate the tensors in the batch in the first dimension:
    # shape: [batch_size * num_instances * num_keypoints, 4]
    batch_indices = tf.concat(batch_indices, axis=0)
    # shape: [batch_size * num_instances * num_keypoints]
    batch_weights = tf.concat(batch_weights, axis=0)
    # shape: [batch_size * num_instances * num_keypoints, 2]
    batch_offsets = tf.concat(batch_offsets, axis=0)
    return (batch_indices, batch_offsets, batch_weights)
