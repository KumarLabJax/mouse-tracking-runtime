"""TensorFlow inference module for mouse tracking."""

from .arena_corners import infer_arena_corner_model
from .food_hopper import infer_food_hopper_model
from .lixit import infer_lixit_model
from .multi_identity import infer_multi_identity_tfs
from .multi_segmentation import infer_multi_segmentation_tfs
from .single_segmentation import infer_single_segmentation_tfs
