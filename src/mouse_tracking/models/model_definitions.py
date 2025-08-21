"""Definitions of trained models."""

from mouse_tracking.core.config import get_config

# Initialize config instance
_config = get_config()

# Backwards compatible model_folder access
model_folder = _config.model_directory

# Backwards compatible model definitions - these now use the config
SINGLE_MOUSE_SEGMENTATION = _config.single_mouse_segmentation
MULTI_MOUSE_SEGMENTATION = _config.multi_mouse_segmentation  
SINGLE_MOUSE_POSE = _config.single_mouse_pose
MULTI_MOUSE_POSE = _config.multi_mouse_pose
MULTI_MOUSE_IDENTITY = _config.multi_mouse_identity
FECAL_BOLI = _config.fecal_boli
STATIC_ARENA_CORNERS = _config.static_arena_corners
STATIC_FOOD_CORNERS = _config.static_food_corners
STATIC_LIXIT = _config.static_lixit
