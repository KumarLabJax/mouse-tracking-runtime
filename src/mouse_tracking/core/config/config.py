"""Configuration for the mouse tracking runtime."""

from functools import lru_cache
from typing import Any

from pydantic_settings import BaseSettings, SettingsConfigDict


class MouseTrackingConfig(BaseSettings):
    """Configuration for the mouse tracking runtime."""

    model_config = SettingsConfigDict(
        case_sensitive=False,
        env_prefix="MOUSE_TRACKING_",
        env_file=".env",
        extra="allow",
    )

    # Model base directory
    model_directory: str = "/kumar_lab_models/models/"

    @property
    def single_mouse_segmentation(self) -> dict[str, dict[str, Any]]:
        """Single mouse segmentation model definitions."""
        return {
            "tracking-paper": {
                "model-name": "full-model-tracking-paper",
                "model-checkpoint": "model.ckpt-415000",
                "tfs-model": self.model_directory
                + "tfs-models/single-mouse-segmentation/tracking-paper/",
            },
        }

    @property
    def multi_mouse_segmentation(self) -> dict[str, dict[str, Any]]:
        """Multi mouse segmentation model definitions."""
        return {
            "social-paper": {
                "model-name": "panoptic-deeplab-res50_v2",
                "model-checkpoint": "ckpt-125000",
                "tfs-model": self.model_directory
                + "tfs-models/multi-mouse-segmentation/panoptic-deeplab/",
            },
        }

    @property
    def single_mouse_pose(self) -> dict[str, dict[str, Any]]:
        """Single mouse pose model definitions."""
        return {
            "gait-paper": {
                "model-name": "gait-model",
                "model-checkpoint": "2019-06-26-param-search/mp-conf4.yaml",
                "pytorch-config": self.model_directory
                + "pytorch-models/single-mouse-pose/gait-model.yaml",
                "pytorch-model": self.model_directory
                + "pytorch-models/single-mouse-pose/gait-model.pth",
            },
        }

    @property
    def multi_mouse_pose(self) -> dict[str, dict[str, Any]]:
        """Multi mouse pose model definitions."""
        return {
            "social-paper-topdown": {
                "model-name": "topdown",
                "model-checkpoint": "multimouse_topdown_1.yaml",
                "pytorch-config": self.model_directory
                + "pytorch-models/multi-mouse-pose/social-topdown.yaml",
                "pytorch-model": self.model_directory
                + "pytorch-models/multi-mouse-pose/social-topdown.pth",
            }
        }

    @property
    def multi_mouse_identity(self) -> dict[str, dict[str, Any]]:
        """Multi mouse identity model definitions."""
        return {
            "social-paper": {
                "model-name": "TrackIDTrain_MNAS_latent16",
                "model-checkpoint": "model.ckpt-183819",
                "tfs-model": self.model_directory
                + "tfs-models/multi-mouse-identity/mnas_2021/",
            },
            "2023": {
                "model-name": "TrackIDTrain_MNAS_latent16",
                "model-checkpoint": "model.ckpt-290566",
                "tfs-model": self.model_directory
                + "tfs-models/multi-mouse-identity/mnas_2023/",
            },
        }

    @property
    def fecal_boli(self) -> dict[str, dict[str, Any]]:
        """Fecal boli model definitions."""
        return {
            "fecal-boli": {
                "model-name": "fecal-boli",
                "model-checkpoint": "fecalboli/fecalboli_2020-06-19_02.yaml",
                "pytorch-config": self.model_directory
                + "pytorch-models/fecal-boli/fecalboli-2020-06-19.yaml",
                "pytorch-model": self.model_directory
                + "pytorch-models/fecal-boli/fecalboli-2020-06-19.pth",
            }
        }

    @property
    def static_arena_corners(self) -> dict[str, dict[str, Any]]:
        """Static arena corners model definitions."""
        return {
            "social-2022-pipeline": {
                "model-name": "obj-api-kp",
                "model-checkpoint": "2022-11-21/ckpt-101",
                "tfs-model": self.model_directory
                + "tfs-models/static-object-arena/obj-api-2022/",
            },
        }

    @property
    def static_food_corners(self) -> dict[str, dict[str, Any]]:
        """Static food corners model definitions."""
        return {
            "social-2022-pipeline": {
                "model-name": "obj-api-seg",
                "model-checkpoint": "2022-11-28/ckpt-101",
                "tfs-model": self.model_directory
                + "tfs-models/static-object-food/obj-api-2022/",
            },
        }

    @property
    def static_lixit(self) -> dict[str, dict[str, Any]]:
        """Static lixit model definitions."""
        return {
            "social-2022-pipeline": {
                "model-name": "dlc-lixit",
                "model-checkpoint": "iteration-0/final-aug-lixitJan3-trainset95shuffle1/train/snapshot-200000",
                "tfs-model": self.model_directory
                + "tfs-models/static-object-lixit/dlc-2022/",
            },
        }


@lru_cache(maxsize=1)
def get_config() -> MouseTrackingConfig:
    """
    Get the application configuration singleton.

    This function is cached and will return the same instance on subsequent calls.
    """
    return MouseTrackingConfig()
