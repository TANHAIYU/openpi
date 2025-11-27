import dataclasses

import einops
import numpy as np

from openpi import transforms
from openpi.models import model as _model


def make_galbot_example() -> dict:
    """Creates a random input example for the galbot_g1 policy."""
    return {
        "observation/state": np.random.rand(8),
        "observation/image": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "observation/left_wrist_image": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "observation/right_wrist_image": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "prompt": "do something",
    }


@dataclasses.dataclass(frozen=True)
class GalbotOutputs(transforms.DataTransformFn):
    """
    This class is used to convert outputs from the model back the the dataset specific format. It is
    used for inference only.

    For your own dataset, you can copy this class and modify the action dimension based on the comments below.
    """

    def __call__(self, data: dict) -> dict:
        # Only return the first N actions -- since we padded actions above to fit the model action
        # dimension, we need to now parse out the correct number of actions in the return dict.
        # For galbot, we only return the first 8 actions (since the rest is padding).
        # For your own dataset, replace `8` with the action dimension of your dataset.
        return {"actions": np.asarray(data["actions"][:, :8])}
