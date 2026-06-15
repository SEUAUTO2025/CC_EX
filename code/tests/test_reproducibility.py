import random

import numpy as np
import torch

from dar_td3bc.utils.seed import set_global_seed


def test_set_global_seed_repeats_python_numpy_and_torch_draws():
    set_global_seed(123)
    first = (random.random(), np.random.rand(), torch.rand(1).item())

    set_global_seed(123)
    second = (random.random(), np.random.rand(), torch.rand(1).item())

    assert first == second
