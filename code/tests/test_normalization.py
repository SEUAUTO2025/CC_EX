import numpy as np
import torch

from dar_td3bc.data.normalization import fit_normalization


def test_fit_normalization_transforms_and_inverts_numpy_values():
    values = np.array([[1.0, 10.0], [3.0, 10.0], [5.0, 10.0]])

    stats = fit_normalization(values)
    transformed = stats.transform_numpy(values)
    restored = stats.inverse_numpy(transformed)

    assert np.allclose(stats.mean, [3.0, 10.0])
    assert stats.std[0] > 0.0
    assert stats.std[1] >= 1e-6
    assert np.allclose(transformed[:, 0].mean(), 0.0)
    assert np.allclose(restored, values)


def test_normalization_supports_torch_tensors_on_same_device():
    values = np.array([[1.0, 2.0], [3.0, 6.0]])
    stats = fit_normalization(values)
    tensor = torch.tensor(values, dtype=torch.float32)

    transformed = stats.transform_torch(tensor)

    assert isinstance(transformed, torch.Tensor)
    assert transformed.device == tensor.device
    assert transformed.dtype == tensor.dtype
    assert transformed.shape == tensor.shape
