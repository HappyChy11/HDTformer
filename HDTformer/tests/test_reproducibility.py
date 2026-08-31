import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from hdtformer.data import load_and_split_data
from hdtformer.model import HDTformer
from hdtformer.msdcd import DynamicConv1D, MSDCD
from hdtformer.temporal_vector import TemporalVector
from sklearn.preprocessing import MinMaxScaler
from tools.run_experiment import update_recursive_inputs


class ReproducibilityTests(unittest.TestCase):
    def test_temporal_vector_selection_and_shape(self):
        torch.manual_seed(12)
        module = TemporalVector(
            sequence_length=20,
            input_feature_count=5,
            linear_feature_index=0,
            periodic_feature_indices=[0, 1, 2, 3],
        )
        inputs = torch.randn(2, 20, 5)
        outputs = module(inputs)
        self.assertEqual(outputs.shape, (2, 20, 10))

        expected_linear = (
            module.weights_linear * inputs[:, :, 0] + module.bias_linear
        )
        expected_periodic = torch.sin(
            module.weights_periodic.unsqueeze(0) * inputs[:, :, :4]
            + module.bias_periodic.unsqueeze(0)
        )
        torch.testing.assert_close(outputs[:, :, 5], expected_linear)
        torch.testing.assert_close(outputs[:, :, 6:], expected_periodic)

    def test_dynamic_weight_dimensions(self):
        module = DynamicConv1D(
            in_channels=1,
            out_channels=1,
            kernel_size=7,
            num_basis_kernels=4,
            basis_kernel_init={"distribution": "normal", "mean": 0.0, "std": 1.0},
        )
        self.assertEqual(tuple(module.weight_generator.weight.shape), (4, 1))
        self.assertEqual(tuple(module.weight_generator.bias.shape), (4,))
        self.assertEqual(tuple(module.basis_kernels.shape), (4, 1, 1, 7))
        self.assertEqual(module(torch.randn(3, 1, 20)).shape, (3, 1, 20))

    def test_msdcd_and_hdtformer_forward_shapes(self):
        msdcd_config = {
            "trend_kernel_size": 20,
            "seasonal_kernel_size": 7,
            "num_basis_kernels": 4,
            "avg_pool_kernel_size": 3,
            "basis_kernel_init": {
                "distribution": "normal",
                "mean": 0.0,
                "std": 1.0,
            },
        }
        decomposition = MSDCD(**msdcd_config)
        components = decomposition(torch.randn(2, 240, 1))
        self.assertTrue(all(component.shape == (2, 240) for component in components))

        model = HDTformer(
            short_lookback=20,
            temporal_vector={
                "input_feature_count": 5,
                "linear_feature_index": 0,
                "periodic_feature_indices": [0, 1, 2, 3],
                "init_mean": 0.0,
                "init_std": 1.0,
            },
            msdcd=msdcd_config,
        )
        output = model(torch.randn(2, 20, 5), torch.randn(2, 240, 1))
        self.assertEqual(output.shape, (2, 1))
        self.assertEqual(model.std.embedding.in_features, 10)

    def test_causal_feature_order(self):
        length = 2000
        timestamps = pd.date_range("2022-01-01", periods=length, freq="min", tz="UTC")
        values = np.linspace(0.1, 1.0, length, dtype=np.float32)
        frame = pd.DataFrame(
            {"timestamp": timestamps.strftime("%Y-%m-%dT%H:%M:%SZ"), "rv": values}
        )
        config = {
            "enabled": True,
            "hema_spans": [60, 720, 1440],
            "include_difference": True,
            "warmup": 1440,
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "rv.csv"
            frame.to_csv(path, index=False)
            prepared = load_and_split_data(path, "rv", "timestamp", 0.8, 0.1, config)
        self.assertEqual(prepared.train.values.shape[1], 5)
        self.assertEqual(prepared.input_scaler.n_features_in_, 5)

    def test_recursive_feature_update(self):
        raw = np.array(
            [
                [1.0, 0.8, 0.7, 0.6, 0.1],
                [2.0, 1.0, 0.9, 0.8, 1.0],
                [3.0, 1.2, 1.1, 1.0, 1.0],
            ],
            dtype=np.float32,
        )
        input_scaler = MinMaxScaler().fit(raw)
        target_scaler = MinMaxScaler().fit(raw[:, :1])
        last_scaled = input_scaler.transform(raw[-1:])
        short_x = torch.tensor(np.repeat(last_scaled[None, :, :], 20, axis=1))
        long_x = torch.tensor(np.repeat(last_scaled[None, :, :1], 240, axis=1))
        predicted_raw = np.array([[4.0]], dtype=np.float32)
        prediction = torch.tensor(target_scaler.transform(predicted_raw))
        new_short, new_long = update_recursive_inputs(
            short_x,
            long_x,
            prediction,
            input_scaler,
            target_scaler,
            {
                "enabled": True,
                "hema_spans": [60, 720, 1440],
                "include_difference": True,
            },
        )
        recovered = input_scaler.inverse_transform(new_short[:, -1, :].numpy())[0]
        self.assertAlmostEqual(recovered[0], 4.0, places=5)
        self.assertAlmostEqual(recovered[-1], 1.0, places=5)
        self.assertEqual(new_short.shape, (1, 20, 5))
        self.assertEqual(new_long.shape, (1, 240, 1))


if __name__ == "__main__":
    unittest.main()
