import unittest
import shutil
import tempfile
import os
from sklearn.linear_model import LinearRegression
import numpy as np
from src.services.model_registry_service import ModelRegistryService
import mlflow

class TestModelRegistry(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        self.registry = ModelRegistryService(storage_path=self.test_dir)
        
    def tearDown(self):
        shutil.rmtree(self.test_dir)
        # Reset MLflow tracking URI to avoid side effects
        mlflow.set_tracking_uri("")

    def test_save_and_load_sklearn_model(self):
        # Create a dummy model
        model = LinearRegression()
        X = np.array([[1, 1], [1, 2], [2, 2], [2, 3]])
        y = np.dot(X, np.array([1, 2])) + 3
        model.fit(X, y)
        
        metadata = {
            'metrics': {'mse': 0.1, 'r2': 0.95},
            'hyperparameters': {'fit_intercept': True},
            'training_config': {'epochs': 10}
        }
        
        # Save model
        model_id = self.registry.save_model(
            model=model,
            symbol="TEST_SYM",
            model_type="LinearRegression",
            metadata=metadata,
            framework="sklearn"
        )
        
        self.assertIsNotNone(model_id)
        
        # Get Metadata
        retrieved_meta = self.registry.get_model_metadata(model_id)
        self.assertEqual(retrieved_meta['symbol'], "TEST_SYM")
        self.assertEqual(retrieved_meta['metrics']['mse'], 0.1)
        self.assertEqual(str(retrieved_meta['hyperparameters']['fit_intercept']), "True")
        
        # Load Model
        loaded_model = self.registry.load_model(model_id)
        self.assertIsInstance(loaded_model, LinearRegression)
        self.assertTrue(np.allclose(model.coef_, loaded_model.coef_))

    def test_get_best_model(self):
        # Save two models with different metrics
        model1 = LinearRegression()
        self.registry.save_model(
            model=model1,
            symbol="TEST_SYM",
            model_type="LinearRegression",
            metadata={'metrics': {'val_loss': 0.5}},
            framework="sklearn"
        )
        
        model2 = LinearRegression()
        self.registry.save_model(
            model=model2,
            symbol="TEST_SYM",
            model_type="LinearRegression",
            metadata={'metrics': {'val_loss': 0.2}},
            framework="sklearn"
        )
        
        best_model = self.registry.get_best_model("TEST_SYM", metric="val_loss", mode="min")
        self.assertIsNotNone(best_model)
        self.assertEqual(best_model['metrics']['val_loss'], 0.2)

if __name__ == '__main__':
    unittest.main()
