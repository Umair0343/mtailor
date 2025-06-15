import unittest
import numpy as np
from model import ImagePreprocessor, OnnxModelHandler, softmax

class TestModelDeployment(unittest.TestCase):
    def setUp(self):
        """
        Set up resources for testing.
        """
        self.model_path = "umair.onnx"  # Replace with the actual ONNX model path
        self.image_path = "n01667114_mud_turtle.JPEG"  # Replace with the actual image path
        self.input_size = (224, 224)
        self.preprocessor = ImagePreprocessor(input_size=self.input_size)
        self.model_handler = OnnxModelHandler(self.model_path)

    def test_image_preprocessing(self):
        """
        Test image preprocessing functionality.
        """
        preprocessed_image = self.preprocessor.preprocess(self.image_path)
        self.assertEqual(preprocessed_image.shape, (1, 3, 224, 224), "Preprocessed image shape is incorrect.")
        self.assertTrue(np.issubdtype(preprocessed_image.dtype, np.floating), "Preprocessed image data type is incorrect.")

    def test_model_loading(self):
        """
        Test ONNX model loading.
        """
        self.assertIsNotNone(self.model_handler.session, "ONNX model session is not initialized.")

    def test_model_prediction(self):
        """
        Test model prediction functionality.
        """
        preprocessed_image = self.preprocessor.preprocess(self.image_path)
        predictions = self.model_handler.predict(preprocessed_image)
        self.assertIsInstance(predictions, np.ndarray, "Model prediction output is not a NumPy array.")
        self.assertGreater(len(predictions), 0, "Model prediction output is empty.")

    def test_softmax_function(self):
        """
        Test softmax function for post-processing.
        """
        logits = np.array([2.0, 1.0, 0.1])
        probabilities = softmax(logits)
        self.assertAlmostEqual(probabilities.sum(), 1.0, places=5, msg="Softmax probabilities do not sum to 1.")
        self.assertTrue(np.all(probabilities >= 0), "Softmax probabilities contain negative values.")

    def test_class_prediction(self):
        """
        Test class prediction functionality for multiple images.
        """
        # Test case 1: n01440764_tench.jpeg belongs to class 0
        image_path_1 = "n01440764_tench.jpeg"
        expected_class_1 = 0
        preprocessed_image_1 = self.preprocessor.preprocess(image_path_1)
        predictions_1 = self.model_handler.predict(preprocessed_image_1)
        probabilities_1 = softmax(predictions_1)
        predicted_class_1 = np.argmax(probabilities_1)
        self.assertEqual(predicted_class_1, expected_class_1, f"Image {image_path_1} was incorrectly classified.")

        # Test case 2: n01667114_mud_turtle.JPEG belongs to class 35
        image_path_2 = "n01667114_mud_turtle.JPEG"
        expected_class_2 = 35
        preprocessed_image_2 = self.preprocessor.preprocess(image_path_2)
        predictions_2 = self.model_handler.predict(preprocessed_image_2)
        probabilities_2 = softmax(predictions_2)
        predicted_class_2 = np.argmax(probabilities_2)
        self.assertEqual(predicted_class_2, expected_class_2, f"Image {image_path_2} was incorrectly classified.")

if __name__ == "__main__":
    unittest.main()