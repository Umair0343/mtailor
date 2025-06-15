import onnxruntime
import numpy as np
from PIL import Image
import torchvision.transforms as transforms

class OnnxModelHandler:
    """
    Handles loading and inference for ONNX models.
    """
    def __init__(self, model_path):
        """
        Initialize the ONNX model handler.

        Args:
            model_path (str): Path to the ONNX model file.
        """
        self.model_path = model_path
        self.session = onnxruntime.InferenceSession(model_path)

    def predict(self, input_data):
        """
        Perform prediction using the ONNX model.

        Args:
            input_data (np.ndarray): Preprocessed input data.

        Returns:
            np.ndarray: Model prediction output.
        """
        input_name = self.session.get_inputs()[0].name
        output_name = self.session.get_outputs()[0].name
        predictions = self.session.run([output_name], {input_name: input_data})
        return predictions[0]


class ImagePreprocessor:
    """
    Handles image preprocessing for model input.
    """
    def __init__(self, input_size=(224, 224)):
        """
        Initialize the image preprocessor.

        Args:
            input_size (tuple): Target size for resizing the image (default: (224, 224)).
        """
        self.input_size = input_size
        self.transform = transforms.Compose([
            transforms.Resize(self.input_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def preprocess(self, image_path):
        """
        Preprocess the image for model input.

        Args:
            image_path (str): Path to the image file.

        Returns:
            np.ndarray: Preprocessed image as a NumPy array.
        """
        image = Image.open(image_path).convert("RGB")
        tensor = self.transform(image).unsqueeze(0)  # Add batch dimension
        return tensor.numpy()

def softmax(logits):
    exp_logits = np.exp(logits - np.max(logits))  # Stability improvement
    return exp_logits / exp_logits.sum(axis=-1, keepdims=True)

# Example usage
if __name__ == "__main__":
    # Initialize the preprocessor and model handler
    preprocessor = ImagePreprocessor(input_size=(224, 224))
    model_handler = OnnxModelHandler("umair.onnx")

    # Preprocess the image
    image_path = "n01667114_mud_turtle.JPEG"
    preprocessed_image = preprocessor.preprocess(image_path)

    # Perform prediction
    predictions = model_handler.predict(preprocessed_image)
    # Apply softmax to convert logits to probabilities

    # Assuming `predictions` is the output from the model
    probabilities = softmax(predictions)
    predicted_class = np.argmax(probabilities)

    print("Predicted Class:", predicted_class)