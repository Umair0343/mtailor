import torch
import torch.onnx

def convert_to_onnx(model_class, pytorch_model_path, onnx_model_path, input_shape):
    """
    Converts a PyTorch model to ONNX format.

    Args:
        model_class (torch.nn.Module): The class of the PyTorch model.
        pytorch_model_path (str): Path to the saved PyTorch model (.pth file).
        onnx_model_path (str): Path to save the ONNX model (.onnx file).
        input_shape (tuple): Shape of the model input (e.g., (1, 3, 224, 224)).
    """
    # Initialize the model architecture
    model = model_class()
    
    # Load the state dictionary into the model
    state_dict = torch.load(pytorch_model_path)
    model.load_state_dict(state_dict)
    model.eval()  # Set the model to evaluation mode

    # Create dummy input matching the input shape
    dummy_input = torch.randn(*input_shape)

    # Export the model to ONNX format
    torch.onnx.export(
        model, 
        dummy_input, 
        onnx_model_path, 
        export_params=True,  # Store the trained parameters
        opset_version=11,    # ONNX version
        do_constant_folding=True,  # Optimize constant folding
        input_names=['input'],  # Name of the input tensor
        output_names=['output'],  # Name of the output tensor
        dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}  # Dynamic batch size
    )

    print(f"Model has been converted to ONNX format and saved at {onnx_model_path}")

# Example usage
if __name__ == "__main__":
    from pytorch_model import Classifier
    pytorch_model_path = "pytorch_model_weights.pth"  # Path to the PyTorch model
    onnx_model_path = "umair.onnx"   # Path to save the ONNX model
    input_shape = (1, 3, 224, 224)   # Example input shape for an image model

    convert_to_onnx(Classifier, pytorch_model_path, onnx_model_path, input_shape)