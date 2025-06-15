import argparse
import requests
import time
import numpy as np

# Replace with your deployed model's endpoint
CEREBRIUM_ENDPOINT = "https://your-cerebrium-model-endpoint.com/predict"

def softmax(logits):
    """
    Apply softmax to logits to get probabilities.
    """
    exp_logits = np.exp(logits - np.max(logits))
    return exp_logits / exp_logits.sum()

def predict_image(image_path):
    """
    Send an image to the deployed model and return the predicted class ID.
    """
    with open(image_path, "rb") as image_file:
        files = {"file": image_file}
        response = requests.post(CEREBRIUM_ENDPOINT, files=files)
    
    if response.status_code == 200:
        predictions = response.json()["predictions"]
        probabilities = softmax(np.array(predictions))
        predicted_class = np.argmax(probabilities)
        return predicted_class
    else:
        raise Exception(f"Failed to get prediction. Status code: {response.status_code}, Response: {response.text}")

def run_custom_tests():
    """
    Run preset custom tests using the deployed model.
    """
    # Test case 1: n01440764_tench.jpeg belongs to class 0
    image_path_1 = "n01440764_tench.jpeg"
    expected_class_1 = 0
    predicted_class_1 = predict_image(image_path_1)
    assert predicted_class_1 == expected_class_1, f"Test failed for {image_path_1}. Expected: {expected_class_1}, Got: {predicted_class_1}"
    print(f"Test passed for {image_path_1}. Predicted class: {predicted_class_1}")

    # Test case 2: n01667114_mud_turtle.JPEG belongs to class 35
    image_path_2 = "n01667114_mud_turtle.JPEG"
    expected_class_2 = 35
    predicted_class_2 = predict_image(image_path_2)
    assert predicted_class_2 == expected_class_2, f"Test failed for {image_path_2}. Expected: {expected_class_2}, Got: {predicted_class_2}"
    print(f"Test passed for {image_path_2}. Predicted class: {predicted_class_2}")

def monitor_deployment():
    """
    Monitor the deployed model's performance and availability.
    """
    # Check endpoint availability
    start_time = time.time()
    response = requests.get(CEREBRIUM_ENDPOINT)
    response_time = time.time() - start_time

    if response.status_code == 200:
        print(f"Endpoint is available. Response time: {response_time:.2f} seconds.")
    else:
        print(f"Endpoint is unavailable. Status code: {response.status_code}, Response: {response.text}")

    # Additional monitoring tests can be added here

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test the deployed model on Cerebrium.")
    parser.add_argument("--image", type=str, help="Path to the image to predict.")
    parser.add_argument("--test", action="store_true", help="Run preset custom tests.")
    parser.add_argument("--monitor", action="store_true", help="Monitor the deployed model.")

    args = parser.parse_args()

    if args.image:
        try:
            predicted_class = predict_image(args.image)
            print(f"Predicted class ID for {args.image}: {predicted_class}")
        except Exception as e:
            print(f"Error: {e}")
    elif args.test:
        run_custom_tests()
    elif args.monitor:
        monitor_deployment()
    else:
        print("Please provide an argument. Use --help for more information.")