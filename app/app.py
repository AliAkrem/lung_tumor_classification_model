import gradio as gr
import tensorflow as tf
import numpy as np
from PIL import Image
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from tensorflow.keras.applications.efficientnet import preprocess_input
import os
import io

# Configuration
MODEL_CONFIG = {
    'input_size': (299, 299),
    'class_names': [
        'adenocarcinoma',
        'large.cell.carcinoma',
        'normal',
        'squamous.cell.carcinoma'
    ],
    'model_path': './lung_tumor_classifier_efficientnet_final.h5'  # Local path for HF Spaces
}

class TumorPredictor:
    def __init__(self):
        """Initialize the predictor with model and class names"""
        self.class_names = MODEL_CONFIG['class_names']
        self.model = None
        self._load_model()

    def _load_model(self):
        """Load the trained model"""
        try:
            model_path = MODEL_CONFIG['model_path']
            if os.path.exists(model_path):
                print(f"Loading model from: {model_path}")
                self.model = tf.keras.models.load_model(model_path, compile=False)
                print("Model loaded successfully!")
                print(f"Model input shape: {self.model.input_shape}")
                print(f"Model output shape: {self.model.output_shape}")
            else:
                print(f"Model file not found at: {model_path}")
                print("Please upload the model file to the root directory")
                self.model = None
        except Exception as e:
            print(f"Error loading model: {str(e)}")
            self.model = None

    def preprocess_image(self, image):
        """Preprocess the uploaded image to exactly match original predict.py"""
        try:
            # Save PIL image to temporary bytes buffer to use with load_img
            # This ensures we use the exact same preprocessing as the original
            if isinstance(image, str):
                # If image is a file path
                img = load_img(image, target_size=(299, 299))
            else:
                # For PIL images from Gradio, save to buffer and reload with load_img
                img_buffer = io.BytesIO()
                image.save(img_buffer, format='PNG')
                img_buffer.seek(0)

                # Use load_img exactly as in the original predict.py
                img = load_img(img_buffer, target_size=(299, 299))

            # Process exactly as in original predict.py
            img_array = img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0)
            img_array = preprocess_input(img_array)

            return img_array

        except Exception as e:
            raise Exception(f"Error preprocessing image: {str(e)}")

    def predict_image(self, image):
        """Make prediction on uploaded image"""
        try:
            if self.model is None:
                return {
                    'error': 'Model not loaded. Please ensure the model file is uploaded.',
                    'class': 'unknown',
                    'confidence': 0.0,
                    'probabilities': {class_name: 0.0 for class_name in self.class_names}
                }

            # Preprocess the image
            processed_image = self.preprocess_image(image)

            # Debug: Print image statistics after preprocessing
            print(f"Processed image shape: {processed_image.shape}")
            print(f"Processed image min: {processed_image.min()}")
            print(f"Processed image max: {processed_image.max()}")
            print(f"Processed image mean: {processed_image.mean()}")

            # Make prediction (exactly as in original predict.py)
            predictions = self.model.predict(processed_image, verbose=0)

            # Debug: Print raw predictions
            print(f"Raw predictions: {predictions[0]}")
            print(f"Predictions shape: {predictions.shape}")

            # Get prediction results (exactly as in original predict.py)
            predicted_class_idx = np.argmax(predictions[0])
            confidence = predictions[0][predicted_class_idx]
            predicted_class = self.class_names[predicted_class_idx]

            print(f"Predicted class: {predicted_class}, confidence: {confidence}")

            # Create results dictionary (exactly as in original predict.py)
            results = {
                'class': predicted_class,
                'confidence': float(confidence),
                'probabilities': {
                    class_name: float(prob)
                    for class_name, prob in zip(self.class_names, predictions[0])
                }
            }

            return results

        except Exception as e:
            print(f"Exception in predict_image: {str(e)}")
            return {
                'error': f'Prediction error: {str(e)}',
                'class': 'unknown',
                'confidence': 0.0,
                'probabilities': {class_name: 0.0 for class_name in self.class_names}
            }

# Initialize the predictor
predictor = TumorPredictor()

def predict_tumor(image):
    """Main prediction function for Gradio interface"""
    if image is None:
        return "Please upload an image."

    try:
        # Get prediction results
        results = predictor.predict_image(image)

        if 'error' in results:
            return f"Error: {results['error']}"

        # Format the output
        output = f"""
## Prediction Results

**Predicted Class:** {results['class']}
**Confidence:** {results['confidence']:.2%}

### Detailed Probabilities:
"""

        # Sort probabilities in descending order
        sorted_probs = sorted(results['probabilities'].items(), key=lambda x: x[1], reverse=True)

        for class_name, prob in sorted_probs:
            percentage = prob * 100
            output += f"- **{class_name}:** {percentage:.2f}%\n"

        return output

    except Exception as e:
        return f"An error occurred: {str(e)}"

def predict_tumor_json(image):
    """API endpoint that returns JSON results"""
    if image is None:
        return {
            'error': 'No image provided',
            'class': 'unknown',
            'confidence': 0.0,
            'probabilities': {class_name: 0.0 for class_name in MODEL_CONFIG['class_names']}
        }

    return predictor.predict_image(image)

# Create Gradio interface
with gr.Blocks(title="Lung Tumor Classification", theme=gr.themes.Soft()) as demo:
    gr.Markdown("""
    # 🫁 Lung Tumor Classification API

    Upload a lung CT scan image to get tumor classification results.

    **Supported Classes:**
    - Adenocarcinoma
    - Large Cell Carcinoma
    - Normal
    - Squamous Cell Carcinoma
    """)

    with gr.Tab("Image Classification"):
        with gr.Row():
            with gr.Column():
                image_input = gr.Image(
                    type="pil",
                    label="Upload Lung CT Scan Image",
                    height=400
                )
                predict_btn = gr.Button("🔍 Analyze Image", variant="primary")

            with gr.Column():
                output_text = gr.Markdown(label="Prediction Results")

        predict_btn.click(
            fn=predict_tumor,
            inputs=image_input,
            outputs=output_text
        )

    with gr.Tab("API Endpoint"):
        gr.Markdown("""
        ## JSON API Endpoint

        This tab demonstrates the JSON API that returns structured prediction results.
        """)

        with gr.Row():
            with gr.Column():
                api_image_input = gr.Image(
                    type="pil",
                    label="Upload Image for API Test",
                    height=300
                )
                api_predict_btn = gr.Button("📡 Get JSON Response", variant="secondary")

            with gr.Column():
                json_output = gr.JSON(label="API Response")

        api_predict_btn.click(
            fn=predict_tumor_json,
            inputs=api_image_input,
            outputs=json_output
        )

    with gr.Tab("Model Info"):
        gr.Markdown("""
        ## Model Information

        **Architecture:** EfficientNet
        **Input Size:** 299x299 pixels
        **Classes:** 4 (adenocarcinoma, large.cell.carcinoma, normal, squamous.cell.carcinoma)

        ### Usage Instructions:
        1. Upload a lung CT scan image
        2. Click "Analyze Image" to get predictions
        3. View results with confidence scores and probabilities

        ### API Response Format:
        ```json
        {
            "class": "adenocarcinoma",
            "confidence": 0.8745,
            "probabilities": {
                "adenocarcinoma": 0.8745,
                "large.cell.carcinoma": 0.0832,
                "normal": 0.0215,
                "squamous.cell.carcinoma": 0.0208
            }
        }
        ```

        ### Note:
        Make sure to upload the model file `lung_tumor_classifier_efficientnet_final.h5`
        to the root directory for the API to work properly.
        """)

# Launch the app
if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=True
    )
