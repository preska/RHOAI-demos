from flask import Flask, request, jsonify
import onnxruntime as ort
import numpy as np
from transformers import AutoTokenizer
import json

# Initialize Flask app
app = Flask(__name__)

# Load the model and tokenizer
MODEL_NAME = "distilbert-base-uncased-finetuned-sst-2-english"
ONNX_MODEL_PATH = "models"
ONNX_MODEL_NAME = "distilbert-base-uncased-finetuned-sst-2-english.onnx"
onnx_model_path = f"{ONNX_MODEL_PATH}/{ONNX_MODEL_NAME}"

# Load the tokenizer and ONNX runtime session
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
session = ort.InferenceSession(onnx_model_path)

# Define the route for predictions
@app.route('/predict', methods=['POST'])
def predict():
    # Extract input text from the request
    data = request.get_json()
    
    if "texts" not in data:
        return jsonify({"error": "No texts provided in request"}), 400
    
    texts = data["texts"]

    # Check the expected input length from the ONNX model
    input_shape = session.get_inputs()[0].shape
    fixed_seq_length = input_shape[1]  # Expected sequence length

    # Tokenize input texts
    inputs = tokenizer(
        texts, 
        padding="max_length",  # Pad all sequences to the fixed length
        truncation=True,       # Truncate if necessary
        max_length=fixed_seq_length,  # Use the fixed sequence length
        return_tensors="np"    # Return NumPy arrays
    )

    # ONNX expects numpy arrays
    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]

    # Prepare the data dictionary with 'input_ids' and 'attention_mask' as separate arrays
    data = {
        "inputs": [
            {
                "name": "input_ids",  # Input name for input_ids
                "shape": input_ids.shape,  # Shape of the input_ids array
                "datatype": "INT64",  # Datatype for input_ids
                "data": input_ids.tolist()  # Convert numpy array to list
            },
            {
                "name": "attention_mask",  # Input name for attention_mask
                "shape": attention_mask.shape,  # Shape of the attention_mask array
                "datatype": "INT64",  # Datatype for attention_mask
                "data": attention_mask.tolist()  # Convert numpy array to list
            }
        ]
    }

    # Send POST request to the model's REST endpoint (model server)
    infer_url = "http://modelmesh-serving.model-serving-demo:8008/v2/models/distilbert/infer"
    response = requests.post(infer_url, data=json.dumps(data), headers={"Content-Type": "application/json"})

    if response.status_code == 200:
        result = response.json()
        logits = result['outputs'][0]['data']
        
        # Reshape logits
        logits = np.array(logits).reshape(len(texts), -1)
        
        # Find predicted class for each text
        predicted_class_ids = np.argmax(logits, axis=1)

        # Assuming the labels are as follows (for SST-2)
        label_map = {0: "NEGATIVE", 1: "POSITIVE"}
        labels = [label_map[class_id] for class_id in predicted_class_ids]

        return jsonify({"predictions": [{"text": text, "label": label} for text, label in zip(texts, labels)]})

    else:
        return jsonify({"error": f"Model inference failed: {response.status_code} - {response.text}"}), 500


if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000)  # Ensure the host is 0.0.0.0 to allow external connections

