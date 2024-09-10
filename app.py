import numpy as np
import faiss
import pickle
from flask import Flask, request, render_template
from PIL import Image
from io import BytesIO
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.preprocessing import image

app = Flask(__name__)

# Load FAISS index and metadata
faiss_index = None
labels_for_index = None
names_for_index = None

try:
    # Load FAISS index
    faiss_index = faiss.read_index('Dataset/lfw_embeddings_index.faiss')
    
    # Debug the type of index
    if isinstance(faiss_index, faiss.Index):
        print("FAISS index loaded successfully.")
    else:
        raise TypeError("The loaded FAISS index is not of type 'faiss.Index'.")

    # Load labels and names
    with open('Dataset/lfw_labels_for_index.pkl', 'rb') as f:
        labels_for_index = pickle.load(f)

    with open('Dataset/lfw_names_for_index.pkl', 'rb') as f:
        names_for_index = pickle.load(f)

except Exception as e:
    print(f"Error loading FAISS index or metadata: {e}")
    faiss_index = None

# Initialize ResNet50 model for embedding
model = ResNet50(weights='imagenet', include_top=False, pooling='avg')

@app.route('/')
def index():
    return render_template('upload.html')

@app.route('/upload', methods=['POST'])
def upload():
    if not faiss_index:
        return "Error: FAISS index not loaded properly."

    file = request.files['file']
    if file:
        try:
            # Load and preprocess the image
            img = Image.open(BytesIO(file.read()))
            img = img.resize((224, 224))  # Resize to match ResNet50 input size
            img_array = image.img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0)
            img_array = preprocess_input(img_array)

            # Generate embedding using ResNet50
            embedding = model.predict(img_array)
            embedding = embedding.flatten()

            # Search for similar faces in FAISS index
            D, I = faiss_index.search(np.array([embedding]), k=1)
            closest_index = I[0][0]
            closest_name = names_for_index[closest_index]

            return f"The closest match is {closest_name}."

        except Exception as e:
            return f"Error processing image: {e}"

    return "No file uploaded."

if __name__ == '__main__':
    app.run(debug=True)
