import os
import pickle
import torch
import numpy as np
from flask import Flask, render_template, request, jsonify, send_from_directory
from PIL import Image
from sklearn.metrics.pairwise import cosine_similarity
import torch.nn.functional as F
from open_clip import create_model_and_transforms, tokenizer
import open_clip
from sklearn.decomposition import PCA

app = Flask(__name__, static_folder='static')

# Get the directory of the current script
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Specify the folder containing images in the static directory
image_folder = os.path.join(BASE_DIR, 'static')

# Ensure the folder exists
if not os.path.exists(image_folder):
    raise FileNotFoundError(f"Image folder not found: {image_folder}")

# Load the model and preprocess functions
device = "cuda" if torch.cuda.is_available() else "cpu"
model_name = "ViT-B-32"
pretrained = "openai"
model, preprocess_train, preprocess_val = create_model_and_transforms(model_name, pretrained=pretrained)
model = model.to(device)
model.eval()

# Load the image embeddings
with open('image_embeddings.pickle', 'rb') as f:
    image_embeddings = pickle.load(f)

# Tokenizer for text queries
tokenizer = open_clip.get_tokenizer(model_name)

def normalize_embedding(embedding):
    """
    Normalize embedding using PyTorch or NumPy depending on input type.
    
    Args:
        embedding (numpy.ndarray or torch.Tensor): Input embedding to normalize
    
    Returns:
        numpy.ndarray: Normalized embedding
    """
    if isinstance(embedding, torch.Tensor):
        # If it's a PyTorch tensor, use torch normalization
        normalized = F.normalize(embedding, p=2, dim=-1)
        return normalized.cpu().numpy()
    elif isinstance(embedding, np.ndarray):
        # If it's a NumPy array, use NumPy normalization
        return embedding / np.linalg.norm(embedding, axis=-1, keepdims=True)
    else:
        raise TypeError("Unsupported embedding type. Must be numpy.ndarray or torch.Tensor")

# Precompute and cache PCA transformation
def compute_pca_embeddings(embeddings, k=50):
    """
    Compute PCA transformation of embeddings.
    
    Args:
        embeddings (numpy.ndarray): Original embeddings
        k (int): Number of principal components to keep
    
    Returns:
        tuple: PCA transformer and transformed embeddings
    """
    pca = PCA(n_components=k)
    pca_embeddings = pca.fit_transform(embeddings)
    return pca, pca_embeddings

# Precompute PCA (this can be cached or saved to a file for efficiency)
original_embeddings = np.array(image_embeddings['embedding'].tolist())
pca_transformer, pca_embeddings = compute_pca_embeddings(original_embeddings, k=50)

# Helper function to calculate cosine similarities for text-to-image search
def text_to_image_search(query_embedding, top_k=5, embedding_type='clip', pca_k=50):
    try:
        # Ensure query_embedding is normalized
        query_embedding = normalize_embedding(query_embedding)
        
        # Choose embedding type
        if embedding_type == 'pca':
            # Transform query embedding to PCA space
            query_embedding = pca_transformer.transform(query_embedding)
            embeddings = pca_embeddings
        else:
            # Use original CLIP embeddings
            embeddings = np.array(image_embeddings['embedding'].tolist())
        
        # Ensure embeddings are normalized
        normalized_embeddings = np.array([normalize_embedding(emb) for emb in embeddings])
        
        # Compute cosine similarities
        similarities = cosine_similarity(query_embedding, normalized_embeddings)[0]
        
        # Get top k indices
        top_indices = similarities.argsort()[-top_k:][::-1]
        
        return [(image_embeddings['file_name'][i], similarities[i]) for i in top_indices]
    
    except Exception as e:
        print(f"Error in text_to_image_search: {e}")
        raise

# Route for the homepage
@app.route('/')
def index():
    return render_template('index.html')

# Route for handling image and text queries
@app.route('/search', methods=['POST'])
def search():
    text_query = request.form.get('text_query')
    image_file = request.files.get('image_query')
    lam = float(request.form.get('lambda', 0.5))  # Weight for blending text and image queries
    embedding_choice = request.form.get('embedding_choice', 'clip')
    pca_k = int(request.form.get('pca_k', 50))  # Default to 50 PCA components
    top_k = 5  # Top K most relevant images

    try:
        # Process text query if provided
        text_embedding = None
        if text_query:
            text_tokens = tokenizer([text_query]).to(device)
            with torch.no_grad():
                text_embedding = model.encode_text(text_tokens).cpu().detach().numpy()
                text_embedding = normalize_embedding(text_embedding)

        # Process image query if provided
        image_embedding = None
        if image_file:
            image = Image.open(image_file).convert('RGB')
            image_tensor = preprocess_val(image).unsqueeze(0).to(device)
            with torch.no_grad():
                image_embedding = model.encode_image(image_tensor).cpu().detach().numpy()
                image_embedding = normalize_embedding(image_embedding)

        # Blend text and image queries if both are provided
        query_embedding = None
        if text_query and image_file:
            query_embedding = lam * text_embedding + (1.0 - lam) * image_embedding
        elif text_query:
            query_embedding = text_embedding
        elif image_file:
            query_embedding = image_embedding

        if query_embedding is None:
            return jsonify({"error": "No valid query provided."}), 400

        # Retrieve top K most similar images
        results = text_to_image_search(
            query_embedding, 
            top_k=top_k, 
            embedding_type=embedding_choice, 
            pca_k=pca_k
        )

        # Format the results with image URLs
        results_with_paths = [
            {"image_path": f"/static/{filename}", "similarity": float(similarity)}
            for filename, similarity in results
        ]

        # Return results as JSON
        return jsonify({"results": results_with_paths})

    except Exception as e:
        print(f"Error in search route: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)