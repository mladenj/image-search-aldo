from sentence_transformers import SentenceTransformer
import pinecone
import pandas as pd
import streamlit as st
import os
from pinecone import Pinecone
from dotenv import load_dotenv

# Loading environment variables
load_dotenv()

# Connecting to Pinecone
pc = Pinecone(
    api_key=os.environ.get("PINECONE_API_KEY")
)

index_name = 'image-search-aldo-metadata'
index = pc.Index(index_name)

# Loading the data
df = pd.read_csv('data/products_aldo_embeddings.csv', index_col=0)

st.image('https://upload.wikimedia.org/wikipedia/commons/3/37/Aldo_Group_logo.svg', width=200)
st.write("""
# Semantic Image Search
""")
st.write("""
Semantic image search uses a *text query* or an *input image* to search a database of images to find images that are semantically similar to the search query. Thanks to advances in machine learning and computer vision, we can better understand what a user is looking for when they search for an image. It's completely independent from all text from the website (categorisation, alt tags, description etc.). 
""")
st.write("""
## Dataset
""")
st.write("""
The dataset contains a snapshot of available products scraped from the Aldo US website June 2024.
""")

st.write("""
## Demo
""")


# Caching the model
@st.cache_resource
def load_model():
    return SentenceTransformer('sentence-transformers/clip-ViT-B-32-multilingual-v1')

# Load the model
text_model = load_model()

query = st.text_input("Enter your search query:")

if st.button("Search images"):
    if query.strip():
        # Encode the query using SentenceTransformer
        xq = text_model.encode(query).tolist()

        # Getting the response from Pinecone
        response_text = index.query(vector=xq, top_k=8, include_values=True)

        # Check if there are any matches
        if response_text.get('matches'):
            # Extract all the ids
            ids = [match['id'] for match in response_text['matches']]

            for i in range(0, len(ids), 4):
                cols = st.columns(4)
                for j, col in enumerate(cols):
                    if i + j < len(ids):
                        id = ids[i + j]
                        image_url = df.loc[int(id)]['IMAGE_URL']
                        product_url = df.loc[int(id)]['PRODUCT_URL']
                        if image_url:
                            # Make image clickable
                            col.markdown(f'<a href="{image_url}" target="_blank"><img src="{image_url}" width="150"/></a>', unsafe_allow_html=True)
                            col.markdown(f'<a href="{product_url}" target="_blank">Product ID: {id}</a>', unsafe_allow_html=True)
                        else:
                            col.write("Image URL not found for this product.")
        else:
            st.write("No matching images found.")
    else:
        st.write("Please enter a valid search query.")
