import streamlit as st
import os
from pinecone import Pinecone, ServerlessSpec
from langchain_community.retrievers import PineconeHybridSearchRetriever
from langchain_huggingface import HuggingFaceEmbeddings
from pinecone_text.sparse import BM25Encoder
from dotenv import load_dotenv
import nltk


# Load environment variables
load_dotenv()
nltk.download('punkt_tab')

# Streamlit interface
st.title("🌐 Pinecone Hybrid Search with Streamlit")
st.subheader("Explore hybrid search across dense and sparse embeddings")

# API keys and index details
api_key = os.getenv("PINECONE_API_KEY")
hf_token = os.getenv("HF_TOKEN")

index_name = "hybrid-search-langchain-pinecone"

# Initialize Pinecone client
pc = Pinecone(api_key=api_key)

# Create the index if it doesn't exist
if index_name not in pc.list_indexes().names():
    st.write("🔄 Creating Pinecone index...")
    pc.create_index(
        name=index_name,
        dimension=384,  # dimensionality of dense model
        metric="dotproduct",  # sparse values supported only for dotproduct
        spec=ServerlessSpec(cloud="aws", region="us-east-1"),
    )
index = pc.Index(index_name)

# Vector embedding and sparse matrix
os.environ["HF_TOKEN"] = hf_token

# Initialize embeddings
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# Initialize BM25 encoder
bm25_encoder = BM25Encoder().default()

# Example sentences
sentences = [
    "In 2023, I visited Paris",
    "In 2022, I visited New York",
    "In 2021, I visited New Orleans",
]

# Fit BM25 encoder on sentences
bm25_encoder.fit(sentences)

# Store values to a JSON file
bm25_encoder.dump("bm25_values.json")

# Load BM25 encoder
bm25_encoder = BM25Encoder().load("bm25_values.json")

# Initialize retriever
retriever = PineconeHybridSearchRetriever(embeddings=embeddings, sparse_encoder=bm25_encoder, index=index)

# Add texts to the retriever
st.write("🔄 Adding texts to the retriever...")
retriever.add_texts(sentences)

# Streamlit input for query
query = st.text_input("🔍 Enter your query:", "What city did I visit first?")

# Run the query and display the result
if st.button("Search"):
    st.write("🔍 Searching...")
    results = retriever.invoke(query)

    if results:
        st.success("Result Found!")
        st.markdown("### **Results:**")
        
        # Initialize a flag to check if a specific match is found
        specific_match_found = False
        
        for res in results:
            # Display only the sentence containing '2023' when queried
            if "2023" in query and "2023" in res.page_content:
                st.markdown(f"**Answer:** {res.page_content}")
                specific_match_found = True
                
            elif "2021" in query and "2021" in res.page_content:
                st.markdown(f"**Answer:** {res.page_content}")
                specific_match_found = True
                
        
        # If no specific match found, show all results
        if not specific_match_found:
            for i, res in enumerate(results, 1):
                st.markdown(f"**{i}.** {res.page_content}")
                st.markdown("---")
    else:
        st.warning("No results found for your query.")

