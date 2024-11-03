import os
import openai
import traceback
from dotenv import load_dotenv
from langchain_community.embeddings import OpenAIEmbeddings  # Updated import

# Load environment variables (like API keys)
load_dotenv()

# Function to convert text to embeddings
def text_to_embedding(text):
    """Converts text to an embedding vector using OpenAI embeddings."""
    print("Converting text to embedding...")
    embeddings_model = OpenAIEmbeddings()  # No need for API key here
    embedding = embeddings_model.embed_query(text)
    print(f"Generated embedding: {embedding}")
    return embedding

# Function to retrieve similar documents from Pinecone
def get_similar_docs(query, index, k=5):
    """
    Query Pinecone to find similar documents based on the input query.
    
    Args:
        query (str): The text query for similarity search.
        index: The Pinecone index to query.
        k (int): The number of results to return.

    Returns:
        list: The list of similar documents and their metadata.
    """
    print(f"Getting similar documents for query: {query}")
    # Convert query to embedding
    embedding = text_to_embedding(query)

    # Query Pinecone index
    response = index.query(vector=embedding, top_k=k, include_metadata=True)
    print(f"Pinecone response: {response}")

    # Extract and return the relevant content from the response
    return [
        {'content': match['metadata'].get('content', '')}
        for match in response['matches']
    ]

# Function to answer query using context from similar documents
def answer_query_with_context(query, index):
    """
    Answer the user's query using the most similar documents as context.

    Args:
        query (str): The user's query.
        index: The Pinecone index to retrieve similar documents.

    Returns:
        str: The formatted generated answer from OpenAI based on the context.
    """
    # Retrieve similar documents from Pinecone
    similar_docs = get_similar_docs(query, index)

    # If no relevant context is found, provide a fallback message
    if not similar_docs or all(doc['content'] == '' for doc in similar_docs):
        return "The current information is not available to me as it is not mentioned in the data provided. Please ask a question related to the available university data."

    # Combine the content of similar documents to create the context
    context = " ".join([doc['content'] for doc in similar_docs if doc['content']])
    print(f"Context for OpenAI: {context}")

    # Initialize the OpenAI client with your API key
    openai.api_key = os.getenv('OPENAI_API_KEY')  # Set the API key
    if not openai.api_key:
        print("Error: OpenAI API key is missing!")
        return "Error: OpenAI API key is missing."

    # Prepare the messages for the chat completion
    messages = [
        {"role": "system", "content": "You are a helpful assistant. Answer the query based ONLY on the provided context. Format the response clearly using bullet points or numbered lists."},
        {"role": "user", "content": f"Context: {context}"},
        {"role": "user", "content": query}
    ]

    try:
        # Use the correct `ChatCompletion.create` method
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=messages,
            max_tokens=300,
            temperature=0.3
        )
        print(f"OpenAI response: {response}")  # Log the full response from OpenAI
        # Extract the answer
        answer = response['choices'][0]['message']['content']

        # Now format the response with HTML tags for better readability
        formatted_answer = answer.replace("1.", "<br><strong>1.</strong>").replace("2.", "<br><strong>2.</strong>").replace("3.", "<br><strong>3.</strong>").replace("4.", "<br><strong>4.</strong>")

        return formatted_answer
    except Exception as e:
        print(f"Error querying OpenAI: {str(e)}")  # Print full exception
        traceback.print_exc()  # Print the full traceback for debugging
        return "Error generating response"
