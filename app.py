import os
from flask import Flask, request, jsonify, render_template
from dotenv import load_dotenv
from query_answer import answer_query_with_context
from recommendation import generate_recommendations  # Import your recommendation function
from pinecone_connect_index import connect_to_pinecone

# Load environment variables
load_dotenv()

# Global variable for chat history
chat_history = []

app = Flask(__name__)

# Route to serve the index.html at the root URL
@app.route("/")
def home():
    return render_template("index.html")

# Route for user query
@app.route("/api/query", methods=["POST"])
def query():
    data = request.json
    user_query = data.get("query")
    
    if not user_query:
        return jsonify({"error": "No query provided"}), 400

    try:
        # Connect to Pinecone
        index = connect_to_pinecone()

        # Call function to generate answer with context from similar docs
        answer = answer_query_with_context(user_query, index)

        # Update global chat history
        chat_history.append((user_query, answer))

        # Call recommendation function
        recommendations = generate_recommendations(user_query, chat_history)

        return jsonify({"answer": answer, "recommendations": recommendations})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Run the Flask app
if __name__ == "__main__":
    app.run(debug=True)
