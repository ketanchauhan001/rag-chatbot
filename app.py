import os
from flask import Flask, render_template, request, jsonify
from rag import get_retriever
from langchain_core.prompts import PromptTemplate
from openai import OpenAI

BOT_NAME = "DubaiEstateBot"

app = Flask(__name__)

# Initialize retriever
try:
    retriever = get_retriever()
    print("✅ Vector store loaded successfully")
except Exception as e:
    print(f"❌ Error loading vector store: {e}")
    retriever = None

# DeepSeek client
client = OpenAI(
    api_key="sk-4972b5b7f4094b77b49f2606605a3595",
    base_url="https://api.deepseek.com"
)


@app.route("/")
def home():
    return render_template("index.html", bot_name=BOT_NAME)


@app.post("/chat")
def chat():
    try:
        if not retriever:
            return jsonify({"error": "Vector store not loaded"}), 500

        user_msg = request.json.get("message")
        print(f"📨 Received message: {user_msg}")

        if not user_msg:
            return jsonify({"error": "No message provided"}), 400

        # Retrieve context - FIXED: Use invoke() instead of get_relevant_documents()
        print("🔍 Retrieving relevant documents...")

        # Try the new method first, fallback to old method
        try:
            # For newer LangChain versions
            docs = retriever.invoke(user_msg)
        except AttributeError:
            # For older LangChain versions
            docs = retriever.get_relevant_documents(user_msg)

        print(f"📄 Found {len(docs)} relevant documents")

        # Extract page content from documents
        context = "\n\n".join([doc.page_content for doc in docs])

        template = """
You are {bot_name}, an expert Dubai Real Estate AI assistant.

Context:
{context}

User Question: {question}

Answer professionally, clearly, and accurately. 
If answer not in context, reply using general real estate knowledge.
Keep responses concise and helpful.
""".strip()

        prompt = PromptTemplate.from_template(template)
        final_prompt = prompt.format(
            bot_name=BOT_NAME,
            context=context,
            question=user_msg
        )

        print("🤖 Calling DeepSeek API...")
        # DeepSeek LLM call
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": "You are a helpful Dubai Real Estate AI assistant."},
                {"role": "user", "content": final_prompt}
            ],
            stream=False
        )

        reply = response.choices[0].message.content
        print(f"✅ Response: {reply}")

        return jsonify({"reply": reply})

    except Exception as e:
        print(f"❌ Error: {e}")
        return jsonify({"error": f"An error occurred: {str(e)}"}), 500


if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=5000)