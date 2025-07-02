from django.http import StreamingHttpResponse, JsonResponse
from django.views.decorators.csrf import csrf_exempt
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores.faiss import FAISS
from langchain_core.messages import HumanMessage
from langchain.callbacks.base import BaseCallbackHandler
from langchain.docstore.document import Document
from django.shortcuts import render
import threading
from langchain_community.docstore.in_memory import InMemoryDocstore
import faiss
import os

#Get OpenAI key 

OPENAIKEY = os.getenv('OPEN_AI_KEY')

# -- Step 1: Initialize embedding model
embedding_model = OpenAIEmbeddings(openai_api_key=OPENAIKEY)  # Replace with your actual key

# -- Step 2: Create FAISS with dummy doc to avoid error, then clear it
# -- Step 2: Create FAISS with dummy doc to avoid error, then recreate properly
# Step 1: Set dimensions (use 1536 for OpenAI embeddings)
embedding_size = 1536
faiss_index = faiss.IndexFlatL2(embedding_size)

# Step 2: Create an empty docstore and mapping
docstore = InMemoryDocstore({})
index_to_docstore_id = {}

# Step 3: Build FAISS vectorstore
vectorstore = FAISS(
    embedding_model.embed_query,
    faiss_index,
    docstore,
    index_to_docstore_id
)
vectorstore.docstore._dict.clear()
vectorstore.index.reset()

# -- Step 3: View for chatbot page
def chatbot(request):
    return render(request, 'chatbot.html')


# -- Step 4: Streaming handler class
class DjangoStreamingHandler(BaseCallbackHandler):
    def __init__(self):
        self.queue = []
        self.streaming = True

    def on_llm_new_token(self, token: str, **kwargs) -> None:
        self.queue.append(token)

    def get_tokens(self):
        while self.streaming or self.queue:
            if self.queue:
                yield self.queue.pop(0)


# -- Step 5: POST /messages/ endpoint with prompt + embeddings + stream
@csrf_exempt
def messages(request):
    if request.method != 'POST':
        return JsonResponse({'status': 'error', 'message': 'Invalid method'}, status=405)

    user_input = request.POST.get('content')

    def event_stream():
        try:
            # Step 1: Retrieve relevant context via vector similarity
            if not vectorstore.docstore._dict:
                similar_docs = []
            else:
                try:
                    similar_docs = vectorstore.similarity_search(user_input, k=2)
                except KeyError:
                    similar_docs = []

            context = "\n".join([doc.page_content for doc in similar_docs]) or "No relevant context found."

            # Step 2: Build prompt using template
            prompt_template = PromptTemplate.from_template(
                "You are a helpful assistant. Here is the context:\n{context}\n\nUser: {question}\nAssistant:"
            )
            formatted_prompt = prompt_template.format(context=context, question=user_input)

            print(formatted_prompt)

            # Step 3: Setup streaming handler and LLM
            stream_handler = DjangoStreamingHandler()
            llm = ChatOpenAI(
                temperature=0.7,
                streaming=True,
                callbacks=[stream_handler],
                openai_api_key=OPENAIKEY  # Replace with your actual key
            )

            # Step 4: Run LLM generation in background thread
            def run_llm():
                try:
                    llm.invoke([HumanMessage(content=formatted_prompt)])
                    full_response = ''.join(stream_handler.queue)
                    print(f"Assistant response: {full_response}")
                finally:
                    stream_handler.streaming = False

            threading.Thread(target=run_llm).start()

            # Step 5: Stream tokens live to frontend
            response_text = ""
            for token in stream_handler.get_tokens():
                print('token',token)
                response_text += token
                yield token

            print(f"Assistant final message:\n{response_text}")

            # Step 6: Save the user input for future context
            if user_input not in [doc.page_content for doc in vectorstore.docstore._dict.values()]:
                vectorstore.add_documents([Document(page_content=user_input)])
        except Exception as e:
            yield f"\n[Error] {str(e)}"

    return StreamingHttpResponse(event_stream(), content_type='text/plain')
