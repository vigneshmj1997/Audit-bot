from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate,PromptTemplate
#from langchain_openai import OpenAIEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
import langchain
from langchain_huggingface.llms import HuggingFacePipeline
from langchain_huggingface.llms import HuggingFacePipeline
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from langchain_huggingface import HuggingFaceEndpoint
import os

langchain.debug = False


model_path = 'Alibaba-NLP/gte-large-en-v1.5'

model_id = "microsoft/Phi-3.5-mini-instruct"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id)
pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, max_length=256)
llm = HuggingFacePipeline(pipeline=pipe)

# llm = HuggingFaceEndpoint(
#     repo_id=model_id,
#     max_length=256,
#     temperature=0.5,
#     huggingfacehub_api_token="__HF__TOKEN",
# )

# using this for prototyping will get same results for when hosted locally


vector_store = Chroma(
    collection_name="example_collection",
    embedding_function=HuggingFaceEmbeddings(model_name=model_path,model_kwargs={"trust_remote_code":True}),
    persist_directory="./chroma_langchain_db",  # Where to save data locally, remove if not neccesary
)


# Change in template
prompt_template = """
<|system|>
You are an assistant for question-answering tasks. 
Use the following pieces of retrieved context to answer 
the question. If you don't know the answer, say that you 
Make the answer descriptive and informative.
Do not mention context in the answer
Include all the points in the answer make it informative
Mention all the clause and rules
{context}
<|end|>

<|user|>

{input}

<|end|>

<|assistant|>
"""

prompt = PromptTemplate(
    template=prompt_template,
    input_variables = ["context", "input"],
)


#print(prompt)
question_answer_chain = create_stuff_documents_chain(llm, prompt)

retrieved_doc = vector_store.as_retriever(search_type="similarity",search_kwargs={"k": 2})

rag_chain = create_retrieval_chain(retrieved_doc, question_answer_chain)

response = rag_chain.invoke({"input": "What are the conditions for a person to obtain separate registration for multiple places of business within a State or Union territory under Rule 11?"})
response["answer"]
print(response["answer"])
