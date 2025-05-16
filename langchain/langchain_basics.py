import os
from pathlib import Path
from dotenv import load_dotenv
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq

load_dotenv()

if not os.getenv("GROQ_API_KEY"):
    os.environ["GROQ_API_KEY"] = input("Enter API key for Groq: ")

# if not os.getenv("HUGGINGFACE_HUB_TOKEN"):
#     token = input("Enter API key for Hugging Face Hub: ")
#     login(token)
# else:
#     login(os.getenv("HUGGINGFACE_HUB_TOKEN"))

llm = ChatGroq(model="meta-llama/llama-4-scout-17b-16e-instruct",
               temperature=0.7)

user_prompt = ChatPromptTemplate.from_template(
    "Write a 4 line funnypoem about {topic}. Just output the poem in plain english without formatting and nothing else. You can add expressions for the reader to dictate the poem better."
)

chain = user_prompt | llm | StrOutputParser()
output = chain.invoke({"topic": "Dreams"})
print(output)
