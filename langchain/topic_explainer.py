import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.document_loaders import AsyncHtmlLoader
from langchain_community.document_transformers import Html2TextTransformer
import asyncio
from typing import List
from langchain.schema import Document


load_dotenv()


def generate_response(
    user_prompt: str,
    temperature: int = 0.7,
    tokens: int = 256,
) -> str:
    if not os.getenv("GROQ_API_KEY"):
        os.environ["GROQ_API_KEY"] = input("Enter API key for Groq: ")

    llm = ChatGroq(
        model="openai/gpt-oss-120b",
        temperature=temperature,
        max_tokens=tokens,
        reasoning_effort="medium",
    )

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a professional explainer who can explain topic in easier to digest words. NEVER return empty response",
            ),
            (
                "human",
                "{user_prompt}",
            ),
        ]
    )

    # Combine prompt
    chain = prompt | llm | StrOutputParser()

    response = chain.invoke({"user_prompt": user_prompt})
    return response


def explain_topic(text: str) -> str:
    summary_prompt = f"Explain the below text for an eight-grade student within 500 characters (no hashtags, no emojis, no links) \
        \nText: {text}"
    formatted_prompt = summary_prompt.format(text)
    # print(f"Formatted Prompt: {formatted_prompt}")
    response = generate_response(formatted_prompt)
    return response


async def load_plain_document(url: str):
    try:
        loader = WebBaseLoader(url)
        docs = []
        async for doc in loader.alazy_load():
            docs.append(doc)
        return docs
    except Exception as error:
        print(f"Error while loading documents: {error}")


async def load_formatted_document(url: str):
    try:
        loader = AsyncHtmlLoader(url)
        transformer = Html2TextTransformer()
        docs: List[Document] = []
        async for doc in loader.alazy_load():
            docs.append(doc)
        transformer = Html2TextTransformer()
        docs_text = transformer.transform_documents(docs)  # batch transform
        return docs_text
    except Exception as error:
        print(f"Error while loading documents: {error}")


async def main():
    formatted_docs = await load_formatted_document(
        "https://en.wikipedia.org/wiki/Marketing_mix_modeling"
    )
    # print(f"Formatted Docs: {formatted_docs}")
    title = formatted_docs[0].metadata.get("title")
    print(f"Formatted title: {title}", end="\n\n")
    final_text = explain_topic(title)
    print(final_text)


if __name__ == "__main__":
    asyncio.run(main())
