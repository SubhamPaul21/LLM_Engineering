import os
from dotenv import load_dotenv
from langchain.prompts import PromptTemplate
from agents.linkedin_lookup_agent import lookup as linkedin_lookup_agent
from third_parties.linkedin import scrape_linkedin_profile
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq


def ice_break_with(name: str):
  if not os.getenv("GROQ_API_KEY"):
    os.environ["GROQ_API_KEY"] = input("Enter API key for Groq: ")

  linkedin_profile_url = linkedin_lookup_agent(name=name)
  linkedin_data = scrape_linkedin_profile(
      linkedin_profile_url=linkedin_profile_url)

  summary_template = """
  given the linkedin information {information} about a person from I want you to create:
    1. A short summary
    2. Two interesting facts about them
    3. A topic that may interest them
    4. 2 creative Ice breakers to open a conversation with them
    """

  summary_prompt_template = PromptTemplate(input_variables=["information"],
                                           template=summary_template)

  llm = ChatGroq(model="meta-llama/llama-4-scout-17b-16e-instruct",
                 temperature=0.7)

  chain = summary_prompt_template | llm | StrOutputParser()
  response = chain.invoke({"information": linkedin_data})
  print(response)


if __name__ == "__main__":
  load_dotenv()

  print("Ice Breaker Enter: ")
  ice_break_with(name="Subham Paul Bell Integration India")
