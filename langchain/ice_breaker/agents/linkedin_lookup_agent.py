import os
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.tools import Tool
from langchain_tavily import TavilySearch
from langchain.agents import AgentExecutor, create_react_agent
from langchain_groq import ChatGroq
from langchain import hub

load_dotenv()


def get_profile_url_tavily(name: str) -> str:
  """Searches for Linkedin or Twitter Profile Page."""
  if not os.getenv("TAVILY_API_KEY"):
    os.environ["TAVILY_API_KEY"] = input("Enter API key for Tavily: ")

  search = TavilySearch()
  res = search.run(f"{name}")
  return res["results"][0]["url"]


def lookup(name: str) -> str:
  if not os.getenv("GROQ_API_KEY"):
    os.environ["GROQ_API_KEY"] = input("Enter API key for Groq: ")

  tools_for_agent = [
      Tool(
          name="Crawl Google for linkedin profile page",
          func=get_profile_url_tavily,
          description="useful for when you need get the Linkedin Page URL",
      )
  ]

  summary_template = """
    given the full linkedin name {name}, I want you to extract the linkedin profile url from the search results. Only return the url 
  """

  prompt_template = PromptTemplate(input_variables=["name"],
                                   template=summary_template)

  llm = ChatGroq(model="meta-llama/llama-4-scout-17b-16e-instruct",
                 temperature=0.7)
  react_prompt = hub.pull("hwchase17/react")

  agent = create_react_agent(llm, tools_for_agent, react_prompt)
  agent_executor = AgentExecutor(agent=agent,
                                 tools=tools_for_agent,
                                 verbose=True,
                                 handle_parsing_errors=True)

  result = agent_executor.invoke(
      input={"input": prompt_template.format_prompt(name=name)})
  linkedin_profile_url = result["output"]
  return linkedin_profile_url


if __name__ == "__main__":
  print(lookup(name="Subham Paul Bell Integration India"))
