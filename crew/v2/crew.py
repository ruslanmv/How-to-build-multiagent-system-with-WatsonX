#from crewai import Crew, Task, Agent
from langchain_ibm import WatsonxLLM
from langchain_community.tools.tavily_search import TavilySearchResults
import os
from dotenv import load_dotenv
from ibm_watsonx_ai.metanames import GenTextParamsMetaNames as GenParams
from ibm_watsonx_ai.foundation_models.utils.enums import DecodingMethods

# Load environment variables from .env file
load_dotenv()

# Load API key and project ID from environment variables
watsonx_api_key = os.getenv("WATSONX_API_KEY")
project_id = os.getenv("PROJECT_ID")
url = os.getenv("WATSONX_URL")

# WatsonxLLM parameters
parameters = {
    GenParams.DECODING_METHOD: DecodingMethods.SAMPLE.value,
    GenParams.MAX_NEW_TOKENS: 100,
    GenParams.MIN_NEW_TOKENS: 50,
    GenParams.TEMPERATURE: 0.7,
    GenParams.TOP_K: 50,
    GenParams.TOP_P: 1
}

if not watsonx_api_key or not project_id:
    raise ValueError("Please set the WATSONX_API_KEY and PROJECT_ID in your .env file.")

# Define LLM models using WatsonxLLM
llm = WatsonxLLM(
    model_id="meta-llama/llama-3-1-70b-instruct",  
    url=url,
    apikey=watsonx_api_key,
    project_id=project_id,
    params=parameters
)

# Tool Initialization for the Researcher
tavily_tool = TavilySearchResults(max_results=5)  

# Router Agent definition
def router_decision(user_query):
    """Router agent that selects either 'Researcher' or 'Creator' based on the user input."""
    if "price" in user_query or "news" in user_query:
        return "Researcher"
    else:
        return "Creator"

router = Agent(
    llm=llm,
    role="Router",
    goal="Decide whether to use the Researcher or Creator agent based on the user query.",
    function=router_decision,
    backstory="You are a router that directs queries to the appropriate agent."
)

# Researcher Agent
researcher = Agent(
    llm=llm,
    role="Researcher",
    goal="Fetch the latest information available on the internet based on the user query.",
    backstory="You are an AI researcher skilled in retrieving the latest information.",
    tools=[tavily_tool]  
)

# Creator Agent
creator = Agent(
    llm=llm,
    role="Creator",
    goal="Generate informative and accurate knowledge-based responses to user queries.",
    backstory="You are an AI assistant specializing in generating knowledge-based answers."
)

# Define tasks for the agents
task_router = Task(
    description="Route the user query to either the Researcher or Creator agent.",
    expected_output="Either 'Researcher' or 'Creator'",
    agent=router
)

task_researcher = Task(
    description="Fetch search results based on the user query.",
    expected_output="A summary of the search results.",
    agent=researcher
)

task_creator = Task(
    description="Generate a response to the user query based on general knowledge.",
    expected_output="An informative response.",
    agent=creator
)

# Assemble the crew with the agents and tasks
crew = Crew(
    agents=[router, researcher, creator],
    tasks=[task_router, task_researcher, task_creator],
    verbose=True,
    output_log_file="crew_log.txt", 
    share_crew=False
)

# Execute the workflow 
def crew_workflow(user_query):
    print("\n==== Starting Crew Workflow ====\n")
    
    # Step 1: Router Agent determines which agent to use
    print(f"[Router] Evaluating the query: '{user_query}'")
    selected_agent = router_decision(user_query)
    print(f"[Router] Selected Agent: {selected_agent}\n")
    
    # Step 2: Based on the selected agent, either Researcher or Creator is invoked
    if selected_agent == "Researcher":
        print(f"[Researcher] Processing query: '{user_query}'")
        search_result = tavily_tool.invoke(user_query)  
        print(f"[Researcher] Result: {search_result}\n")
    elif selected_agent == "Creator":
        creator_prompt = f"Please provide an informative response for: '{user_query}'"
        result = llm.invoke(creator_prompt) 
        print(f"[Creator] Result: {result}\n")

# Example usage
crew_workflow("Fetch the bitcoin price over the past 5 days.") 
crew_workflow("Explain what Bitcoin is.")