from crewai import Crew, Task, Agent
from langchain_ibm import WatsonxLLM
import os
from langchain.tools import tool
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Set up WatsonxLLM with API details
os.environ["WATSONX_APIKEY"] = "your-api-key"

# Define the LLM models
llm = WatsonxLLM(
    model_id="meta-llama/llama-3-70b-instruct",
    url="https://us-south.ml.cloud.ibm.com",
    params={
        "decoding_method": "sample",
        "max_new_tokens": 100,
        "temperature": 0.7,
        "top_k": 50,
        "top_p": 1,
    },
    project_id="your-project-id",
)

# Function to perform an internet search
@tool("Search in Internet")
def search_internet(topic: str) -> str:
    """Search service provides live searches on the internet for a specific topic."""
    # Here we simulate the output as we would get from an API
    result = f"Search results for '{topic}': [IBM Quantum Computing Research](https://www.ibm.com/topics/quantum-computing)"
    return result

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
    tools=[search_internet]
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
    output_log_file=True,
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
        search_result = search_internet(user_query)
        print(f"[Researcher] Result: {search_result}\n")
    elif selected_agent == "Creator":
        creator_prompt = f"Please provide an informative response for: '{user_query}'"
        result = llm.predict(creator_prompt)
        print(f"[Creator] Result: {result}\n")

# Example usage
crew_workflow("Fetch the bitcoin price over the past 5 days.")  # Should trigger the Researcher agent
crew_workflow("Explain what Bitcoin is.")  # Should trigger the Creator agent
