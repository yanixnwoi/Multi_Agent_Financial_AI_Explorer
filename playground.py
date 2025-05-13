from agno.models.groq import Groq
from agno.agent import Agent
from agno.tools.duckduckgo import DuckDuckGoTools
from agno.tools.yfinance import YFinanceTools
from agno.playground import Playground, serve_playground_app
from dotenv import load_dotenv
import os

# Load environment variables from the .env file
load_dotenv()

# Set up API keys for Groq and Agno from environment variables
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")
os.environ["AGNO_API_KEY"] = os.getenv("AGNO_API_KEY")

# Define an agent specialized in web search, using DuckDuckGo tools
internet_agent = Agent(
    name="Internet Search Specialist",  # Renamed for a different but similar meaning
    role="Explore online sources for knowledge",  # Updated wording for the role
    model=Groq(id="llama-3.3-70b-versatile"),  # Specify the model ID
    tools=[DuckDuckGoTools()],  # Utilize DuckDuckGo for web searches
    instructions="Ensure that sources are cited in every response",  # Updated instruction phrasing
    show_tool_calls=True,  # Display tool calls for transparency
    markdown=True,  # Format responses in Markdown
    add_datetime_to_instructions=True,  # Append the date and time to the instructions
    add_history_to_messages=True,  # Include the conversation history in the context
    num_history_responses=5,  # Number of prior messages to include
)

# Define an agent specialized in financial data analysis, using Yahoo Finance tools
financial_data_agent = Agent(
    name="Finance Agent",  # Renamed to a different but equivalent meaning
    role="Retrieve and analyze financial information",  # Updated role description
    model=Groq(id="llama-3.3-70b-versatile"),  # Specify the model ID
    tools=[
        YFinanceTools(
            stock_price=True,  # Enable stock price retrieval
            analyst_recommendations=True,  # Enable analyst recommendations
            stock_fundamentals=True,  # Enable stock fundamentals retrieval
            company_info=True,  # Enable company information retrieval
        )
    ],
    instructions="Present data in tabular format whenever appropriate",  # Updated instruction phrasing
    show_tool_calls=True,  # Display tool calls for transparency
    markdown=True,  # Format responses in Markdown
    add_datetime_to_instructions=True,  # Append the date and time to the instructions
    add_history_to_messages=True,  # Include the conversation history in the context
    num_history_responses=5,  # Number of prior messages to include
)

# Define a combined agent team for handling complex tasks requiring both web and financial expertise
team_agent = Agent(
    team=[internet_agent, financial_data_agent],  # Combine the two agents
    model=Groq(id="llama-3.3-70b-versatile"),  # Specify the model ID
    instructions=[
        "Always provide citations for information sources",  # Updated instruction phrasing
        "Whenever possible, organize data in tables",  # Updated instruction phrasing
    ],
    show_tool_calls=True,  # Display tool calls for transparency
    markdown=True,  # Format responses in Markdown
)

# Define a playground app to interact with the agents
app = Playground(agents=[internet_agent, financial_data_agent]).get_app()

# Run the app using a development server
if __name__ == "__main__":
    serve_playground_app("playground:app", reload=True)
