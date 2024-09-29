from psycopg import Connection
from langgraph.checkpoint.postgres import PostgresSaver
from langgraph.prebuilt import create_react_agent
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from typing import Literal

# Define the database URI and connection parameters
DB_URI = "postgresql://torazocode:@localhost:5432/postgres?sslmode=disable"
connection_kwargs = {
    "autocommit": True,
    "prepare_threshold": 0,
}

# Define a tool function
@tool
def get_weather(city: Literal["nyc", "sf"]):
    """Get the weather of the city"""
    if city == "nyc":
        return "It might be cloudy in nyc"
    elif city == "sf":
        return "It's always sunny in sf"
    else:
        raise AssertionError("Unknown city")

# Function to format and display the results
def display_results(res, checkpoint_tuple):
    # Extract relevant details
    conversation_id = checkpoint_tuple.checkpoint['id']
    
    # Get the latest AI message for the response
    chat_response = res['messages'][-1].content
    token_usage = res['messages'][-1].response_metadata['token_usage']
    
    input_tokens = token_usage['prompt_tokens']
    output_tokens = token_usage['completion_tokens']
    
    # Display the results
    print(f"\nConversation ID: {conversation_id}")
    print(f"Chat Response: {chat_response}")
    print(f"Input Tokens: {input_tokens}")
    print(f"Output Tokens: {output_tokens}\n")

# Main interaction logic
def main():
    # Using a synchronous connection
    with Connection.connect(DB_URI, **connection_kwargs) as conn:
        # Initialize PostgresSaver
        checkpointer = PostgresSaver(conn)
        
        # Setup checkpointer schema if it's the first time
        checkpointer.setup()

        # Create the LangGraph agent
        model = ChatOpenAI(model_name="gpt-4o-mini", temperature=0)
        tools = [get_weather]
        graph = create_react_agent(model, tools=tools, checkpointer=checkpointer)

        # Define the configuration
        config = {"configurable": {"thread_id": "2"}}
        
        # Loop to take input and invoke the agent
        while True:
            user_input = input("Enter your prompt (or type 'exit' to quit): ")
            if user_input.lower() == "exit":
                break

            # Invoke the agent with user input
            res = graph.invoke({"messages": [("human", user_input)]}, config)
            
            # Get the latest checkpoint
            checkpoint_tuple = checkpointer.get_tuple(config)

            # Display the formatted results
            display_results(res, checkpoint_tuple)

# Run the interaction
if __name__ == "__main__":
    main()
