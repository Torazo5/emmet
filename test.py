import os
from dotenv import load_dotenv

# Load the .env file
load_dotenv()

# Print the variable
print(os.getenv("OPENAI_API_KEY"))