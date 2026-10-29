# Live Gemini chat test using automatic function calling
import os, sys, json
from google import genai
from google.genai import types

# Load .env for credentials
from dotenv import load_dotenv
load_dotenv()

api_key = os.getenv('GEMINI_API_KEY')
if not api_key:
    print('ERROR: GEMINI_API_KEY not set')
    sys.exit(1)
model_name = os.getenv('GEMINI_MODEL', 'gemini-3.6-flash')

# Ensure project root is on PYTHONPATH for relative imports
project_root = os.path.abspath(os.path.join(__file__, '..', '..'))
if project_root not in sys.path:
    sys.path.append(project_root)

# Wrapper for the existing tool function
def get_store_rankings(top_n: int = 5, ascending: bool = False):
    """Return store rankings using the internal assistant tool.
    The function signature is used by Gemini to map arguments.
    """
    from src.retailiq.assistant import tools as atools
    return atools.get_store_rankings(top_n=top_n, ascending=ascending)

# Initialize the client
client = genai.Client(api_key=api_key)

# Create a chat with the tool attached
chat = client.chats.create(
    model=model_name,
    config=types.GenerateContentConfig(
        tools=[get_store_rankings],
        temperature=0.0,
    ),
)

# First user message that should trigger the tool
prompt = "Which store has the highest cumulative historical sales? Use the get_store_rankings tool to get the top 1 store."
response = chat.send_message(prompt)

# The SDK automatically executes the function and returns a final text response.
print('--- Chat response ---')
print(response.text)

# For transparency, also print the function call details if present
if getattr(response, 'function_calls', None):
    print('\nFunction calls made:')
    for fc in response.function_calls:
        print('Name:', fc.name)
        print('Args:', json.dumps(fc.args))
        # Execute locally to show the raw tool output (should match what SDK used)
        result = get_store_rankings(**fc.args)
        print('Raw tool result success:', result.get('success'))
        print('Summary:', result.get('summary'))
        print('Evidence source:', result.get('evidence_source'))

