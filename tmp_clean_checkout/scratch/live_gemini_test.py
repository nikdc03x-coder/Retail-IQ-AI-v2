# Live Gemini verification script
import os, json, sys
from google import genai

# Load .env file for credentials
from dotenv import load_dotenv
load_dotenv()
# Load API key and model from environment (already set in .env)
api_key = os.getenv('GEMINI_API_KEY')
if not api_key:
    print('ERROR: GEMINI_API_KEY not set')
    sys.exit(1)
model_name = os.getenv('GEMINI_MODEL', 'gemini-3.6-flash')

# Configure client
client = genai.Client(api_key=api_key)

# Define a wrapper for the existing tool function
def get_store_rankings(top_n: int = 5, ascending: bool = False):
    from src.retailiq.assistant import tools as atools
    return atools.get_store_rankings(top_n=top_n, ascending=ascending)

# Prompt that encourages function calling
prompt = "Which store has the highest cumulative historical sales? Use the get_store_rankings tool to retrieve the top 1 store."

# Generate content with function calling enabled
response = client.models.generate_content(
    model=model_name,
    contents=prompt,
    config=genai.types.GenerateContentConfig(
        tools=[get_store_rankings],
        temperature=0.0,
    ),
)

if getattr(response, 'function_calls', None):
    for fc in response.function_calls:
        print('Function call:', fc.name)
        print('Args:', json.dumps(fc.args))
        # Execute locally
        result = get_store_rankings(**fc.args)
        print('Tool result success:', result.get('success'))
        print('Summary:', result.get('summary'))
        print('Evidence source:', result.get('evidence_source'))
else:
    # No function call ??? print raw response
    print('Model response (no function call):')
    print(response.text)
