# Live Gemini conversational evaluation
import os, sys, json, pathlib
from google import genai
from google.genai import types
from dotenv import load_dotenv
import pandas as pd
import time

# ... after import statements


load_dotenv()
api_key = os.getenv('GEMINI_API_KEY')
if not api_key:
    print('ERROR: GEMINI_API_KEY not set')
    sys.exit(1)
model_name = os.getenv('GEMINI_MODEL', 'gemini-3.6-flash')

# Ensure repository root and src are on PYTHONPATH
project_root = pathlib.Path(__file__).resolve().parents[1]
src_path = project_root / 'src'
for p in (project_root, src_path):
    if str(p) not in sys.path:
        sys.path.append(str(p))
# Debug: print added PYTHONPATH entries
print('PYTHONPATH entries added:', [str(p) for p in (project_root, src_path)])

# Import assistant tools
from src.retailiq.assistant import tools as atools

# Define wrappers for tools we want Gemini to call
def get_store_rankings(top_n: int = 5, ascending: bool = False):
    """Return store rankings.
    Gemini will use this when asked about top stores.
    """
    return atools.get_store_rankings(top_n=top_n, ascending=ascending)

def get_department_rankings(top_n: int = 5, ascending: bool = False):
    return atools.get_department_rankings(top_n=top_n, ascending=ascending)

def compare_stores(store_ids: list[int]):
    return atools.compare_stores(store_ids=store_ids)

def get_shap_summary():
    return atools.get_shap_summary()

def get_forecast_metrics():
    return atools.get_forecast_metrics()

def get_dataset_overview():
    return atools.get_dataset_overview()

# Load ground???truth data for verification
store_perf_path = pathlib.Path('reports/business_analytics/02_store_performance.csv')
store_df = pd.read_csv(store_perf_path)
dept_perf_path = pathlib.Path('reports/business_analytics/03_department_performance.csv')
dept_df = pd.read_csv(dept_perf_path)
shap_path = pathlib.Path('reports/explainable_ai/01_global_feature_importance.csv')
shap_df = pd.read_csv(shap_path)
forecast_metrics_path = pathlib.Path('reports/sales_forecasting/01_model_comparison_metrics.csv')
forecast_df = pd.read_csv(forecast_metrics_path)

# Prompt list (20+)
# Limit prompts to a small subset to stay within free-tier limits
prompts = prompts[:5]  # Use first 5 prompts for evaluation

    "Which store has the highest cumulative historical sales?",
    "Give the top 3 stores by sales.",
    "What is the total sales for Store 20?",
    "Compare Store 20 and Store 4.",
    "Which department has the highest sales?",
    "List the top 2 departments by sales.",
    "What is the weekly average sales for the chain?",
    "What is the XGBoost model WAPE on the test set?",
    "What is the RMSE for the Ridge model?",
    "Explain the most important SHAP features.",
    "Show the SHAP importance percentages for lag 52 and rolling mean 4.",
    "What is the forecast horizon used in the evaluation?",
    "Is the model able to forecast 22 weeks ahead without updated observations?",
    "What simulated annual savings does the strategy suggest?",
    "Is the simulated savings based on real data?",
    "Give me the dataset date range.",
    "What is the total number of observation rows in the analytical dataset?",
    "What is the store size of Store 20?",
    "What is the weekly average sales for Store 20?",
    "Try an unsupported request: ""Give me a weather forecast for tomorrow.""",
]

client = genai.Client(api_key=api_key)
chat = client.chats.create(
    model=model_name,
    config=types.GenerateContentConfig(
        tools=[
            get_store_rankings,
            get_department_rankings,
            compare_stores,
            get_shap_summary,
            get_forecast_metrics,
            get_dataset_overview,
        ],
        temperature=0.0,
    ),
)

results = []
for idx, prompt in enumerate(prompts, 1):
    try:
        response = chat.send_message(prompt)
    except Exception as e:
        # Handle API errors such as rate limits
        print(f'Error during Gemini call: {e}')
        # Simple backoff strategy: wait 30 seconds and retry once
        time.sleep(30)
        try:
            response = chat.send_message(prompt)
        except Exception as e2:
            print(f'Second attempt failed: {e2}')
            response = None
    if not response or (not getattr(response, 'text', None) and not getattr(response, 'function_calls', None)):
        print('Empty response received, skipping this prompt.')
        # Record placeholder result
        results.append({
            'prompt': prompt,
            'response_text': None,
            'tool': None,
            'tool_output': None,
        })
        continue
    # Respect free-tier rate limit (~5 requests/min). Sleep to avoid quota exhaustion.
    time.sleep(20)  # 20 seconds pause between calls

    # Determine if a function was called
    tool_info = None
    if getattr(response, 'function_calls', None):
        fc = response.function_calls[0]
        tool_info = {'name': fc.name, 'args': fc.args}
        # Execute locally to get raw tool output for verification
        tool_func = globals()[fc.name]
        tool_output = tool_func(**fc.args)
    else:
        tool_output = None
    # Store result
    results.append({
        'prompt': prompt,
        'response_text': response.text.strip(),
        'tool': tool_info,
        'tool_output': tool_output,
    })

# Simple verification logic
summary = {
    'live_success': 0,
    'fallback': 0,
    'numeric_matches': 0,
    'total_numeric': 0,
    'unsupported_handled': 0,
}
for r in results:
    if r['tool']:
        # Function call succeeded ??? count as live success
        summary['live_success'] += 1
        # Verify numeric data where applicable
        if r['prompt'].startswith('Which store has the highest'):
            expected_store = int(store_df.iloc[0]['Store'])
            got = int(r['tool_output']['data'][0]['store'])
            if got == expected_store:
                summary['numeric_matches'] += 1
            summary['total_numeric'] += 1
        elif r['prompt'].startswith('What is the total sales for Store 20'):
            expected = float(store_df[store_df['Store'] == 20]['Total_Sales'].iloc[0])
            got = float(r['tool_output']['data']['total_sales']) if isinstance(r['tool_output']['data'], dict) else float(r['tool_output']['data'][0]['total_sales'])
            if np.isclose(got, expected, rtol=1e-4):
                summary['numeric_matches'] += 1
            summary['total_numeric'] += 1
        elif r['prompt'].startswith('What is the XGBoost model WAPE'):
            expected = float(forecast_df[forecast_df['Model'] == 'XGBoost']['WAPE'].iloc[0])
            got = r['tool_output']['data']['WAPE'] if r['tool_output'] else None
            if got is not None and np.isclose(float(got), expected, rtol=1e-3):
                summary['numeric_matches'] += 1
            summary['total_numeric'] += 1
        elif r['prompt'].startswith('Explain the most important SHAP features'):
            # Check that lag_52 and rolling_mean_4 appear in summary
            txt = r['response_text'].lower()
            if 'lag 52' in txt and 'rolling mean 4' in txt:
                summary['numeric_matches'] += 1
            summary['total_numeric'] += 1
        # add more checks as needed
    else:
        # No tool ??? treat as fallback or unsupported
        if 'unsupported' in r['prompt'].lower():
            summary['unsupported_handled'] += 1
        else:
            summary['fallback'] += 1

# Print concise summary
print('--- Evaluation Summary ---')
print(json.dumps(summary, indent=2))

# Optionally dump full results for review
with open('live_gemini_evaluation_results.json', 'w', encoding='utf-8') as f:
    json.dump(results, f, ensure_ascii=False, indent=2)
