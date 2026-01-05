import os
from dotenv import load_dotenv
load_dotenv()
import json
import requests
from google import genai
from google.genai import types
from groq import Groq
import re

# ---------- Configuration ----------
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent"
GEMINI_MODELS = [
    ("gemini-2.5-flash", "Gemini 2.5 Flash"),
    ("gemini-2.5-pro", "Gemini 2.5 Pro"),
    ("gemini-2.5-flash-lite", "Gemini 2.5 Flash Lite"),
    ("gemini-2.0-flash", "Gemini 2.0 Flash"),
    ("gemini-2.0-flash-lite", "Gemini 2.0 Flash Lite")
]

# ---------- Groq Model Support ----------
def load_groq_models(json_path="grok_models.json"):
    """Loads Groq models from a JSON file and returns a list of (id, label) tuples."""
    try:
        with open(json_path, "r") as f:
            data = json.load(f)
        models = []
        for entry in data.get("data", []):
            model_id = entry.get("id")
            label = model_id if model_id else "Unknown Model"
            models.append((model_id, label))
        return models
    except Exception as e:
        print(f"Error loading Groq models: {e}")
        return []

GROQ_MODELS = load_groq_models()

# ---------- Prompts ----------

# ---------- Solver Options ----------
ALL_SOLVERS = [
    "atlantis-free", "cbc-free", "cbc-par", "choco-solver__cp_-fd", "choco-solver__cp_-free", "choco-solver__cp_-par",
    "choco-solver__cp-sat_-fd", "choco-solver__cp-sat_-free", "choco-solver__cp-sat_-par", "chuffed-fd", "chuffed-free",
    "cp_optimizer-free", "cp_optimizer-par", "cplex-free", "cplex-par", "gecode-fd", "gecode-par", "gecode_dexter-open",
    "gurobi-free", "gurobi-par", "highs-free", "highs-par", "huub-fd", "huub-free", "izplus-free", "izplus-par",
    "jacop-fd", "jacop-free", "or-tools_cp-sat-fd", "or-tools_cp-sat-free", "or-tools_cp-sat-par", "or-tools_cp-sat_ls-free",
    "or-tools_cp-sat_ls-par", "picatsat-free", "pumpkin-fd", "pumpkin-free", "scip-free", "scip-par", "sicstus_prolog-fd",
    "sicstus_prolog-free", "yuck-free", "yuck-par"
]

FREE_SOLVERS = [
    "atlantis-free", "cbc-free", "choco-solver__cp_-free", "choco-solver__cp_-par", "choco-solver__cp-sat_-free", "chuffed-free",
    "cp_optimizer-free", "cplex-free","gurobi-free", "highs-free", "huub-free", "izplus-free", "jacop-free", "or-tools_cp-sat-free", 
    "or-tools_cp-sat_ls-free", "picatsat-free", "pumpkin-free", "scip-free", "sicstus_prolog-free", "yuck-free"
]

SIGNIFICATIVE_SOLVERS = [
    "cbc-free", "choco-solver__cp_-free", "choco-solver__cp_-par", "choco-solver__cp-sat_-free",
    "cp_optimizer-free", "cplex-free","gurobi-free", "highs-free", "izplus-free", "jacop-free",  
    "pumpkin-free", "scip-free", "sicstus_prolog-free"
]

MINIZINC_SOLVERS = [
    "Gecode", "Chuffed", "Google OR-Tools CP-SAT", "HiGHS", "COIN-OR CBC"
]

def get_solver_prompt(solver_list=None, name_only=False):
    """
    Returns a prompt string for the given solver list.
    If name_only is True, restricts output to names in brackets.
    """
    if solver_list is None:
        # default to the set of free/public solvers for lightweight benchmarking
        solver_list = FREE_SOLVERS
    solver_lines = '\n'.join(f"- {s}" for s in solver_list)
    if name_only:
        return (
            f"The goal is to determine which constraint programming solver would be best suited for this problem, considering the following options:\n\n"
            f"{solver_lines}\n\n"
            "Answer only with the name of the 3 best solvers inside square brackets separated by comma and nothing else."
        )
    else:
        return (
            f"The goal is to determine which constraint programming solver would be best suited for this problem, considering the following options:\n\n"
            f"{solver_lines}\n\n"
            "Please analyze the model and recommend the best solver for this problem, explaining your reasoning."
        )

# ---------- Utility Functions ----------
def load_problems(json_path: str):
    """Reads the json file containing problem descriptions."""
    with open(json_path, "r") as f:
        return json.load(f)
    

def get_model_label(model_id):
    """Returns the readable label for a given model ID."""
    for mid, label in GEMINI_MODELS:
        if mid == model_id:
            return label
    return model_id

def get_problem_script(problem_entry: dict):
    """Reads the MiniZinc model content, even if it's a file path."""
    script = problem_entry.get('script', '')
    if script.startswith('./'):
        try:
            with open(script[2:], 'r') as sf:
                return sf.read()
        except Exception as e:
            return f"[Error reading {script}: {e}]"
    return script

def query_gemini(prompt_text: str, model_name: str = "gemini-2.5-flash", temperature: float | None = None):
    """Sends prompt to Gemini using the selected model and returns the response text."""
    client = genai.Client()
    cfg = types.GenerateContentConfig(temperature=temperature) if temperature is not None else None
    response = client.models.generate_content(model=model_name, config=cfg, contents=prompt_text)
    return getattr(response, "text", str(response))

def get_groq_model_label(model_id):
    for mid, label in GROQ_MODELS:
        if mid == model_id:
            return label
    return model_id

def query_groq(prompt_text: str, model_name: str = "llama-3-70b-versatile", temperature: float | None = None):
    """Sends prompt to Groq using the selected model and returns the response text."""
    client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
    kwargs = {}
    if temperature is not None:
        kwargs["temperature"] = temperature
    chat_completion = client.chat.completions.create(
        messages=[{"role": "user", "content": prompt_text}],
        model=model_name,
        **kwargs,
    )
    return chat_completion.choices[0].message.content

def evaluate_response(selected_problem: str, response_text: str, problems: dict):
    """Evaluates Gemini's response against the known top 3 solvers."""
    challenge_top3 = problems[selected_problem].get('top3_solvers', [])
    # Extract the top 3 solvers from the response, which is expected in the format [s1,s2,s3]
    match = re.search(r"\[([^\]]+)\]", response_text)
    if match:
        llm_top3 = [s.strip() for s in match.group(1).split(',')]
    else:
        llm_top3 = []
    # log the extracted top 3 for debugging
    print(f"LLM Response: {response_text}")
    print(f"LLM Top 3: {llm_top3}")
    print(f"Challenge Top 3: {challenge_top3}")

    order_match = llm_top3 == challenge_top3
    
    order_match_count = sum(g == c for g, c in zip(llm_top3, challenge_top3))
    unordered_match_count = len(set(llm_top3) & set(challenge_top3))
    
    match_info = (
        f"<b>MiniZinc Challenge Top 3:</b> {', '.join(challenge_top3)}<br>"
        f"<b>LLM's Top 3:</b> {', '.join(llm_top3)}<br>"
        f"<b>Exact order matches:</b> {order_match_count} / 3<br>"
        f"<b>Unordered matches:</b> {unordered_match_count} / 3"
    )
    if order_match:
        match_info += "<br><b>Suggestion is correct!</b>"
    elif not order_match and unordered_match_count == 3:
        match_info += "<br><b>Order is incorrect.</b>"
    elif unordered_match_count > 0 and unordered_match_count < 3:
        match_info += "<br><b>Suggestion is partially correct.</b>"
    else:
        match_info += "<br><b>No matches found, incorrect suggestion.</b>"

    return match_info