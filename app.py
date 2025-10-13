from flask import Flask, request, render_template
import logging
import markdown as md
from utils import *

# Setup logging
logging.basicConfig(level=logging.INFO)
app = Flask(__name__)
# Markdown filter for Jinja2
@app.template_filter('markdown')
def markdown_filter(text):
    return md.markdown(text)

# Load problems once
problems = load_problems("mznc2025_probs/problems_with_descriptions.json")

@app.route('/', methods=['GET', 'POST'])

def index():
    response_text = None
    selected_problem = None
    prompt_type = 'full'
    prompt_text = None
    selected_provider = 'gemini'
    selected_model = GEMINI_MODELS[0][0]
    solver_option = 'minizinc'  # default
    script_version = 'uncommented'  # default

    if request.method == 'POST':
        logging.info(f"Raw form data: {dict(request.form)}")
        selected_provider = request.form.get('provider', 'gemini')
        selected_model = request.form.get('model')
        logging.info(f"Provider selected: {selected_provider}")
        logging.info(f"Model selected: {selected_model}")
        selected_problem = request.form.get('problem')
        prompt_type = request.form.get('prompt_type', 'full')
        solver_option = request.form.get('solver_option', 'minizinc')
        script_version = request.form.get('script_version', 'uncommented')
        custom_description = request.form.get('custom_description', '').strip()
        custom_model = request.form.get('custom_model', '').strip()
        include_description = request.form.get('include_description') == 'yes'

        if custom_description and custom_model:
            description, script = custom_description, custom_model
        else:
            prob = problems.get(selected_problem, {})
            description = prob.get('description', '')
            # Choose which script to use
            if script_version == 'commented':
                script_path = prob.get('script_commented', '')
            else:
                script_path = prob.get('script', '')
            # Use get_problem_script logic but with explicit path
            if script_path.startswith('./'):
                try:
                    with open(script_path[2:], 'r') as sf:
                        script = sf.read()
                except Exception as e:
                    script = f"[Error reading {script_path}: {e}]"
            else:
                script = script_path

        # Choose solver list
        if solver_option == 'all':
            solver_list = ALL_SOLVERS
        else:
            solver_list = MINIZINC_SOLVERS
        solver_prompt = get_solver_prompt(solver_list, name_only=(prompt_type == 'name'))

        if include_description:
            prompt_text = f"Description:\n{description}\n\nMiniZinc model:\n{script}\n\n{solver_prompt}"
        else:
            prompt_text = f"MiniZinc model:\n{script}\n\n{solver_prompt}"

        if selected_provider == 'gemini':
            response_text = query_gemini(prompt_text, model_name=selected_model)
        elif selected_provider == 'groq':
            response_text = query_groq(prompt_text, model_name=selected_model)
        else:
            response_text = f"[Unknown provider: {selected_provider}]"

        if prompt_type == 'name' and selected_problem and selected_problem in problems:
            response_text = evaluate_response(selected_problem, response_text, problems)

    else:
        logging.info("GET request: Initial state")
        logging.info(f"Provider default: {selected_provider}")
        logging.info(f"Model default: {selected_model}")

    return render_template('index.html',
                           problems=problems,
                           response=response_text,
                           selected_problem=selected_problem,
                           prompt_type=prompt_type,
                           prompt_text=prompt_text,
                           gemini_models=GEMINI_MODELS,
                           groq_models=GROQ_MODELS,
                           selected_model=selected_model,
                           selected_provider=selected_provider,
                           solver_option=solver_option,
                           script_version=script_version)

if __name__ == '__main__':
    app.run(debug=True)