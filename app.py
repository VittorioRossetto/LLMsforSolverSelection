from flask import Flask, request, render_template
import markdown as md
from utils import *

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
    selected_model = request.form.get('model', GEMINI_MODELS[0][0])

    if request.method == 'POST':
        selected_problem = request.form.get('problem')
        prompt_type = request.form.get('prompt_type', 'full')
        custom_description = request.form.get('custom_description', '').strip()
        custom_model = request.form.get('custom_model', '').strip()

        if custom_description and custom_model:
            description, script = custom_description, custom_model
        else:
            prob = problems.get(selected_problem, {})
            description = prob.get('description', '')
            script = get_problem_script(prob)

        solver_prompt = SOLVER_PROMPT_NAME_ONLY if prompt_type == 'name' else SOLVER_PROMPT
        prompt_text = f"Description:\n{description}\n\nMiniZinc model:\n{script}\n\n{solver_prompt}"
        response_text = query_gemini(prompt_text, model_name=selected_model)

        # Optional: compare Gemini’s prediction with MiniZinc top 3
        if prompt_type == 'name' and selected_problem and selected_problem in problems:
            response_text = evaluate_response(selected_problem, response_text, problems)
            
    return render_template('index.html',
                           problems=problems,
                           response=response_text,
                           selected_problem=selected_problem,
                           prompt_type=prompt_type,
                           prompt_text=prompt_text,
                           gemini_models=GEMINI_MODELS,
                           selected_model=selected_model)

if __name__ == '__main__':
    app.run(debug=True)
