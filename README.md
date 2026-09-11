# Large Language Models For Solver Selection
The idea of this research is based on a novel paradigm for solving complex, NP-hard problems (e.g., schedul- ing, routing) by leveraging Large Language Models (LLMs) as dynamic orchestrators in Agentic solvers, as proposed in a position Paper by professor Roberto Amadini and Simone Gazza.

This repository contains the experimental code for the associated thesis, *"Large Language Models for Solver Selection: A Preliminary Study"* (see `Large Language Models for Solver Selection_ A Preliminary Study.pdf`), which investigates whether general-purpose LLMs can select an appropriate constraint solver for a MiniZinc model without task-specific fine-tuning.

## Results

Across several input representations (raw MiniZinc scripts, structured feature vectors via `mzn2feat`, natural-language problem descriptions, and FlatZinc-derived natural-language descriptions via `fzn2nl/`) and prompting/inference configurations, the study found:

- **General-purpose LLMs show a limited but non-negligible ability** to infer solver choices from a model's structural properties, but none of the tested configurations were competitive with the single best solver (SBS) on the full portfolio.
- **Input representation matters**: naive prompting with raw MiniZinc scripts underperformed the SBS; structured natural-language descriptions gave modest improvements over raw scripts.
- **Feature vectors (`mzn2feat`) did not outperform** the best natural-language-based configurations, whether used alone or combined with other representations.
- **Temperature tuning produced only small gains**, though controlled stochasticity does influence solver-selection behaviour.
- **LLMs tend to converge on a small subset of globally dominant solvers** rather than discriminating between candidates on a per-instance basis; this became more apparent (and more costly) when tested on a restricted solver portfolio, where the approach consistently underperformed the best solver in that reduced set.
- **`fzn2nl`** (introduced in this work) converts FlatZinc models into deterministic natural-language descriptions, matching the performance of other input modalities while using substantially fewer tokens.
- **Solver names strongly bias model choices**: permuting solver descriptions while keeping names fixed barely changed selections, while anonymizing solver names increased the influence of the descriptive content itself.
- **Qualitatively, the LLM shows a coherent high-level understanding** of constraint models and solver paradigms (e.g., correctly identifying CP vs. MILP vs. CP-SAT approaches) but struggles to discriminate between solvers within the same paradigm or to map fine-grained instance features to empirically optimal choices.

**Overall**: current general-purpose LLMs are not yet competitive with specialized solver-selection methods, but they demonstrate a meaningful capacity to interpret constraint-model structure and solver paradigms — suggesting LLM-based components could become viable in CP workflows with further domain-specific adaptation (e.g., fine-tuning on solver-performance data) and larger/more capable models. See `thesis/sections/conclusions/conclusions2.tex` (or the compiled `thesis/main.pdf`) for the full discussion, limitations, and future work.

## `fzn2nl`

`fzn2nl/` is a parser and text generator, developed as part of this research, that converts a FlatZinc (`.fzn`) model into a deterministic natural-language description of the underlying problem. It was built to address a limitation observed when prompting LLMs directly with raw MiniZinc/FlatZinc code or with `mzn2feat` feature vectors: solver names and recognizable "famous" problem patterns in the raw code can bias the LLM's solver choice independently of the model's actual structure, and raw code is verbose relative to the structural information it conveys. To avoid this, `fzn2nl` works against a custom-patched MiniZinc compiler that produces a solver-agnostic pseudo-FlatZinc representation (global constraints are kept undecomposed rather than expanded into a specific solver's propagator-level form), so the description doesn't leak which solver — or solver family — was used to compile the model. From that representation, `fzn2nl` extracts variables and domains, constraints (optionally grouped by category via `fzn_descriptions_categorized.json`), the objective function, and the search strategy, and renders each as natural-language text (see `variables.py`, `constraints.py`, `objFunction.py`, `searchStrat.py`, `nlMappings.py`). In the thesis experiments, this representation matched the performance of the other input modalities tested while using substantially fewer tokens, making it a more context-efficient way to expose problem structure to an LLM. It can be run standalone from the command line:
```bash
python fzn2nl/main.py path/to/model.fzn [--categorize-constraints]
```

## Flask App (`app.py`)

This app provides an interface to get solver recommendations for various MiniZinc problems using an LLM, via either the Gemini or Groq API.

### How it works
- Loads a set of MiniZinc problems and their descriptions from `mznc2025_probs/problems_with_descriptions.json`.
- For each problem, the app displays a button to request a solver recommendation.
- You choose a provider (Gemini or Groq) and model, and the app sends the problem description and MiniZinc model code to the selected LLM, asking for the best solver(s) and optionally a reasoning.
- The response is shown in the browser, along with the prompt that was sent.
- You can choose between a detailed analysis or just the top 3 solver names.

### How to use
1. Make sure you have Python and Flask installed.
2. Set the API key(s) for the provider(s) you want to use as environment variables: `GEMINI_API_KEY` for Gemini, `GROQ_API_KEY` for Groq.
3. Run the app:
   ```bash
   python app.py
   ```
4. Open your browser and go to `http://127.0.0.1:5000/`.
5. Select a problem, provider/model, and prompt type, then view the LLM's recommendation and reasoning.

## Author
### Vittorio Rossetto
 - [GitHub](https://github.com/VittorioRossetto)
 - [Linkedin](https://www.linkedin.com/in/vittorio-rossetto-508086333/)