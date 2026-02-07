import argparse
from parser import parse_fzn

from constraints import describe_constraints
from objFunction import describe_problem
from searchStrat import describe_search
from variables import describe_variables_detailed

def summarize(model, categorize_constraints: bool = False):
    # print("=" * 60)
    # print("PROBLEM SUMMARY")
    # print("=" * 60)

    print("\nProblem:")
    problem_txt = (describe_problem(model) or "").strip()
    search_txt = (describe_search(model.search, model=model) or "").strip()
    if search_txt:
        if search_txt.startswith("The model suggests"):
            search_txt = "Where the model suggests" + search_txt[len("The model suggests"):]
        combined = (problem_txt.rstrip(".") + ". " + search_txt) if problem_txt else search_txt
    else:
        combined = problem_txt
    print(" ", combined)

    print("\nVariables:")
    print(describe_variables_detailed(model))

    print("\nConstraints:")
    print(describe_constraints(model, categorize=categorize_constraints))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Parse and summarize a FlatZinc (.fzn) model")
    parser.add_argument("fzn", help="Path to the .fzn file")
    parser.add_argument(
        "--categorize-constraints",
        action="store_true",
        help="Group constraints by category from fzn_descriptions_categorized.json",
    )
    args = parser.parse_args()

    model = parse_fzn(args.fzn)
    summarize(model, categorize_constraints=args.categorize_constraints)
