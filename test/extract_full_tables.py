import json
from bs4 import BeautifulSoup

def extract_fixed_structure(html_path, output_path="./data/allTables_free.json"):
    """
    Extracts MiniZinc Challenge-style HTML tables into hierarchical JSON:
    { Problem: { Instance: [ {Solver, Status, Time, ...} ] } }
    Uses fixed field names.
    """

    FIELD_NAMES = [
        "Problem",
        "Instance",
        "Solver",
        "Status",
        "Time",
        "Objective",
        "Score",
        "Score Incomplete",
        "Score Area"
    ]

    with open(html_path, "r", encoding="utf-8") as f:
        html = f.read()

    soup = BeautifulSoup(html, "html.parser")
    tbodies = soup.find_all("tbody")

    results = {}
    current_problem = None
    current_instance = None

    for tbody in tbodies:
        rows = tbody.find_all("tr")
        for row in rows:
            cols = [c.get_text(strip=True) for c in row.find_all("td")]
            if not cols:
                continue

            # If first column has a value, it's a new problem
            if cols[0]:
                current_problem = cols[0]
                if current_problem not in results:
                    results[current_problem] = {}

            # If second column has a value, it's a new instance
            if len(cols) > 1 and cols[1]:
                current_instance = cols[1]
                if current_problem and current_instance not in results[current_problem]:
                    results[current_problem][current_instance] = []

            # Skip TOTAL rows
            if len(cols) > 2 and cols[2].strip().lower() == "total":
                continue

            # Build row data dictionary using fixed field names
            entry = {
                "Solver": cols[2] if len(cols) > 2 else "",
                "Status": cols[3] if len(cols) > 3 else "",
                "Time": cols[4] if len(cols) > 4 else "",
                "Objective": cols[5] if len(cols) > 5 else "",
                "Score": cols[6] if len(cols) > 6 else "",
                "Score Incomplete": cols[7] if len(cols) > 7 else "",
                "Score Area": cols[8] if len(cols) > 8 else "",
            }

            # Add to current problem/instance
            if current_problem and current_instance:
                results[current_problem][current_instance].append(entry)

    # Save to JSON
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"Extracted structured data saved to {output_path}")


if __name__ == "__main__":
    html_file = "./data/allTables_free.html"  # Path to your HTML file
    output_json = "./data/allTables_free.json"
    extract_fixed_structure(html_file, output_json)
