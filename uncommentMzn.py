import os
import re
import shutil

def remove_comments(content: str) -> str:
    """Remove single-line (%) and multi-line (/* */) comments from MiniZinc code."""
    # Remove multi-line comments
    content = re.sub(r'/\*.*?\*/', '', content, flags=re.DOTALL)
    # Remove single-line comments
    content = re.sub(r'%.*', '', content)
    # Clean up trailing whitespace and blank lines
    cleaned = '\n'.join(line.rstrip() for line in content.splitlines() if line.strip())
    return cleaned


def process_mzn_file(file_path: str):
    """Rename the original file and create a cleaned version without comments."""
    base, ext = os.path.splitext(file_path)
    commented_path = f"{base}_commented{ext}"

    # Skip if this is already a commented backup
    if file_path.endswith("_commented.mzn"):
        return

    # Rename the original file
    shutil.move(file_path, commented_path)
    print(f"🔹 Renamed original to: {commented_path}")

    # Read and clean the commented version
    with open(commented_path, "r", encoding="utf-8") as f:
        content = f.read()

    cleaned = remove_comments(content)

    # Write the cleaned version back under the original filename
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(cleaned)

    print(f"Created uncommented version: {file_path}")


def clean_mzn_comments_in_dir(directory: str):
    """Recursively find and process all .mzn files in the directory."""
    for root, _, files in os.walk(directory):
        for filename in files:
            if filename.endswith(".mzn") and not filename.endswith("_commented.mzn"):
                full_path = os.path.join(root, filename)
                process_mzn_file(full_path)


if __name__ == "__main__":
    target_dir = input("Enter the path to the directory: ").strip()
    if os.path.isdir(target_dir):
        clean_mzn_comments_in_dir(target_dir)
        print("\nAll .mzn files processed successfully!")
    else:
        print("Error: The provided path is not a valid directory.")
