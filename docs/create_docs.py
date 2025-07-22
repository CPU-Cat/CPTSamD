import ast
from pathlib import Path


def extract_docstrings(folder_path, save_folder="docs"):
    folder = Path(folder_path)
    py_files = list(folder.glob("*.py"))
    print(f"Found {len(py_files)} Python files in {folder_path}")
    docs = []

    for file in py_files:
        with open(file, "r", encoding="utf-8") as f:
            source = f.read()

        tree = ast.parse(source)
        file_doc = ast.get_docstring(tree)
        docs.append(f"# File: {file.name}\n")

        # Module-level docstring
        if file_doc:
            docs.append(f"**Module Docstring:**\n\n{file_doc}\n")

        # Function and class docstrings
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                func_name = node.name
                func_doc = ast.get_docstring(node)
                if func_doc:
                    docs.append(f"## Function: {func_name}\n\n{func_doc}\n")
            elif isinstance(node, ast.ClassDef):
                class_name = node.name
                class_doc = ast.get_docstring(node)
                if class_doc:
                    docs.append(f"## Class: {class_name}\n\n{class_doc}\n")

        docs.append("\n---\n")

    # Write to docs.md
    output_file = "docs.md"
    Path(save_folder).mkdir(parents=True, exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        f.write("\n".join(docs))


extract_docstrings("../CPTSamD")
