import ast
import math
import operator
from pathlib import Path
from langchain.tools import tool
from langchain_community.tools import DuckDuckGoSearchRun


search_engine = DuckDuckGoSearchRun()


@tool
def web_search(query: str) -> str:
    """Search the internet for current information about a topic.

    Use this tool for recent facts, research, and web sources.
    Input should be a clear search query.
    """
    try:
        result = search_engine.run(query)
        return str(result)[:10000]
    except Exception as exc:
        return f"Web search failed: {exc}"


ALLOWED_OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}

ALLOWED_FUNCTIONS = {
    "sqrt": math.sqrt,
    "sin": math.sin,
    "cos": math.cos,
    "tan": math.tan,
    "log": math.log,
    "log10": math.log10,
    "ceil": math.ceil,
    "floor": math.floor,
    "fabs": math.fabs,
}

ALLOWED_NAMES = {
    "pi": math.pi,
    "e": math.e,
}


def evaluate_expression(node):
    """Evaluate only explicitly permitted mathematical syntax."""

    if isinstance(node, ast.Expression):
        return evaluate_expression(node.body)

    if isinstance(node, ast.Constant):
        if type(node.value) not in (int, float):
            raise ValueError("Only numeric values are allowed.")
        return node.value

    if isinstance(node, ast.BinOp):
        operation = ALLOWED_OPERATORS.get(type(node.op))

        if operation is None:
            raise ValueError("Unsupported mathematical operator.")

        left = evaluate_expression(node.left)
        right = evaluate_expression(node.right)

        if isinstance(node.op, ast.Pow) and abs(right) > 1000:
            raise ValueError("Exponent is too large.")

        result = operation(left, right)

        if isinstance(result, (int, float)) and abs(result) > 1e100:
            raise ValueError("Result is too large.")

        return result

    if isinstance(node, ast.UnaryOp):
        operation = ALLOWED_OPERATORS.get(type(node.op))

        if operation is None:
            raise ValueError("Unsupported unary operator.")

        return operation(evaluate_expression(node.operand))

    if isinstance(node, ast.Name):
        if node.id not in ALLOWED_NAMES:
            raise ValueError(f"Unknown constant: {node.id}")

        return ALLOWED_NAMES[node.id]

    if isinstance(node, ast.Call):
        if not isinstance(node.func, ast.Name):
            raise ValueError("Unsupported function call.")

        function = ALLOWED_FUNCTIONS.get(node.func.id)

        if function is None:
            raise ValueError("Mathematical function is not allowed.")

        if node.keywords:
            raise ValueError("Keyword arguments are not allowed.")

        arguments = [
            evaluate_expression(argument)
            for argument in node.args
        ]

        return function(*arguments)

    raise ValueError("Unsupported expression.")


@tool
def python_calculator(expression: str) -> str:
    """Calculate mathematical expressions accurately.

    Supports arithmetic, parentheses, powers, pi, e, and functions
    such as sqrt, sin, cos, tan, log, ceil, and floor.
    Example: (125 * 4) + sqrt(144)
    """
    try:
        if len(expression) > 300:
            return "Error: Expression is too long."

        tree = ast.parse(expression, mode="eval")
        result = evaluate_expression(tree)

        return f"{expression} = {result}"

    except Exception as exc:
        return f"Calculation error: {exc}"


DATA_DIR = (Path(__file__).resolve().parent / "data").resolve()

ALLOWED_EXTENSIONS = {".txt", ".md", ".json"}


@tool
def read_local_file(filename: str) -> str:
    """Read a local TXT, Markdown, or JSON file from the project's
    data folder. Input should be the filename, such as notes.txt.
    Use this tool when the answer depends on local document content.
    """
    try:
        DATA_DIR.mkdir(parents=True, exist_ok=True)

        requested_path = Path(filename)

        if requested_path.name != filename or requested_path.is_absolute():
            return "Error: Provide a filename located in the data folder."

        file_path = (DATA_DIR / requested_path).resolve()

        if not file_path.is_relative_to(DATA_DIR):
            return "Error: Access outside the data folder is forbidden."

        if file_path.suffix.lower() not in ALLOWED_EXTENSIONS:
            return "Error: Only TXT, MD, and JSON files are supported."

        if not file_path.is_file():
            return f"Error: File '{filename}' was not found in data/."

        if file_path.stat().st_size > 1_000_000:
            return "Error: File exceeds the 1 MB size limit."

        content = file_path.read_text(encoding="utf-8")

        return content[:12000] or "(The file is empty.)"

    except Exception as exc:
        return f"File reading error: {exc}"


TOOLS = [
    web_search,
    python_calculator,
    read_local_file,
]
