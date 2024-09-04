import json
import argparse
import subprocess
import tempfile


def lint_code(language, code):
    """Lint code based on its language and return (has_errors, error_message)."""
    if language == 'Python':
        return lint_python(code)
    elif language == 'JavaScript':
        return lint_javascript(code)
    elif language == 'TypeScript':
        return lint_typescript(code)
    elif language == 'C++':
        return lint_cpp(code)
    elif language == 'C#':
        return None, None  # Uncomment and implement if needed
    elif language == 'Ruby':
        return lint_ruby(code)
    elif language == 'Java':
        return None, None  # Uncomment and implement if needed
    elif language == 'PHP':
        return lint_php(code)
    else:
        print(f"Unsupported language: {language}")
        return None, None


def lint_python(code):
    """Lint Python code using pylint."""
    with tempfile.NamedTemporaryFile(suffix='.py', delete=False, mode='w') as temp_file:
        temp_file.write(code)
        temp_file_path = temp_file.name

    result = subprocess.run(['pylint', temp_file_path], capture_output=True, text=True)
    return result.returncode != 0, result.stdout + result.stderr


def lint_javascript(code):
    """Lint JavaScript code using eslint."""
    with tempfile.NamedTemporaryFile(suffix='.js', delete=False, mode='w') as temp_file:
        temp_file.write(code)
        temp_file_path = temp_file.name

    result = subprocess.run(['./node_modules/.bin/eslint', temp_file_path], capture_output=True, text=True)
    return result.returncode != 0, result.stdout + result.stderr


def lint_typescript(code):
    """Lint TypeScript code using eslint."""
    with tempfile.NamedTemporaryFile(suffix='.ts', delete=False, mode='w') as temp_file:
        temp_file.write(code)
        temp_file_path = temp_file.name

    result = subprocess.run(['./node_modules/.bin/eslint', temp_file_path], capture_output=True, text=True)
    return result.returncode != 0, result.stdout + result.stderr


def lint_cpp(code):
    """Lint C++ code using cpplint."""
    with tempfile.NamedTemporaryFile(suffix='.cpp', delete=False, mode='w') as temp_file:
        temp_file.write(code)
        temp_file_path = temp_file.name

    result = subprocess.run(['cpplint', temp_file_path], capture_output=True, text=True)
    return result.returncode != 0, result.stdout + result.stderr


def lint_ruby(code):
    """Lint Ruby code using rubocop."""
    with tempfile.NamedTemporaryFile(suffix='.rb', delete=False, mode='w') as temp_file:
        temp_file.write(code)
        temp_file_path = temp_file.name

    result = subprocess.run(['rubocop', temp_file_path], capture_output=True, text=True)
    return result.returncode != 0, result.stdout + result.stderr


def lint_php(code):
    """Lint PHP code using php -l."""
    with tempfile.NamedTemporaryFile(suffix='.php', delete=False, mode='w') as temp_file:
        temp_file.write(code)
        temp_file_path = temp_file.name

    result = subprocess.run(['php', '-l', temp_file_path], capture_output=True, text=True)
    return result.returncode != 0, result.stdout + result.stderr


def process_jsonl(file_path, output_path):
    """Process JSONL file, lint code and add lint_error and lint_message keys."""
    output_data = []

    with open(file_path, 'r', encoding='utf-8') as f:
        for index, line in enumerate(f, start=1):
            data = json.loads(line)
            language = data.get('language')
            code = data.get('text')

            if language and code:
                lint_error, lint_message = lint_code(language, code)
                data['lint_error'] = lint_error
                data['lint_message'] = lint_message
            else:
                data['lint_error'] = None
                data['lint_message'] = None

            output_data.append(data)

            if index % 100 == 0:
                write_results(output_data, output_path, mode='a')
                output_data = []

    return output_data


def write_results(data, output_path, mode='a'):
    with open(output_path, mode, encoding='utf-8') as file:
        for entry in data:
            json.dump(entry, file, ensure_ascii=False)
            file.write('\n')


def main():
    parser = argparse.ArgumentParser(description='Lint code in a JSONL file based on detected language.')
    parser.add_argument('--file-path', type=str, help='Path to the JSONL file')
    parser.add_argument('--output-path', type=str, help='Path to the output JSONL file')
    args = parser.parse_args()

    processed_data = process_jsonl(args.file_path, args.output_path)

    write_results(processed_data, args.output_path)

    print(f"Processing complete. Output saved to {args.output_path}.")

if __name__ == '__main__':
    main()
