# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "typer",
#     "datasets",
#     "loguru",
#     "huggingface_hub",
#     "google-genai",
#     "aiofiles",
#     "rich",
# ]
# ///
from __future__ import annotations

import asyncio
import json
import os
import random
import shutil
import tempfile
import urllib.error
import urllib.request
from pathlib import Path

import aiofiles
import pyarrow as pa
import typer
from datasets import Dataset, load_dataset
from gemini_function_call import gemini_function_call
from huggingface_hub import HfApi
from loguru import logger

app = typer.Typer()

# Configure logger to a rotating file.
logger.add('app.log', rotation='1 MB')


async def get_jsonl_files(directory: Path) -> list[Path]:
    """Return all JSONL files in the given directory."""
    return list(directory.glob('*.jsonl'))


async def validate_jsonl_file(file: Path) -> tuple[bool, list[dict], list[int]]:
    """Validate a JSONL file.

    Returns:
      - has_error (bool): True if any JSONDecodeError occurred.
      - valid_data (list): List of valid JSON objects.
      - error_lines (list): Line numbers where errors were detected.
    """
    valid_data = []
    error_lines = []
    try:
        async with aiofiles.open(file, encoding='utf-8') as f:
            i = 1
            async for line in f:
                try:
                    record = json.loads(line)
                    valid_data.append(record)
                except json.JSONDecodeError as e:
                    logger.error(f'Error parsing line {i} in {file.name}: {e}')
                    error_lines.append(i)
                i += 1
    except Exception as e:
        logger.error(f'Failed to read {file.name}: {e}')
        error_lines.append(-1)
    return (len(error_lines) > 0), valid_data, error_lines


async def attempt_easy_fix(file: Path, valid_data: list[dict]) -> bool:
    """Attempt an easy fix on a JSONL file by rewriting it with only valid JSON lines.
    Returns True if the file was fixed (and is non-empty), otherwise False.
    """
    if not valid_data:
        logger.error(f'No valid data found in {file.name} after attempting fix.')
        return False
    try:
        async with aiofiles.open(file, 'w', encoding='utf-8') as f:
            for record in valid_data:
                await f.write(json.dumps(record) + '\n')
        logger.info(f'Easy fix applied to {file.name}.')
        return True
    except Exception as e:
        logger.error(f'Failed to write fixed data to {file.name}: {e}')
        return False


async def fix_via_json(dataset: Dataset) -> Dataset:
    """Alternative method to fix empty structs using JSON serialization."""
    logger.info('Using JSON serialization method to fix empty structs')

    # Serialize to JSON and back
    json_data = dataset.to_json()

    # Add dummy fields to empty dicts in the JSON data
    def fix_json_dict(obj):
        if isinstance(obj, dict):
            # If this is an empty dict, add a dummy field
            if len(obj) == 0:
                return {'_dummy': None}
            # Otherwise recursively fix all values
            return {k: fix_json_dict(v) for k, v in obj.items()}
        if isinstance(obj, list):
            # Recursively fix all items in the list
            return [fix_json_dict(item) for item in obj]
        # Other types don't need fixing
        return obj

    fixed_json_data = [fix_json_dict(item) for item in json_data]

    # Create a new dataset from the fixed JSON data
    return Dataset.from_list(fixed_json_data)


async def fix_empty_structs(dataset: Dataset) -> Dataset:
    """Fix empty struct fields in a dataset that would cause PyArrow errors when writing to Parquet.

    This addresses the 'Cannot write struct type with no child field to Parquet' error
    by directly modifying the PyArrow schema to add dummy fields to empty structs.

    Args:
        dataset: The dataset to fix

    Returns:
        The fixed dataset
    """
    # Log the schema to help diagnose issues
    logger.info('Checking dataset schema for empty structs')

    # Get the PyArrow table from the dataset
    try:
        table = dataset.data.table
        schema = table.schema

        # Function to recursively fix empty structs in a field
        def fix_field_type(field_type, path=''):
            if pa.types.is_struct(field_type) and len(field_type) == 0:
                # This is an empty struct, add a dummy field
                logger.warning(f'Found empty struct at path: {path}')
                return pa.struct([pa.field('_dummy', pa.null())])
            if pa.types.is_list(field_type):
                # For list types, fix the value type
                value_type = fix_field_type(field_type.value_type, f'{path}[item]')
                return pa.list_(value_type)
            if pa.types.is_struct(field_type):
                # For non-empty structs, recursively fix each field
                new_fields = []
                for i in range(len(field_type)):
                    subfield = field_type[i]
                    new_path = f'{path}.{subfield.name}' if path else subfield.name
                    new_type = fix_field_type(subfield.type, new_path)
                    new_fields.append(pa.field(subfield.name, new_type))
                return pa.struct(new_fields)
            # Other types don't need fixing
            return field_type

        # Create a new schema with fixed field types
        new_fields = []
        for field in schema:
            new_type = fix_field_type(field.type, field.name)
            new_fields.append(pa.field(field.name, new_type))

        new_schema = pa.schema(new_fields, schema.metadata)

        # Create a new table with the fixed schema
        # This is a bit tricky - we need to cast the table to the new schema
        # First, let's check if there are any changes
        if new_schema != schema:
            logger.warning('Found empty structs in the schema, adding dummy fields')
            try:
                # Try to cast the table to the new schema
                # This might fail if the schemas are incompatible
                new_table = table.cast(new_schema)
                # Create a new dataset with the fixed table
                return Dataset(
                    pa.Table.from_arrays(
                        new_table.columns, names=new_table.column_names, metadata=new_table.schema.metadata
                    )
                )
            except Exception as e:
                logger.error(f'Error casting table to new schema: {e}')
                logger.warning('Falling back to alternative method')
                return await fix_via_json(dataset)

        # If no changes were needed, return the original dataset
        return dataset

    except (AttributeError, TypeError) as e:
        logger.warning(f'Could not access PyArrow table: {e}')
        logger.info('Falling back to JSON serialization method')
        # Fall back to JSON serialization method
        return await fix_via_json(dataset)


async def create_archive_directory(directory: Path, processed_files: list[Path], archive_folder: str) -> None:
    """Create an archive directory (named after the repository) and move the processed files into it."""
    archive_dir = directory / archive_folder
    try:
        archive_dir.mkdir(exist_ok=True)
        for file in processed_files:
            shutil.move(str(file), str(archive_dir / file.name))
        logger.info(f'Archived files to {archive_dir}.')
    except (OSError, shutil.Error) as e:
        logger.warning(f'Could not archive files: {e}')


async def estimate_line_size(file_path: Path) -> int:
    """Estimate the average size of a line in a JSONL file.

    Args:
        file_path: Path to the JSONL file

    Returns:
        Estimated average line size in characters
    """
    try:
        # Get file size for logging purposes
        file_size = file_path.stat().st_size
        logger.debug(f'File size of {file_path.name}: {file_size} bytes')

        lines = []
        async with aiofiles.open(file_path, encoding='utf-8') as f:
            # Read up to 10 lines to get a sample
            for _i in range(10):
                line = await f.readline()
                if not line:
                    break
                lines.append(line)

        if not lines:
            logger.warning(f'No lines found in {file_path.name}')
            return 0

        # Calculate average line size
        avg_line_size = sum(len(line) for line in lines) / len(lines)
        logger.debug(f'Average line size in {file_path.name}: {int(avg_line_size)} characters')
        return int(avg_line_size)
    except Exception as e:
        logger.error(f'Error estimating line size for {file_path}: {e}')
        return 500  # Default fallback value


async def sample_jsonl_lines(file_path: Path, num_lines: int = 5) -> list[dict]:
    """Sample random lines from a JSONL file.

    Args:
        file_path: Path to the JSONL file
        num_lines: Number of lines to sample

    Returns:
        List of sampled JSON objects
    """
    try:
        lines = []
        async with aiofiles.open(file_path, encoding='utf-8') as f:
            # Read all lines but limit to 1000 for very large files
            i = 0
            async for line in f:
                if i >= 1000:
                    break
                lines.append(line)

        if not lines:
            return []

        # Sample random lines
        sampled_lines = random.sample(lines, min(num_lines, len(lines)))

        # Parse JSON
        return [json.loads(line) for line in sampled_lines]
    except Exception as e:
        logger.error(f'Error sampling lines from {file_path}: {e}')
        return []


async def generate_dataset_card(
    file_path: Path, repo_name: str, user_name: str, author: str, email_author: str
) -> str:
    """Generate a dataset card using Gemini API.

    Args:
        file_path: Path to the JSONL file
        repo_name: Repository name
        user_name: HuggingFace username
        author: Author name for citation
        email_author: Author email for citation

    Returns:
        Generated dataset card content
    """
    logger.info(f'Generating dataset card for {repo_name}')

    # Estimate line size to determine how many samples to use
    line_size = await estimate_line_size(file_path)
    num_samples = 5 if line_size > 10000 else 7

    # Sample lines from the JSONL file
    samples = await sample_jsonl_lines(file_path, num_samples)

    if not samples:
        logger.warning(f'Could not sample lines from {file_path}')
        samples = [{'warning': 'Could not sample data from file'}]

    # Define function for Gemini
    function_def = {
        'name': 'generate_dataset_card',
        'description': 'Generate a dataset card based on dataset information',
        'parameters': {
            'type': 'OBJECT',
            'properties': {
                'pretty_name': {'type': 'STRING', 'description': 'Pretty name for the dataset'},
                'is_instruction_confusing': {
                    'type': 'STRING',
                    'description': 'Whether the instruction is confusing and why',
                },
                'dataset_summary': {'type': 'STRING', 'description': 'A summary of the dataset'},
                'dataset_structure': {'type': 'STRING', 'description': 'Description of the dataset structure'},
                'languages': {
                    'type': 'ARRAY',
                    'items': {'type': 'STRING'},
                    'description': 'Languages used in the dataset',
                },
                'tags': {'type': 'ARRAY', 'items': {'type': 'STRING'}, 'description': 'Tags for the dataset'},
            },
            'required': [
                'pretty_name',
                'dataset_summary',
                'dataset_structure',
                'languages',
                'tags',
                'is_instruction_confusing',
            ],
        },
    }

    # Create prompt for Gemini
    prompt = f"""
    You are a data scientist creating a dataset card for a dataset being uploaded to Hugging Face. You do that by calling the function generate_dataset_card with the correct parameters. Therefore, you must use the tool with the correct parameters.

    Dataset Information:
    - Repository Name: {repo_name}
    - Username: {user_name}
    - Author: {author}
    - Email: {email_author}

    Here are {len(samples)} sample records from the dataset:
    ```
    {json.dumps(samples, indent=2)}
    ```

    Based on these samples, generate a comprehensive dataset card with the following information:
    1. A pretty name for the dataset (based on the samples and repository name, it's called "pretty" for a reason)
    2. A summary of what the dataset contains (in English and the other languages present in the samples)
    3. A detailed description of the dataset structure, including list of fields and their types and a few illustrative examples (formatted for easy reading and unformatted for easy parsing) (in English and the other languages present in the samples)
    5. Languages used in the dataset (infer from the samples)
    6. Appropriate tags for the dataset (infer from the samples)
    7. For the citation link, use the repository link (https://huggingface.co/{user_name}/{repo_name})
    8. All of your text should include line breaks because it will be used as a variable (so add \\n where appropriate)
    9. Please answer if these instructions are confusing and why, including whether it was clear that you must call the function (use the tool) with the correct parameters. Indicate if you prefer to use "call function" or "use tool" or something else (must provide one of these three options). If confusing, indicate which items I should modify.

    YOU MUST USE THE TOOL WITH THE CORRECT PARAMETERS!!! DO IT NO MATTER WHAT!
    Use the generate_dataset_card tool (make the function call) with the correct parameters **NOW**
    """

    try:
        # Use the correct function name: gemini_function_call instead of call_gemini_function
        result = await gemini_function_call(prompt=prompt, functions=[function_def], model='gemini-2.0-flash')

        if not result:
            logger.error('Failed to generate dataset card with Gemini')
            return ''  # Return empty string on failure

        # Format the dataset card using the template
        return await format_dataset_card(result, repo_name, user_name, author, email_author)

    except Exception as e:
        logger.error(f'Error generating dataset card: {e}')
        return ''


async def format_dataset_card(card_data: dict, repo_name: str, user_name: str, author: str, email_author: str) -> str:
    """Format the dataset card using the template.

    Args:
        card_data: Dictionary containing dataset card information
        repo_name: Repository name
        user_name: HuggingFace username
        author: Author name for citation
        email_author: Author email for citation

    Returns:
        Formatted dataset card content
    """
    # Extract data from the result
    pretty_name = card_data.get('pretty_name', 'Dataset Name')
    dataset_summary_coded = card_data.get('dataset_summary', '')
    dataset_structure_coded = card_data.get('dataset_structure', '[More Information Needed]')
    is_instruction_confusing = card_data.get('is_instruction_confusing', '')
    logger.info(f'Is instruction confusing: {is_instruction_confusing}')

    # Safely handle Unicode escape sequences and encoding
    def safe_decode(text: str) -> str:
        r"""Safely decode text with proper handling of both escape sequences and character encoding.

        This handles:
        1. Unicode escape sequences (\\uXXXX)
        2. Newline escape sequences (\\n)
        3. Proper encoding of international characters
        """
        try:
            # Check if text actually contains escape sequences
            if '\\u' in text:
                # Unicode escape sequences present, decode them
                try:
                    return text.encode('utf-8').decode('unicode_escape')
                except UnicodeDecodeError:
                    # If decoding fails, just replace newlines
                    return text.replace('\\n', '\n')
            elif '\\n' in text:
                # Only newline escapes present
                return text.replace('\\n', '\n')
            else:
                # No escape sequences, return the text as is
                # to preserve original encoding of accented characters
                return text
        except Exception as e:
            logger.warning(f'Error in safe_decode: {e}')
            # Last resort fallback
            return text

    # Apply safe decoding to text fields
    dataset_summary = safe_decode(dataset_summary_coded)
    dataset_structure = safe_decode(dataset_structure_coded)

    # Get languages and tags
    languages = card_data.get('languages', ['en'])
    tags = card_data.get('tags', ['dataset'])

    # Format languages and tags
    languages_str = '\n'.join([f'- {lang}' for lang in languages])
    tags_str = '\n'.join([f'- {tag}' for tag in tags])

    # Create the dataset card
    return f"""---
language:
{languages_str}
tags:
{tags_str}
---

# Dataset Card for {pretty_name}

<!-- Provide a quick summary of the dataset. -->

{dataset_summary}

## Dataset Structure

<!-- This section provides a description of the dataset fields, and additional information about the dataset structure such as criteria used to create the splits, relationships between data points, etc. -->

{dataset_structure}

## Additional Information

### Licensing Information

The dataset is released under Apache 2.0.

### Citation Information

```bibtex
@misc{{{repo_name}}},
      title={{{pretty_name}}},
      author={{{author}}},
      year={{2025}},
      primaryClass={{cs.LG}}
}}
```

"""


async def upload_dataset_card(content: str, repo_id: str, token: str, create_pr: bool = False) -> bool:
    """Upload a dataset card to a Hugging Face repository.

    Args:
        content: Dataset card content
        repo_id: Repository ID (username/repo_name)
        token: HuggingFace token
        create_pr: Whether to create a PR instead of committing directly

    Returns:
        True if upload was successful, False otherwise
    """
    logger.info(f'Uploading dataset card to {repo_id}')

    try:
        # Create a temporary file for the README.md
        with tempfile.NamedTemporaryFile(mode='w', suffix='.md', delete=False, encoding='utf-8') as temp_file:
            temp_file.write(content)
            temp_path = temp_file.name

        # Initialize the Hugging Face API
        api = HfApi(token=token)

        # Upload the file
        api.upload_file(
            path_or_fileobj=temp_path,
            path_in_repo='README.md',
            repo_id=repo_id,
            repo_type='dataset',
            create_pr=create_pr,
        )

        # Clean up the temporary file using Path.unlink() for consistency
        Path(temp_path).unlink()
        logger.info(f'Temporary file {temp_path} cleaned up')

        logger.info(f'Dataset card uploaded successfully to {repo_id}')
        return True

    except Exception as e:
        logger.error(f'Error uploading dataset card: {e}')
        return False


async def repo_exists(repo_id: str, token: str) -> bool:
    """Check if a dataset repository exists on Hugging Face Hub.
    Uses the API endpoint for datasets and includes the token in the Authorization header.
    """
    url = f'https://huggingface.co/api/datasets/{repo_id}'
    req = urllib.request.Request(url)
    req.add_header('Authorization', f'Bearer {token}')
    try:
        with urllib.request.urlopen(req) as response:
            if response.getcode() == 200:
                return True
    except urllib.error.HTTPError as e:
        if e.code == 404:
            return False
        logger.error(f'HTTPError while checking repo: {e}')
        return True
    except Exception as e:
        logger.error(f'Error checking repo existence: {e}')
        return True
    return False


# Define module-level variables to avoid B008 linting errors
DEFAULT_DIRECTORY = Path.cwd()
DIRECTORY_ARGUMENT = typer.Argument(None, help='Directory containing JSONL files')


async def fix_key_encoding(dataset: Dataset, keys_to_fix: list[str]) -> Dataset:
    """Fix encoding issues for specific keys in the dataset.

    Args:
        dataset: The dataset to fix
        keys_to_fix: List of keys whose values need encoding fixes

    Returns:
        Dataset with fixed encoding for specified keys
    """
    logger.info(f'Fixing encoding for keys: {keys_to_fix}')

    # Function to safely decode text that might have encoding issues
    def safe_text_decode(text):
        if not isinstance(text, str):
            return text

        try:
            # Check if text actually contains escape sequences
            if '\\u' in text:
                # Unicode escape sequences present, decode them
                try:
                    return text.encode('utf-8').decode('unicode_escape')
                except UnicodeDecodeError:
                    # If decoding fails, just replace newlines
                    return text.replace('\\n', '\n')
            elif '\\n' in text:
                # Only newline escapes present
                return text.replace('\\n', '\n')
            else:
                # For text with potential encoding issues but no escape sequences
                # Try encoding as utf-8 then decoding to fix potential issues
                try:
                    # This helps with Latin-1 encoded text appearing as wrong characters
                    test_decode = text.encode('latin-1').decode('utf-8')
                    # If this succeeded and changed something, return it
                    if test_decode != text:
                        return test_decode
                except (UnicodeEncodeError, UnicodeDecodeError):
                    pass

                # Return original if no fix was needed/possible
                return text
        except Exception as e:
            logger.warning(f'Error while fixing encoding: {e}')
            return text

    # Apply fix to each record in the dataset
    def fix_record(record):
        for key in keys_to_fix:
            if key in record:
                if isinstance(record[key], str):
                    record[key] = safe_text_decode(record[key])
                elif isinstance(record[key], list):
                    # Handle list of strings
                    record[key] = [safe_text_decode(item) if isinstance(item, str) else item for item in record[key]]
                elif isinstance(record[key], dict):
                    # Handle nested dictionaries (recursive fix for all string values)
                    def fix_dict(d):
                        for k, v in d.items():
                            if isinstance(v, str):
                                d[k] = safe_text_decode(v)
                            elif isinstance(v, dict):
                                fix_dict(v)
                            elif isinstance(v, list) and all(isinstance(i, str) for i in v):
                                d[k] = [safe_text_decode(i) for i in v]
                        return d

                    record[key] = fix_dict(record[key])
        return record

    # Apply the fix to the dataset
    return dataset.map(fix_record)


@app.command()
def process_jsonl(
    directory: Path = DIRECTORY_ARGUMENT,
    repo_name: str | None = typer.Option(
        None, '--repo_name', '-r', help="Custom repository name. If not provided, uses the first file's name."
    ),
    user_name: str = typer.Option('cicero-im', '--user_name', '-u', help='HuggingFace username'),
    should_compile: bool = typer.Option(
        True,
        '--compile',
        help='Compile all JSONL files into a single dataset and push once (default True). If False, each file is pushed as its own dataset.',
    ),
    flatten: bool = typer.Option(
        False, '--flatten', help='Flatten nested JSON objects using datasets.flatten(max_depth=16)'
    ),
    private: bool | None = typer.Option(
        None,
        '--private',
        help="Whether to make the repo private. If None (default), the repo will be public unless the organization's default is private.",
    ),
    create_pr: bool = typer.Option(
        False, '--create-pr', help='Create a PR with the uploaded files instead of directly committing'
    ),
    hf_token: str | None = typer.Option(
        None, '--hf-token', help='HuggingFace token (defaults to the HF_TOKEN env variable)'
    ),
    single_file: str | None = typer.Option(
        None, '--single-file', help='Process only a specific JSONL file instead of all files in the directory'
    ),
    author: str = typer.Option('Arthur Souza Rodrigues', '--author', help='Author name for dataset card citation'),
    email_author: str = typer.Option(
        'arthrod@umich.edu', '--email-author', help='Author email for dataset card citation'
    ),
    fix_coding_first: list[str] = typer.Option(
        None,
        '--fix-coding-first',
        help='List of field names whose values should have encoding fixed before any other operation',
    ),
) -> None:
    """Process JSONL files and upload to HuggingFace Hub.

    The script validates each JSONL file and, if errors are found, prompts whether to proceed, interrupt, or attempt an easy fix.

    - When **--single-file** is provided:
      Only the specified file is processed, instead of all JSONL files in the directory.

    - When **--compile** is True (default):
      All JSONL files (or just the single file if specified) are normalized and compiled into a single dataset. The repository (and archive folder) is named based on the first file's name (if no custom repo name is provided). **push_to_hub** is then called once.
      Note: Internally this uses the `should_compile` parameter.

    - When **--compile** is False:
      Each valid JSONL file is processed separately and pushed as its own dataset. The repository for each file is named using the file's stem (or the provided repo_name plus an underscore and the file's stem).

    - When **--fix-coding-first** is provided with a list of field names:
      The script will attempt to fix encoding issues for those specific fields' values before any other processing.
      Example: --fix-coding-first content --fix-coding-first description

    In both cases, if the repository already exists on Hugging Face, you are prompted to choose one of:
      - **[a]ppend (default):** Append a numeric suffix to the repo name.
      - **[o]verwrite:** Overwrite the existing repository.
      - **[p]r:** Create a pull request with the uploaded files.

    The HuggingFace token is passed to both load_dataset and push_to_hub.
    """
    asyncio.run(
        async_process_jsonl(
            directory,
            repo_name,
            user_name,
            should_compile,
            flatten,
            private,
            create_pr,
            hf_token,
            single_file,
            author,
            email_author,
            fix_coding_first,
        )
    )


async def async_process_jsonl(
    directory: Path,
    repo_name: str | None,
    user_name: str,
    should_compile: bool,
    flatten: bool,
    private: bool | None,
    create_pr: bool,
    hf_token: str | None,
    single_file: str | None,
    author: str,
    email_author: str,
    fix_coding_first: list[str] | None = None,
) -> None:
    """Async implementation of process_jsonl."""
    # Use the DEFAULT_DIRECTORY if directory is None
    if directory is None:
        directory = DEFAULT_DIRECTORY
        logger.info(f'Using default directory: {directory}')
    else:
        logger.info(f'Using provided directory: {directory}')

    token = hf_token if hf_token is not None else os.environ.get('HF_TOKEN')
    if token is None:
        typer.echo(
            typer.style(
                'Error: HF_TOKEN is not set in the environment and no --hf-token provided.', fg='red', bold=True
            )
        )
        raise typer.Exit(1)

    if single_file:
        single_file_path = directory / single_file if not Path(single_file).is_absolute() else Path(single_file)
        if not single_file_path.exists():
            typer.echo(typer.style(f'File {single_file_path} does not exist.', fg='red', bold=True))
            raise typer.Exit(1)
        if single_file_path.suffix.lower() != '.jsonl':
            typer.echo(typer.style(f'File {single_file_path} is not a JSONL file.', fg='red', bold=True))
            raise typer.Exit(1)
        files = [single_file_path]
        logger.info(f'Processing single file: {single_file_path}')
    else:
        files = await get_jsonl_files(directory)
        if not files:
            typer.echo(typer.style('No JSONL files found in the directory.', fg='red', bold=True))
            raise typer.Exit(1)

    valid_files = []
    for file in files:
        has_error, valid_data, error_lines = await validate_jsonl_file(file)
        if has_error:
            msg = (
                f"File '{file.name}' has syntax errors on lines: {error_lines}.\n"
                "Choose an option: [p]roceed without this file, [i]nterrupt, [e]asy fix (default 'e')"
            )
            choice = typer.prompt(msg, default='e')
            if choice.lower() == 'i':
                logger.error(f'User chose to interrupt due to errors in {file.name}.')
                raise typer.Exit(1)
            if choice.lower() == 'p':
                logger.info(f'User chose to proceed without file {file.name}.')
                continue
            if await attempt_easy_fix(file, valid_data):
                logger.info(f'File {file.name} fixed and validated.')
            else:
                logger.error(f'Easy fix failed for {file.name}. Skipping file.')
                continue
        valid_files.append(file)

    if not valid_files:
        typer.echo(typer.style('No valid JSONL files to process after validation.', fg='red', bold=True))
        raise typer.Exit(1)

    # Log the compilation mode
    logger.info(f'Compilation mode: {"enabled" if should_compile else "disabled"}')

    if should_compile:
        # For compiling, use one repo.
        first_file = valid_files[0]
        if repo_name is None:
            repo_name = first_file.stem
        archive_folder = repo_name
        repo_id = f'{user_name}/{repo_name}'

        if await repo_exists(repo_id, token):
            choice = typer.prompt(
                typer.style(
                    f"Repository '{repo_id}' already exists. Choose an action: [a]ppend (default), [o]verwrite, [p]ull request",
                    fg='yellow',
                    bold=True,
                ),
                default='a',
            )
            # Check if the choice is valid
            if choice.lower() not in {'a', 'o', 'p'}:
                typer.echo(typer.style('Invalid option. Please try again.', fg='red', bold=True))
                choice = typer.prompt(
                    typer.style(
                        f"Repository '{repo_id}' already exists. Choose an action: [a]ppend (default), [o]verwrite, [p]r",
                        fg='yellow',
                        bold=True,
                    ),
                    default='a',
                )
                if choice.lower() not in {'a', 'o', 'p'}:
                    typer.echo(typer.style('Invalid option. Aborting.', fg='red', bold=True))
                    raise typer.Exit(1)

            if choice.lower() == 'a':
                base = repo_id
                counter = 1
                new_repo_id = f'{base}_{counter}'
                while await repo_exists(new_repo_id, token):
                    counter += 1
                    new_repo_id = f'{base}_{counter}'
                repo_id = new_repo_id
                typer.echo(typer.style(f'New repository id set to: {repo_id}', fg='blue', bold=True))
            elif choice.lower() == 'o':
                typer.echo(typer.style(f"Overwriting repository '{repo_id}'.", fg='red', bold=True))
            elif choice.lower() == 'p':
                create_pr = True
                typer.echo(typer.style(f"Will create a PR for repository '{repo_id}'.", fg='blue', bold=True))
            else:
                typer.echo(typer.style('Invalid option. Aborting.', fg='red', bold=True))
                raise typer.Exit(1)

        combined_dataset = load_dataset('json', data_files=[str(file) for file in valid_files], token=token)['train']

        # Apply encoding fix if requested
        if fix_coding_first:
            typer.echo(typer.style(f'Fixing encoding for fields: {fix_coding_first}', fg='blue', bold=True))
            combined_dataset = await fix_key_encoding(combined_dataset, fix_coding_first)

        if flatten:
            combined_dataset = combined_dataset.flatten(max_depth=16)
            # Compute the new flattened columns compared to the union of all columns.
            flattened_columns = set(combined_dataset.column_names) - set().union(*[
                set(load_dataset('json', data_files=str(file), token=token)['train'].column_names)
                for file in valid_files
            ])
            typer.echo(typer.style(f'Flattened columns added: {sorted(flattened_columns)}', fg='blue', bold=True))

        # Fix any empty struct fields that would cause PyArrow errors
        try:
            combined_dataset = await fix_empty_structs(combined_dataset)
        except Exception as e:
            logger.warning(f'Error while fixing empty structs: {e}')

        # Generate and upload dataset card with simplified timer
        typer.echo(typer.style('Uploading compiled dataset...', fg='magenta', bold=True))
        combined_dataset.push_to_hub(repo_id, token=token, private=private, create_pr=create_pr)

        # Get user preference with the new timeout function
        upload_card = await get_user_input_with_timeout(
            'Would you like to upload a dataset card (README.md)? [Y/n] (5s timeout)', 'Y'
        )

        if upload_card.lower() in {'y', 'yes', ''}:
            # Generate dataset card
            card_content = await generate_dataset_card(first_file, repo_name, user_name, author, email_author)

            # Upload dataset card
            if await upload_dataset_card(card_content, repo_id, token, create_pr):
                typer.echo(typer.style('Dataset card uploaded successfully.', fg='green', bold=True))
            else:
                typer.echo(typer.style('Failed to upload dataset card.', fg='red', bold=True))
        else:
            typer.echo(typer.style('Dataset card upload skipped.', fg='yellow', bold=True))
    else:
        # Process each file separately.
        for file in valid_files:
            file_stem = file.stem
            local_repo_name = f'{repo_name}_{file_stem}' if repo_name is not None else file_stem
            local_repo_id = f'{user_name}/{local_repo_name}'

            if await repo_exists(local_repo_id, token):
                choice = typer.prompt(
                    typer.style(
                        f"Repository '{local_repo_id}' already exists. Choose an action: [a]ppend (default), [o]verwrite, [p]r",
                        fg='yellow',
                        bold=True,
                    ),
                    default='a',
                )
                if choice.lower() == 'a':
                    base = local_repo_id
                    counter = 1
                    new_repo_id = f'{base}_{counter}'
                    while await repo_exists(new_repo_id, token):
                        counter += 1
                        new_repo_id = f'{base}_{counter}'
                    local_repo_id = new_repo_id
                    typer.echo(
                        typer.style(f'New repository id for {file.name} set to: {local_repo_id}', fg='blue', bold=True)
                    )
                elif choice.lower() == 'o':
                    typer.echo(
                        typer.style(f"Overwriting repository '{local_repo_id}' for {file.name}.", fg='red', bold=True)
                    )
                elif choice.lower() == 'p':
                    create_pr = True
                    typer.echo(
                        typer.style(
                            f"Will create a PR for repository '{local_repo_id}' for {file.name}.", fg='blue', bold=True
                        )
                    )
                else:
                    typer.echo(typer.style('Invalid option. Aborting.', fg='red', bold=True))
                    raise typer.Exit(1)

            ds = load_dataset('json', data_files=str(file), token=token)['train']

            # Apply encoding fix if requested
            if fix_coding_first:
                typer.echo(
                    typer.style(f'Fixing encoding for fields: {fix_coding_first} in {file.name}', fg='blue', bold=True)
                )
                ds = await fix_key_encoding(ds, fix_coding_first)

            original_columns = set(ds.column_names)
            if flatten:
                ds = ds.flatten(max_depth=16)
                flattened_columns = set(ds.column_names) - original_columns
                typer.echo(
                    typer.style(
                        f'For file {file.name}, flattened columns added: {sorted(flattened_columns)}',
                        fg='blue',
                        bold=True,
                    )
                )
            # Fix any empty struct fields that would cause PyArrow errors
            try:
                ds = await fix_empty_structs(ds)
            except Exception as e:
                logger.warning(f'Error while fixing empty structs in {file.name}: {e}')

            typer.echo(
                typer.style(
                    f'Processing and uploading {file.name} with columns: {sorted(ds.column_names)}...',
                    fg='magenta',
                    bold=True,
                )
            )
            ds.push_to_hub(local_repo_id, token=token, private=private, create_pr=create_pr)

            # Generate and upload dataset card
            upload_card = typer.prompt(
                typer.style(
                    f'Would you like to upload a dataset card (README.md) for {file.name}? [Y/n]', fg='cyan', bold=True
                ),
                default='Y',
            )

            if upload_card.lower() in {'y', 'yes', ''}:
                # Generate dataset card
                card_content = await generate_dataset_card(file, local_repo_name, user_name, author, email_author)

                # Upload dataset card
                if await upload_dataset_card(card_content, local_repo_id, token, create_pr):
                    typer.echo(
                        typer.style(f'Dataset card uploaded successfully for {file.name}.', fg='green', bold=True)
                    )
                else:
                    typer.echo(typer.style(f'Failed to upload dataset card for {file.name}.', fg='red', bold=True))
            else:
                typer.echo(typer.style(f'Dataset card upload skipped for {file.name}.', fg='yellow', bold=True))
        archive_folder = repo_name if repo_name is not None else 'individual'

    await create_archive_directory(directory, valid_files, archive_folder)
    typer.echo(typer.style('Upload complete and files archived.', fg='cyan', bold=True))


# Replace timer implementation in async_process_jsonl
async def get_user_input_with_timeout(prompt_text: str, default_value: str, timeout_seconds: int = 5) -> str:
    """Display a prompt with a countdown timer and return user input or default after timeout.

    Args:
        prompt_text: Text to display in the prompt
        default_value: Default value to return if no input is received
        timeout_seconds: Number of seconds to wait for input

    Returns:
        User input or default value after timeout
    """
    try:
        # Print the main prompt
        typer.echo(typer.style(prompt_text, fg='cyan', bold=True))

        # Simple countdown without using rich or panels
        remaining = timeout_seconds

        # Set up a task to handle the timeout
        async def timeout_handler() -> None:
            nonlocal remaining
            while remaining > 0:
                # Print the countdown
                await asyncio.sleep(1)
                remaining -= 1
            # Clear the line when done

        # Start the timeout handler
        timeout_task = asyncio.create_task(timeout_handler())

        # Get user input with asyncio
        input_queue = asyncio.Queue()

        async def get_input() -> None:
            # Run input in a separate thread to not block the event loop
            loop = asyncio.get_running_loop()
            user_input = await loop.run_in_executor(None, input)
            await input_queue.put(user_input)

        # Start the input task
        input_task = asyncio.create_task(get_input())

        # Wait for either timeout or input
        done, pending = await asyncio.wait([timeout_task, input_task], return_when=asyncio.FIRST_COMPLETED)

        # Cancel the pending task
        for task in pending:
            task.cancel()

        # Get the result if input task completed
        if input_task in done:
            user_input = await input_queue.get()
            return user_input if user_input.strip() else default_value

        # Timeout occurred
        typer.echo(typer.style(f'No input received, using default: {default_value}', fg='yellow'))
        return default_value

    except Exception as e:
        typer.echo(typer.style(f'Error during prompt: {e!r}, using default: {default_value}', fg='yellow'))
        return default_value


if __name__ == '__main__':
    app()
