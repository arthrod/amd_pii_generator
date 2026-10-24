#!/usr/bin/env python3
# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "typer",
#     "rich",
# ]
# ///
"""JSONL Processor - A tool to modify JSONL files by adding or renaming keys, and merging multiple files.

This script uses Typer and Rich to provide a beautiful CLI experience for processing
JSONL files. It can add new key-value pairs to each record, rename existing keys,
and merge multiple JSONL files based on common ID fields.
"""

from __future__ import annotations

import json
import sys
import traceback
from pathlib import Path

import typer
from rich.console import Console
from rich.panel import Panel
from rich.progress import BarColumn, Progress, TaskProgressColumn, TextColumn
from rich.syntax import Syntax
from rich.table import Table

# Initialize Typer app with help information
app = typer.Typer(
    help='Process JSONL files by adding new key-value pairs, renaming existing keys, or merging multiple files',
    add_completion=False,
)

# Initialize Rich console for beautiful output
console = Console()

# Define options outside of function parameter list
INPUT_FILE_OPTION = typer.Option(
    None, '--input-file', '-i', help='Path to the input JSONL file (not used with --merge)', exists=True, readable=True
)

OUTPUT_FILE_OPTION = typer.Option(
    None,
    '--output-file',
    '-o',
    help="Path to the output JSONL file (defaults to adding '_modified' suffix to input file)",
)

KEY_CONTENT_OPTION = typer.Option(
    None,
    '--key-content-to-add',
    '-a',
    help='JSON string with keys and values to add (e.g., \'{"new_key": "value", "another_key": 123}\')',
)

KEYS_TO_MODIFY_OPTION = typer.Option(
    None,
    '--keys-to-modify',
    '-m',
    help='JSON string with keys to rename (e.g., \'{"old_key": "new_key", "another_old_key": "another_new_key"}\')',
)

DELETE_LINE_IF_VALUE_OPTION = typer.Option(
    None,
    '--delete-line-if-value',
    '-d',
    help='JSON string with key-value pairs that will cause a line to be deleted if matched (e.g., \'{"generated_sample_PII": "input_text"}\')',
)

CONVERT_TO_CHATML_OPTION = typer.Option(
    None,
    '--convert-to-chatml',
    '-c',
    help='JSON string defining ChatML conversion: {"new_key": ["system_field", "user_field", "assistant_field"]} or {"new_key": ["user_field", "assistant_field"]}',
)

MERGE_OPTION = typer.Option(
    None,
    '--merge',
    '-M',
    help='Merge 2 or more JSONL files. Provide file paths separated by commas, or "all" to merge all JSONL files in current directory',
)

PREVIEW_OPTION = typer.Option(False, '--preview', '-p', help='Preview changes without writing to output file')


def parse_json_arg(value: str) -> dict:
    """Parse a JSON string argument from the command line.

    Args:
        value: A JSON string (e.g., '{"key": "value"}')

    Returns:
        Dict containing the parsed JSON

    Raises:
        typer.BadParameter: If the JSON string is invalid
    """
    if not value:
        return {}

    try:
        return json.loads(value)
    except json.JSONDecodeError as err:
        msg = f'Invalid JSON format: {err!s}'
        raise typer.BadParameter(msg) from err


def validate_jsonl_file(file_path: Path) -> bool:
    """Validate that a file is a proper JSONL file.

    Args:
        file_path: Path to the JSONL file

    Returns:
        True if file is valid JSONL, False otherwise
    """
    try:
        with file_path.open('r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line_stripped = line.strip()
                if not line_stripped:
                    continue
                try:
                    json.loads(line_stripped)
                except json.JSONDecodeError:
                    console.print(f'[bold red]Error:[/] Invalid JSON at line {line_num} in {file_path}')
                    return False
        return True
    except (OSError, PermissionError, FileNotFoundError) as err:
        console.print(f'[bold red]Error:[/] Failed to read file {file_path}: {err!s}')
        return False


def get_id_field_name(record: dict) -> str | None:
    """Get the ID field name from a record, checking in order: _id, uuid, id.

    Args:
        record: The record to check

    Returns:
        The field name if found, None otherwise
    """
    for field in ['_id', 'uuid', 'id']:
        if field in record:
            return field
    return None


def collect_ids_from_file(file_path: Path) -> tuple[set[str], str]:
    """Collect all ID values from a JSONL file in a memory-efficient way.

    Args:
        file_path: Path to the JSONL file

    Returns:
        Tuple of (set of ID values, ID field name used)

    Raises:
        ValueError: If no valid ID field is found
    """
    ids = set()
    id_field_name = None

    with file_path.open('r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line_stripped = line.strip()
            if not line_stripped:
                continue

            try:
                record = json.loads(line_stripped)

                # Determine ID field name from first record
                if id_field_name is None:
                    id_field_name = get_id_field_name(record)
                    if id_field_name is None:
                        msg = f'No valid ID field (_id, uuid, or id) found in {file_path}'
                        raise ValueError(msg)

                # Collect the ID value
                id_value = record.get(id_field_name)
                if id_value is not None:
                    ids.add(str(id_value))  # Convert to string to handle different types

            except json.JSONDecodeError:
                console.print(f'[yellow]Warning:[/] Skipping invalid JSON at line {line_num} in {file_path}')
                continue

    if not id_field_name:
        msg = f'No valid records found in {file_path}'
        raise ValueError(msg)

    return ids, id_field_name


def merge_jsonl_files(file_paths: list[Path], output_path: Path) -> None:
    """Merge multiple JSONL files by combining all unique records (no duplicates based on ID).

    Args:
        file_paths: List of paths to JSONL files to merge
        output_path: Path for the merged output file
    """
    # Collect IDs from each file first
    console.print(f'\n[bold]Analyzing {len(file_paths)} files for merging...[/]')

    all_ids_by_file = {}
    id_fields_by_file = {}
    seen_ids = set()  # Track all IDs we've already written

    # First pass: collect IDs from each file
    with Progress(
        TextColumn('[bold blue]{task.description}'),
        BarColumn(),
        '[progress.percentage]{task.percentage:>3.0f}%',
        TaskProgressColumn(),
        console=console,
    ) as progress:
        task_id = progress.add_task('[cyan]Collecting IDs from files...', total=len(file_paths))

        for file_path in file_paths:
            try:
                ids, id_field = collect_ids_from_file(file_path)
                all_ids_by_file[file_path] = ids
                id_fields_by_file[file_path] = id_field
                console.print(f'[green]✓[/] Found {len(ids)} records with field "{id_field}" in {file_path}')
            except ValueError as e:
                console.print(f'[red]✗[/] {e}')
                raise typer.Exit(code=1)
            progress.update(task_id, advance=1)

    # Calculate statistics
    total_unique_ids = len(set.union(*all_ids_by_file.values()))
    console.print(f'\n[green]Total unique records across all files: {total_unique_ids}[/]\n')

    # Second pass: write records, skipping duplicates
    total_written = 0
    total_skipped = 0

    with output_path.open('w', encoding='utf-8') as outfile:
        for file_idx, file_path in enumerate(file_paths):
            id_field = id_fields_by_file[file_path]
            written_from_file = 0
            skipped_from_file = 0

            with Progress(
                TextColumn('[bold blue]{task.description}'),
                BarColumn(),
                '[progress.percentage]{task.percentage:>3.0f}%',
                TaskProgressColumn(),
                console=console,
            ) as progress:
                # Count lines for progress
                with file_path.open('r', encoding='utf-8') as f:
                    total_lines = sum(1 for _ in f)

                task_id = progress.add_task(
                    f'[cyan]Processing {file_path.name} (file {file_idx + 1}/{len(file_paths)})...', total=total_lines
                )

                with file_path.open('r', encoding='utf-8') as infile:
                    for line in infile:
                        line_stripped = line.strip()
                        if not line_stripped:
                            progress.update(task_id, advance=1)
                            continue

                        try:
                            record = json.loads(line_stripped)
                            id_value = record.get(id_field)

                            if id_value is not None:
                                id_str = str(id_value)

                                # Check if we've already seen this ID
                                if id_str not in seen_ids:
                                    # New unique record - write it
                                    outfile.write(json.dumps(record, ensure_ascii=False) + '\n')
                                    seen_ids.add(id_str)
                                    written_from_file += 1
                                    total_written += 1
                                else:
                                    # Duplicate - skip it
                                    skipped_from_file += 1
                                    total_skipped += 1

                        except json.JSONDecodeError:
                            pass

                        progress.update(task_id, advance=1)

            console.print(
                f'[green]✓[/] File {file_path.name}: wrote {written_from_file} new records, skipped {skipped_from_file} duplicates'
            )

    console.print(
        Panel(
            f'[bold green]✓[/] Successfully merged {len(file_paths)} files\n'
            f'[bold green]✓[/] Total unique records written: {total_written}\n'
            f'[bold green]✓[/] Total duplicate records skipped: {total_skipped}\n'
            f'[bold green]✓[/] Output saved to [cyan]{output_path}[/]',
            title='Merge Complete',
            border_style='green',
        )
    )


def add_keys(record: dict, keys_to_add: dict) -> dict:
    """Add new key-value pairs to a record.

    Args:
        record: The original record
        keys_to_add: Dict of key-value pairs to add

    Returns:
        Updated record with new key-value pairs
    """
    result = record.copy()
    for key, value in keys_to_add.items():
        result[key] = value
    return result


def modify_keys(record: dict, keys_to_modify: dict) -> dict:
    """Rename keys in a record.

    Args:
        record: The original record
        keys_to_modify: Dict mapping old key names to new key names

    Returns:
        Updated record with renamed keys
    """
    result = {}
    for key, value in record.items():
        if key in keys_to_modify:
            result[keys_to_modify[key]] = value
        else:
            result[key] = value
    return result


def convert_to_chatml_function(record: dict, chatml_config: dict) -> dict:
    """Convert specified fields to ChatML format and add as a new key.

    Args:
        record: The original record
        chatml_config: Dict with new key name and list of source field names

    Returns:
        Updated record with new ChatML formatted field
    """
    result = record.copy()

    for new_key, source_fields in chatml_config.items():
        if not isinstance(source_fields, list):
            continue

        if len(source_fields) not in {2, 3}:
            continue

        messages = []

        if len(source_fields) == 3:
            system_field, user_field, assistant_field = source_fields
            if record.get(system_field):
                messages.append({'role': 'system', 'content': record[system_field]})
        else:
            user_field, assistant_field = source_fields

        if record.get(user_field):
            messages.append({'role': 'user', 'content': record[user_field]})

        if record.get(assistant_field):
            messages.append({'role': 'assistant', 'content': record[assistant_field]})

        if messages:
            result[new_key] = messages

    return result


def should_delete_record(record: dict, delete_conditions: dict) -> bool:
    """Check if a record should be deleted based on key-value conditions.

    Args:
        record: The record to check
        delete_conditions: Dict where keys are field names and values to match

    Returns:
        True if the record should be deleted, False otherwise
    """
    if not delete_conditions:
        return False

    for key, value_condition in delete_conditions.items():
        if key in record:
            record_value = record[key]
            if isinstance(value_condition, list):
                if record_value in value_condition:
                    return True
            elif record_value == value_condition:
                return True

    return False


def show_changes_preview(
    original: dict,
    modified: dict | None,
    keys_added: dict | None = None,
    keys_modified: dict | None = None,
    delete_conditions: dict | None = None,
    chatml_config: dict | None = None,
) -> None:
    """Display a preview of changes made to a record."""
    table = Table(title='Changes Preview', expand=True)
    table.add_column('Type', style='cyan')
    table.add_column('Details', style='green')

    if keys_added:
        for key, value in keys_added.items():
            table.add_row('Added Key', f'[bold]{key}[/bold]: {json.dumps(value)}')

    if keys_modified:
        for old_key, new_key in keys_modified.items():
            if old_key in original:
                table.add_row('Renamed Key', f'[bold]{old_key}[/bold] → [bold]{new_key}[/bold]')

    if chatml_config:
        for new_key, source_fields in chatml_config.items():
            if isinstance(source_fields, list):
                table.add_row(
                    'ChatML Conversion',
                    f'Created [bold]{new_key}[/bold] from fields: {", ".join(f"[bold]{f}[/bold]" for f in source_fields)}',
                )

    if delete_conditions:
        table.add_row('Delete Condition', f'Would delete if matches: {json.dumps(delete_conditions)}')

    console.print(table)

    if modified:
        console.print('Example of modified record:')
        syntax = Syntax(
            json.dumps(modified, indent=2, ensure_ascii=False),
            'json',
            theme='monokai',
            line_numbers=True,
            word_wrap=True,
        )
        console.print(Panel(syntax, expand=False))


def get_schema_from_file(file_path: Path) -> tuple[dict | None, int]:
    """Extract schema (keys) from the first valid line of a JSONL file and count lines.

    Args:
        file_path: Path to the JSONL file

    Returns:
        Tuple of (Dictionary with the keys from the first record or None, line count)
    """
    schema = None
    line_count = 0
    
    try:
        with file_path.open('r', encoding='utf-8') as f:
            for line in f:
                line_count += 1
                line_stripped = line.strip()
                if not line_stripped:
                    continue
                if schema is None:  # Only extract schema from first valid line
                    try:
                        record = json.loads(line_stripped)
                        schema = {key: type(value).__name__ for key, value in record.items()}
                    except json.JSONDecodeError:
                        continue
    except (OSError, PermissionError, FileNotFoundError) as err:
        console.print(f'[bold red]Error:[/] Failed to read file {file_path}: {err!s}')
    
    return schema, line_count


def find_jsonl_files(path: Path, recursive: bool = True) -> list[Path]:
    """Find all JSONL files in a directory.

    Args:
        path: Path to search in
        recursive: Whether to search recursively in subdirectories

    Returns:
        List of paths to JSONL files
    """
    if path.is_file():
        return [path] if path.suffix == '.jsonl' else []

    if recursive:
        return sorted(path.rglob('*.jsonl'))
    else:
        return sorted(path.glob('*.jsonl'))


@app.command()
def see_schema(
    path: Path = typer.Argument(..., help='Path to a JSONL file or directory to analyze'),
    recursive: bool = typer.Option(True, '--recursive/--no-recursive', '-r/-R', help='Search subdirectories (default: True)'),
) -> None:
    """Display the schema (keys and types) from JSONL files.

    Analyzes the first line of each JSONL file to extract the schema.
    Can process a single file or all JSONL files in a directory (including subdirectories).
    """
    if not path.exists():
        console.print(f'[bold red]Error:[/] Path does not exist: {path}')
        raise typer.Exit(code=1)

    # Find all JSONL files
    jsonl_files = find_jsonl_files(path, recursive=recursive)

    if not jsonl_files:
        console.print(f'[bold yellow]Warning:[/] No JSONL files found in {path}')
        raise typer.Exit(code=0)

    console.print(f'\n[bold]Found {len(jsonl_files)} JSONL file(s)[/]\n')

    # Analyze each file
    for file_path in jsonl_files:
        schema, line_count = get_schema_from_file(file_path)

        # Determine display name
        if path.is_dir():
            try:
                display_name = str(file_path.relative_to(path))
            except ValueError:
                display_name = file_path.name
        else:
            display_name = file_path.name

        # Get file metadata
        try:
            stat_info = file_path.stat()
            size_mb = stat_info.st_size / (1024 * 1024)
            created_time = stat_info.st_birthtime
            modified_time = stat_info.st_mtime
            
            from datetime import datetime
            created_str = datetime.fromtimestamp(created_time).strftime('%Y-%m-%d %H:%M:%S')
            modified_str = datetime.fromtimestamp(modified_time).strftime('%Y-%m-%d %H:%M:%S')
        except (OSError, AttributeError) as e:
            size_mb = 0
            created_str = 'N/A'
            modified_str = 'N/A'

        if schema is None:
            console.print(f'[bold red]✗[/] {display_name}: No valid JSON found')
            continue

        # Create a table for this file's schema
        file_info = f'{display_name}\n[dim]Lines: {line_count} | Size: {size_mb:.2f} MB | Created: {created_str} | Modified: {modified_str}[/dim]'
        table = Table(title=file_info, show_header=True)
        table.add_column('Key', style='cyan', no_wrap=True)
        table.add_column('Type', style='green')

        for key, value_type in schema.items():
            table.add_row(key, value_type)

        console.print(table)
        console.print()  # Add spacing between files


@app.command()
def process(
    input_file: Path | None = INPUT_FILE_OPTION,
    output_file: Path | None = OUTPUT_FILE_OPTION,
    key_content_to_add: str | None = KEY_CONTENT_OPTION,
    keys_to_modify: str | None = KEYS_TO_MODIFY_OPTION,
    delete_line_if_value: str | None = DELETE_LINE_IF_VALUE_OPTION,
    convert_to_chatml_config: str | None = CONVERT_TO_CHATML_OPTION,
    merge: str | None = MERGE_OPTION,
    preview: bool = PREVIEW_OPTION,
) -> None:
    """Process JSONL files by adding new key-value pairs, renaming existing keys,
    filtering out records that match specific key-value pairs, or merging multiple files.
    """
    # Handle merge operation
    if merge:
        # In the merge operation section, modify the "all" handling:
        if merge.lower() == 'all':
            # Find all JSONL files in current directory
            jsonl_files = list(Path.cwd().glob('*.jsonl'))

            # Filter out previously generated merge files to avoid loops
            jsonl_files = [f for f in jsonl_files if not f.stem.endswith('_merge')]

            if len(jsonl_files) < 2:
                console.print('[bold red]Error:[/] Need at least 2 JSONL files in current directory to merge')
                raise typer.Exit(code=1)
            file_paths = jsonl_files
        else:
            # Parse comma-separated file paths
            file_names = [f.strip() for f in merge.split(',')]
            file_paths = []
            for fname in file_names:
                fpath = Path(fname)
                if not fpath.exists():
                    console.print(f'[bold red]Error:[/] File not found: {fname}')
                    raise typer.Exit(code=1)
                if not fname.endswith('.jsonl'):
                    console.print(f'[bold red]Error:[/] Not a JSONL file: {fname}')
                    raise typer.Exit(code=1)
                file_paths.append(fpath)

            if len(file_paths) < 2:
                console.print('[bold red]Error:[/] Need at least 2 files to merge')
                raise typer.Exit(code=1)

        # Validate all files
        for fpath in file_paths:
            if not validate_jsonl_file(fpath):
                raise typer.Exit(code=1)

        # Determine output path
        if output_file:
            merge_output = output_file
        else:
            first_file = file_paths[0]
            merge_output = first_file.with_stem(f'{first_file.stem}_merge')

        # Perform merge
        merge_jsonl_files(file_paths, merge_output)
        return

    # Regular processing (non-merge)
    if not input_file:
        console.print('[bold red]Error:[/] --input-file is required when not using --merge')
        raise typer.Exit(code=1)

    # Verify input file is valid JSONL
    if not validate_jsonl_file(input_file):
        raise typer.Exit(code=1)

    # Parse JSON arguments
    keys_to_add_dict = parse_json_arg(key_content_to_add) if key_content_to_add else {}
    keys_to_modify_dict = parse_json_arg(keys_to_modify) if keys_to_modify else {}
    delete_conditions_dict = parse_json_arg(delete_line_if_value) if delete_line_if_value else {}
    chatml_config_dict = parse_json_arg(convert_to_chatml_config) if convert_to_chatml_config else {}

    # If no operations specified, show error and exit
    if not keys_to_add_dict and not keys_to_modify_dict and not delete_conditions_dict and not chatml_config_dict:
        console.print(
            '[bold red]Error:[/] No operations specified. Please provide --key-content-to-add, --keys-to-modify, '
            '--delete-line-if-value, or --convert-to-chatml'
        )
        raise typer.Exit(code=1)

    # Validate ChatML config
    for new_key, fields in chatml_config_dict.items():
        if not isinstance(fields, list) or len(fields) not in {2, 3}:
            console.print(
                f'[bold red]Error:[/] Invalid ChatML configuration for key "{new_key}". '
                f'Expected a list of 2 or 3 field names, got: {fields}'
            )
            raise typer.Exit(code=1)

    # Determine output file path if not provided
    if not output_file:
        stem = input_file.stem
        output_file = input_file.with_stem(f'{stem}_modified')

    # Read input file to count lines for progress bar
    with input_file.open('r', encoding='utf-8') as f:
        total_lines = sum(1 for _ in f)

    # Display operation information
    console.print(
        Panel.fit(
            f'[bold green]JSONL Processor[/]\n\n'
            f'Input file: [cyan]{input_file}[/]\n'
            f'Output file: [cyan]{output_file}[/]\n'
            f'Keys to add: [cyan]{json.dumps(keys_to_add_dict, indent=2) if keys_to_add_dict else "None"}[/]\n'
            f'Keys to modify: [cyan]{json.dumps(keys_to_modify_dict, indent=2) if keys_to_modify_dict else "None"}[/]\n'
            f'Delete conditions: [cyan]{json.dumps(delete_conditions_dict, indent=2) if delete_conditions_dict else "None"}[/]\n'
            f'ChatML conversion: [cyan]{json.dumps(chatml_config_dict, indent=2) if chatml_config_dict else "None"}[/]\n'
            f'Preview mode: [cyan]{"Yes" if preview else "No"}[/]',
            title='Operation Details',
            border_style='green',
        )
    )

    # Preview only? Just read the first record and show changes
    if preview:
        with input_file.open('r', encoding='utf-8') as f:
            for line in f:
                line_content = line.strip()
                if line_content:
                    try:
                        record = json.loads(line_content)

                        # Check if record would be deleted
                        if should_delete_record(record, delete_conditions_dict):
                            show_changes_preview(record, None, None, None, delete_conditions_dict)
                            break

                        # Apply modifications
                        modified_record = record.copy()
                        if keys_to_add_dict:
                            modified_record = add_keys(modified_record, keys_to_add_dict)
                        if keys_to_modify_dict:
                            modified_record = modify_keys(modified_record, keys_to_modify_dict)
                        if chatml_config_dict:
                            modified_record = convert_to_chatml_function(modified_record, chatml_config_dict)

                        show_changes_preview(
                            record,
                            modified_record,
                            keys_to_add_dict,
                            keys_to_modify_dict,
                            delete_conditions_dict,
                            chatml_config_dict,
                        )
                        break
                    except json.JSONDecodeError:
                        continue
        return

    # Process the file with a progress bar
    records_processed = 0
    records_deleted = 0
    with Progress(
        TextColumn('[bold blue]{task.description}'),
        BarColumn(),
        '[progress.percentage]{task.percentage:>3.0f}%',
        TaskProgressColumn(),
        TextColumn('({task.completed}/{task.total})'),
        console=console,
    ) as progress:
        task_id = progress.add_task('[cyan]Processing JSONL...', total=total_lines)

        with input_file.open('r', encoding='utf-8') as infile, output_file.open('w', encoding='utf-8') as outfile:
            for line in infile:
                line_content = line.strip()
                if not line_content:
                    outfile.write('\n')
                    progress.update(task_id, advance=1)
                    continue

                try:
                    record = json.loads(line_content)

                    # Check if record should be deleted
                    if should_delete_record(record, delete_conditions_dict):
                        records_deleted += 1
                        progress.update(task_id, advance=1)
                        continue

                    # Apply modifications
                    modified_record = record.copy()
                    if keys_to_add_dict:
                        modified_record = add_keys(modified_record, keys_to_add_dict)
                    if keys_to_modify_dict:
                        modified_record = modify_keys(modified_record, keys_to_modify_dict)
                    if chatml_config_dict:
                        modified_record = convert_to_chatml_function(modified_record, chatml_config_dict)

                    # Write modified record to output file with ensure_ascii=False to preserve Unicode
                    outfile.write(json.dumps(modified_record, ensure_ascii=False) + '\n')
                    records_processed += 1
                except json.JSONDecodeError:
                    # Write invalid lines unchanged
                    outfile.write(line)
                    if not line.endswith('\n'):
                        outfile.write('\n')

                progress.update(task_id, advance=1)

    # Show summary
    console.print(
        Panel(
            f'[bold green]✓[/] Successfully processed {records_processed} records\n'
            f'[bold green]✓[/] Deleted {records_deleted} records\n'
            f'[bold green]✓[/] Output written to [cyan]{output_file}[/]',
            title='Processing Complete',
            border_style='green',
        )
    )


def main() -> None:
    """Entry point for the application."""
    try:
        app()
    except KeyboardInterrupt:
        console.print('[bold yellow]Process interrupted by user.[/]')
        sys.exit(130)
    except Exception as e:
        console.print(f'[bold red]Error:[/] {e!s}')
        console.print(f'[red]{traceback.format_exc()}[/]')
        sys.exit(1)


if __name__ == '__main__':
    main()
