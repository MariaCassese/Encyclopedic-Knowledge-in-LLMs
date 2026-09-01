import json
import re
import string
from datetime import datetime
from pathlib import Path

import pandas as pd


MISSING_VALUE = ""


def _is_missing(x) -> bool:
    """Return whether a value should be treated as missing."""
    if x is None:
        return True

    if isinstance(x, float) and pd.isna(x):
        return True

    if isinstance(x, str) and x.strip() == "":
        return True

    return False


def _resolve_nested_prompt(prompt_content, placeholders):
    """
    Resolve a prompt that may contain multiple date-dependent variants.

    Prompt variants ending in .1, .2, or .3 are selected according to the
    availability of the supported date pairs:
    - start_of_office / end_of_office
    - birth_date / death_date
    """
    # Standard list-based prompt
    if isinstance(prompt_content, list):
        return prompt_content

    if isinstance(prompt_content, dict):
        variant_keys = [
            key
            for key in prompt_content.keys()
            if re.match(r".*\.\d+$", key)
        ]

        if variant_keys:
            # President case: start_of_office / end_of_office
            office_start = placeholders.get("start_of_office")
            office_end = placeholders.get("end_of_office")

            # Actor case: birth_date / death_date
            birth = placeholders.get("birth_date")
            death = placeholders.get("death_date")

            has_office_pair = not (
                _is_missing(office_start)
                and _is_missing(office_end)
            )
            has_life_pair = not (
                _is_missing(birth)
                and _is_missing(death)
            )

            if has_office_pair:
                first_date = office_start
                second_date = office_end
            elif has_life_pair:
                first_date = birth
                second_date = death
            else:
                return None

            first_missing = _is_missing(first_date)
            second_missing = _is_missing(second_date)

            # Variant mapping:
            # .1 = both dates are available
            # .2 = only the first date is available
            # .3 = only the second date is available
            if not first_missing and not second_missing:
                want_suffix = "1"
            elif not first_missing and second_missing:
                want_suffix = "2"
            elif first_missing and not second_missing:
                want_suffix = "3"
            else:
                return None

            # Select the matching prompt variant
            for key in prompt_content.keys():
                match = re.match(r"^(.*)\.(\d+)$", key)

                if match and match.group(2) == want_suffix:
                    chosen = prompt_content[key]
                    return chosen if isinstance(chosen, list) else [chosen]

            # Fallback to the first available variant
            first_key = sorted(variant_keys)[0]
            chosen = prompt_content[first_key]

            return chosen if isinstance(chosen, list) else [chosen]

        # Standard message dictionary: {role, content}
        if "role" in prompt_content and "content" in prompt_content:
            return [prompt_content]

    return [
        {
            "role": "user",
            "content": str(prompt_content),
        }
    ]


def _format_field_root(field_name: str | None) -> str | None:
    """
    Extract the root placeholder name from a Python format field.

    Examples:
        full_name -> full_name
        person.name -> person
        items[0] -> items
    """
    if not field_name:
        return None

    return re.split(r"[.\[]", field_name, maxsplit=1)[0]


def _placeholders_in_format_string(text: str) -> set[str]:
    """
    Return placeholders used by a format string.

    string.Formatter handles escaped braces such as {{ and }} correctly.
    """
    found: set[str] = set()
    formatter = string.Formatter()

    try:
        parsed = list(formatter.parse(text))
    except ValueError as e:
        excerpt = text if len(text) <= 240 else f"{text[:237]}..."
        raise ValueError(
            f"{e} in prompt text: {excerpt!r}"
        ) from e

    for _, field_name, format_spec, _ in parsed:
        root = _format_field_root(field_name)

        if root:
            found.add(root)

        # Support nested placeholders in format specs, e.g. {value:{width}}
        if format_spec:
            found.update(
                _placeholders_in_format_string(format_spec)
            )

    return found


def _required_placeholders(
    prompt,
    path: str = "prompt",
) -> set[str]:
    """
    Recursively collect placeholders used by the selected prompt.

    This makes row skipping prompt-aware:
    - if the prompt contains {full_name}, full_name is required;
    - if the prompt contains only {country}, full_name is not required.
    """
    if isinstance(prompt, str):
        try:
            return _placeholders_in_format_string(prompt)
        except ValueError as e:
            raise ValueError(
                f"Invalid format string at {path}: {e}"
            ) from e

    if isinstance(prompt, list):
        required: set[str] = set()

        for idx, item in enumerate(prompt):
            required.update(
                _required_placeholders(
                    item,
                    f"{path}[{idx}]",
                )
            )

        return required

    if isinstance(prompt, dict):
        required: set[str] = set()

        for key, value in prompt.items():
            required.update(
                _required_placeholders(
                    value,
                    f"{path}.{key}",
                )
            )

        return required

    return set()


def prepare_prompt(prompt, **values):
    """
    Apply values to placeholders in a prompt recursively.

    The prompt can be a dictionary, a list, or a string.
    """
    if isinstance(prompt, dict):
        out = {}

        for key, value in prompt.items():
            if key == "type" and value == "image":
                return {
                    "type": "image",
                    "image_path": values.get("image_path"),
                }

            out[key] = prepare_prompt(
                value,
                **values,
            )

        return out

    if isinstance(prompt, list):
        return [
            prepare_prompt(item, **values)
            for item in prompt
        ]

    if isinstance(prompt, str):
        return prompt.format(**values)

    raise TypeError(
        f"Unexpected element: {prompt!r} "
        f"(type {type(prompt).__name__})"
    )


def _extract_iso_date(value: str):
    """
    Extract the date part from an ISO-like string.

    Accepted examples:
        1950-01-12
        1950-01-12T00:00:00Z
        born on 1950-01-12

    Returns:
        A datetime.date instance, or None if no valid ISO date is found.
    """
    if not isinstance(value, str):
        return None

    match = re.search(r"\d{4}-\d{2}-\d{2}", value)

    if not match:
        return None

    try:
        return datetime.fromisoformat(
            match.group(0)
        ).date()
    except ValueError:
        return None


def _maybe_format_iso_date(value):
    """
    Normalize an ISO-like date to YYYY-MM-DD.

    The representation is intentionally language-neutral so that the same
    format can be used consistently across prompts in different languages.
    """
    date_value = _extract_iso_date(value)

    if date_value is None:
        return value

    return date_value.isoformat()


def _normalise_value(value):
    """Normalize missing dataframe values."""
    if pd.isna(value):
        return MISSING_VALUE

    return value


def _resume_key(record: dict) -> tuple:
    """
    Build a stable key for checkpoint/resume.

    source_row_index is preferred when available because rows with missing
    localized names can otherwise collide.

    The legacy fields preserve backward compatibility with older outputs
    that did not store source_row_index.
    """
    source_row_index = record.get("source_row_index")

    if not _is_missing(source_row_index):
        try:
            source_row_index = int(
                float(source_row_index)
            )
        except (TypeError, ValueError):
            source_row_index = str(
                source_row_index
            ).strip()

        return (
            "source_row_index",
            source_row_index,
        )

    return (
        "legacy",
        *_legacy_resume_key(record),
    )


def _legacy_resume_key(record: dict) -> tuple:
    """Build a resume key compatible with older writer versions."""
    values = (
        record.get("full_name"),
        record.get("country"),
        record.get("country_of_citizenship"),
        record.get("birth_date"),
        record.get("death_date"),
        record.get("start_of_office"),
        record.get("end_of_office"),
        record.get("image_path"),
    )

    return tuple(
        (
            MISSING_VALUE
            if _is_missing(value)
            else _maybe_format_iso_date(value)
        )
        for value in values
    )


def save_results(
    model,
    model_name: str,
    csv_path: str,
    prompt_content: dict,
    output_csv: str,
    mapping_json_path: str,
    save_interval: int = 100,
    mapping_key: str | None = None,
    csv_encoding: str = "utf-8",
):
    """
    Generate model answers from a CSV file and save the results.

    The function reads a CSV file, builds placeholders for each row using
    the mapping defined in a JSON file, prepares the selected prompt,
    calls model.predict, and saves the generated answers.

    Args:
        model: Model instance used for generation.
        model_name: Name of the model.
        csv_path: Path to the input CSV file.
        prompt_content: Prompt configuration used for generation.
        output_csv: Path to the output CSV file.
        mapping_json_path: Path to label_placeholders.json.
        save_interval: Number of records between intermediate saves.
        mapping_key: Mapping entry to use from label_placeholders.json.
            If None, Path(csv_path).stem is used.
        csv_encoding: Encoding used to read and write CSV files.

    Date handling:
        - ISO-like dates are normalized to YYYY-MM-DD.
        - No English month names are introduced.
        - No locale or Babel dependency is required.

    Missing-value handling:
        - A row is skipped only if the selected prompt requires a
          missing placeholder.
        - A missing full_name does not cause prompts that do not use
          {full_name} to be skipped.

    Example:
        csv_path = actor_with_labels.csv
        mapping_key = actor_it

        The CSV actor_with_labels.csv is read while the mapping stored
        under "actor_it" in label_placeholders.json is used.

    Returns:
        A dataframe containing the generated records.
    """
    with open(
        mapping_json_path,
        "r",
        encoding="utf-8",
    ) as f:
        mapping_dict = json.load(f)

    csv_stem = Path(csv_path).stem
    query_type = mapping_key or csv_stem

    if query_type not in mapping_dict:
        available = ", ".join(
            sorted(mapping_dict.keys())
        )

        raise ValueError(
            f"No mapping found for query type '{query_type}' "
            f"in {mapping_json_path}. "
            f"Available mappings: {available}"
        )

    field_mapping = mapping_dict[query_type]

    df = pd.read_csv(
        csv_path,
        encoding=csv_encoding,
    )

    missing_columns = [
        csv_field
        for csv_field in field_mapping
        if csv_field not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"The CSV file {csv_path} is missing columns required by mapping "
            f"'{query_type}': {missing_columns}. "
            "Either add these columns to the CSV or change "
            "label_placeholders.json."
        )

    records = []
    existing_keys = set()
    existing_legacy_keys = set()

    # Load an existing checkpoint, if available
    if Path(output_csv).exists():
        try:
            existing_df = pd.read_csv(
                output_csv,
                encoding=csv_encoding,
            )
            records = existing_df.to_dict("records")

            for record in records:
                existing_keys.add(
                    _resume_key(record)
                )
                existing_legacy_keys.add(
                    _legacy_resume_key(record)
                )

            print(
                f"Resuming from checkpoint: "
                f"{len(records)} records already processed"
            )

        except Exception as e:
            print(
                f"Could not load existing output file: {e}. "
                "Starting fresh."
            )

            records = []
            existing_keys = set()
            existing_legacy_keys = set()

    supports_img = bool(
        getattr(model, "supports_image", False)
    )

    skipped_count = 0
    skipped_missing_required_count = 0
    skipped_checkpoint_count = 0

    for i, row in df.iterrows():
        placeholders = {}

        # Build placeholders from the current CSV row
        for csv_field, placeholder_name in field_mapping.items():
            value = _normalise_value(
                row[csv_field]
            )
            value = _maybe_format_iso_date(value)

            placeholders[placeholder_name] = value

        # Optional date placeholders prevent KeyError when a prompt refers
        # to a date field that is not available in the current CSV.
        for optional_key in [
            "start_of_office",
            "end_of_office",
            "birth_date",
            "death_date",
        ]:
            placeholders.setdefault(
                optional_key,
                "",
            )

        # Preserve the original CSV row index in the output.
        # This makes checkpoint/resume safer when localized names are missing.
        placeholders["source_row_index"] = int(i)

        chosen_messages = _resolve_nested_prompt(
            prompt_content,
            placeholders,
        )

        # Skip the row if neither supported date pair is available
        if chosen_messages is None:
            skipped_count += 1
            skipped_missing_required_count += 1

            print(
                "Skipping row: missing both supported date pairs "
                "(start_of_office/end_of_office and "
                "birth_date/death_date)"
            )

            continue

        # Skip only when placeholders required by the selected prompt
        # are actually missing.
        required = _required_placeholders(
            chosen_messages
        )

        missing_required = sorted(
            key
            for key in required
            if _is_missing(placeholders.get(key))
        )

        if missing_required:
            skipped_count += 1
            skipped_missing_required_count += 1

            print(
                f"Skipping row {i}: missing required placeholder(s) "
                f"for current prompt: {missing_required}"
            )

            continue

        # Skip records already present in the checkpoint
        key = _resume_key(placeholders)
        legacy_key = _legacy_resume_key(placeholders)

        if (
            key in existing_keys
            or legacy_key in existing_legacy_keys
        ):
            skipped_count += 1
            skipped_checkpoint_count += 1
            continue

        # Materialize the prompt and generate the model answer
        materialized_prompt = prepare_prompt(
            chosen_messages,
            **placeholders,
        )

        image_arg = (
            placeholders.get("image_path")
            if supports_img
            else None
        )

        model_answer = model.predict(
            materialized_prompt,
            image_arg,
        )

        if model_answer is None:
            continue

        record = {
            "model_name": model_name,
            "query": query_type,
            "csv_stem": csv_stem,
            **placeholders,
            "model_answer": model_answer,
        }

        records.append(record)

        existing_keys.add(key)
        existing_legacy_keys.add(legacy_key)

        # Save an intermediate checkpoint
        if len(records) % save_interval == 0:
            out_df = pd.DataFrame(records)

            out_df.to_csv(
                output_csv,
                index=False,
                encoding=csv_encoding,
            )

            print(
                f"Saved {len(records)} records to {output_csv}"
            )

    # Final save
    out_df = pd.DataFrame(records)

    out_df.to_csv(
        output_csv,
        index=False,
        encoding=csv_encoding,
    )

    print(
        f"Final save: {len(records)} total records "
        f"(skipped {skipped_count}; "
        f"missing required placeholders: "
        f"{skipped_missing_required_count}; "
        f"already processed: {skipped_checkpoint_count})"
    )

    return out_df
