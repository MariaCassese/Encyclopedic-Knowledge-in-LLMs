import argparse
import datetime
import json
import logging
import time
from logging.handlers import RotatingFileHandler
from pathlib import Path

from model_loader import get_model
from writer import save_results
from utils.utils import prompt_requires_image


def setup_logger(log_path: str) -> logging.Logger:
    """
    Configure the application logger.

    Logs are written both to stdout and to a rotating log file.
    The file is rotated when it exceeds 5 MB, keeping up to three backups.

    Args:
        log_path: Path to the log file.

    Returns:
        Configured logger instance.
    """
    logger = logging.getLogger("ModelEval")
    logger.setLevel(logging.INFO)

    # Avoid duplicated handlers when the logger is initialized multiple times
    if logger.handlers:
        logger.handlers.clear()

    fmt = "%(asctime)s %(levelname)-8s %(message)s"
    datefmt = "%Y-%m-%dT%H:%M:%S"
    formatter = logging.Formatter(fmt, datefmt=datefmt)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    file_handler = RotatingFileHandler(
        log_path,
        maxBytes=5_000_000,
        backupCount=3,
        encoding="utf-8",
    )
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger


def main():
    parser = argparse.ArgumentParser(
        description="Generate answers to a textual or visual query"
    )
    parser.add_argument(
        "--csv_path",
        required=True,
        help="Path of the input CSV file",
    )
    parser.add_argument(
        "--prompts_file",
        default=None,
        help="Path of the JSON file with the list of prompts",
    )
    parser.add_argument(
        "--mapping_key",
        default=None,
        help=(
            "Key to use in modeling/config/label_placeholders.json. "
            "Use this when the CSV stem is different from the mapping key, "
            "for example actor_it, actor_es, actor_ar, actor_zh. "
            "If omitted, the CSV stem is used as before."
        ),
    )
    parser.add_argument(
        "--model_name",
        default="meta-llama/Llama-3.1-8B-Instruct",
        help="The model",
    )
    parser.add_argument(
        "--device",
        default="cuda:1",
        help="The GPU in use",
    )
    parser.add_argument(
        "--log_file",
        default="model_eval.log",
    )
    parser.add_argument(
        "--force-overwrite",
        action="store_true",
        help="Overwrite existing results instead of resuming from checkpoint",
    )
    parser.add_argument(
        "--csv_encoding",
        default="utf-8",
        help=(
            "Encoding used to read the input CSV and existing checkpoints, "
            "and to write results."
        ),
    )
    args = parser.parse_args()

    logger = setup_logger(args.log_file)
    logger.info("Script started")

    csv_stem = Path(args.csv_path).stem
    output_dataset_name = args.mapping_key or csv_stem

    # Prompt file resolution
    if args.prompts_file is None:
        dataset_name = output_dataset_name
        args.prompts_file = f"modeling/config/prompts/{dataset_name}.json"

        if not Path(args.prompts_file).exists():
            args.prompts_file = f"modeling/config/prompts/{csv_stem}.json"

        if not Path(args.prompts_file).exists():
            args.prompts_file = "modeling/config/prompts/prime_minister.json"
            logger.warning(
                f"Prompt file not found for '{dataset_name}' or '{csv_stem}', "
                "using default prime_minister.json"
            )

    t0 = time.time()
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S-%f")

    logger.info("Run timestamp: %s", timestamp)
    logger.info("CSV path: %s", args.csv_path)
    logger.info("Prompts file: %s", args.prompts_file)
    logger.info("Mapping key: %s", args.mapping_key or csv_stem)
    logger.info("Date format: iso")
    logger.info("CSV encoding: %s", args.csv_encoding)

    # Load prompts
    with open(args.prompts_file, "r", encoding="utf-8") as f:
        prompts = json.load(f)
        prompt_names = list(prompts.keys())

    prompt_file_stem = Path(args.prompts_file).stem

    model_instance = get_model(
        args.model_name,
        device=args.device,
        logger=logger,
        do_sample=True,
        top_p=0.9,
        temperature=0.7,
        max_new_tokens=100,
    )

    for prompt in prompt_names:
        prompt_content = prompts[prompt]

        if (
            prompt_requires_image(prompt_content)
            and not model_instance.supports_image
        ):
            logger.info(
                "Skipping '%s' on %s: model does not support images.",
                prompt,
                args.model_name,
            )
            continue

        out_csv = (
            f"modeling/{output_dataset_name}/{args.model_name}/generation_results/"
            f"model_answer_{output_dataset_name}_{prompt_file_stem}_{prompt}.csv"
        )
        Path(out_csv).parent.mkdir(parents=True, exist_ok=True)

        if Path(out_csv).exists() and not args.force_overwrite:
            logger.info(
                "Resuming from existing checkpoint: %s",
                out_csv,
            )
        else:
            if args.force_overwrite and Path(out_csv).exists():
                logger.info(
                    "Force overwrite enabled, removing existing file: %s",
                    out_csv,
                )
                Path(out_csv).unlink()

            logger.info(
                "Starting new results file: %s",
                out_csv,
            )

        save_results(
            model_instance,
            args.model_name,
            args.csv_path,
            prompt_content,
            out_csv,
            mapping_json_path="modeling/config/label_placeholders.json",
            mapping_key=args.mapping_key,
            csv_encoding=args.csv_encoding,
        )

        logger.info(
            "Completed processing: %s",
            out_csv,
        )

    total_time = time.time() - t0
    print(f"Total generation time: {total_time:.2f}s")


if __name__ == "__main__":
    main()

