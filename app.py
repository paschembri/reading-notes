# -*- coding: utf-8 -*-
import re
import argparse
import os
from llm_core.assistants import Summarizer, LLaMACPPAssistant
from unstructured.partition.auto import partition

exclude_list = [
    r"^reading_notes\.txt",
    r"^\..*",
]


def get_files(directory):
    for filename in os.listdir(directory):
        filepath = os.path.join(directory, filename)
        if os.path.isfile(filepath):
            exclusion_checks = [
                re.match(pattern, filename) for pattern in exclude_list
            ]

            if any(exclusion_checks):
                print(f"Excluding {filename}.")
                continue

            yield filename, filepath


def main():
    parser = argparse.ArgumentParser(
        description="Recursively create reading notes from a directory."
    )
    parser.add_argument("dir_path", type=str, help="The directory path.")
    parser.add_argument(
        "--model_name",
        type=str,
        default="mistral-7b-instruct-v0.1.Q4_K_M.gguf",
        help="The model file in ~/.cache/py-llm-core/models.",
    )
    args = parser.parse_args()

    notes_file = os.path.join(args.dir_path, "reading_notes.txt")

    summarizer = Summarizer(
        model=args.model_name, assistant_cls=LLaMACPPAssistant
    )

    with open(notes_file, "a") as file:
        for filename, path in get_files(args.dir_path):
            print(f"Summarizing {filename}")
            try:
                elements = partition(
                    filename=path,
                    strategy="auto",
                )
            except ValueError:
                print(f"Skipping {filename}")
                continue

            content = "\n\n".join(str(e) for e in elements)
            summary = summarizer.fast_summarize(content)
            file.write("# File: {}\n\n".format(filename))
            file.write("{}\n\n".format(summary.content))
            file.flush()


if __name__ == "__main__":
    main()
