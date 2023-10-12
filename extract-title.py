#!/Users/pas/development/advanced-stack/experiments/reading-notes/venv/bin/python3
# -*- coding: utf-8 -*-
import argparse
import os
from dataclasses import dataclass
from llm_core.parsers import LLaMACPPParser
from unstructured.partition.auto import partition


UNABLE_TO_PARSE = "unable-to-parse"


@dataclass
class Document:
    title: str


def extract_content(filename):
    try:
        elements = partition(
            filename=filename,
            strategy="auto",
        )
    except ValueError:
        print(f"Skipping {filename}")
        content = UNABLE_TO_PARSE

    content = "\n\n".join(str(e) for e in elements)
    return content


def main():
    parser = argparse.ArgumentParser(description="Extract title from document")
    parser.add_argument("filename", type=str)
    parser.add_argument(
        "--model_name",
        type=str,
        default="mistral-7b-instruct-v0.1.Q4_K_M.gguf",
        help="The model file in ~/.cache/py-llm-core/models.",
    )
    args = parser.parse_args()
    content = extract_content(args.filename)
    with LLaMACPPParser(Document, args.model_name) as parser:
        document = parser.parse(content[:1000])

    if document.title == UNABLE_TO_PARSE:
        return

    # Get the original file extension
    _, extension = os.path.splitext(args.filename)

    # Get the original file directory
    directory = os.path.dirname(args.filename)

    # Generate the new file name using the document title and original extension
    new_filename = f"{directory}/{document.title.title()}{extension}"

    # Rename the file
    os.rename(args.filename, new_filename)

    print(f"File renamed to: {new_filename}")


if __name__ == "__main__":
    main()
