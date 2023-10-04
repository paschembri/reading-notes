# -*- coding: utf-8 -*-
import argparse
import os
import urllib.request
import hashlib
from dataclasses import dataclass
from llm_core.parsers import LLamaParser
from llm_core.splitters import TokenSplitter
from unstructured.partition.auto import partition
from typing import List

# Define the directory and file
dir_path = os.path.expanduser("~/.cache/reading-notes/")
file_url = "https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.1-GGUF/resolve/main/mistral-7b-instruct-v0.1.Q4_K_M.gguf"
file_path = os.path.join(dir_path, "mistral-7b-instruct-v0.1.Q4_K_M.gguf")
model_path = file_path
# Create the directory if it doesn't exist
os.makedirs(dir_path, exist_ok=True)


# Function to check the MD5 hash of the file
def check_md5(file_path, expected_md5):
    hasher = hashlib.md5()
    with open(file_path, "rb") as afile:
        buf = afile.read()
        hasher.update(buf)
    return hasher.hexdigest() == expected_md5


# Function to download the file
def download_file(url, dest):
    print("Downloading model from HF...")
    urllib.request.urlretrieve(url, dest)
    if not check_md5(file_path, "9cfab7b0e378473415241229dad8de47"):
        raise ValueError(
            "MD5 hash of the file does not match the expected hash."
        )
    print("Done.")


# Download the file only if it doesn't exist
if not os.path.exists(file_path):
    download_file(file_url, file_path)


def get_files(directory):
    for filename in os.listdir(directory):
        filepath = os.path.join(directory, filename)
        if os.path.isfile(filepath):
            yield filepath


@dataclass
class Document:
    document_classification: str
    document_entity: str
    top_5_keywords: List[str]
    document_summary: str

    def to_markdown(self):
        markdown = f"**Classification:** {self.document_classification}\n\n"
        markdown += f"**Entity:** {self.document_entity}\n\n"
        markdown += f"**Summary:** {self.document_summary}\n\n"
        markdown += f"**Keywords:** {', '.join(self.top_5_keywords)}\n\n"

        return markdown

    @classmethod
    def from_directory(cls, directory):
        with LLamaParser(Document, model_path=model_path) as parser:
            splitter = TokenSplitter(chunk_size=3000)
            for filename in get_files(directory):
                if "reading_notes.txt" in filename:
                    continue

                print(f"Parsing file {filename}")
                try:
                    elements = partition(
                        filename=filename,
                        strategy="auto",
                    )
                except ValueError:
                    print(f"Skipping {filename}")
                    continue

                text = "\n\n".join(str(e) for e in elements)
                extract = next(splitter.chunkify(text))

                formatted_extract = f"""
                FILENAME: `{filename}`
                CONTENT:```{extract}```

                > Parse and summarize information"""

                print(f"Indexing file {filename}")
                yield filename, parser.parse(formatted_extract)
                print("Done.")


def main():
    parser = argparse.ArgumentParser(
        description="Recursively create reading notes from a directory."
    )
    parser.add_argument("dir_path", type=str, help="The directory path.")
    args = parser.parse_args()

    notes_file = os.path.join(args.dir_path, "reading_notes.txt")

    with open(notes_file, "a") as file:
        for filename, document in Document.from_directory(args.dir_path):
            markdown = document.to_markdown()
            file.write(markdown)
            file.write("\nFile: {}\n".format(filename))
            file.write("\n\n---\n\n")
            file.flush()


if __name__ == "__main__":
    main()
