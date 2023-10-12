# Reading Notes AI

AI Assistant that runs locally and provides reading notes for a given folder

## What's this ?

This is an experiment to show how an AI-augmented knowledge management system can provide insights and information about a large document collection.

Reading Notes AI uses the recently released model from Mistral AI and runs locally (i.e. offline).

## Disclaimer

This is still experimental: hallucinations are to be expected.

## How does it work ?

I use the unstructured library to convert any supported document into text (it even works on images, see the `assets` directory), then I use Mistral AI Instruct to generate a summary with PyLLMCore:

```python
from llm_core.assistants import Summarizer, LLaMACPPAssistant

...

summarizer = Summarizer(
    model=args.model_name, assistant_cls=LLaMACPPAssistant
)

...

summarizer.fast_summarize(content)

```

## Why is it slow ?

Running this script on a M1 MacBook Pro can go as fast as 13 tokens / sec (for token sampling) - that is approx 1 min 30s per page generated.

As an example processing this repo took 2min24sec.

## Launching the analysis on a folder

1. Install:

```shell
python3 -m venv venv
venv/bin/python3 -m pip install -r requirements.txt

# If you have a M1/M2 Apple Silicon

CMAKE_ARGS="-DCMAKE_OSX_ARCHITECTURES=arm64" venv/bin/python3 -m install --upgrade --verbose --force-reinstall --no-cache-dir llama-cpp-python
```

2. Launch

```shell
venv/bin/python3 app.py /path/to/a/directory/containing/any/documents
```

3. Wait (it will take a while)

Go look for the file reading_notes.txt

A sample result from running this code on the repository itself is in reading_notes.txt
