# Quora Question Answering Model

## Overview

This project aims to develop a state-of-the-art question-answering model leveraging the Quora Question Answer Dataset. The objective is to create an AI system capable of understanding and generating accurate responses to a variety of user queries, mimicking human-like interaction.

## Dataset

The dataset used for this project is the [Quora Question Answer Dataset](https://huggingface.co/datasets/toughdata/quora-question-answer-dataset) available on Hugging Face. The dataset contains questions and their corresponding answers.

## Models

The models used for this project are:
- GPT-2
- BERT
- T5

These models are from the Hugging Face `transformers` library.

## Installation

To run this project, you need to install the following libraries:

```bash
pip install nltk
pip install datasets
pip install transformers[torch]
pip install tokenizers
pip install rouge_score
pip install sentencepiece
pip install huggingface_hub
```

## Data Preprocessing

The dataset is preprocessed by tokenizing the questions and answers. The preprocessing steps include:

- Adding a prefix to each question to form a complete input sentence.
- Tokenizing the input sentences and answers.
- Padding and truncating the sequences to a fixed length.

## Training

The models are fine-tuned on the Quora Question Answer dataset using the following training parameters:

- Learning Rate: 5e-5
- Batch Size: 2
- Number of Epochs: 3

The training is performed using the Trainer class from the `transformers` library.

## Evaluation

The models' performance is evaluated using the ROUGE metric. The `datasets` library's `load_metric` function is used to compute the ROUGE scores for the generated answers compared to the reference answers.

## Usage

To use the fine-tuned models for generating answers to new questions, you can run the following code:

### GPT-2

```python
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import torch

# Load the fine-tuned model
finetuned_model = GPT2LMHeadModel.from_pretrained("./results/gpt2")
tokenizer = GPT2Tokenizer.from_pretrained("./results/gpt2")

# Generate an answer to a sample question
my_question = "What do you think about the benefit of Artificial Intelligence?"
inputs = "Please answer this question: " + my_question
inputs = tokenizer(inputs, return_tensors="pt")
outputs = finetuned_model.generate(**inputs)
answer = tokenizer.decode(outputs[0], skip_special_tokens=True)

print(answer)
```
### BERT
```python
from transformers import BertTokenizer, BertForQuestionAnswering
import torch

# Load the fine-tuned model
finetuned_model = BertForQuestionAnswering.from_pretrained("./results/bert")
tokenizer = BertTokenizer.from_pretrained("./results/bert")

# Generate an answer to a sample question
my_question = "What do you think about the benefit of Artificial Intelligence?"
inputs = tokenizer.encode_plus(my_question, return_tensors="pt")
outputs = finetuned_model(**inputs)
answer_start = torch.argmax(outputs.start_logits)
answer_end = torch.argmax(outputs.end_logits) + 1
answer = tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(inputs.input_ids[0][answer_start:answer_end]))

print(answer)
```
### T5
```python
from transformers import T5Tokenizer, T5ForConditionalGeneration
import torch

# Load the fine-tuned model
finetuned_model = T5ForConditionalGeneration.from_pretrained("./results/t5")
tokenizer = T5Tokenizer.from_pretrained("./results/t5")

# Generate an answer to a sample question
my_question = "What do you think about the benefit of Artificial Intelligence?"
inputs = "Please answer this question: " + my_question
inputs = tokenizer(inputs, return_tensors="pt")
outputs = finetuned_model.generate(**inputs)
answer = tokenizer.decode(outputs[0], skip_special_tokens=True)

print(answer)
```

## Results
The models' performance is evaluated using the ROUGE metric, which provides scores for ROUGE-1, ROUGE-2, and ROUGE-L. These scores help assess the quality of the generated answers in terms of their similarity to the reference answers.

## License
This project is licensed under the MIT License. See the LICENSE file for details.

## Acknowledgements
- Hugging Face for providing the transformers library and the Quora Question Answer Dataset.
- OpenAI for developing the GPT-2 model.
- The contributors of the nltk, datasets, rouge_score, and sentencepiece libraries.
