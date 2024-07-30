# Quora Question Answering Model

## Overview

This project aims to develop a state-of-the-art question-answering model leveraging the Quora Question Answer Dataset. The objective is to create an AI system capable of understanding and generating accurate responses to a variety of user queries, mimicking human-like interaction.

## Dataset

The dataset used for this project is the [Quora Question Answer Dataset](https://huggingface.co/datasets/toughdata/quora-question-answer-dataset) available on Hugging Face. The dataset contains questions and their corresponding answers.

## Model

The model used for this project is GPT-2 from the Hugging Face `transformers` library. GPT-2 is a large-scale transformer-based language model trained by OpenAI.

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

1. Adding a prefix to each question to form a complete input sentence.
2. Tokenizing the input sentences and answers.
3. Padding and truncating the sequences to a fixed length.

## Training

The model is fine-tuned on the Quora Question Answer dataset using the following training parameters:

- **Learning Rate**: 5e-5
- **Batch Size**: 2
- **Number of Epochs**: 3

The training is performed using the `Trainer` class from the `transformers` library.

## Evaluation

The model's performance is evaluated using the ROUGE metric. The `datasets` library's `load_metric` function is used to compute the ROUGE scores for the generated answers compared to the reference answers.

## Usage

To use the fine-tuned model for generating answers to new questions, you can run the following code:

```python
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import torch

# Load the fine-tuned model
finetuned_model = GPT2LMHeadModel.from_pretrained("./results")
tokenizer = GPT2Tokenizer.from_pretrained("./results")

# Generate an answer to a sample question
my_question = "What do you think about the benefit of Artificial Intelligence?"
inputs = "Please answer this question: " + my_question
inputs = tokenizer(inputs, return_tensors="pt")
outputs = finetuned_model.generate(**inputs)
answer = tokenizer.decode(outputs[0], skip_special_tokens=True)

print(answer)
```
## Results

The model's performance is evaluated using the ROUGE metric, which provides scores for ROUGE-1, ROUGE-2, and ROUGE-L. These scores help assess the quality of the generated answers in terms of their similarity to the reference answers.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Acknowledgements

- [Hugging Face](https://huggingface.co/) for providing the `transformers` library and the Quora Question Answer Dataset.
- [OpenAI](https://www.openai.com/) for developing the GPT-2 model.
- The contributors of the `nltk`, `datasets`, `rouge_score`, and `sentencepiece` libraries.
