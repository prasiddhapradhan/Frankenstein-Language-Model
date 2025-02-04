# Frankenstein-Language-Model

Overview

This project is a fine-tuned language model based on DistilGPT-2 (or GPT-2) trained on a dataset for text generation. The goal is to create a more coherent and context-aware text generation model. The model is trained using the Hugging Face Transformers library and optimized for lower perplexity.

Features

Text Generation: Generates text based on given prompts.

Fine-Tuned on Custom Dataset: Improves over the base model for better coherence.

Low Perplexity (~27.88): Decent performance but room for improvement.

Uses Transformer-based Tokenization: Efficient text processing.

Installation

Ensure you have Python installed, then install dependencies:

pip install transformers datasets torch accelerate

Training the Model

Run the following script to train the model:

from transformers import TrainingArguments, Trainer

training_args = TrainingArguments(
    output_dir="./frankenstein_model",
    evaluation_strategy="epoch",
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    num_train_epochs=5,  # Adjust as needed
    save_strategy="epoch",
    logging_dir="./logs",
    logging_steps=100,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["validation"],
)

trainer.train()

Loading the Model for Inference

After training, load the model for text generation:

from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("./frankenstein_model")
tokenizer = AutoTokenizer.from_pretrained("distilgpt2")

input_text = "Once upon a time"
input_ids = tokenizer.encode(input_text, return_tensors="pt")

output = model.generate(input_ids, max_length=100)
print(tokenizer.decode(output[0], skip_special_tokens=True))

Troubleshooting

Issue 1: Padding Token Error

Add a padding token before training:

tokenizer.pad_token = tokenizer.eos_token

Issue 2: Missing Dependencies

Run:

pip install transformers[torch] accelerate

Issue 3: Model Not Recognized

Ensure the config.json file is present in ./frankenstein_model.

Try reloading with:

model = AutoModelForCausalLM.from_pretrained("distilgpt2")

Next Steps

Reduce Perplexity by tuning hyperparameters (epochs, batch size, learning rate).

Train on Larger Dataset for improved performance.

Experiment with GPT-2 instead of DistilGPT-2.

Author

Prasiddha Pradhan

License

MIT License
