"""Runs ML workflow to fine-tune BERT model on a dataset for sentiment analysis.

AI Pipeline:
Create container image
Download dataset
Download model weights
Visualize dataset
Train model
Evaluate model
Save model to Hugging Face hub
Predict sentiment from text

Note: this workflow can be broken into modular python files for better organization.
"""

# %% import libraries & Create container image
# ---------------------------
from pathlib import Path
import os
from flytekit import (Deck, ImageSpec, Resources, Secret, current_context,
                      task, workflow)
from flytekit.types.directory import FlyteDirectory
from flytekit.types.file import FlyteFile
from transformers import BertForSequenceClassification

image = ImageSpec(
    packages=[
        "union==0.1.64",
        "transformers==4.39.1",
        "datasets==2.18.0",
        "matplotlib==3.8.3",
        "torch==2.0.1",
        "accelerate==0.27.2",
        "scikit-learn==1.5.1"
    ],
)

# %% download dataset
# ---------------------------
@task(
    container_image=image,
    cache=True,
    cache_version="v3",
    requests=Resources(cpu="2", mem="2Gi"),
)
def download_dataset() -> FlyteDirectory:
    from datasets import load_dataset
   
    working_dir = Path(current_context().working_directory)
    dataset_cache_dir = working_dir / "dataset_cache"

    load_dataset("imdb", cache_dir=dataset_cache_dir)

    return dataset_cache_dir

# %% download model
# ---------------------------
@task(
    container_image=image,
    cache=True,
    cache_version="v3",
    requests=Resources(cpu="2", mem="2Gi"),
)
def download_model(model: str) -> FlyteDirectory:
    from transformers import AutoTokenizer, AutoModelForSequenceClassification

    working_dir = Path(current_context().working_directory)
    model_cache_dir = working_dir / "model_cache"

    AutoTokenizer.from_pretrained(model, cache_dir=model_cache_dir)
    AutoModelForSequenceClassification.from_pretrained(model, cache_dir=model_cache_dir)
    return model_cache_dir


# %% visualize data
# ---------------------------
@task(
    container_image=image,
    enable_deck=True,
    requests=Resources(cpu="2", mem="2Gi"),
)
def visualize_data(dataset_cache_dir: FlyteDirectory):
    from datasets import load_dataset
    import matplotlib.pyplot as plt
    import pandas as pd
    import base64
    from textwrap import dedent

    ctx = current_context()

    # Load the dataset
    dataset = load_dataset("imdb", cache_dir=dataset_cache_dir)
    train_df = pd.DataFrame(dataset['train'])
    test_df = pd.DataFrame(dataset['test'])

    # Create the deck for visualization
    deck = Deck("Dataset Analysis")

    # Sample one review from each class (positive and negative) from the training and test datasets
    train_positive_review = train_df[train_df['label'] == 1].iloc[0]['text']
    train_negative_review = train_df[train_df['label'] == 0].iloc[0]['text']
    test_positive_review = test_df[test_df['label'] == 1].iloc[0]['text']
    test_negative_review = test_df[test_df['label'] == 0].iloc[0]['text']

    # Visualize label distribution for training data
    plt.figure(figsize=(10, 5))
    train_df['label'].value_counts().plot(kind='bar', color='skyblue')
    plt.title('Train Data Label Distribution')
    plt.xlabel('Label')
    plt.ylabel('Count')
    plt.tight_layout()
    train_label_dist_path = "/tmp/train_label_distribution.png"
    plt.savefig(train_label_dist_path)
    plt.close()

    # Visualize label distribution for test data
    plt.figure(figsize=(10, 5))
    test_df['label'].value_counts().plot(kind='bar', color='lightgreen')
    plt.title('Test Data Label Distribution')
    plt.xlabel('Label')
    plt.ylabel('Count')
    plt.tight_layout()
    test_label_dist_path = "/tmp/test_label_distribution.png"
    plt.savefig(test_label_dist_path)
    plt.close()

    # Convert images to base64 and embed in HTML
    def image_to_base64(image_path):
        with open(image_path, "rb") as img_file:
            return base64.b64encode(img_file.read()).decode("utf-8")

    train_image_base64 = image_to_base64(train_label_dist_path)
    test_image_base64 = image_to_base64(test_label_dist_path)

    # HTML report with styled text, tables, and embedded images
    html_report = dedent(f"""
    <div style="font-family: Arial, sans-serif; line-height: 1.6;">
        <h2 style="color: #2C3E50;">Dataset Analysis</h2>
        
        <h3 style="color: #2980B9;">Training Data Summary</h3>
        <p>Below is a summary of the training dataset including the distribution of labels.</p>
        Shape: {train_df.shape} <br>
        Columns: {train_df.columns} <br>
        Label Distribution: {train_df['label'].value_counts()} <br>
        
        <h3 style="color: #2980B9;">Sample Reviews from Training Data</h3>
        <p><strong>Positive Review:</strong> {train_positive_review}</p>
        <p><strong>Negative Review:</strong> {train_negative_review}</p>

        <h3 style="color: #2980B9;">Training Data Label Distribution</h3>
        <p>The following bar chart shows the distribution of labels in the training dataset:</p>
        <img src="data:image/png;base64,{train_image_base64}" alt="Train Data Label Distribution" width="600">

        <h3 style="color: #2980B9;">Test Data Summary</h3>
        <p>Below is a summary of the test dataset including the distribution of labels.</p>
        Shape: {test_df.shape} <br>
        Columns: {test_df.columns} <br>
        Label Distribution: {test_df['label'].value_counts()} <br>
        
        <h3 style="color: #2980B9;">Sample Reviews from Test Data</h3>
        <p><strong>Positive Review:</strong> {test_positive_review}</p>
        <p><strong>Negative Review:</strong> {test_negative_review}</p>

        <h3 style="color: #2980B9;">Test Data Label Distribution</h3>
        <p>The following bar chart shows the distribution of labels in the test dataset:</p>
        <img src="data:image/png;base64,{test_image_base64}" alt="Test Data Label Distribution" width="600">
    </div>
    """)

    # Append HTML content to the deck
    deck.append(html_report)

    # Insert the deck into the context
    ctx.decks.insert(0, deck)


# %% train model
# ---------------------------
@task(
    container_image=image,
    requests=Resources(cpu="4", mem="12Gi", gpu="1"),
)
def train_model(model_name: str, 
                dataset_cache_dir: FlyteDirectory,
                model_cache_dir: FlyteDirectory,
                epochs: int = 3
    ) -> BertForSequenceClassification:
    from datasets import load_dataset
    import numpy as np
    from transformers import(
        AutoTokenizer,
        AutoModelForSequenceClassification,
        TrainingArguments,
        Trainer,
    )
    ctx = current_context()

    working_dir = Path(ctx.working_directory)
    train_dir = working_dir / "models"

    dataset = load_dataset("imdb", cache_dir=dataset_cache_dir)
    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=model_cache_dir)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=2,
        id2label={0: "NEGATIVE", 1: "POSITIVE"},
        label2id={"NEGATIVE": 0, "POSITIVE": 1},
        cache_dir=model_cache_dir,
    )

    def tokenizer_function(examples):
        return tokenizer(examples["text"], padding="max_length", truncation=True)

    # Use a small subset such that finetuning completes
    small_train_dataset = (
        dataset["train"].shuffle(seed=42).select(range(500)).map(tokenizer_function)
    )
    small_eval_dataset = (
        dataset["test"].shuffle(seed=42).select(range(100)).map(tokenizer_function)
    )

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        return {"accuracy": np.mean(predictions == labels)}
    
    training_args = TrainingArguments(
        output_dir=train_dir,
        evaluation_strategy="epoch",
        num_train_epochs=epochs,
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=small_train_dataset,
        eval_dataset=small_eval_dataset,
        compute_metrics=compute_metrics,
    )
    trainer.train()

    return model

# %% Evaluate model
# ---------------------------
@task(
    container_image=image,
    enable_deck=True,
    requests=Resources(cpu="2", mem="12Gi", gpu="1"),
)
def evaluate_model(
    model: BertForSequenceClassification,
    dataset_cache_dir: FlyteDirectory,
    model_cache_dir: FlyteDirectory,
) -> dict:
    from datasets import load_dataset
    from transformers import AutoTokenizer, Trainer, TrainingArguments
    from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
    import numpy as np
    import torch

    # Load the test dataset and tokenizer
    dataset = load_dataset("imdb", cache_dir=dataset_cache_dir)
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased", cache_dir=model_cache_dir)

    def tokenize_function(examples):
        return tokenizer(examples["text"], padding="max_length", truncation=True)

    # Use a small subset (200 examples) for evaluation
    eval_dataset = dataset["test"].shuffle(seed=42).select(range(200)).map(tokenize_function)

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        accuracy = accuracy_score(labels, predictions)
        f1 = f1_score(labels, predictions, average="weighted")
        precision = precision_score(labels, predictions, average="weighted")
        recall = recall_score(labels, predictions, average="weighted")
        return {
            "accuracy": accuracy,
            "f1": f1,
            "precision": precision,
            "recall": recall,
        }

    # Initialize Trainer for evaluation
    training_args = TrainingArguments(
        output_dir=".", 
        per_device_eval_batch_size=16, 
        dataloader_drop_last=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics,
    )

    # Run evaluation
    eval_results = trainer.evaluate()

    print(f"Evaluation results on 100 examples: {eval_results}")

    return eval_results


# %% save model
# ---------------------------
@task(
    container_image=image,
    requests=Resources(cpu="2", mem="2Gi"),
    secret_requests=[Secret(group=None, key="hf_token")],
)
def save_model(model: BertForSequenceClassification, repo_name: str) -> str:
    from huggingface_hub import HfApi

    ctx = current_context()
    
    working_dir = Path(ctx.working_directory)
    model_path = working_dir / "model"
    model.save_pretrained(model_path)

    # Ensure the model files are saved
    model_files = ["model.safetensors", "config.json"]
    for file_name in model_files:
        file_path = model_path / file_name
        if not file_path.exists():
            raise FileNotFoundError(f"Model file not found: {file_path}")
    
    # dir list for debug
    dir_list = os.listdir(model_path)
    print("Files and directories in '", model_path, "' :")
    print(dir_list)


     # set hf_token from local or union secret
    hf_token = os.getenv("HF_TOKEN")
    if hf_token is None:
        # If HF_TOKEN is not found, attempt to get it from the Flyte secrets
        hf_token = ctx.secrets.get(key="hf_token")
        print("Using Hugging Face token from Union secrets.")
    else:
        print("Using Hugging Face token from env.")

    # Create a new repository (if it doesn't exist)
    api = HfApi()
    api.create_repo(repo_name, token=hf_token, exist_ok=True)

    # Upload the model to the HF repository
    # Upload each model file to the HF repository
    for file_name in model_files:
        file_path = model_path / file_name
        api.upload_file(
            path_or_fileobj=str(file_path),
            path_in_repo=file_name,
            repo_id=repo_name,
            commit_message=f"Upload {file_name}",
            repo_type=None,
            token=hf_token
        )
    return f"Model uploaded to Hugging Face Hub: {repo_name}"

# %% Predict sentiment
# ---------------------------
@task(
    container_image=image,
    requests=Resources(cpu="2", mem="2Gi", gpu="1"),
    retries=2,
)
def predict_sentiment(model: BertForSequenceClassification, text: str, model_cache_dir: FlyteDirectory) -> dict:
    from transformers import AutoTokenizer, pipeline

    # Load the tokenizer
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased", cache_dir=model_cache_dir)

    # Initialize the pipeline for sentiment analysis
    nlp_pipeline = pipeline("text-classification", model=model, tokenizer=tokenizer)

    # Perform the prediction
    prediction = nlp_pipeline(text)

    return prediction[0] 

# %% Workflow
@workflow
def bert_ft(model: str = "bert-base-uncased",
            repo_name: str = "my-model",
            test_text: str = "I love this movie!",
            epochs: int = 3
            ) -> (BertForSequenceClassification, FlyteDirectory, dict, dict):
    
    dataset_cache_dir = download_dataset()
    model_cache_dir = download_model(model)
    visualize_data(dataset_cache_dir=dataset_cache_dir)
    model = train_model(model_name=model, 
                        dataset_cache_dir=dataset_cache_dir, 
                        model_cache_dir=model_cache_dir,
                        epochs=epochs)
    eval_results = evaluate_model(model=model, 
                                  dataset_cache_dir=dataset_cache_dir, 
                                  model_cache_dir=model_cache_dir)
    save_model(model=model, repo_name=repo_name)
    
    # Pass the model and other needed arguments directly to the predict_sentiment task
    prediction = predict_sentiment(model=model, text=test_text, model_cache_dir=model_cache_dir)
    
    # Return results as a dict
    return model, model_cache_dir, eval_results, prediction