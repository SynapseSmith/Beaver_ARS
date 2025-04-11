from transformers import pipeline
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

model_path = '/home/user09/beaver/data/shared_files/241205_menu/custom_dataset/bert-topic-cls'  # Change this to your model's path or a Hugging Face model name

# Load the pipeline for text classification
classifier = pipeline("text-classification", model=model_path, tokenizer=model_path)

# Example sentences
sentences = [
    "The stock market is reaching new heights.",
    "The new sports car has been unveiled at the auto show.",
    "The tech company announced its latest gadget yesterday."
]

# Make predictions
predictions = classifier(sentences)

# Print the predictions using the label map
for sentence, prediction in zip(sentences, predictions):
    # Map the predicted label to the actual class name
    # class_name = label_map[prediction['label']]
    print(f"Sentence: '{sentence}'")
    print(f"Predicted Label: '{prediction['label']}' with score {prediction['score']:.4f}\n")