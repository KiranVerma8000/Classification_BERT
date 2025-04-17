import torch
from transformers import BertTokenizer, BertForSequenceClassification
import pandas as pd
import json
from tqdm import tqdm
import time
path = '/Users/mitul/Documents/study/sem 4/DSSE/Assignmets/Assignments 3/assignment3_repo/ds4se3-group6'
inputfilepath = '/Users/mitul/Documents/study/sem 4/DSSE/Assignmets/Assignments 3/assignment3_repo/ds4se3-group6/datasets/split_dataset/k-cross/test_fold/test_fold_7.xlsx'
class CustomBertForSequenceClassification(BertForSequenceClassification):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = 4
        self.classifier = torch.nn.Linear(config.hidden_size, self.num_labels)

# Start timing
start_time = time.time()

# Step 1: Load the BERT Model with 4 output classes
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model_path = f'{path}/utils/model/bert_model_fold_7.bin'  # Updated path

# Use the custom model class
model = CustomBertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=4)
model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
model.eval()

# Step 2: Load the Excel File
df = pd.read_excel(inputfilepath)

# Ensure all entries in 'preprocessed_Text' are strings
df['preprocessed_Text'] = df['preprocessed_Text'].astype(str)

# Measure time for preprocessing and prediction
preprocess_and_predict_start = time.time()

# Step 3: Preprocess the Data
def preprocess_text(text):
    return tokenizer(text, padding=True, truncation=True, return_tensors="pt")

# Step 4: Predict Architectural Posts
def predict_architecture(post):
    inputs = preprocess_text(post)
    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits
    predicted_class = torch.argmax(logits, dim=1).item()
    return predicted_class

# Apply the prediction to all posts with a progress bar
df['predicted_class'] = [predict_architecture(text) for text in tqdm(df['preprocessed_Text'], desc="Predicting")]

# Step 5: Classify Types of Architectural Posts (if applicable)
def predict_architecture_type(row):
    if row['predicted_class'] == 1:
        if row['Synthesis'] == 1:
            return 'Synthesis'
        elif row['Evaluation'] == 1:
            return 'Evaluation'
        elif row['Analysis'] == 1:
            return 'Analysis'
    return 'Non-Architectural'

df['predicted_subcategory'] = df.apply(predict_architecture_type, axis=1)

# Measure end of preprocessing and prediction time
preprocess_and_predict_end = time.time()

# Step 6: Analyze Results
# Count the number of architectural posts
num_arch_posts = int(df['predicted_class'].sum())  # Convert to int to ensure JSON serialization

# Get the most common types of architectural posts
arch_post_types = df[df['predicted_class'] == 1]['predicted_subcategory'].value_counts().to_dict()

# Determine correctness of predictions
df['correct_prediction'] = df['Post_Type'] == df['predicted_class']

# Determine subcategory prediction correctness
df['subcategory_correct'] = ((df['Post_Type'] == 1) & 
                             ((df['Synthesis'] == 1) & (df['predicted_subcategory'] == 'Synthesis') |
                             (df['Evaluation'] == 1) & (df['predicted_subcategory'] == 'Evaluation') |
                             (df['Analysis'] == 1) & (df['predicted_subcategory'] == 'Analysis')))

# Count correct and incorrect predictions
num_correct_predictions = df['correct_prediction'].sum()
num_incorrect_predictions = len(df) - num_correct_predictions

# Count subcategory prediction failures
num_subcategory_failures = df[(df['Post_Type'] == 1) & (~df['subcategory_correct'])].shape[0]

# Step 7: Store Predicted Data in Excel
output_excel_file = f'{path}/datasets/model_prediction/predicted_stackoverflow_posts_7_test.xlsx'
df.to_excel(output_excel_file, index=False)

# Step 8: Store Overall Model Results in JSON
# Prepare the result data
result_data = {
    "number_of_architectural_posts": num_arch_posts,
    "most_common_architectural_post_types": {k: int(v) for k, v in arch_post_types.items()},  # Convert to int
    "number_of_correct_predictions": int(num_correct_predictions),  # Convert to int
    "number_of_incorrect_predictions": int(num_incorrect_predictions),  # Convert to int
    "subcategory_prediction_failures": int(num_subcategory_failures)  # Convert to int
}

# Save the result data to a JSON file
output_json_file = f'{path}/datasets/model_prediction/model_results_test_7.json'
with open(output_json_file, 'w') as json_file:
    json.dump(result_data, json_file, indent=4)

# End timing
end_time = time.time()

print(f"Predicted data saved to {output_excel_file}")
print(f"Model results saved to {output_json_file}")
print(f"Total time taken: {end_time - start_time:.2f} seconds")
print(f"Preprocessing and prediction time: {preprocess_and_predict_end - preprocess_and_predict_start:.2f} seconds")
