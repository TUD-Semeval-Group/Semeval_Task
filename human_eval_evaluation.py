import pandas as pd
from sklearn.metrics import f1_score

def evaluate_classification(ground_truth_file, classified_file):
    # Load CSV files
    ground_truth = pd.read_csv(ground_truth_file)
    classified = pd.read_csv(classified_file)

    # Normalize column names (strip spaces, lowercase)
    ground_truth.columns = ground_truth.columns.str.strip().str.lower()
    classified.columns = classified.columns.str.strip().str.lower()

    # Print column names for debugging
    print("Ground truth columns:", ground_truth.columns.tolist())
    print("Classified columns:", classified.columns.tolist())

    # Merge files on 'id'
    merged = ground_truth.merge(classified, on='id', suffixes=('_true', '_pred'))

    print(merged)
    
    # Print merged columns for verification
    print("Merged columns:", merged.columns.tolist())

    # Define label columns based on normalized column names
    label_cols = ['anger', 'fear', 'joy', 'sadness', 'surprise']
    y_true = merged[[col + '_true' for col in label_cols]].values
    y_pred = merged[[col + '_pred' for col in label_cols]].values

    # Calculate accuracy (all labels must match)
    correct_predictions = (y_true == y_pred).all(axis=1)
    accuracy = correct_predictions.mean()

    # Calculate F1 score
    f1 = f1_score(y_true, y_pred, average='macro')

    return accuracy, f1

# Example usage
accuracy, f1 = evaluate_classification('data/track_a/dev/eng.csv', 'human_classification_robert.csv')
print(f"Accuracy: {accuracy:.2%}")
print(f"F1 Score: {f1:.4f}")
