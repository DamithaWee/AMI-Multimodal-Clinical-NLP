import joblib
import config
import os

def save_model(mode_name, model, path):
    folder_path = os.path.join(config.results_path, 'models', path)
    os.makedirs(folder_path, exist_ok=True)
    path = os.path.join(folder_path, f'{mode_name}.joblib')
    joblib.dump(model, path)
    print(f"Model saved successfully at: {path}")


def save_report(model_name, report, path, auc=None, f1=None):
    folder_path = os.path.join(config.results_path, 'metrics', path)
    os.makedirs(folder_path, exist_ok=True)
    file_path = os.path.join(folder_path, f'{model_name}.txt')
    
    with open(file_path, 'w') as f:
        if auc is not None:
            f.write(f"Area Under the Curve (AUC): {auc:.4f}\n")
        if f1 is not None:
            f.write(f"F1-Score: {f1:.4f}\n")
        f.write("\nClassification Report:\n")
        f.write(report)
        
    print(f"Report saved successfully at: {file_path}")


def save_figure(model_name, fig, path, filename_suffix='confusion_matrix'):
    folder_path = os.path.join(config.results_path, 'figures', path)
    os.makedirs(folder_path, exist_ok=True)
    file_path = os.path.join(folder_path, f'{model_name}_{filename_suffix}.png')
    
    fig.savefig(file_path, dpi=300, bbox_inches='tight')
    print(f"Figure saved successfully at: {file_path}")


def tokenize_function(texts):
    return tokenizer(texts, padding="max_length", truncation=True, max_length=MAX_LEN)