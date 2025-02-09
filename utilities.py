import pandas as pd
from sklearn.model_selection import train_test_split
import torch
from torch import nn

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from transformers import AutoTokenizer, AutoModelForMaskedLM
from transformers import RobertaModel, RobertaTokenizer, TrainingArguments, Trainer, RobertaConfig


import os
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl

import re
import numpy as np 

from sklearn.metrics import accuracy_score, precision_recall_fscore_support, multilabel_confusion_matrix


if torch.backends.mps.is_available():
    device = torch.device('mps') # for m series mac 
elif torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

def show_tensorBoard_data(log_dir_list = ["./logs/pretraining/roberta-base_2025-01-29_13-50-06", "./logs/pretraining/roberta-base_2025-01-29_10-39-30"], 
                            plot_name_list = ["RoBERTa-base", "RoBERTa-large"],  
                            metric = "loss",
                            mode="eval", 
                            title = "Pretraining loss",
                            font = "Times New Roman", 
                            line_styles = ["-", "--", "-.", ":"],
                            line_width = 6,
                            axis_width = 2,
                            title_font_size = 40,
                            ticks_font_size = 20,
                            axis_label_font_size = 30, 
                            legend_size = 25):
    
    fig, ax = plt.subplots()

    for i, log_dir in enumerate(log_dir_list):
        event_accumulator = EventAccumulator(log_dir)
        event_accumulator.Reload()

        events = event_accumulator.Scalars(f"{mode}/{metric.lower()}")
        
        x = [x.step for x in events]
        y = [x.value for x in events]

        df = pd.DataFrame({"step": x, metric: y})

        ax.plot(df["step"].values[:-1], df[metric].values[:-1], label = plot_name_list[i], linestyle = line_styles[i], linewidth = line_width)


    # change the font style and size of the legend 
    mpl.rc('font', family = font)
    

    # change axe thickness 
    # change all spines
    for axis in ['top','bottom','left','right']:
        ax.spines[axis].set_linewidth(axis_width)

    # increase tick width
    ax.tick_params(width=axis_width)

    plt.xlabel("Steps", fontname = font, fontsize = axis_label_font_size)
    plt.ylabel(metric, fontname = font, fontsize = axis_label_font_size)
    plt.xticks(fontsize = ticks_font_size, fontname = font)
    plt.yticks(fontsize = ticks_font_size, fontname = font)
    plt.ylim(top = 3)
    plt.title(title, fontname = font, fontsize = title_font_size)
    plt.legend(prop={"size" : legend_size})
    plt.grid(linewidth = axis_width)
    plt.show()

    return 

def load_and_split_dataset(dataset_path, split_ratio=0.8):
    dataset = pd.read_csv(dataset_path)
    dataset = dataset.dropna()
    dataset = dataset.reset_index(drop=True)
    dataset = clean_dataset(dataset)
    print(dataset.head())
    
    if (split_ratio == 1.0):
        training_set = dataset.reset_index(drop=True)
        return training_set , []
    
    training_set, validation_set = train_test_split(dataset, test_size=float(1.0-split_ratio), random_state=123)
    
    training_set = training_set.reset_index(drop=True)
    validation_set = validation_set.reset_index(drop=True)
    
    return training_set, validation_set

def tokenize_dataset_for_pretraining(dataset, tokenizer): 
    
    tokenized = [tokenizer(text) for text in dataset["text"]]
    
    return tokenized 
def clean_text(text):
    # Convert to lowercase
    text = text.lower()
    # Remove hyperlinks
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    # Remove punctuation
    text = re.sub(r'[^\w\s]', '', text)
    return text

def clean_dataset(dataset):
    if 'text' in dataset.columns:
        dataset['text'] = dataset['text'].apply(clean_text)
    else:
        raise ValueError("Dataset must contain a 'text' column.")
    
    return dataset

def tokenize_dataset(dataset, tokenizer, max_len):
    targets = torch.tensor(
        list(zip(
            dataset["anger"],
            dataset["fear"],
            dataset["joy"],
            dataset["sadness"],
            dataset["surprise"]
        )),
        dtype=torch.float
    )

    print(dataset["text"])
    
    tokenized = [
        _prepare_data(text, label, tokenizer,max_len) for text, label in zip(dataset["text"], targets)
    ]
    
    return tokenized

def _prepare_data(text, label,tokenizer,max_len):
    tokenized = tokenizer(text, truncation=True, add_special_tokens=True,padding='max_length',
                          max_length=max_len,return_token_type_ids=True, return_tensors="pt")
    tokenized['labels'] = label
    tokenized['input_ids'] = tokenized['input_ids'].squeeze(0)
    tokenized['attention_mask'] = tokenized['attention_mask'].squeeze(0)
    tokenized['token_type_ids'] = tokenized['token_type_ids'].squeeze(0)
    return tokenized


import torch
from transformers import RobertaConfig, AutoModel
import torch
from transformers import RobertaConfig, RobertaModel  # oder das entsprechende Model

class CustomClassifier(torch.nn.Module):
    def __init__(
        self, 
        model_name, 
        model_type, 
        classifier_size, 
        dropout_rate=0.3, 
        num_classes=5, 
        head_type="fc", 
        attention_dim=64,
        classification_layers_cnt=2,
        num_attention_heads=1
    ):
        super(CustomClassifier, self).__init__()
        
        # Roberta-Konfiguration und Basismodell laden
        self.config = RobertaConfig.from_pretrained(model_name)
        self.l1 = model_type.from_pretrained(model_name)
        self.head_type = head_type.lower()

        # **FC Head**
        if self.head_type == "fc":
            layers = []
            # Erste Schicht: LazyLinear, ReLU und Dropout
            layers.append(torch.nn.LazyLinear(classifier_size))
            layers.append(torch.nn.ReLU())
            layers.append(torch.nn.Dropout(dropout_rate))
            
            # Zusätzliche Schichten, falls gewünscht
            for _ in range(classification_layers_cnt - 2):
                layers.append(torch.nn.Linear(classifier_size, classifier_size))
                layers.append(torch.nn.ReLU())
                layers.append(torch.nn.Dropout(dropout_rate))
            
            # Finale Ausgabeschicht
            layers.append(torch.nn.Linear(classifier_size, num_classes))
            self.classification_layers = torch.nn.Sequential(*layers)

        # **Self-Attention-Based Head**
        elif self.head_type == "attention":
            self.projection = torch.nn.Linear(self.config.hidden_size, attention_dim)
            self.self_attention = torch.nn.MultiheadAttention(
                embed_dim=attention_dim, 
                num_heads=num_attention_heads, 
                batch_first=True
            )
            self.attention_classifier = torch.nn.Linear(attention_dim, num_classes)
        else:
            raise ValueError("Invalid head_type. Choose between 'fc' and 'attention'.")

        self.sigmoid = torch.nn.Sigmoid()

    def forward(
        self, 
        labels=None, 
        input_ids=None, 
        attention_mask=None, 
        token_type_ids=None, 
        inputs_embeds=None, 
        output_attentions=None, 
        output_hidden_states=None, 
        return_dict=None
    ):
        # Übergabe der Eingaben an das Basismodell
        if inputs_embeds is not None:
            output_1 = self.l1(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict
            )
        else:
            output_1 = self.l1(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict
            )

        hidden_state = output_1[0]  # Sequenzoutput (batch_size, seq_len, hidden_size)

        if self.head_type == "fc":
            # CLS-Token extrahieren (erste Tokenposition)
            pooler = hidden_state[:, 0]
            # Durch den FC-Head leiten
            output = self.classification_layers(pooler)
            
        elif self.head_type == "attention":
            # Projektion und Self-Attention
            projected_tokens = self.projection(hidden_state)
            attn_output, _ = self.self_attention(
                projected_tokens, projected_tokens, projected_tokens,
                key_padding_mask=(~attention_mask.bool() if attention_mask is not None else None)
            )
            pooled_output = torch.mean(attn_output, dim=1)  # Mittelwert über die Token
            output = self.attention_classifier(pooled_output)

        # Berechnung des Loss, falls Labels vorhanden sind
        if labels is not None:
            loss_fn = torch.nn.BCEWithLogitsLoss()
            loss = loss_fn(output, labels)
            return loss, output

        return output





class T5Classifier(CustomClassifier):
    
    def forward(self, labels, input_ids, attention_mask, token_type_ids):
        output_1 = self.l1(input_ids=input_ids, attention_mask=attention_mask, decoder_input_ids=token_type_ids)
        hidden_state = output_1[0]
        pooler = hidden_state[:, 0]
        pooler = self.pre_classifier(pooler)
        pooler = torch.nn.ReLU()(pooler)
        pooler = self.dropout(pooler)
        output = self.classifier(pooler)
        loss = nn.BCEWithLogitsLoss()(output, labels)
        #print(f"{loss.shape=} {output.shape=}")
        return loss, output
    
class CustomTrainer(Trainer):
    
    #def compute_loss(self, model, inputs, return_outputs=False,num_items_in_batch=None):
    #    #labels = inputs.pop("labels")
    #    labels = inputs.get("labels")
    #    #print(inputs["input_ids"].shape)
    #    outputs = model(**inputs)
    #    loss = nn.BCEWithLogitsLoss()(outputs, labels)
    #    outputs  = torch.cat((torch.zeros(1,outputs.size(1),device=device), outputs), dim=0) ## Add a column of zeros to the beginning of the tensor, Necessary because of a weird implementation of the Trainer class
    #    #print(f"{outputs.shape=}")
    #    return (loss, outputs) if return_outputs else loss
    #
    
    def save_model(self, output_dir: str = None, _internal_call=False):
        """
        Override the default `save_model` method to ensure proper saving of shared memory tensors.
        """
        if not output_dir:
            output_dir = self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)
        print(f"{output_dir=}")
        model_path = os.path.join(output_dir, "pytorch_model.bin")
        torch.save(self.model.state_dict(), model_path)
        if self.tokenizer is not None:
            torch.save(self.tokenizer, output_dir)
    
def compute_metrics(p):
        pred, labels = p
        pred = torch.tensor(pred)
        labels = torch.tensor(labels)
        pred = torch.sigmoid(pred)
        pred = torch.round(pred)
        accuracy = ((pred == labels).sum(axis=1) == 5).sum() / len(labels)
        #print(f"{accuracy=}")
        return {
             'accuracy': accuracy
        }

def compute_metrics_f1(p): 
    predictions, labels = p
    #print(f"{predictions=} {labels=}")
    labels = torch.tensor(labels)
    predictions = torch.tensor(predictions)
    predictions = torch.sigmoid(predictions)
    predictions = torch.round(predictions)

    # can raise a warning when there are no positive labels or predictions for one 
    # emotion in the passed data 
    precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average='weighted')
    labels = np.array(labels)
    predictions = np.array(predictions)
    acc = accuracy_score(labels, predictions)
    return {'accuracy': acc, 'f1': f1, 'precision': precision, 'recall': recall}
        
        
def get_save_file_path(model_name, category):
    """
    Generates the save file path from the current timestamp. 

    Args:
        model_name: Name of the model to be saved. 
        category: The category of the model to save. 1 stands for 'pretraining' and 2 stands for 'classification'. 
    """
    from datetime import datetime

    # Get the current date and time
    now = datetime.now()

    # Format the date and time
    formatted_date_time = now.strftime("%Y-%m-%d_%H-%M-%S")

    # choose category: 1 = pretraining, 2 = classification 
    if category == 1: 
        category = "pretraining"
    elif category == 2: 
        category = "classification"
    else: 
        print(f"{category} is no valid category. Valid categories are 1 for pretraining and 2 for classification.")

    model_name = f"{model_name}_{formatted_date_time}"
        
    return f"./results/{category}/{model_name}", model_name


def remove_all_files_and_folders_except_best_model(folder_path):
    import os
    import shutil
    best_model = None

    for file in os.listdir(folder_path):
        if file == "best_model.pth":
            best_model = file
        elif file == "results.yaml": 
            pass
        elif file == "best_model": 
            pass
        else:
            try:
                shutil.rmtree(f"{folder_path}/{file}")
            except:
                os.remove(f"{folder_path}/{file}")
    return best_model

def plot_confusion_matrix(predictions, save_path = None, file_name = None, show = True): 

    preds = predictions.predictions
    preds = torch.tensor(preds,device=device)
    preds = torch.sigmoid(preds)
    preds = torch.round(preds)
    ids = torch.tensor(predictions.label_ids,device=device)

    label_map = {
    'LABEL_0': 'anger',
    'LABEL_1': 'fear',
    'LABEL_2': 'joy',
    'LABEL_3': 'sadness',
    'LABEL_4': 'surprise'
    }

    cm = multilabel_confusion_matrix(ids, preds)

    # label_map to labels
    labels = [label_map[f'LABEL_{i}'] for i in range(len(label_map))]

    # Confusion Matrix
    plt.figure(figsize=(20, 10))
    
    for i, label in enumerate(labels): 
        plt.subplot(2, 3, i + 1)
        sns.heatmap(cm[i], annot=True, fmt='d', cmap='Blues', xticklabels=["False", "True"], yticklabels=["False", "True"])
        plt.xlabel('Predicted value')
        plt.ylabel('True value')
        plt.title(f"Confusion Matrix for '{label}' class")


    if save_path is not None and file_name is not None: 
        plt.savefig(f"{save_path}/{file_name}.png", bbox_inches='tight')

    if show: 
        plt.show()
    
if __name__ == "__main__": 

    show_tensorBoard_data(log_dir_list = ["./results/final_eval/pretraining_logs/bert-base-uncased_pre_fine",
                                          "./results/final_eval/pretraining_logs/bert-large-uncased_pre_fine",
                                          "./results/final_eval/pretraining_logs/roberta-base_pre_fine",
                                          "./results/final_eval/pretraining_logs/roberta-large_pre_fine"],
                            plot_name_list = ["BERT-base-uncased", "BERT-large", "RoBERTa-base", "RoBERTa-large"],
                            mode = "train", 
                            metric = "Loss",
                            title = "Trainings Loss")