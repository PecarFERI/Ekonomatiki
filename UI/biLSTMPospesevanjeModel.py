import torch
import torch.nn as nn
import torch.optim as optim
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import numpy as np
import csv
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from datetime import datetime, timedelta
import os
import pandas as pd
import gc
from sklearn.metrics import classification_report, confusion_matrix


class AccelerationEconomyModel(nn.Module):
    def __init__(self, input_size=1, hidden_size=64, num_layers=2, num_classes=6):  # Changed to 6 classes
        super(AccelerationEconomyModel, self).__init__()
        # Improved BiLSTM with proper dropout
        self.bilstm = nn.LSTM(input_size=input_size, 
                             hidden_size=hidden_size,
                             num_layers=num_layers, 
                             batch_first=True,
                             dropout=0.3 if num_layers > 1 else 0,  # Increased dropout
                             bidirectional=True)
        
        # Improved attention mechanism
        self.attention = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_size, 1),
            nn.Softmax(dim=1)
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.BatchNorm1d(hidden_size),  
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_size, num_classes)  
        )
        
        self._init_weights()
        
    def _init_weights(self):
        for name, param in self.named_parameters():
            if 'weight' in name and param.dim() > 1:
                nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                nn.init.zeros_(param)
        
    def forward(self, x, mask=None):
        # BiLSTM output
        lstm_out, (hn, cn) = self.bilstm(x)
        
        # Apply mask if provided
        if mask is not None:
            lstm_out = lstm_out * mask.unsqueeze(-1)

        # Attention mechanism
        attention_weights = self.attention(lstm_out)

        # Normalize attention weights with mask
        if mask is not None:
            attention_weights = attention_weights * mask.unsqueeze(-1)
            attention_weights = attention_weights / (attention_weights.sum(dim=1, keepdim=True) + 1e-8)

        # Context vector
        context_vector = torch.sum(attention_weights * lstm_out, dim=1)
        
        # Final classification
        out = self.classifier(context_vector)
        return out


examples_X = []
examples_y = []
model = None
sequence_length = 20  


def create_mask(data):
    mask = (data != 0.0).float()
    return mask


def normalize_data_improved(data):
    """Improved normalization using robust statistics"""
    data = np.array(data)
    non_zero_mask = data != 0.0

    if not np.any(non_zero_mask):
        return data

    non_zero_data = data[non_zero_mask]

    # Use robust statistics (median and IQR instead of mean and std)
    q25, q75 = np.percentile(non_zero_data, [25, 75])
    median = np.median(non_zero_data)
    iqr = q75 - q25

    if iqr == 0:
        iqr = 1

    normalized = np.zeros_like(data)
    normalized[non_zero_mask] = (data[non_zero_mask] - median) / iqr

    return normalized


def calculate_detailed_accuracy(model, X_tensor, y_tensor):
    model.eval()
    with torch.no_grad():
        masks = torch.stack([create_mask(x.squeeze()) for x in X_tensor])
        
        output = model(X_tensor, masks)
        _, predicted = torch.max(output.data, 1)

        total = y_tensor.size(0)
        correct = (predicted == y_tensor).sum().item()
        accuracy = 100 * correct / total

        y_true = y_tensor.cpu().numpy()
        y_pred = predicted.cpu().numpy()

        # Updated class names for 6 levels
        class_names = ['Zelo ekonomično', 'Ekonomično', 'Zmerno ekonomično', 
                      'Neekonomično', 'Zelo neekonomično', 'Ekstremno neekonomično']
        report = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)

        return accuracy, report, y_true, y_pred


def add_example_manually():
    try:
        accel_str = accel_input.get("1.0", tk.END).strip()
        if not accel_str:
            raise ValueError("Prosim vnesite pospeške.")
            
        accel_values = [float(x) for x in accel_str.replace(',', ' ').split()]
        
        if len(accel_values) != sequence_length:
            raise ValueError(f"Vnesite natanko {sequence_length} vrednosti pospeškov (dobljenih {len(accel_values)})")
            
        economy_level = int(economy_var.get())
        
        # Use improved normalization
        normalized_accel = normalize_data_improved(accel_values)
        
        examples_X.append(normalized_accel)
        examples_y.append(economy_level)
        
        status_label.config(text=f"Dodan primer: #{len(examples_X)} (Ocena ekonomičnosti: {economy_level})")
        
        accel_input.delete("1.0", tk.END)
        update_plot()
        
    except Exception as e:
        messagebox.showerror("Error", f"Invalid input: {e}")


def load_from_csv():
    """Load examples from a CSV file with improved error handling"""
    try:
        filename = filedialog.askopenfilename(
            title="Izberi CSV file",
            filetypes=(("CSV files", "*.csv"), ("All files", "*.*"))
        )
        
        if not filename:
            return
        
        loaded_count = 0
        
        with open(filename, 'r') as f:
            reader = csv.reader(f)
            try:
                header = next(reader)
            except StopIteration:
                messagebox.showerror("Error", "CSV file is empty")
                return
            
            for row_idx, row in enumerate(reader, start=2):
                try:
                    if not row or len(row) < 21:  # Need at least 20 values + 1 label
                        print(f"Row {row_idx}: Expected 21 columns, got {len(row)}")
                        continue

                    # Last element is the label
                    efficiency_rating = int(float(row[-1]))
                    if efficiency_rating < 0 or efficiency_rating > 5:  # Changed to 0-5
                        print(f"Row {row_idx}: Invalid efficiency rating {efficiency_rating}, should be 0-5")
                        continue

                    # First 20 elements are acceleration values
                    accel_values = [float(x) for x in row[:20]]
                    
                    # Use improved normalization
                    normalized_accel = normalize_data_improved(accel_values)
                    
                    examples_X.append(normalized_accel)
                    examples_y.append(efficiency_rating)
                    loaded_count += 1
                    
                except Exception as e:
                    print(f"Error processing row {row_idx}: {e}")
                    continue
        
        if loaded_count == 0:
            messagebox.showwarning("No data loaded", "No valid data rows were found in the CSV file.")
        else:
            status_label.config(text=f"Naloženih {loaded_count} primerov iz CSV.")
            update_plot()
        
    except Exception as e:
        messagebox.showerror("Error", f"Failed to load from CSV: {e}")


def save_model():
    global model
    if model is None:
        messagebox.showinfo("Ni modela.", "Sprva naučite model.")
        return
    
    try:
        filename = filedialog.asksaveasfilename(
            title="Save Model",
            defaultextension=".pt",
            filetypes=(("PyTorch Model", "*.pt"), ("All files", "*.*"))
        )
        
        if filename:
            torch.save(model.state_dict(), filename)
            status_label.config(text=f"Model shranjen v {filename}")
            
    except Exception as e:
        messagebox.showerror("Error", f"Failed to save model: {e}")


def load_model():
    global model
    
    try:
        filename = filedialog.askopenfilename(
            title="Naloži model",
            filetypes=(("PyTorch Model", "*.pt"), ("All files", "*.*"))
        )
        
        if not filename:
            return
            
        if any(ord(c) > 127 for c in filename):
            messagebox.showerror("Error", "File path contains special characters. Please move the file to a simpler path.")
            return
            
        model = AccelerationEconomyModel(
            input_size=1,
            hidden_size=128,  
            num_layers=3,
            num_classes=6  # Changed to 6 classes
        )
        
        model.load_state_dict(torch.load(filename, map_location=torch.device('cpu')))
        model.eval()
        
        status_label.config(text=f"Model successfully loaded from {filename}")
            
    except Exception as e:
        messagebox.showerror("Error", f"Failed to load model: {str(e)}")
        print(f"Detailed error: {repr(e)}")


def train_model():
    global model
    
    if len(examples_X) < 20:  # Increased minimum examples
        messagebox.showwarning("Ni dovolj primerov", "Naložite vsaj 20 primerov podatkov za robustno učenje.")
        return

    try:
        X_tensor = torch.tensor(np.array(examples_X), dtype=torch.float32).unsqueeze(2)
        y_tensor = torch.tensor(examples_y, dtype=torch.long)

        if model is None:
            model = AccelerationEconomyModel(hidden_size=128, num_layers=3, num_classes=6)  # Changed to 6 classes
            # Weight initialization is done in constructor
        
        # Weighted loss for balanced classes (updated for 6 classes)
        class_counts = np.bincount(examples_y, minlength=6)
        class_weights = 1.0 / (class_counts + 1e-8)
        class_weights = torch.tensor(class_weights, dtype=torch.float32)
        criterion = nn.CrossEntropyLoss(weight=class_weights)
        
        # Better optimizer settings
        optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=15, factor=0.5)

        epochs = 800
        best_loss = float('inf')
        patience = 30
        patience_counter = 0
        losses = []
        
        progress = ttk.Progressbar(window, orient="horizontal", length=300, mode="determinate")
        progress.grid(row=9, column=0, columnspan=2, pady=5)
        progress["maximum"] = epochs
        
        for epoch in range(epochs):
            model.train()
            optimizer.zero_grad()
            
            # Create masks for each sequence
            masks = torch.stack([create_mask(x.squeeze()) for x in X_tensor])
            
            output = model(X_tensor, masks)
            loss = criterion(output, y_tensor)
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step(loss)
            losses.append(loss.item())
            
            # Early stopping
            if loss.item() < best_loss:
                best_loss = loss.item()
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    status_label.config(text=f"Early stopping at epoch {epoch}")
                    break
            
            if epoch % 20 == 0:
                progress["value"] = epoch
                status_label.config(text=f"Epoch: {epoch}/{epochs}, Loss: {loss.item():.4f}, LR: {optimizer.param_groups[0]['lr']:.6f}")
                window.update()

        # Calculate detailed accuracy
        accuracy, report, y_true, y_pred = calculate_detailed_accuracy(model, X_tensor, y_tensor)
        
        result_text = f"✓ Model naučen. Izguba: {loss.item():.4f}\n"
        result_text += f"Accuracy: {accuracy:.2f}%\n"
        result_text += f"Macro Avg F1-Score: {report['macro avg']['f1-score']:.3f}\n"
        result_text += f"Weighted Avg F1-Score: {report['weighted avg']['f1-score']:.3f}"

        status_label.config(text=result_text)
        progress.destroy()
        plot_loss(losses)
        
        clear_memory()
        
    except Exception as e:
        messagebox.showerror("Error", f"Error during training: {e}")


def predict():
    global model
    
    if model is None:
        messagebox.showinfo("Ni modela", "Sprva naučite model.")
        return
    
    try:
        accel_str = predict_input.get("1.0", tk.END).strip()
        if not accel_str:
            raise ValueError("Vnesite pospeške")
            
        accel_values = [float(x) for x in accel_str.replace(',', ' ').split()]
        
        if len(accel_values) != sequence_length:
            raise ValueError(f"Vnesite natanko {sequence_length} pospeškov (dobljenih {len(accel_values)})")
        
        # Use improved normalization
        normalized_accel = normalize_data_improved(accel_values)
        
        input_tensor = torch.tensor(np.array([normalized_accel]), dtype=torch.float32).unsqueeze(2)
        
        model.eval()
        with torch.no_grad():
            mask = create_mask(torch.tensor(normalized_accel)).unsqueeze(0)
            output = model(input_tensor, mask)
            probabilities = torch.nn.functional.softmax(output, dim=1)
            _, predicted = torch.max(output.data, 1)
            predicted_level = predicted.item()
            
        # Updated descriptions for 6 levels
        descriptions = [
            f"Ekonomična ocena 0 - Zelo ekonomično (Optimalna vožnja)",
            f"Ekonomična ocena 1 - Ekonomično (Manjše izboljšave možne)",
            f"Ekonomična ocena 2 - Zmerno ekonomično (Nekaj pridobitev možnih)", 
            f"Ekonomična ocena 3 - Neekonomično (Opazne izboljšave potrebne)",
            f"Ekonomična ocena 4 - Zelo neekonomično (Večje izboljšave potrebne)",
            f"Ekonomična ocena 5 - Ekstremno neekonomično (Kritične izboljšave potrebne)"
        ]
        
        result_text = f"Napovedano: {descriptions[predicted_level]}\n"
        result_text += f"Zaupanje: {probabilities[0][predicted_level].item():.1%}\n"
        
        result_text += "\nVse verjetnosti:"
        for i, prob in enumerate(probabilities[0]):
            result_text += f"\nLevel {i}: {prob.item():.1%}"
        
        result_label.config(text=result_text)
        
    except Exception as e:
        messagebox.showerror("Error", f"Prediction error: {e}")


def predict_from_csv():
    """Improved CSV prediction with results display window"""
    global model
    
    if model is None:
        messagebox.showinfo("Ni modela.", "Trenirajte ali pa naložite že naučen model.")
        return
    
    try:
        filename = filedialog.askopenfilename(
            title="Izberi CSV file za napoved",
            filetypes=(("CSV files", "*.csv"), ("All files", "*.*"))
        )
        
        if not filename:
            return
        
        base_name = os.path.splitext(os.path.basename(filename))[0]
        output_filename = f"{base_name}_predictions.csv"
        
        csv_filename = filedialog.asksaveasfilename(
            title="Shrani napovedi kot",
            initialfile=output_filename,
            defaultextension=".csv",
            filetypes=(("CSV files", "*.csv"), ("All files", "*.*"))
        )
        
        if not csv_filename:
            return
        
        all_rows = []
        predictions = []
        probabilities_list = []
        processed_count = 0
        
        with open(filename, 'r') as f:
            reader = csv.reader(f)
            
            try:
                header = next(reader)
            except StopIteration:
                raise ValueError("CSV file is empty")
            
            model.eval()
            
            with open(csv_filename, 'w', newline='') as output_file:
                writer = csv.writer(output_file)
                
                # Write header - maintaining original format plus prediction
                if header:
                    if len(header) > sequence_length:
                        writer.writerow(header[:sequence_length] + ["label"])
                    else:
                        writer.writerow(header + ["label"] if len(header) == sequence_length else 
                                       [f"p{i+1}" for i in range(sequence_length)] + ["label"])
                else:
                    writer.writerow([f"p{i+1}" for i in range(sequence_length)] + ["label"])
                
                for row_idx, row in enumerate(reader):
                    try:
                        original_row = row.copy()
                        
                        # Check if row has sufficient data
                        if len(row) < sequence_length:
                            if len(row) == sequence_length + 1:
                                accel_values = [float(x) for x in row[:-1]]
                            else:
                                raise ValueError(f"Row {row_idx + 1} has insufficient data")
                        else:
                            accel_values = [float(x) for x in row[:sequence_length]]

                        normalized_accel = normalize_data_improved(accel_values)

                        input_tensor = torch.tensor(np.array([normalized_accel]), dtype=torch.float32).unsqueeze(2)

                        with torch.no_grad():
                            mask = create_mask(torch.tensor(normalized_accel)).unsqueeze(0)
                            output = model(input_tensor, mask)
                            probabilities = torch.nn.functional.softmax(output, dim=1)
                            _, predicted = torch.max(output.data, 1)
                            predicted_level = predicted.item()
                            confidence = probabilities[0][predicted_level].item()

                        predictions.append(predicted_level)
                        probabilities_list.append(probabilities[0].tolist())

                        # Write result with confidence
                        output_row = [str(val) for val in accel_values] + [str(predicted_level)]
                        writer.writerow(output_row)
                        all_rows.append(output_row)
                        processed_count += 1

                    except Exception as e:
                        print(f"Error processing row {row_idx + 2}: {e}")
                        predictions.append(-1)  # Error marker
                        probabilities_list.append([0, 0, 0, 0, 0, 0])  # Updated for 6 classes
                        continue
        
        # Show results window
        show_prediction_results(predictions, probabilities_list, csv_filename)
        
        status_label.config(text=f"Napovedanih {processed_count} vrstic. Rezultati so shranjeni v {csv_filename}")
        
    except Exception as e:
        messagebox.showerror("Error", f"CSV prediction error: {e}")


def show_prediction_results(predictions, probabilities_list, output_filename):
    """Show prediction results in a new window"""
    result_window = tk.Toplevel(window)
    result_window.title("Rezultati napovedi")
    result_window.geometry("800x600")
    
    # Text frame with results
    text_frame = ttk.Frame(result_window)
    text_frame.pack(fill="both", expand=True, padx=10, pady=10)
    
    text_widget = tk.Text(text_frame, wrap=tk.WORD)
    scrollbar = ttk.Scrollbar(text_frame, orient="vertical", command=text_widget.yview)
    text_widget.configure(yscrollcommand=scrollbar.set)
    
    text_widget.pack(side="left", fill="both", expand=True)
    scrollbar.pack(side="right", fill="y")
    
    # Add results
    text_widget.insert(tk.END, f"Rezultati napovedi\n")
    text_widget.insert(tk.END, f"========================\n\n")
    text_widget.insert(tk.END, f"Število obravnavanih vrstic: {len(predictions)}\n\n")
    
    # Statistics for 6 levels
    valid_predictions = [p for p in predictions if p != -1]
    if valid_predictions:
        level_descriptions = [
            "Level 0 - Zelo ekonomično",
            "Level 1 - Ekonomično", 
            "Level 2 - Zmerno ekonomično",
            "Level 3 - Neekonomično",
            "Level 4 - Zelo neekonomično",
            "Level 5 - Ekstremno neekonomično"
        ]
        
        text_widget.insert(tk.END, "Distribucija rezultatov:\n")
        text_widget.insert(tk.END, "-" * 30 + "\n")
        
        for level in range(6):
            count = valid_predictions.count(level)
            percentage = count/len(valid_predictions)*100 if valid_predictions else 0
            text_widget.insert(tk.END, f"{level_descriptions[level]}: {count} segments ({percentage:.1f}%)\n")
    
    text_widget.insert(tk.END, f"\nRezultati shranjeni v: {output_filename}\n\n")
    
    # Detailed results
    text_widget.insert(tk.END, "Detaljni rezultati (prvih 100 vrstic):\n")
    text_widget.insert(tk.END, "-" * 50 + "\n")
    
    for i, (pred, probs) in enumerate(zip(predictions[:100], probabilities_list[:100])):
        if pred == -1:
            text_widget.insert(tk.END, f"Vrstica {i+1}: ERROR\n")
        else:
            max_prob = max(probs) if probs else 0
            text_widget.insert(tk.END, f"Vrstica {i+1}: Ocena {pred} (Verjetnost: {max_prob:.2%})\n")
    
    if len(predictions) > 100:
        text_widget.insert(tk.END, f"\n... in {len(predictions) - 100} več vrstic\n")
    
    text_widget.config(state=tk.DISABLED)
    
    # Close button
    ttk.Button(result_window, text="Zapri", command=result_window.destroy).pack(pady=10)


def update_plot():
    if not examples_X or not examples_y:
        return
        
    plt.figure(figsize=(12, 8))
    
    # Updated colors and class names for 6 levels
    colors = ['#2E8B57', '#32CD32', '#FFD700', '#FFA500', '#FF6347', '#8B0000']
    class_names = ['Zelo ekonomično', 'Ekonomično', 'Zmerno ekonomično', 
                   'Neekonomično', 'Zelo neekonomično', 'Ekstremno neekonomično']
    
    plotted_classes = set()
    
    for i, (x, y) in enumerate(zip(examples_X, examples_y)):
        if y not in plotted_classes:
            plt.plot(x, label=f"Level {y}: {class_names[y]}", 
                     color=colors[y], alpha=0.8, linewidth=2)
            plotted_classes.add(y)
        else:
            plt.plot(x, color=colors[y], alpha=0.3, linewidth=1)
    
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.title("Acceleration Patterns by Economy Level (6 Levels)", fontsize=14, fontweight='bold')
    plt.xlabel("Time Steps (1-20)", fontsize=12)
    plt.ylabel("Normalized Acceleration", fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if hasattr(update_plot, 'canvas'):
        update_plot.canvas.get_tk_widget().destroy()
    
    canvas = FigureCanvasTkAgg(plt.gcf(), master=plot_frame)
    canvas.draw()
    canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
    update_plot.canvas = canvas
    
    plt.close()


def plot_loss(losses):
    plt.figure(figsize=(10, 6))
    plt.plot(losses, 'b-', linewidth=2)
    plt.title("Training Loss Over Time", fontsize=14, fontweight='bold')
    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel("Loss", fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    try:
        plt.savefig("training_loss.png", dpi=300, bbox_inches='tight')
    except:
        pass
    
    plt.show()


def clear_data():
    global examples_X, examples_y
    
    if messagebox.askyesno("Confirm", "Are you sure you want to clear all training data?"):
        examples_X = []
        examples_y = []
        status_label.config(text="All training data cleared")
        update_plot()
        clear_memory()


def clear_memory():
    """Clear memory and cache"""
    try:
        torch.cuda.empty_cache()
    except:
        pass
    
    gc.collect()
    status_label.config(text=status_label.cget("text") + " | Memory cleared")


def exit_application():
    if messagebox.askyesno("Exit", "Are you sure you want to exit?"):
        clear_memory()
        window.destroy()


# GUI Setup
window = tk.Tk()
window.title("Improved BiLSTM Acceleration Economy Model (6 Levels)")
window.geometry("1000x800")

notebook = ttk.Notebook(window)
notebook.grid(row=0, column=0, columnspan=2, sticky="nsew", padx=10, pady=10)

# Tabs
training_tab = ttk.Frame(notebook)
prediction_tab = ttk.Frame(notebook)
batch_prediction_tab = ttk.Frame(notebook)
visualization_tab = ttk.Frame(notebook)

notebook.add(training_tab, text="Training")
notebook.add(prediction_tab, text="Single Prediction")
notebook.add(batch_prediction_tab, text="Batch Prediction")
notebook.add(visualization_tab, text="Visualization")

# Training Tab
examples_frame = ttk.LabelFrame(training_tab, text="Add Training Examples")
examples_frame.pack(fill="both", expand=True, padx=10, pady=10)

tk.Label(examples_frame, text="Enter exactly 20 acceleration values (space or comma separated):").pack(anchor="w", padx=5, pady=5)
accel_input = tk.Text(examples_frame, height=4, width=80)
accel_input.pack(fill="both", expand=True, padx=5, pady=5)

# Economy level selection (updated for 6 levels)
economy_frame = ttk.Frame(examples_frame)
economy_frame.pack(fill="x", padx=5, pady=5)

tk.Label(economy_frame, text="Economy Level:").pack(anchor="w")
economy_var = tk.StringVar(value="0")

radio_frame1 = ttk.Frame(economy_frame)
radio_frame1.pack(fill="x", pady=2)
radio_frame2 = ttk.Frame(economy_frame)
radio_frame2.pack(fill="x", pady=2)

descriptions = [
    "Zelo ekonomično (0)", "Ekonomično (1)", "Zmerno ekonomično (2)", 
    "Neekonomično (3)", "Zelo neekonomično (4)", "Ekstremno neekonomično (5)"
]

for i, desc in enumerate(descriptions[:3]):
    tk.Radiobutton(radio_frame1, text=desc, variable=economy_var, value=str(i)).pack(side="left", padx=10)

for i, desc in enumerate(descriptions[3:], 3):
    tk.Radiobutton(radio_frame2, text=desc, variable=economy_var, value=str(i)).pack(side="left", padx=10)

# Buttons frame
button_frame = ttk.Frame(examples_frame)
button_frame.pack(fill="x", padx=5, pady=10)

ttk.Button(button_frame, text="Add Example", command=add_example_manually).pack(side="left", padx=5)
ttk.Button(button_frame, text="Load from CSV", command=load_from_csv).pack(side="left", padx=5)
ttk.Button(button_frame, text="Clear Data", command=clear_data).pack(side="left", padx=5)

# Model training frame
model_frame = ttk.LabelFrame(training_tab, text="BiLSTM Model Training")
model_frame.pack(fill="both", expand=True, padx=10, pady=10)

ttk.Button(model_frame, text="Train Model", command=train_model).pack(side="left", padx=10, pady=10)
ttk.Button(model_frame, text="Save Model", command=save_model).pack(side="left", padx=10, pady=10)
ttk.Button(model_frame, text="Load Model", command=load_model).pack(side="left", padx=10, pady=10)
ttk.Button(model_frame, text="Clear Memory", command=clear_memory).pack(side="left", padx=10, pady=10)

# Single Prediction Tab
predict_frame = ttk.LabelFrame(prediction_tab, text="Predict Single Segment")
predict_frame.pack(fill="both", expand=True, padx=10, pady=10)

tk.Label(predict_frame, text="Enter exactly 20 acceleration values (space or comma separated):").pack(anchor="w", padx=5, pady=5)
predict_input = tk.Text(predict_frame, height=4, width=80)
predict_input.pack(fill="both", expand=True, padx=5, pady=5)

single_predict_frame = ttk.Frame(predict_frame)
single_predict_frame.pack(fill="x", padx=5, pady=5)

ttk.Button(single_predict_frame, text="Predict", command=predict).pack(side="left", padx=5)

result_label = tk.Label(predict_frame, text="", font=("Arial", 11), justify="left")
result_label.pack(fill="both", expand=True, padx=5, pady=5)

# Batch Prediction Tab
batch_frame = ttk.LabelFrame(batch_prediction_tab, text="Batch Prediction from CSV")
batch_frame.pack(fill="both", expand=True, padx=10, pady=10)

tk.Label(batch_frame, 
         text="Load CSV file to predict economy rating for each row.\nCSV should contain 20 acceleration values per row.",
         font=("Arial", 10), justify="left").pack(anchor="w", padx=10, pady=10)

ttk.Button(batch_frame, text="Load CSV and Predict All", 
          command=predict_from_csv).pack(padx=10, pady=20)

# Visualization Tab
plot_frame = ttk.LabelFrame(visualization_tab, text="Acceleration Patterns by Economy Level")
plot_frame.pack(fill="both", expand=True, padx=10, pady=10)

# Status bar
status_label = tk.Label(window, text="Improved BiLSTM Model Ready", bd=1, relief=tk.SUNKEN, anchor=tk.W)
status_label.grid(row=10, column=0, columnspan=2, sticky="we")

# Exit button
ttk.Button(window, text="Exit", command=exit_application).grid(row=11, column=0, columnspan=2, pady=5)

# Configure grid weights
window.grid_rowconfigure(0, weight=1)
window.grid_columnconfigure(0, weight=1)

update_plot()
window.protocol("WM_DELETE_WINDOW", exit_application)

if __name__ == "__main__":
    window.mainloop()