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


class AccelerationEconomyModel(nn.Module):
    def __init__(self, input_size=1, hidden_size=64, num_layers=2, num_classes=5):
        super(AccelerationEconomyModel, self).__init__()
        # BiLSTM namesto običajnega LSTM
        self.bilstm = nn.LSTM(input_size=input_size, 
                             hidden_size=hidden_size,
                             num_layers=num_layers, 
                             batch_first=True,
                             dropout=0.2 if num_layers > 1 else 0,
                             bidirectional=True)  # Dodano bidirectional=True
        
        # Attention layer - mora upoštevati bidirectional (2 * hidden_size)
        self.attention = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),  # * 2 zaradi bidirectional
            nn.Tanh(),
            nn.Linear(hidden_size, 1),
            nn.Softmax(dim=1)
        )
        
        # Fully connected layer - tudi mora upoštevati bidirectional
        self.fc = nn.Linear(hidden_size * 2, num_classes)
        
    def forward(self, x):
        # BiLSTM output
        bilstm_out, (hn, cn) = self.bilstm(x)  # shape: (batch, seq_len, hidden_size * 2)
        
        # Attention mechanism
        attention_weights = self.attention(bilstm_out)
        context_vector = torch.sum(attention_weights * bilstm_out, dim=1)
        
        # Final classification
        out = self.fc(context_vector)
        return out

examples_X = []
examples_y = []
model = None
sequence_length = 20  

def normalize_data(data):
    mean = np.mean(data)
    std = np.std(data)
    if std == 0:
        std = 1  
    return (data - mean) / std

def add_example_manually():
    try:
        accel_str = accel_input.get("1.0", tk.END).strip()
        if not accel_str:
            raise ValueError("Prosim vnesite pospeške.")
            
        accel_values = [float(x) for x in accel_str.replace(',', ' ').split()]
        
        if len(accel_values) < 5:
            raise ValueError("Vnesite vsaj 5 pospeškov.")
            
        economy_level = int(economy_var.get())
        
        if len(accel_values) != sequence_length:
            accel_values = np.interp(
                np.linspace(0, 1, sequence_length),
                np.linspace(0, 1, len(accel_values)),
                accel_values
            )
        
        normalized_accel = normalize_data(accel_values)
        
        examples_X.append(normalized_accel)
        examples_y.append(economy_level)
        
        status_label.config(text=f"Dodan primer: #{len(examples_X)} (Ocena ekonomičnosti: {economy_level})")
        
        accel_input.delete("1.0", tk.END)
        
        update_plot()
        
    except Exception as e:
        messagebox.showerror("Error", f"Invalid input: {e}")


def load_from_csv():
    """Load examples from a CSV file"""
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
            header = next(reader)  # skip header
            
            for row in reader:
                try:
                    # Skip empty or incomplete rows
                    if len(row) < sequence_length + 1:
                        raise ValueError("Prekratka vrstica")

                    # Zadnji element je label (ekonomska raven)
                    label = int(row[-1])

                    # Ostali elementi so pospeški
                    accel_values = [float(x) for x in row[:-1]]

                    # Prilagodi dolžino če je potrebno
                    if len(accel_values) != sequence_length:
                        accel_values = np.interp(
                            np.linspace(0, 1, sequence_length),
                            np.linspace(0, 1, len(accel_values)),
                            accel_values
                        )
                    
                    normalized_accel = normalize_data(accel_values)

                    examples_X.append(normalized_accel)
                    examples_y.append(label)
                    loaded_count += 1
                    
                except Exception as e:
                    print(f"Error processing row: {e}")
                    continue
        
        status_label.config(text=f"Naloženih {loaded_count} primerov iz CSV.")
        update_plot()
        
    except Exception as e:
        messagebox.showerror("Error", f"Failed to load from CSV: {e}")


def predict_from_csv():
    """Nova funkcija za napovedovanje celotnega CSV fila"""
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
        
        all_rows = []
        predictions = []
        probabilities_list = []
        
        with open(filename, 'r') as f:
            reader = csv.reader(f)
            header = next(reader, None)  # skip header if exists
            
            row_count = 0
            for row in reader:
                try:
                    original_row = row.copy()
                    # Preveri ali ima vrstica dovolj podatkov
                    if len(row) < sequence_length:
                        # Če ima vrstica label na koncu, ga odstrani
                        if len(row) == sequence_length + 1:
                            accel_values = [float(x) for x in row[:-1]]
                        else:
                            raise ValueError(f"Row {row_count + 1} has insufficient data")
                    else:
                        # Vzemi samo prvih sequence_length vrednosti
                        accel_values = [float(x) for x in row[:sequence_length]]
                    
                    # Prilagodi dolžino če je potrebno
                    if len(accel_values) != sequence_length:
                        accel_values = np.interp(
                            np.linspace(0, 1, sequence_length),
                            np.linspace(0, 1, len(accel_values)),
                            accel_values
                        )
                    
                    normalized_accel = normalize_data(accel_values)
                    
                    # Napovej
                    input_tensor = torch.tensor(np.array([normalized_accel]), dtype=torch.float32).unsqueeze(2)
                    model.eval()
                    with torch.no_grad():
                        output = model(input_tensor)
                        probs = torch.nn.functional.softmax(output, dim=1)
                        _, predicted = torch.max(output.data, 1)
                        
                        predictions.append(predicted.item())
                        probabilities_list.append(probs[0].tolist())
                    
                    row_count += 1

                    output_row = original_row[:sequence_length] + [str(predicted.item())]
                    all_rows.append(output_row)
                    
                except Exception as e:
                    print(f"Error processing row {row_count + 1}: {e}")
                    predictions.append(-1)  # Error marker
                    probabilities_list.append([0, 0, 0, 0, 0])
                    row_count += 1
                    continue
        
          # Save results - maintaining original format plus prediction
        output_filename = filename.replace('.csv', '_predictions.csv')
        
        with open(output_filename, 'w', newline='') as f:
            writer = csv.writer(f)
            
            # Write header
            if header:
                # If header has more than 30 columns, use first 20 + "label"
                if len(header) > sequence_length:
                    writer.writerow(header[:sequence_length] + ["label"])
                else:
                    # If header has exactly 30 columns, add "label"
                    writer.writerow(header + ["label"] if len(header) == sequence_length else 
                                   [f"p{i+1}" for i in range(sequence_length)] + ["label"])
            else:
                # No header - create default p1-p30 + label
                writer.writerow([f"p{i+1}" for i in range(sequence_length)] + ["label"])

            
            writer.writerows(all_rows)
        
        # Prikaži rezultate v novem oknu
        show_prediction_results(predictions, probabilities_list, output_filename)
        
        status_label.config(text=f"Napovedanih {row_count} vrstic. Rezulatati so shranjeni v {output_filename}")
        
    except Exception as e:
        messagebox.showerror("Error", f"Failed to predict from CSV: {e}")


def show_prediction_results(predictions, probabilities_list, output_filename):
    """Prikaže rezultate napovedov v novem oknu"""
    result_window = tk.Toplevel(window)
    result_window.title("Rezultati napovedi")
    result_window.geometry("800x600")
    
    # Besedilo z rezultati
    text_frame = ttk.Frame(result_window)
    text_frame.pack(fill="both", expand=True, padx=10, pady=10)
    
    text_widget = tk.Text(text_frame, wrap=tk.WORD)
    scrollbar = ttk.Scrollbar(text_frame, orient="vertical", command=text_widget.yview)
    text_widget.configure(yscrollcommand=scrollbar.set)
    
    text_widget.pack(side="left", fill="both", expand=True)
    scrollbar.pack(side="right", fill="y")
    
    # Dodaj rezultate
    text_widget.insert(tk.END, f"Rezultati napovedi\n")
    text_widget.insert(tk.END, f"========================\n\n")
    text_widget.insert(tk.END, f"Število obravnavanih vrstic: {len(predictions)}\n")
    
    # Statistike
    valid_predictions = [p for p in predictions if p != -1]
    if valid_predictions:
        level_counts = {}
        for level in range(5):
            count = valid_predictions.count(level)
            level_counts[level] = count
            text_widget.insert(tk.END, f"Level {level}: {count} segments ({count/len(valid_predictions)*100:.1f}%)\n")
    
    text_widget.insert(tk.END, f"\nRezultati shranjeni v: {output_filename}\n\n")
    
    # Detaljni rezultati
    text_widget.insert(tk.END, "Detailed Results:\n")
    text_widget.insert(tk.END, "-" * 50 + "\n")
    
    for i, (pred, probs) in enumerate(zip(predictions[:100], probabilities_list[:100])):  # Prikaži samo prvih 100
        if pred == -1:
            text_widget.insert(tk.END, f"Vrstica {i+1}: ERROR\n")
        else:
            max_prob = max(probs)
            text_widget.insert(tk.END, f"Vrstica {i+1}: Ocena {pred} (Verjetnost: {max_prob:.2%})\n")
    
    if len(predictions) > 100:
        text_widget.insert(tk.END, f"\n... in {len(predictions) - 100} več vrstic\n")
    
    text_widget.config(state=tk.DISABLED)
    
    # Gumb za zapiranje
    ttk.Button(result_window, text="Zapri", command=result_window.destroy).pack(pady=10)


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
            num_classes=5
        )
        
        model.load_state_dict(torch.load(filename, map_location=torch.device('cpu')))
        model.eval()
        
        status_label.config(text=f"BiLSTM model successfully loaded from {filename}")
            
    except Exception as e:
        messagebox.showerror("Error", f"Failed to load model: {str(e)}")
        print(f"Detailed error: {repr(e)}")

def train_model():
    global model
    
    if len(examples_X) < 10: 
        messagebox.showwarning("Ni dovolj primerov", "Naložite vsaj 10 primerov podatkov.")
        return

    try:
        X_tensor = torch.tensor(np.array(examples_X), dtype=torch.float32).unsqueeze(2)
        y_tensor = torch.tensor(examples_y, dtype=torch.long)

        if model is None:
            model = AccelerationEconomyModel(hidden_size=128, num_layers=3, num_classes=5)
            for name, param in model.named_parameters():
                if 'weight' in name:
                    nn.init.xavier_normal_(param)
        
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.AdamW(model.parameters(), lr=0.0001, weight_decay=0.01)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=10, factor=0.5)

        epochs = 500  
        best_loss = float('inf')
        patience = 20
        patience_counter = 0
        losses = []
        
        progress = ttk.Progressbar(window, orient="horizontal", length=300, mode="determinate")
        progress.grid(row=9, column=0, columnspan=2, pady=5)
        progress["maximum"] = epochs
        
        for epoch in range(epochs):
            optimizer.zero_grad()
            output = model(X_tensor)
            loss = criterion(output, y_tensor)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0) 
            optimizer.step()
            scheduler.step(loss)
            losses.append(loss.item())
            
            if loss.item() < best_loss:
                best_loss = loss.item()
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    status_label.config(text=f"Early stopping at epoch {epoch}")
                    break
            
            if epoch % 10 == 0:
                progress["value"] = epoch
                status_label.config(text=f"Epoch: {epoch}/{epochs}, Loss: {loss.item():.4f}, LR: {optimizer.param_groups[0]['lr']:.6f}")
                window.update()

        with torch.no_grad():
            output = model(X_tensor)
            _, predicted = torch.max(output.data, 1)
            total = y_tensor.size(0)
            correct = (predicted == y_tensor).sum().item()
            accuracy = 100 * correct / total

        status_label.config(text=f"✓ BiLSTM model naučen. Final Loss: {loss.item():.4f}, Accuracy: {accuracy:.2f}%")
        progress.destroy()
        plot_loss(losses)
        
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
        
        if len(accel_values) < 5:
            raise ValueError("Vnesite vsaj 5 pospeškov")
            
        if len(accel_values) != sequence_length:
            accel_values = np.interp(
                np.linspace(0, 1, sequence_length),
                np.linspace(0, 1, len(accel_values)),
                accel_values
            )
        
        normalized_accel = normalize_data(accel_values)
        
        input_tensor = torch.tensor(np.array([normalized_accel]), dtype=torch.float32).unsqueeze(2)        
        model.eval()
        with torch.no_grad():
            output = model(input_tensor)
            probabilities = torch.nn.functional.softmax(output, dim=1)
            _, predicted = torch.max(output.data, 1)
            predicted_level = predicted.item()
            
        descriptions = [
            f"Ekonomična ocena 0 - Zelo ekonomično",
            f"Ekonomična ocena 1 - Ekonomično",
            f"Ekonomična ocena 2 - Zmerno ekonomično", 
            f"Ekonomična ocena 3 - Neekonomično",
            f"Ekonomična ocena 4 - Zelo neekonomično"
        ]
        
        result_text = f"Napovedano: {descriptions[predicted_level]}"
        
        result_text += "\n\nVerjetnosti:"
        for i, prob in enumerate(probabilities[0]):
            result_text += f"\nLevel {i}: {prob.item():.2%}"
        
        result_label.config(text=result_text)
        
    except Exception as e:
        messagebox.showerror("Error", f"Prediction error: {e}")

def update_plot():
    if not examples_X or not examples_y:
        return
        
    plt.figure(figsize=(8, 4))
    
    levels_shown = set()
    for i, (x, y) in enumerate(zip(examples_X, examples_y)):
        if y not in levels_shown and len(levels_shown) < 5:
            plt.plot(x, label=f"Level {y}")
            levels_shown.add(y)
            
            if len(levels_shown) >= 5:
                break
    
    plt.legend()
    plt.title("Example Acceleration Patterns (BiLSTM Training Data)")
    plt.xlabel("Time Steps")
    plt.ylabel("Normalized Acceleration")
    
    if hasattr(update_plot, 'canvas'):
        update_plot.canvas.get_tk_widget().destroy()
    
    canvas = FigureCanvasTkAgg(plt.gcf(), master=plot_frame)
    canvas.draw()
    canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
    update_plot.canvas = canvas

def plot_loss(losses):
    plt.figure(figsize=(8, 4))
    plt.plot(losses)
    plt.title("BiLSTM Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True)
    
    try:
        plt.savefig("bilstm_training_loss.png")
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


# GUI Setup
window = tk.Tk()
window.title("BiLSTM Acceleration Economy Model")
window.geometry("900x750")

notebook = ttk.Notebook(window)
notebook.grid(row=0, column=0, columnspan=2, sticky="nsew", padx=10, pady=10)

# Tabs
training_tab = ttk.Frame(notebook)
prediction_tab = ttk.Frame(notebook)
batch_prediction_tab = ttk.Frame(notebook)  # Nova tab za batch prediction
visualization_tab = ttk.Frame(notebook)

notebook.add(training_tab, text="Training")
notebook.add(prediction_tab, text="Single Prediction")
notebook.add(batch_prediction_tab, text="Batch Prediction")  # Nova tab
notebook.add(visualization_tab, text="Visualization")

# Training Tab
examples_frame = ttk.LabelFrame(training_tab, text="Add Training Examples")
examples_frame.pack(fill="both", expand=True, padx=10, pady=10)

tk.Label(examples_frame, text="Enter acceleration values (space or comma separated):").pack(anchor="w", padx=5, pady=5)
accel_input = tk.Text(examples_frame, height=5, width=60)
accel_input.pack(fill="both", expand=True, padx=5, pady=5)

# Economy level selection
economy_frame = ttk.Frame(examples_frame)
economy_frame.pack(fill="x", padx=5, pady=5)

tk.Label(economy_frame, text="Economy Level:").pack(side="left")
economy_var = tk.StringVar(value="0")
for i, desc in enumerate(["Very Economical (0)", "Economical (1)", "Moderate (2)", "Uneconomical (3)", "Very Uneconomical (4)"]):
    tk.Radiobutton(economy_frame, text=desc, variable=economy_var, value=str(i)).pack(side="left", padx=5)

# Buttons frame
button_frame = ttk.Frame(examples_frame)
button_frame.pack(fill="x", padx=5, pady=10)

ttk.Button(button_frame, text="Add Example", command=add_example_manually).pack(side="left", padx=5)
ttk.Button(button_frame, text="Load from CSV", command=load_from_csv).pack(side="left", padx=5)
ttk.Button(button_frame, text="Clear Data", command=clear_data).pack(side="left", padx=5)

# Model training frame
model_frame = ttk.LabelFrame(training_tab, text="BiLSTM Model Training")
model_frame.pack(fill="both", expand=True, padx=10, pady=10)

ttk.Button(model_frame, text="Train BiLSTM Model", command=train_model).pack(side="left", padx=10, pady=10)
ttk.Button(model_frame, text="Save Model", command=save_model).pack(side="left", padx=10, pady=10)
ttk.Button(model_frame, text="Load Model", command=load_model).pack(side="left", padx=10, pady=10)

# Single Prediction Tab
predict_frame = ttk.LabelFrame(prediction_tab, text="Predict Single Segment")
predict_frame.pack(fill="both", expand=True, padx=10, pady=10)

tk.Label(predict_frame, text="Enter acceleration values (space or comma separated):").pack(anchor="w", padx=5, pady=5)
predict_input = tk.Text(predict_frame, height=5, width=60)
predict_input.pack(fill="both", expand=True, padx=5, pady=5)

ttk.Button(predict_frame, text="Predict", command=predict).pack(padx=5, pady=10)

result_label = tk.Label(predict_frame, text="", font=("Arial", 12), justify="left")
result_label.pack(fill="both", expand=True, padx=5, pady=5)

# Batch Prediction Tab - NOVA FUNKCIONALNOST
batch_frame = ttk.LabelFrame(batch_prediction_tab, text="Batch Prediction from CSV")
batch_frame.pack(fill="both", expand=True, padx=10, pady=10)

tk.Label(batch_frame, 
         text="Naložite CSV datoteko da bi lahko napovedali oceno ekonomičnosti vsake vrstice.\n",
         font=("Arial", 10), justify="left").pack(anchor="w", padx=10, pady=10)

ttk.Button(batch_frame, text="Load CSV and Predict All", 
          command=predict_from_csv).pack(padx=10, pady=20)

# Visualization Tab
plot_frame = ttk.LabelFrame(visualization_tab, text="Acceleration Patterns")
plot_frame.pack(fill="both", expand=True, padx=10, pady=10)

# Status bar
status_label = tk.Label(window, text="BiLSTM Model Ready", bd=1, relief=tk.SUNKEN, anchor=tk.W)
status_label.grid(row=10, column=0, columnspan=2, sticky="we")

# Configure grid weights
window.grid_rowconfigure(0, weight=1)
window.grid_columnconfigure(0, weight=1)

update_plot()

if __name__ == "__main__":
    window.mainloop()
    torch.cuda.empty_cache()