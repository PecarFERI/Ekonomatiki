import torch
import torch.nn as nn
import torch.optim as optim
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import numpy as np
import csv
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import gc
import os
from sklearn.metrics import classification_report, confusion_matrix


class BiLSTMSpeedEconomyModel(nn.Module):
    def __init__(self, input_size=1, hidden_size=64, num_layers=2, num_classes=6):
        super(BiLSTMSpeedEconomyModel, self).__init__()
        self.lstm = nn.LSTM(input_size=input_size,
                            hidden_size=hidden_size,
                            num_layers=num_layers,
                            batch_first=True,
                            dropout=0.3 if num_layers > 1 else 0,
                            bidirectional=True)

        self.attention = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_size, 1),
            nn.Softmax(dim=1)
        )

        self.classifier = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.BatchNorm1d(hidden_size), #stabilizacijo
            nn.ReLU(), #regularizacijo
            nn.Dropout(0.3),
            nn.Linear(hidden_size, num_classes)
        )

    def forward(self, x, mask=None):
        lstm_out, (hn, cn) = self.lstm(x)

        if mask is not None: #maskiranje pove katere so veljavne in katere ne
            lstm_out = lstm_out * mask.unsqueeze(-1)

        attention_weights = self.attention(lstm_out)

        if mask is not None:
            attention_weights = attention_weights * mask.unsqueeze(-1) #da sestevek 1 tudi pri maskiranju
            attention_weights = attention_weights / (attention_weights.sum(dim=1, keepdim=True) + 1e-8)

        context_vector = torch.sum(attention_weights * lstm_out, dim=1)

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
    data = np.array(data)
    non_zero_mask = data != 0.0

    if not np.any(non_zero_mask):
        return data

    non_zero_data = data[non_zero_mask]

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

        class_names = ['Excellent', 'Very Good', 'Good', 'Moderate', 'Poor', 'Very Poor']
        report = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)

        return accuracy, report, y_true, y_pred


def add_example_manually():
    try:
        speed_str = speed_input.get("1.0", tk.END).strip()
        if not speed_str:
            raise ValueError("Please enter speed values")

        speed_values = [float(x) for x in speed_str.replace(',', ' ').split()]

        if len(speed_values) != 20:
            raise ValueError(f"Please enter exactly 20 speed values (got {len(speed_values)})")

        normalized_speed = normalize_data_improved(speed_values)

        examples_X.append(normalized_speed)
        examples_y.append(int(economy_var.get()))

        status_label.config(text=f"Added example #{len(examples_X)} (Economy Level: {economy_var.get()})")

        speed_input.delete("1.0", tk.END)
        update_plot()

    except Exception as e:
        messagebox.showerror("Error", f"Invalid input: {e}")


def load_from_csv():
    try:
        filename = filedialog.askopenfilename(
            title="Select CSV file",
            filetypes=(("CSV files", "*.csv"), ("All files", "*.*"))
        )

        if not filename:
            return

        loaded_count = 0

        with open(filename, 'r') as f:
            reader = csv.reader(f)
            header = next(reader)

            for row_num, row in enumerate(reader, start=2):
                try:
                    if len(row) < 21:
                        print(f"Row {row_num}: Expected 21 columns, got {len(row)}")
                        continue

                    efficiency_rating = int(float(row[-1]))
                    if efficiency_rating < 0 or efficiency_rating > 5:
                        print(f"Row {row_num}: Invalid efficiency rating {efficiency_rating}, should be 0-5")
                        continue

                    speed_values = [float(x) for x in row[:20]]
                    normalized_speed = normalize_data_improved(speed_values)

                    examples_X.append(normalized_speed)
                    examples_y.append(efficiency_rating)
                    loaded_count += 1

                except Exception as e:
                    print(f"Error processing row {row_num}: {e}")
                    continue

        status_label.config(text=f"Loaded {loaded_count} examples from CSV")
        update_plot()

    except Exception as e:
        messagebox.showerror("Error", f"Failed to load from CSV: {e}")


def save_model():
    global model
    if model is None:
        messagebox.showinfo("No Model", "Please train a model first.")
        return

    try:
        filename = filedialog.asksaveasfilename(
            title="Save Model",
            defaultextension=".pt",
            filetypes=(("PyTorch Model", "*.pt"), ("All files", "*.*"))
        )

        if filename:
            torch.save(model.state_dict(), filename)
            status_label.config(text=f"Model saved to {filename}")

    except Exception as e:
        messagebox.showerror("Error", f"Failed to save model: {e}")


def load_model():
    global model

    try:
        filename = filedialog.askopenfilename(
            title="Load Model",
            filetypes=(("PyTorch Model", "*.pt"), ("All files", "*.*"))
        )

        if not filename:
            return

        if any(ord(c) > 127 for c in filename):
            messagebox.showerror("Error",
                                 "File path contains special characters. Please move the file to a simpler path.")
            return

        model = BiLSTMSpeedEconomyModel(
            input_size=1,
            hidden_size=128,
            num_layers=3,
            num_classes=6
        )

        model.load_state_dict(torch.load(filename, map_location=torch.device('cpu')))
        model.eval()

        status_label.config(text=f"Model successfully loaded from {filename}")

    except Exception as e:
        messagebox.showerror("Error", f"Failed to load model: {str(e)}")
        print(f"Detailed error: {repr(e)}")


def train_model():
    global model

    if len(examples_X) < 20:
        messagebox.showwarning("Insufficient Data", "Please add at least 20 examples for robust training.")
        return

    try:
        X_tensor = torch.tensor(np.array(examples_X), dtype=torch.float32).unsqueeze(2)
        y_tensor = torch.tensor(examples_y, dtype=torch.long)

        if model is None:
            model = BiLSTMSpeedEconomyModel(hidden_size=128, num_layers=3, num_classes=6)
            for name, param in model.named_parameters():
                if 'weight' in name and param.dim() > 1:
                    nn.init.xavier_uniform_(param)
                elif 'bias' in name:
                    nn.init.zeros_(param)

        #weighted loss za uravnotežene razrede
        class_counts = np.bincount(examples_y, minlength=6)
        class_weights = 1.0 / (class_counts + 1e-8)
        class_weights = torch.tensor(class_weights, dtype=torch.float32)
        criterion = nn.CrossEntropyLoss(weight=class_weights)

        optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01) #da se nea prenauci
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

            # Ustvari maske
            masks = torch.stack([create_mask(x.squeeze()) for x in X_tensor])

            output = model(X_tensor, masks)
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

            if epoch % 20 == 0:
                progress["value"] = epoch
                status_label.config(
                    text=f"Epoch: {epoch}/{epochs}, Loss: {loss.item():.4f}, LR: {optimizer.param_groups[0]['lr']:.6f}")
                window.update()

        accuracy, report, y_true, y_pred = calculate_detailed_accuracy(model, X_tensor, y_tensor)

        result_text = f"✓ Model trained. Final Loss: {loss.item():.4f}\n"
        result_text += f"Overall Accuracy: {accuracy:.2f}%\n"
        result_text += f"Macro Avg F1-Score: {report['macro avg']['f1-score']:.3f}\n"
        result_text += f"Weighted Avg F1-Score: {report['weighted avg']['f1-score']:.3f}"

        status_label.config(text=result_text)
        progress.destroy()
        plot_loss(losses)

        clear_memory()

    except Exception as e:
        messagebox.showerror("Error", f"Error during training: {e}")


def predict_single():
    global model

    if model is None:
        messagebox.showinfo("No Model", "Please train a model first.")
        return

    try:
        speed_str = predict_input.get("1.0", tk.END).strip()
        if not speed_str:
            raise ValueError("Please enter speed values")

        speed_values = [float(x) for x in speed_str.replace(',', ' ').split()]

        if len(speed_values) != 20:
            raise ValueError(f"Please enter exactly 20 speed values (got {len(speed_values)})")

        normalized_speed = normalize_data_improved(speed_values)
        input_tensor = torch.tensor(np.array([normalized_speed]), dtype=torch.float32).unsqueeze(2)

        model.eval()
        with torch.no_grad():
            mask = create_mask(torch.tensor(normalized_speed)).unsqueeze(0)
            output = model(input_tensor, mask)
            probabilities = torch.nn.functional.softmax(output, dim=1)
            _, predicted = torch.max(output.data, 1)
            predicted_level = predicted.item()

        descriptions = [
            f"Efficiency Level 0 - Excellent (Optimal driving)",
            f"Efficiency Level 1 - Very Good (Minor improvements possible)",
            f"Efficiency Level 2 - Good (Some efficiency gains possible)",
            f"Efficiency Level 3 - Moderate (Noticeable improvements needed)",
            f"Efficiency Level 4 - Poor (Significant improvements required)",
            f"Efficiency Level 5 - Very Poor (Major efficiency issues)"
        ]

        result_text = f"Predicted: {descriptions[predicted_level]}\n"
        result_text += f"Confidence: {probabilities[0][predicted_level].item():.1%}\n"

        result_text += "\nAll Probabilities:"
        for i, prob in enumerate(probabilities[0]):
            result_text += f"\nLevel {i}: {prob.item():.1%}"

        result_label.config(text=result_text)

    except Exception as e:
        messagebox.showerror("Error", f"Prediction error: {e}")


def predict_csv_file():
    global model

    if model is None:
        messagebox.showinfo("No Model", "Please train a model first.")
        return

    try:
        csv_filename = filedialog.askopenfilename(
            title="Select CSV file for prediction (without efficiency_rating column)",
            filetypes=(("CSV files", "*.csv"), ("All files", "*.*"))
        )

        if not csv_filename:
            return

        base_name = os.path.splitext(os.path.basename(csv_filename))[0]
        output_filename = f"{base_name}_output.txt"

        txt_filename = filedialog.asksaveasfilename(
            title="Save predictions as",
            initialfile=output_filename,
            defaultextension=".txt",
            filetypes=(("Text files", "*.txt"), ("All files", "*.*"))
        )

        if not txt_filename:
            return

        processed_count = 0

        with open(csv_filename, 'r') as f:
            reader = csv.reader(f)

            try:
                header = next(reader)
            except StopIteration:
                raise ValueError("CSV file is empty")

            model.eval()

            with open(txt_filename, 'w') as output_file:
                for row_idx, row in enumerate(reader):
                    try:
                        if len(row) < 20:
                            print(f"Row {row_idx + 2}: Expected 20 speed values, got {len(row)}")
                            continue

                        speed_values = [float(x) for x in row[:20]]
                        normalized_speed = normalize_data_improved(speed_values)

                        input_tensor = torch.tensor(np.array([normalized_speed]), dtype=torch.float32).unsqueeze(2)

                        with torch.no_grad():
                            mask = create_mask(torch.tensor(normalized_speed)).unsqueeze(0)
                            output = model(input_tensor, mask)
                            probabilities = torch.nn.functional.softmax(output, dim=1)
                            _, predicted = torch.max(output.data, 1)
                            predicted_level = predicted.item()
                            confidence = probabilities[0][predicted_level].item()

                        # Zapiši rezultat z zaupanjem
                        original_values = row[:20]
                        line = ",".join(original_values) + f",{predicted_level}"
                        output_file.write(line + "\n")

                        processed_count += 1

                    except Exception as e:
                        print(f"Error processing row {row_idx + 2}: {e}")
                        continue

        messagebox.showinfo("Success",
                            f"Predictions saved to {txt_filename}\n"
                            f"Processed {processed_count} rows\n"
                            f"Format: original_20_values,predicted_level,confidence")

        status_label.config(text=f"CSV prediction completed: {processed_count} rows processed")

    except Exception as e:
        messagebox.showerror("Error", f"CSV prediction error: {e}")

def update_plot():
    if not examples_X or not examples_y:
        return

    plt.figure(figsize=(12, 8))

    colors = ['#2E8B57', '#32CD32', '#FFD700', '#FFA500', '#FF6347', '#DC143C']
    class_names = ['Excellent', 'Very Good', 'Good', 'Moderate', 'Poor', 'Very Poor']

    plotted_classes = set()

    for i, (x, y) in enumerate(zip(examples_X, examples_y)):
        if y not in plotted_classes:
            plt.plot(x, label=f"Level {y}: {class_names[y]}",
                     color=colors[y], alpha=0.8, linewidth=2)
            plotted_classes.add(y)
        else:
            plt.plot(x, color=colors[y], alpha=0.3, linewidth=1)

    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.title("Speed Patterns by Efficiency Level", fontsize=14, fontweight='bold')
    plt.xlabel("Time Steps (1-20)", fontsize=12)
    plt.ylabel("Normalized Speed (0.0 = padding)", fontsize=12)
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


window = tk.Tk()
window.title("BiLSTM model hitrosti")
window.geometry("1000x800")

notebook = ttk.Notebook(window)
notebook.grid(row=0, column=0, columnspan=2, sticky="nsew", padx=10, pady=10)

training_tab = ttk.Frame(notebook)
prediction_tab = ttk.Frame(notebook)
visualization_tab = ttk.Frame(notebook)

notebook.add(training_tab, text="Training")
notebook.add(prediction_tab, text="Prediction")
notebook.add(visualization_tab, text="Visualization")

examples_frame = ttk.LabelFrame(training_tab, text="Add Training Examples")
examples_frame.pack(fill="both", expand=True, padx=10, pady=10)

tk.Label(examples_frame, text="Enter exactly 20 speed values").pack(anchor="w", padx=5,
                                                                                                pady=5)
speed_input = tk.Text(examples_frame, height=4, width=80)
speed_input.pack(fill="both", expand=True, padx=5, pady=5)

economy_frame = ttk.Frame(examples_frame)
economy_frame.pack(fill="x", padx=5, pady=5)

tk.Label(economy_frame, text="Efficiency Level:").pack(anchor="w")
economy_var = tk.StringVar(value="0")

radio_frame1 = ttk.Frame(economy_frame)
radio_frame1.pack(fill="x", pady=2)
radio_frame2 = ttk.Frame(economy_frame)
radio_frame2.pack(fill="x", pady=2)

descriptions = ["Excellent (0)", "Very Good (1)", "Good (2)", "Moderate (3)", "Poor (4)", "Very Poor (5)"]
for i, desc in enumerate(descriptions[:3]):
    tk.Radiobutton(radio_frame1, text=desc, variable=economy_var, value=str(i)).pack(side="left", padx=10)

for i, desc in enumerate(descriptions[3:], 3):
    tk.Radiobutton(radio_frame2, text=desc, variable=economy_var, value=str(i)).pack(side="left", padx=10)

button_frame = ttk.Frame(examples_frame)
button_frame.pack(fill="x", padx=5, pady=10)

ttk.Button(button_frame, text="Add Example", command=add_example_manually).pack(side="left", padx=5)
ttk.Button(button_frame, text="Load from CSV", command=load_from_csv).pack(side="left", padx=5)
ttk.Button(button_frame, text="Clear Data", command=clear_data).pack(side="left", padx=5)

model_frame = ttk.LabelFrame(training_tab, text="Model Training")
model_frame.pack(fill="both", expand=True, padx=10, pady=10)

ttk.Button(model_frame, text="Train Model", command=train_model).pack(side="left", padx=10, pady=10)
ttk.Button(model_frame, text="Save Model", command=save_model).pack(side="left", padx=10, pady=10)
ttk.Button(model_frame, text="Load Model", command=load_model).pack(side="left", padx=10, pady=10)
ttk.Button(model_frame, text="Clear Memory", command=clear_memory).pack(side="left", padx=10, pady=10)

predict_frame = ttk.LabelFrame(prediction_tab, text="Single Prediction")
predict_frame.pack(fill="both", expand=True, padx=10, pady=10)

tk.Label(predict_frame, text="Enter exactly 20 speed values (space or comma separated):").pack(anchor="w", padx=5,
                                                                                               pady=5)
predict_input = tk.Text(predict_frame, height=4, width=80)
predict_input.pack(fill="both", expand=True, padx=5, pady=5)

single_predict_frame = ttk.Frame(predict_frame)
single_predict_frame.pack(fill="x", padx=5, pady=5)

ttk.Button(single_predict_frame, text="Predict Single", command=predict_single).pack(side="left", padx=5)

result_label = tk.Label(predict_frame, text="", font=("Arial", 11), justify="left")
result_label.pack(fill="both", expand=True, padx=5, pady=5)

csv_predict_frame = ttk.LabelFrame(prediction_tab, text="CSV File Prediction")
csv_predict_frame.pack(fill="both", expand=True, padx=10, pady=10)

tk.Label(csv_predict_frame,
         text="Process CSV file with 20 speed columns (automatically names output with _output suffix):").pack(
    anchor="w", padx=5, pady=5)
ttk.Button(csv_predict_frame, text="Predict CSV File", command=predict_csv_file).pack(padx=5, pady=10)

plot_frame = ttk.LabelFrame(visualization_tab, text="Speed Patterns by Efficiency Level")
plot_frame.pack(fill="both", expand=True, padx=10, pady=10)

status_label = tk.Label(window, text="Enhanced BiLSTM model ready - Improved padding handling and accuracy calculation",
                        bd=1, relief=tk.SUNKEN, anchor=tk.W)
status_label.grid(row=10, column=0, columnspan=2, sticky="we")

ttk.Button(window, text="Exit", command=exit_application).grid(row=11, column=0, columnspan=2, pady=5)

window.grid_rowconfigure(0, weight=1)
window.grid_columnconfigure(0, weight=1)

update_plot()
window.protocol("WM_DELETE_WINDOW", exit_application)
window.mainloop()