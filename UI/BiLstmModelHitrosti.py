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


class BiLSTMSpeedEconomyModel(nn.Module):
    def __init__(self, input_size=1, hidden_size=64, num_layers=2, num_classes=5):
        super(BiLSTMSpeedEconomyModel, self).__init__()
        self.lstm = nn.LSTM(input_size=input_size,
                            hidden_size=hidden_size,
                            num_layers=num_layers,
                            batch_first=True,  # vhodne oblike so batch size, sequence length, input size
                            dropout=0.2 if num_layers > 1 else 0,
                            bidirectional=True)  # BiLSTM, gre skozi zaporedje naprej in nazaj

        self.attention = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size * 2),  # podvojim zaradi bidirectional
            nn.Tanh(),
            nn.Linear(hidden_size * 2, 1),
            nn.Softmax(dim=1)
        )
        self.fc = nn.Linear(hidden_size * 2, num_classes)

    def forward(self, x):
        lstm_out, (hn, cn) = self.lstm(x)  # batch, seq_len, hidden_size*2

        attention_weights = self.attention(lstm_out)
        context_vector = torch.sum(attention_weights * lstm_out, dim=1)

        out = self.fc(context_vector)
        return out


examples_X = []
examples_y = []
model = None
sequence_length = 30


def normalize_data(data):
    non_zero_data = [x for x in data if x != 0.0]

    if len(non_zero_data) == 0:
        return data

    mean = np.mean(non_zero_data)
    std = np.std(non_zero_data)
    if std == 0:
        std = 1

    normalized = []
    for val in data:
        if val == 0.0:
            normalized.append(0.0)
        else:
            normalized.append((val - mean) / std)

    return np.array(normalized)


def prepare_sequence(speed_values, target_length=30):
    if len(speed_values) == target_length:
        return speed_values
    elif len(speed_values) < target_length:
        padded = speed_values + [0.0] * (target_length - len(speed_values))
        return padded
    else:
        interpolated = np.interp(
            np.linspace(0, 1, target_length),
            np.linspace(0, 1, len(speed_values)),
            speed_values
        )
        return interpolated.tolist()


def add_example_manually():
    try:
        speed_str = speed_input.get("1.0", tk.END).strip()
        if not speed_str:
            raise ValueError("Please enter speed values")

        speed_values = [float(x) for x in speed_str.replace(',', ' ').split()]

        if len(speed_values) < 5:
            raise ValueError("Please enter at least 5 speed values")

        prepared_sequence = prepare_sequence(speed_values, sequence_length)
        normalized_speed = normalize_data(prepared_sequence)

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
            header = next(reader)  # skippam header

            for row in reader:
                try:
                    if len(row) < 2:
                        raise ValueError("Row too short")

                    # zadnji element je ocena
                    label = int(row[-1])
                    speed_values = [float(x) for x in row[:-1]]

                    # Pripravi zaporedje z paddingom ali interpolacijo
                    prepared_sequence = prepare_sequence(speed_values, sequence_length)
                    normalized_speed = normalize_data(prepared_sequence)

                    examples_X.append(normalized_speed)
                    examples_y.append(label)
                    loaded_count += 1

                except Exception as e:
                    print(f"Error processing row: {e}")
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
            num_classes=5
        )

        model.load_state_dict(torch.load(filename, map_location=torch.device('cpu')))
        model.eval()

        status_label.config(text=f"Model successfully loaded from {filename}")

    except Exception as e:
        messagebox.showerror("Error", f"Failed to load model: {str(e)}")
        print(f"Detailed error: {repr(e)}")


def train_model():
    global model

    if len(examples_X) < 10:
        messagebox.showwarning("Insufficient Data", "Please add at least 10 examples.")
        return

    try:
        X_tensor = torch.tensor(np.array(examples_X), dtype=torch.float32).unsqueeze(2)
        y_tensor = torch.tensor(examples_y, dtype=torch.long)

        if model is None:
            model = BiLSTMSpeedEconomyModel(hidden_size=128, num_layers=3, num_classes=5)
            for name, param in model.named_parameters():
                if 'weight' in name:
                    nn.init.xavier_normal_(param)  # stabilizira malo

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
                status_label.config(
                    text=f"Epoch: {epoch}/{epochs}, Loss: {loss.item():.4f}, LR: {optimizer.param_groups[0]['lr']:.6f}")
                window.update()

        # ocena natančnosti
        with torch.no_grad():
            output = model(X_tensor)
            _, predicted = torch.max(output.data, 1)
            total = y_tensor.size(0)
            correct = (predicted == y_tensor).sum().item()
            accuracy = 100 * correct / total

        status_label.config(text=f"✓ Model trained. Final Loss: {loss.item():.4f}, Accuracy: {accuracy:.2f}%")
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

        if len(speed_values) < 5:
            raise ValueError("Please enter at least 5 speed values")

        prepared_sequence = prepare_sequence(speed_values, sequence_length)
        normalized_speed = normalize_data(prepared_sequence)

        input_tensor = torch.tensor(np.array([normalized_speed]), dtype=torch.float32).unsqueeze(2)

        model.eval()
        with torch.no_grad():
            output = model(input_tensor)
            probabilities = torch.nn.functional.softmax(output, dim=1)
            _, predicted = torch.max(output.data, 1)
            predicted_level = predicted.item()

        descriptions = [
            f"Economy Level 0",
            f"Economy Level 1",
            f"Economy Level 2",
            f"Economy Level 3",
            f"Economy Level 4"
        ]

        result_text = f"Predicted Economy Level: {predicted_level}\n{descriptions[predicted_level]}"

        result_text += "\n\nProbabilities:"
        for i, prob in enumerate(probabilities[0]):
            result_text += f"\nLevel {i}: {prob.item():.2%}"

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

    plt.figure(figsize=(8, 4))

    levels_shown = set()
    for i, (x, y) in enumerate(zip(examples_X, examples_y)):
        if y not in levels_shown and len(levels_shown) < 5:
            plt.plot(x, label=f"Level {y}")
            levels_shown.add(y)

            if len(levels_shown) >= 5:
                break

    plt.legend()
    plt.title("Example Speed Patterns")
    plt.xlabel("Time Steps")
    plt.ylabel("Normalized Speed (0.0 = padding)")

    if hasattr(update_plot, 'canvas'):
        update_plot.canvas.get_tk_widget().destroy()

    canvas = FigureCanvasTkAgg(plt.gcf(), master=plot_frame)
    canvas.draw()
    canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
    update_plot.canvas = canvas

    plt.close()


def plot_loss(losses):
    plt.figure(figsize=(8, 4))
    plt.plot(losses)
    plt.title("Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True)

    try:
        plt.savefig("training_loss.png")
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
        torch.cuda.empty_cache()  # Če bi bil model na CUDA
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
window.title("BiLSTM Speed Economy Model - Enhanced")
window.geometry("800x700")

notebook = ttk.Notebook(window)
notebook.grid(row=0, column=0, columnspan=2, sticky="nsew", padx=10, pady=10)

training_tab = ttk.Frame(notebook)
prediction_tab = ttk.Frame(notebook)
visualization_tab = ttk.Frame(notebook)

notebook.add(training_tab, text="Training")
notebook.add(prediction_tab, text="Prediction")
notebook.add(visualization_tab, text="Visualization")

# Training Tab
examples_frame = ttk.LabelFrame(training_tab, text="Add Training Examples")
examples_frame.pack(fill="both", expand=True, padx=10, pady=10)

tk.Label(examples_frame, text="Enter speed values (space or comma separated):").pack(anchor="w", padx=5, pady=5)
speed_input = tk.Text(examples_frame, height=5, width=60)
speed_input.pack(fill="both", expand=True, padx=5, pady=5)

economy_frame = ttk.Frame(examples_frame)
economy_frame.pack(fill="x", padx=5, pady=5)

tk.Label(economy_frame, text="Economy Level:").pack(side="left")
economy_var = tk.StringVar(value="0")
for i, desc in enumerate(
        ["Very Economical (0)", "Economical (1)", "Moderate (2)", "Uneconomical (3)", "Very Uneconomical (4)"]):
    tk.Radiobutton(economy_frame, text=desc, variable=economy_var, value=str(i)).pack(side="left", padx=5)

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

# Prediction Tab
predict_frame = ttk.LabelFrame(prediction_tab, text="Single Prediction")
predict_frame.pack(fill="both", expand=True, padx=10, pady=10)

tk.Label(predict_frame, text="Enter speed values (space or comma separated):").pack(anchor="w", padx=5, pady=5)
predict_input = tk.Text(predict_frame, height=5, width=60)
predict_input.pack(fill="both", expand=True, padx=5, pady=5)

single_predict_frame = ttk.Frame(predict_frame)
single_predict_frame.pack(fill="x", padx=5, pady=5)

ttk.Button(single_predict_frame, text="Predict Single", command=predict_single).pack(side="left", padx=5)

result_label = tk.Label(predict_frame, text="", font=("Arial", 12), justify="left")
result_label.pack(fill="both", expand=True, padx=5, pady=5)

# CSV Prediction Frame
csv_predict_frame = ttk.LabelFrame(prediction_tab, text="CSV File Prediction")
csv_predict_frame.pack(fill="both", expand=True, padx=10, pady=10)

tk.Label(csv_predict_frame, text="Process entire CSV file and save predictions to TXT file:").pack(anchor="w", padx=5,
                                                                                                   pady=5)
ttk.Button(csv_predict_frame, text="Predict CSV File", command=predict_csv_file).pack(padx=5, pady=10)

# Visualization Tab
plot_frame = ttk.LabelFrame(visualization_tab, text="Speed Patterns")
plot_frame.pack(fill="both", expand=True, padx=10, pady=10)

# Status and Exit
status_label = tk.Label(window, text="Ready - Enhanced with CSV prediction and padding support", bd=1, relief=tk.SUNKEN,
                        anchor=tk.W)
status_label.grid(row=10, column=0, columnspan=2, sticky="we")

ttk.Button(window, text="Exit", command=exit_application).grid(row=11, column=0, columnspan=2, pady=5)

window.grid_rowconfigure(0, weight=1)
window.grid_columnconfigure(0, weight=1)

update_plot()

window.protocol("WM_DELETE_WINDOW", exit_application)

window.mainloop()