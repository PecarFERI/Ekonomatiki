import torch
import torch.nn as nn
import torch.optim as optim
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import numpy as np
import csv
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg


class SpeedEconomyModel(nn.Module):
    def __init__(self, input_size=1, hidden_size=64, num_layers=2, num_classes=5):
        super(SpeedEconomyModel, self).__init__()
        self.lstm = nn.LSTM(input_size=input_size,
                            hidden_size=hidden_size,
                            num_layers=num_layers,
                            batch_first=True, #vhodne oblike so batch size, sequence length, input size
                            dropout=0.2 if num_layers > 1 else 0)
        self.attention = nn.Sequential( #utezi najbolj pomembne casovne korake
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1), #vsakemu koraku
            nn.Softmax(dim=1)
        )
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        lstm_out, (hn, cn) = self.lstm(x)  #batch, seq_len, hidden_size

        attention_weights = self.attention(lstm_out)
        context_vector = torch.sum(attention_weights * lstm_out, dim=1)

        out = self.fc(context_vector)
        return out


examples_X = []
examples_y = []
model = None
sequence_length = 30


def normalize_data(data):
    mean = np.mean(data)
    std = np.std(data)
    if std == 0:
        std = 1
    return (data - mean) / std


def add_example_manually():
    try:
        speed_str = speed_input.get("1.0", tk.END).strip()
        if not speed_str:
            raise ValueError("Please enter speed values")

        #dam v float
        speed_values = [float(x) for x in speed_str.replace(',', ' ').split()]

        if len(speed_values) < 5:
            raise ValueError("Please enter at least 5 speed values")

        #ce ni prave velikosti
        if len(speed_values) != sequence_length:
            speed_values = np.interp(
                np.linspace(0, 1, sequence_length),
                np.linspace(0, 1, len(speed_values)),
                speed_values
            )

        normalized_speed = normalize_data(speed_values)

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
            header = next(reader)  #skippam header

            for row in reader:
                try:
                    if len(row) < sequence_length + 1:
                        raise ValueError("Row too short")

                   #zadnji element je ocena
                    label = int(row[-1])

                    speed_values = [float(x) for x in row[:-1]]

                    if len(speed_values) != sequence_length:
                        speed_values = np.interp(
                            np.linspace(0, 1, sequence_length),
                            np.linspace(0, 1, len(speed_values)),
                            speed_values
                        )

                    normalized_speed = normalize_data(speed_values)

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

        model = SpeedEconomyModel(
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
        #pretvorim v tenzorje pytorch
        X_tensor = torch.tensor(np.array(examples_X), dtype=torch.float32).unsqueeze(2)
        y_tensor = torch.tensor(examples_y, dtype=torch.long)

        if model is None:
            model = SpeedEconomyModel(hidden_size=128, num_layers=3, num_classes=5)
            for name, param in model.named_parameters():
                if 'weight' in name:
                    nn.init.xavier_normal_(param) #stabilizira malo

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
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # Gradient clipping
            optimizer.step()
            scheduler.step(loss)
            losses.append(loss.item())

            #prej zaustavim
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

        #ocena natancnosyi
        with torch.no_grad():
            output = model(X_tensor)
            _, predicted = torch.max(output.data, 1)
            total = y_tensor.size(0)
            correct = (predicted == y_tensor).sum().item()
            accuracy = 100 * correct / total

        status_label.config(text=f"✓ Model trained. Final Loss: {loss.item():.4f}, Accuracy: {accuracy:.2f}%")
        progress.destroy()
        plot_loss(losses)

    except Exception as e:
        messagebox.showerror("Error", f"Error during training: {e}")


def predict():
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

        if len(speed_values) != sequence_length:
            speed_values = np.interp(
                np.linspace(0, 1, sequence_length),
                np.linspace(0, 1, len(speed_values)),
                speed_values
            )

        normalized_speed = normalize_data(speed_values)

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
    plt.ylabel("Normalized Speed")

    if hasattr(update_plot, 'canvas'):
        update_plot.canvas.get_tk_widget().destroy()

    canvas = FigureCanvasTkAgg(plt.gcf(), master=plot_frame)
    canvas.draw()
    canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
    update_plot.canvas = canvas


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
    """Clear all training data"""
    global examples_X, examples_y

    if messagebox.askyesno("Confirm", "Are you sure you want to clear all training data?"):
        examples_X = []
        examples_y = []
        status_label.config(text="All training data cleared")
        update_plot()


window = tk.Tk()
window.title("Speed Economy Model")
window.geometry("800x700")

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

predict_frame = ttk.LabelFrame(prediction_tab, text="Predict Economy Level")
predict_frame.pack(fill="both", expand=True, padx=10, pady=10)

tk.Label(predict_frame, text="Enter speed values (space or comma separated):").pack(anchor="w", padx=5, pady=5)
predict_input = tk.Text(predict_frame, height=5, width=60)
predict_input.pack(fill="both", expand=True, padx=5, pady=5)

ttk.Button(predict_frame, text="Predict", command=predict).pack(padx=5, pady=10)

result_label = tk.Label(predict_frame, text="", font=("Arial", 12), justify="left")
result_label.pack(fill="both", expand=True, padx=5, pady=5)

plot_frame = ttk.LabelFrame(visualization_tab, text="Speed Patterns")
plot_frame.pack(fill="both", expand=True, padx=10, pady=10)

# Status bar
status_label = tk.Label(window, text="Ready", bd=1, relief=tk.SUNKEN, anchor=tk.W)
status_label.grid(row=10, column=0, columnspan=2, sticky="we")

# Configure grid weights
window.grid_rowconfigure(0, weight=1)
window.grid_columnconfigure(0, weight=1)

# Update plot on start
update_plot()

# Start the GUI
window.mainloop()