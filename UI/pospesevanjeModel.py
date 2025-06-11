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


class AccelerationEconomyModel(nn.Module):
    def __init__(self, input_size=1, hidden_size=64, num_layers=2, num_classes=5):
        super(AccelerationEconomyModel, self).__init__()
        self.lstm = nn.LSTM(input_size=input_size, 
                           hidden_size=hidden_size,
                           num_layers=num_layers, 
                           batch_first=True,
                           dropout=0.2 if num_layers > 1 else 0)
        self.attention = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1),
            nn.Softmax(dim=1)
        )
        self.fc = nn.Linear(hidden_size, num_classes)
        
    def forward(self, x):
        lstm_out, (hn, cn) = self.lstm(x)  # lstm_out shape: (batch, seq_len, hidden_size)
        
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
        accel_str = accel_input.get("1.0", tk.END).strip()
        if not accel_str:
            raise ValueError("Please enter acceleration values")
            
        accel_values = [float(x) for x in accel_str.replace(',', ' ').split()]
        
        if len(accel_values) < 5:
            raise ValueError("Please enter at least 5 acceleration values")
            
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
        
        status_label.config(text=f"Added example #{len(examples_X)} (Economy Level: {economy_level})")
        
        accel_input.delete("1.0", tk.END)
        
        update_plot()
        
    except Exception as e:
        messagebox.showerror("Error", f"Invalid input: {e}")


def load_from_csv():
    """Load examples from a CSV file"""
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
            header = next(reader)  # skip header
            
            for row in reader:
                try:
                    # Skip empty or incomplete rows
                    if len(row) < sequence_length + 1:
                        raise ValueError("Row too short")

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
            model = AccelerationEconomyModel(hidden_size=128, num_layers=3, num_classes=5)
            for name, param in model.named_parameters():
                if 'weight' in name:
                    nn.init.xavier_normal_(param)
        
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.AdamW(model.parameters(), lr=0.0001, weight_decay=0.01)  # Nižji learning rate
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
        accel_str = predict_input.get("1.0", tk.END).strip()
        if not accel_str:
            raise ValueError("Please enter acceleration values")
            
        accel_values = [float(x) for x in accel_str.replace(',', ' ').split()]
        
        if len(accel_values) < 5:
            raise ValueError("Please enter at least 5 acceleration values")
            
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
    plt.title("Example Acceleration Patterns")
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
    plt.title("Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True)
    
    try:
        plt.savefig("training_loss.png")
    except:
        pass
    
    plt.show()

def generate_sample_data():
    global examples_X, examples_y
    
    try:
        if messagebox.askyesno("Generate Data", "This will create synthetic acceleration data WITHOUT preset economy levels. You'll need to assign economy levels manually. Continue?"):
            num_samples = 10
            
            for i in range(num_samples):
                pattern = np.zeros(sequence_length)
                pattern_type = i % 5  # 5 different pattern types
                
                if pattern_type == 0:
                    pattern = np.sin(np.linspace(0, 3*np.pi, sequence_length)) * 5
                elif pattern_type == 1:
                    pattern = np.sin(np.linspace(0, 2*np.pi, sequence_length)) * 15
                elif pattern_type == 2:
                    pattern = np.sin(np.linspace(0, 4*np.pi, sequence_length)) * 25
                    spikes = np.random.randint(0, sequence_length, 3)
                    for spike in spikes:
                        pattern[spike] *= 1.5
                elif pattern_type == 3:
                    midpoint = sequence_length // 2
                    pattern[:midpoint] = np.linspace(0, 30, midpoint)
                    pattern[midpoint:] = np.linspace(30, 0, sequence_length - midpoint)
                else:
                    pattern = np.cumsum(np.random.normal(0, 5, sequence_length))
                    pattern = pattern - np.mean(pattern)  # Center around zero
                
                # Add some noise
                pattern += np.random.normal(0, 3, sequence_length)
                
                normalized = normalize_data(pattern)
                
                examples_X.append(normalized)
                
                plt.figure(figsize=(8, 4))
                plt.plot(pattern)
                plt.title(f"Sample Pattern {i+1}")
                plt.xlabel("Time Steps")
                plt.ylabel("Acceleration")
                plt.grid(True)
                
                plt.savefig(f"temp_pattern_{i}.png")
                plt.close()
                
                rating_window = tk.Toplevel(window)
                rating_window.title(f"Rate Pattern {i+1}")
                rating_window.geometry("400x500")
                
                img = tk.PhotoImage(file=f"temp_pattern_{i}.png")
                img_label = tk.Label(rating_window, image=img)
                img_label.image = img  
                img_label.pack(pady=10)
                
                tk.Label(rating_window, 
                       text="How would you rate this driving pattern?",
                       font=("Arial", 12)).pack(pady=10)
                
                rating_var = tk.IntVar(value=-1)
                
                for j, desc in enumerate(["0 - Very Economical", "1 - Economical", 
                                        "2 - Moderate", "3 - Uneconomical", 
                                        "4 - Very Uneconomical"]):
                    tk.Radiobutton(rating_window, text=desc, variable=rating_var, 
                                 value=j, font=("Arial", 10)).pack(anchor="w", padx=20)
                
                rate_button = tk.Button(rating_window, text="Submit Rating", 
                                      font=("Arial", 12), bg="#4CAF50", fg="white",
                                      command=lambda: rating_window.destroy())
                rate_button.pack(pady=20)
                
                window.wait_window(rating_window)
                
                rating = rating_var.get()
                if rating >= 0: 
                    examples_y.append(rating)
                else:
                    examples_X.pop()
                    messagebox.showinfo("Rating Skipped", 
                                      f"Pattern {i+1} was skipped.")
                
                try:
                    os.remove(f"temp_pattern_{i}.png")
                except:
                    pass
            
            status_label.config(text=f"Generated and rated {len(examples_y)} sample examples")
            update_plot()
        
    except Exception as e:
        messagebox.showerror("Error", f"Failed to generate samples: {e}")
        try:
            for i in range(10):
                temp_file = f"temp_pattern_{i}.png"
                if os.path.exists(temp_file):
                    os.remove(temp_file)
        except Exception as cleanup_error:
            print(f"Cleanup error: {cleanup_error}")


def clear_data():
    global examples_X, examples_y
    
    if messagebox.askyesno("Confirm", "Are you sure you want to clear all training data?"):
        examples_X = []
        examples_y = []
        status_label.config(text="All training data cleared")
        update_plot()


window = tk.Tk()
window.title("Acceleration Economy Model")
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


tk.Label(examples_frame, text="Enter acceleration values (space or comma separated):").pack(anchor="w", padx=5, pady=5)
accel_input = tk.Text(examples_frame, height=5, width=60)
accel_input.pack(fill="both", expand=True, padx=5, pady=5)


economy_frame = ttk.Frame(examples_frame)
economy_frame.pack(fill="x", padx=5, pady=5)

tk.Label(economy_frame, text="Economy Level:").pack(side="left")
economy_var = tk.StringVar(value="0")
for i, desc in enumerate(["Very Economical (0)", "Economical (1)", "Moderate (2)", "Uneconomical (3)", "Very Uneconomical (4)"]):
    tk.Radiobutton(economy_frame, text=desc, variable=economy_var, value=str(i)).pack(side="left", padx=5)


button_frame = ttk.Frame(examples_frame)
button_frame.pack(fill="x", padx=5, pady=10)

ttk.Button(button_frame, text="Add Example", command=add_example_manually).pack(side="left", padx=5)
ttk.Button(button_frame, text="Load from CSV", command=load_from_csv).pack(side="left", padx=5)
ttk.Button(button_frame, text="Generate Sample Data", command=generate_sample_data).pack(side="left", padx=5)
ttk.Button(button_frame, text="Clear Data", command=clear_data).pack(side="left", padx=5)


model_frame = ttk.LabelFrame(training_tab, text="Model Training")
model_frame.pack(fill="both", expand=True, padx=10, pady=10)


ttk.Button(model_frame, text="Train Model", command=train_model).pack(side="left", padx=10, pady=10)
ttk.Button(model_frame, text="Save Model", command=save_model).pack(side="left", padx=10, pady=10)
ttk.Button(model_frame, text="Load Model", command=load_model).pack(side="left", padx=10, pady=10)


predict_frame = ttk.LabelFrame(prediction_tab, text="Predict Economy Level")
predict_frame.pack(fill="both", expand=True, padx=10, pady=10)


tk.Label(predict_frame, text="Enter acceleration values (space or comma separated):").pack(anchor="w", padx=5, pady=5)
predict_input = tk.Text(predict_frame, height=5, width=60)
predict_input.pack(fill="both", expand=True, padx=5, pady=5)


ttk.Button(predict_frame, text="Predict", command=predict).pack(padx=5, pady=10)

result_label = tk.Label(predict_frame, text="", font=("Arial", 12), justify="left")
result_label.pack(fill="both", expand=True, padx=5, pady=5)


plot_frame = ttk.LabelFrame(visualization_tab, text="Acceleration Patterns")
plot_frame.pack(fill="both", expand=True, padx=10, pady=10)


status_label = tk.Label(window, text="Ready", bd=1, relief=tk.SUNKEN, anchor=tk.W)
status_label.grid(row=10, column=0, columnspan=2, sticky="we")


window.grid_rowconfigure(0, weight=1)
window.grid_columnconfigure(0, weight=1)

update_plot()

window.mainloop()
torch.cuda.empty_cache()