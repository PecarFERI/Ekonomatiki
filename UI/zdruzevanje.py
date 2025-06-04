import torch
import torch.nn as nn
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import numpy as np
import csv
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import gc
import os
from sklearn.metrics import classification_report
import subprocess
import sys

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
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_size, num_classes)
        )

    def forward(self, x, mask=None):
        lstm_out, (hn, cn) = self.lstm(x)

        if mask is not None:
            lstm_out = lstm_out * mask.unsqueeze(-1)

        attention_weights = self.attention(lstm_out)

        if mask is not None:
            attention_weights = attention_weights * mask.unsqueeze(-1)
            attention_weights = attention_weights / (attention_weights.sum(dim=1, keepdim=True) + 1e-8)

        context_vector = torch.sum(attention_weights * lstm_out, dim=1)
        out = self.classifier(context_vector)
        return out


class AccelerationEconomyModel(nn.Module):
    def __init__(self, input_size=1, hidden_size=64, num_layers=2, num_classes=6):
        super(AccelerationEconomyModel, self).__init__()
        self.bilstm = nn.LSTM(input_size=input_size,
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
        lstm_out, (hn, cn) = self.bilstm(x)

        if mask is not None:
            lstm_out = lstm_out * mask.unsqueeze(-1)

        attention_weights = self.attention(lstm_out)

        if mask is not None:
            attention_weights = attention_weights * mask.unsqueeze(-1)
            attention_weights = attention_weights / (attention_weights.sum(dim=1, keepdim=True) + 1e-8)

        context_vector = torch.sum(attention_weights * lstm_out, dim=1)

        out = self.classifier(context_vector)
        return out


class UnifiedEconomyPredictor:
    def __init__(self):
        self.speed_model = None
        self.acceleration_model = None
        self.class_names = [
            'Zelo ekonomično', 'Ekonomično', 'Zmerno ekonomično',
            'Neekonomično', 'Zelo neekonomično', 'Ekstremno neekonomično'
        ]

    def load_speed_model(self, model_path):
        try:
            self.speed_model = BiLSTMSpeedEconomyModel(
                input_size=1,
                hidden_size=128,
                num_layers=3,
                num_classes=6
            )
            self.speed_model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
            self.speed_model.eval()
            print(f"Model za hitrost uspešno naložen: {model_path}")
            return True
        except Exception as e:
            print(f"Napaka pri nalaganju modela za hitrost: {e}")
            return False

    def load_acceleration_model(self, model_path):
        try:
            self.acceleration_model = AccelerationEconomyModel(
                input_size=1,
                hidden_size=128,
                num_layers=3,
                num_classes=6
            )
            self.acceleration_model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
            self.acceleration_model.eval()
            print(f"Model za pospešek uspešno naložen: {model_path}")
            return True
        except Exception as e:
            print(f"Napaka pri nalaganju modela za pospešek: {e}")
            return False

    def create_mask(self, data):
        mask = (data != 0.0).float()
        return mask

    def normalize_data_improved(self, data):
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

    def predict_single(self, speed_data, acceleration_data):
        results = {}

        if self.speed_model is not None and len(speed_data) == 20:
            try:
                normalized_speed = self.normalize_data_improved(speed_data)
                input_tensor = torch.tensor(np.array([normalized_speed]), dtype=torch.float32).unsqueeze(2)

                with torch.no_grad():
                    mask = self.create_mask(torch.tensor(normalized_speed)).unsqueeze(0)
                    output = self.speed_model(input_tensor, mask)
                    probabilities = torch.nn.functional.softmax(output, dim=1)
                    _, predicted = torch.max(output.data, 1)

                    results['speed'] = {
                        'prediction': predicted.item(),
                        'confidence': probabilities[0][predicted.item()].item(),
                        'probabilities': probabilities[0].tolist()
                    }
                    print(
                        f"Napoved hitrosti: {predicted.item()}, zaupanje: {probabilities[0][predicted.item()].item():.3f}")
            except Exception as e:
                print(f"Napaka pri napovedi hitrosti: {e}")

        if self.acceleration_model is not None and len(acceleration_data) == 20:
            try:
                normalized_acceleration = self.normalize_data_improved(acceleration_data)
                input_tensor = torch.tensor(np.array([normalized_acceleration]), dtype=torch.float32).unsqueeze(2)

                with torch.no_grad():
                    mask = self.create_mask(torch.tensor(normalized_acceleration)).unsqueeze(0)
                    output = self.acceleration_model(input_tensor, mask)
                    probabilities = torch.nn.functional.softmax(output, dim=1)
                    _, predicted = torch.max(output.data, 1)

                    results['acceleration'] = {
                        'prediction': predicted.item(),
                        'confidence': probabilities[0][predicted.item()].item(),
                        'probabilities': probabilities[0].tolist()
                    }
                    print(
                        f"Napoved pospeška: {predicted.item()}, zaupanje: {probabilities[0][predicted.item()].item():.3f}")
            except Exception as e:
                print(f"Napaka pri napovedi pospeška: {e}")

        if 'speed' in results and 'acceleration' in results:
            try:
                combined_probabilities = []
                for i in range(6):
                    #dodana utez 0,6 za model hittosti
                    weighted_prob = (results['speed']['probabilities'][i] * 0.6 +
                                     results['acceleration']['probabilities'][i] * 0.4)
                    combined_probabilities.append(weighted_prob)

                combined_prediction = combined_probabilities.index(max(combined_probabilities))
                results['combined'] = {
                    'prediction': combined_prediction,
                    'confidence': max(combined_probabilities),
                    'probabilities': combined_probabilities
                }
                print(f"Skupna napoved: {combined_prediction}, zaupanje: {max(combined_probabilities):.3f}")
            except Exception as e:
                print(f"Napaka pri skupni napovedi: {e}")

        return results


predictor = UnifiedEconomyPredictor()


def load_speed_model():
    filename = filedialog.askopenfilename(
        title="Izberi model za hitrost",
        filetypes=(("PyTorch Model", "*.pt"), ("All files", "*.*"))
    )

    if filename:
        if predictor.load_speed_model(filename):
            speed_model_label.config(text=f"Model za hitrost: {os.path.basename(filename)}", fg="green")
            status_label.config(text="Model za hitrost uspešno naložen")
        else:
            messagebox.showerror("Napaka", "Napaka pri nalaganju modela za hitrost")


def load_acceleration_model():
    filename = filedialog.askopenfilename(
        title="Izberi model za pospešek",
        filetypes=(("PyTorch Model", "*.pt"), ("All files", "*.*"))
    )

    if filename:
        if predictor.load_acceleration_model(filename):
            accel_model_label.config(text=f"Model za pospešek: {os.path.basename(filename)}", fg="green")
            status_label.config(text="Model za pospešek uspešno naložen")
        else:
            messagebox.showerror("Napaka", "Napaka pri nalaganju modela za pospešek")


def predict_manual():
    try:
        speed_str = speed_input.get("1.0", tk.END).strip()
        speed_values = []
        if speed_str:
            speed_values = [float(x) for x in speed_str.replace(',', ' ').split()]
            if len(speed_values) != 20:
                raise ValueError(f"Hitrost: potrebno je natanko 20 vrednosti (dobil {len(speed_values)})")

        accel_str = accel_input.get("1.0", tk.END).strip()
        accel_values = []
        if accel_str:
            accel_values = [float(x) for x in accel_str.replace(',', ' ').split()]
            if len(accel_values) != 20:
                raise ValueError(f"Pospešek: potrebno je natanko 20 vrednosti (dobil {len(accel_values)})")

        if not speed_values and not accel_values:
            raise ValueError("Vnesi podatke za hitrost in/ali pospešek")

        if not speed_values and predictor.speed_model is None:
            raise ValueError("Model za hitrost ni naložen, vnesi podatke za pospešek")
        if not accel_values and predictor.acceleration_model is None:
            raise ValueError("Model za pospešek ni naložen, vnesi podatke za hitrost")

        results = predictor.predict_single(speed_values, accel_values)

        if not results:
            raise ValueError("Ni bilo mogoče narediti nobene napovedi")

        display_results(results)

    except Exception as e:
        messagebox.showerror("Napaka", f"Napaka pri napovedovanju: {e}")


def generate_output_filename(input_filename):
    base_name = os.path.basename(input_filename)
    name_without_ext = os.path.splitext(base_name)[0]
    if name_without_ext.endswith('_predict'):
        output_name = name_without_ext.replace('_predict', '_output') + '.csv'
    else:
        output_name = name_without_ext + '_output.csv'

    input_dir = os.path.dirname(input_filename)
    return os.path.join(input_dir, output_name)


def predict_csv():
    if predictor.speed_model is None and predictor.acceleration_model is None:
        messagebox.showwarning("Opozorilo", "Naloži vsaj en model")
        return

    csv_filename = filedialog.askopenfilename(
        title="Izberi CSV datoteko",
        filetypes=(("CSV files", "*.csv"), ("All files", "*.*"))
    )

    if not csv_filename:
        return

    suggested_output = generate_output_filename(csv_filename)

    output_filename = filedialog.asksaveasfilename(
        title="Shrani rezultate",
        initialfile=os.path.basename(suggested_output),
        initialdir=os.path.dirname(suggested_output),
        defaultextension=".csv",
        filetypes=(("CSV files", "*.csv"), ("All files", "*.*"))
    )

    if not output_filename:
        return

    try:
        processed_count = 0
        error_count = 0
        model_differences = []

        with open(csv_filename, 'r') as f:
            reader = csv.reader(f)
            try:
                header = next(reader)
            except StopIteration:
                messagebox.showerror("Napaka", "CSV datoteka je prazna")
                return

            with open(output_filename, 'w', newline='') as out_f:
                writer = csv.writer(out_f)

                out_header = [f'speed_{i + 1}' for i in range(20)]
                out_header.append('predicted_label')  # dodaj label na konec
                writer.writerow(out_header)

                for row_idx, row in enumerate(reader):
                    try:
                        if len(row) < 20:
                            print(f"Vrstica {row_idx + 2}: premalo podatkov ({len(row)} < 20)")
                            error_count += 1
                            continue

                        speed_data = [float(x) for x in row[:20]]

                        accel_data = []
                        if len(row) >= 40:
                            accel_data = [float(x) for x in row[20:40]]
                        else:
                            remaining_data = row[20:] if len(row) > 20 else []
                            accel_data = [float(x) for x in remaining_data]
                            accel_data.extend([0.0] * (20 - len(accel_data)))

                        results = predictor.predict_single(speed_data, accel_data)

                        if not results:
                            print(f"Vrstica {row_idx + 2}: ni rezultatov")
                            error_count += 1
                            continue

                        predicted_label = ""
                        if 'combined' in results:
                            predicted_label = results['combined']['prediction']
                            #za razlike med modeli
                            if 'speed' in results and 'acceleration' in results:
                                speed_pred = results['speed']['prediction']
                                accel_pred = results['acceleration']['prediction']
                                if speed_pred != accel_pred:
                                    model_differences.append({
                                        'row': row_idx + 2,
                                        'speed_pred': speed_pred,
                                        'accel_pred': accel_pred,
                                        'combined_pred': predicted_label,
                                        'speed_conf': results['speed']['confidence'],
                                        'accel_conf': results['acceleration']['confidence']
                                    })
                        elif 'speed' in results:
                            predicted_label = results['speed']['prediction']
                        elif 'acceleration' in results:
                            predicted_label = results['acceleration']['prediction']

                        out_row = speed_data + [predicted_label]
                        writer.writerow(out_row)
                        processed_count += 1

                    except Exception as e:
                        print(f"Napaka v vrstici {row_idx + 2}: {e}")
                        error_count += 1
                        continue

        display_csv_analysis(processed_count, error_count, model_differences)

        update_stats_label(processed_count, error_count, model_differences)

        message = f"Obdelanih {processed_count} vrstic"
        if error_count > 0:
            message += f", {error_count} napak"
        message += f"\nRezultati shranjeni: {output_filename}"

        messagebox.showinfo("Uspeh", message)
        status_label.config(text=f"CSV napoved končana: {processed_count} vrstic obdelanih, {error_count} napak")

    except Exception as e:
        messagebox.showerror("Napaka", f"Napaka pri obdelavi CSV: {e}")


def draw_map():
    try:
        if os.path.exists("izrisZemljevidaPoStopnjah.py"):
            status_label.config(text="Zaganjam izris zemljevida...")

            process = subprocess.Popen([sys.executable, "izrisZemljevidaPoStopnjah.py"],
                                       stdout=subprocess.PIPE,
                                       stderr=subprocess.PIPE)

            stdout, stderr = process.communicate()

            if process.returncode == 0:
                status_label.config(text="Zemljevid uspešno izrisan")
                messagebox.showinfo("Uspeh", "Zemljevid je bil uspešno izrisan!")
            else:
                error_msg = stderr.decode('utf-8') if stderr else "Neznana napaka"
                status_label.config(text="Napaka pri izrisu zemljevida")
                messagebox.showerror("Napaka", f"Napaka pri izrisu zemljevida:\n{error_msg}")

        else:
            messagebox.showerror("Napaka", "Datoteka 'izrisZemljevidaPoStopnjah.py' ni bila najdena v trenutni mapi!")

    except Exception as e:
        status_label.config(text="Napaka pri zaganjanju zemljevida")
        messagebox.showerror("Napaka", f"Napaka pri zaganjanju skripte za zemljevid:\n{str(e)}")




def update_stats_label(processed_count, error_count, model_differences):
    if processed_count == 0:
        stats_text = "Ni bilo obdelanih vrstic"
    else:
        same_predictions = processed_count - len(model_differences)
        different_predictions = len(model_differences)

        stats_text = f"📊 Obdelanih vrstic: {processed_count} | Napake: {error_count}\n"

        if predictor.speed_model is not None and predictor.acceleration_model is not None:
            stats_text += f"🤝 Enake napovedi: {same_predictions} ({same_predictions / processed_count * 100:.1f}%)\n"
            stats_text += f"⚖️ Različne napovedi: {different_predictions} ({different_predictions / processed_count * 100:.1f}%)"
        else:
            stats_text += "ℹ️ Uporabljen samo en model"

    stats_label.config(text=stats_text)


def display_csv_analysis(processed_count, error_count, model_differences):
    result_text = f"📊 ANALIZA CSV OBDELAVE:\n\n"
    result_text += f"Skupno obdelanih vrstic: {processed_count}\n"
    result_text += f"Napake: {error_count}\n\n"

    if model_differences:
        result_text += f"🔍 RAZLIKE MED MODELI:\n"
        result_text += f"Število vrstic z različnimi napovedmi: {len(model_differences)}\n"
        result_text += f"Odstotek razlik: {len(model_differences) / processed_count * 100:.1f}%\n\n"

        result_text += "Prvih 10 razlik:\n"
        for i, diff in enumerate(model_differences[:10]):
            result_text += f"Vrstica {diff['row']}: "
            result_text += f"Hitrost={diff['speed_pred']}, "
            result_text += f"Pospešek={diff['accel_pred']}, "
            result_text += f"Skupno={diff['combined_pred']}\n"

        if len(model_differences) > 10:
            result_text += f"... in še {len(model_differences) - 10} razlik\n"

        result_text += "\n📈 PORAZDELITEV RAZLIK:\n"
        speed_preds = [d['speed_pred'] for d in model_differences]
        accel_preds = [d['accel_pred'] for d in model_differences]

        for i in range(6):
            speed_count = speed_preds.count(i)
            accel_count = accel_preds.count(i)
            if speed_count > 0 or accel_count > 0:
                result_text += f"Stopnja {i}: Hitrost={speed_count}x, Pospešek={accel_count}x\n"
    else:
        result_text += "✅ Vsi modeli so se strinjali pri vseh napovedih!\n"

    result_text += f"\n💡 Utež kombinacije: 60% hitrost, 40% pospešek"

    result_display.config(state='normal')
    result_display.delete('1.0', tk.END)
    result_display.insert('1.0', result_text)
    result_display.config(state='disabled')


def display_results(results):
    result_text = ""

    if 'speed' in results:
        speed_pred = results['speed']['prediction']
        speed_conf = results['speed']['confidence']
        result_text += f"🚗 MODEL ZA HITROST:\n"
        result_text += f"Napoved: Stopnja {speed_pred} - {predictor.class_names[speed_pred]}\n"
        result_text += f"Zaupanje: {speed_conf:.1%}\n\n"

    if 'acceleration' in results:
        accel_pred = results['acceleration']['prediction']
        accel_conf = results['acceleration']['confidence']
        result_text += f"⚡ MODEL ZA POSPEŠEK:\n"
        result_text += f"Napoved: Stopnja {accel_pred} - {predictor.class_names[accel_pred]}\n"
        result_text += f"Zaupanje: {accel_conf:.1%}\n\n"

    if 'combined' in results:
        combined_pred = results['combined']['prediction']
        combined_conf = results['combined']['confidence']
        result_text += f"🎯 SKUPNA OCENA (60% hitrost, 40% pospešek):\n"
        result_text += f"Napoved: Stopnja {combined_pred} - {predictor.class_names[combined_pred]}\n"
        result_text += f"Zaupanje: {combined_conf:.1%}\n\n"

        result_text += "Podrobne verjetnosti:\n"
        for i, prob in enumerate(results['combined']['probabilities']):
            result_text += f"Stopnja {i} ({predictor.class_names[i]}): {prob:.1%}\n"

    result_display.config(state='normal')
    result_display.delete('1.0', tk.END)
    result_display.insert('1.0', result_text)
    result_display.config(state='disabled')


def clear_inputs():
    speed_input.delete('1.0', tk.END)
    accel_input.delete('1.0', tk.END)
    result_display.config(state='normal')
    result_display.delete('1.0', tk.END)
    result_display.config(state='disabled')


def test_models():
    result_text = "TESTIRANJE MODELOV:\n\n"

    if predictor.speed_model is not None:
        result_text += "✓ Model za hitrost je naložen\n"
    else:
        result_text += "✗ Model za hitrost ni naložen\n"

    if predictor.acceleration_model is not None:
        result_text += "✓ Model za pospešek je naložen\n"
    else:
        result_text += "✗ Model za pospešek ni naložen\n"

    result_text += "\nTEST Z NAKLJUČNIMI PODATKI:\n"

    test_speed = np.random.randn(20).tolist()
    test_accel = np.random.randn(20).tolist()

    try:
        results = predictor.predict_single(test_speed, test_accel)
        result_text += f"Test uspešen - dobljeno {len(results)} rezultatov\n"
        for key in results:
            result_text += f"- {key}: napoved {results[key]['prediction']}\n"
    except Exception as e:
        result_text += f"Test neuspešen: {e}\n"

    result_display.config(state='normal')
    result_display.delete('1.0', tk.END)
    result_display.insert('1.0', result_text)
    result_display.config(state='disabled')

def exit_application():
    if messagebox.askyesno("Izhod", "Ali si prepričan, da se želiš odjaviti?"):
        gc.collect()
        window.destroy()


window = tk.Tk()
window.title("Združen model za ekonomičnost vožnje - Popravljena verzija")
window.geometry("1200x900")

notebook = ttk.Notebook(window)
notebook.pack(fill="both", expand=True, padx=10, pady=10)

model_tab = ttk.Frame(notebook)
notebook.add(model_tab, text="Modeli")

predict_tab = ttk.Frame(notebook)
notebook.add(predict_tab, text="Napovedi")

csv_tab = ttk.Frame(notebook)
notebook.add(csv_tab, text="CSV Obdelava")

model_frame = ttk.LabelFrame(model_tab, text="Nalaganje modelov")
model_frame.pack(fill="both", expand=True, padx=10, pady=10)

ttk.Button(model_frame, text="Naloži model za hitrost", command=load_speed_model).pack(pady=10)
speed_model_label = tk.Label(model_frame, text="Model za hitrost: ni naložen", fg="red")
speed_model_label.pack(pady=5)

ttk.Button(model_frame, text="Naloži model za pospešek", command=load_acceleration_model).pack(pady=10)
accel_model_label = tk.Label(model_frame, text="Model za pospešek: ni naložen", fg="red")
accel_model_label.pack(pady=5)

ttk.Button(model_frame, text="Testiraj modele", command=test_models).pack(pady=10)

input_frame = ttk.LabelFrame(predict_tab, text="Vhodni podatki")
input_frame.pack(fill="both", expand=True, padx=10, pady=10)

tk.Label(input_frame, text="Hitrost (20 vrednosti, ločenih s presledki ali vejicami):").pack(anchor="w", padx=5, pady=5)
speed_input = tk.Text(input_frame, height=3, width=80)
speed_input.pack(fill="x", padx=5, pady=5)

tk.Label(input_frame, text="Pospešek (20 vrednosti, ločenih s presledki ali vejicami):").pack(anchor="w", padx=5,
                                                                                              pady=5)
accel_input = tk.Text(input_frame, height=3, width=80)
accel_input.pack(fill="x", padx=5, pady=5)

button_frame = ttk.Frame(input_frame)
button_frame.pack(fill="x", padx=5, pady=10)

ttk.Button(button_frame, text="Napovej", command=predict_manual).pack(side="left", padx=5)
ttk.Button(button_frame, text="Počisti", command=clear_inputs).pack(side="left", padx=5)

result_frame = ttk.LabelFrame(predict_tab, text="Rezultati")
result_frame.pack(fill="both", expand=True, padx=10, pady=10)

result_display = tk.Text(result_frame, height=15, width=80, state='disabled')
result_display.pack(fill="both", expand=True, padx=5, pady=5)

csv_frame = ttk.LabelFrame(csv_tab, text="CSV obdelava")
csv_frame.pack(fill="both", expand=True, padx=10, pady=10)

csv_info = tk.Label(csv_frame,
                    text="Format CSV datoteke:\n" +
                         "- Če imaš oba modela: prvih 20 stolpcev = hitrost, naslednjih 20 = pospešek\n" +
                         "- Če imaš samo model za hitrost: prvih 20 stolpcev = hitrost\n" +
                         "- Če imaš samo model za pospešek: prvih 20 stolpcev = pospešek\n" +
                         "- Ostali stolpci bodo prepisani v rezultat\n" +
                         "- Output datoteka bo poimenovana: ime_output.csv",
                    justify="left")
csv_info.pack(anchor="w", padx=10, pady=10)

ttk.Button(csv_frame, text="Obdelaj CSV datoteko", command=predict_csv).pack(pady=20)
ttk.Button(csv_frame, text="Izris zemljevida", command=draw_map).pack(pady=20)

stats_frame = ttk.LabelFrame(csv_frame, text="Statistika zadnje obdelave")
stats_frame.pack(fill="x", padx=10, pady=10)

global stats_label
stats_label = tk.Label(stats_frame, text="Še ni bilo obdelave CSV datoteke",
                       justify="left", anchor="w", bg="white", relief="sunken", padx=10, pady=10)
stats_label.pack(fill="x", padx=5, pady=5)

# Status bar
status_label = tk.Label(window, text="Pripravljeno - naloži modele za začetek",
                        bd=1, relief=tk.SUNKEN, anchor=tk.W)
status_label.pack(side="bottom", fill="x")

ttk.Button(window, text="Izhod", command=exit_application).pack(side="bottom", pady=5)

window.protocol("WM_DELETE_WINDOW", exit_application)

if __name__ == "__main__":
    window.mainloop()