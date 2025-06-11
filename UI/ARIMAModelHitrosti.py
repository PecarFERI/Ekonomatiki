import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import numpy as np
import csv
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import gc
from datetime import datetime
import os
import joblib
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
from sklearn.metrics import mean_squared_error
import warnings

warnings.filterwarnings("ignore")

sequence_length = 30

examples_X = []
examples_y = []
model = None

p = 1
d = 1
q = 1


def check_stacionarnost(timeseries):
    try:
        result = adfuller(timeseries, autolag='AIC')
        return result[1] <= 0.05  # p <= 0.05 pomeni, da je serija stacionarna
    except:
        return False


def diferencialna_serija(serije, interval=1):
    return np.array([serije[i] - serije[i - interval] for i in range(interval, len(serije))])


def normalize_data(data):
    mean = np.mean(data)
    std = np.std(data)
    if std == 0:
        std = 1
    return (data - mean) / std


def find_best_arima_params(series, max_p=3, max_d=2, max_q=3):
    best_p, best_d, best_q = 1, 1, 1
    best_aic = float('inf')

    series_copy = np.array(series)
    d_optimal = 0

    for d_test in range(max_d + 1): #za d, najde stacionarnost
        test_series = series_copy.copy()

        for _ in range(d_test):
            if len(test_series) > 1:
                test_series = diferencialna_serija(test_series)
            else:
                break

        if len(test_series) > 5 and check_stacionarnost(test_series):
            d_optimal = d_test
            break

    if d_optimal > max_d:
        d_optimal = max_d

    total_combinations = (max_p + 1) * (max_q + 1)
    tested = 0

    for p_test in range(max_p + 1): #poisce najboljso kombinacijo p, d q
        for q_test in range(max_q + 1):
            if p_test == 0 and q_test == 0:
                continue

            try:
                test_series = series_copy.copy()

                for _ in range(d_optimal):
                    if len(test_series) > 1:
                        test_series = diferencialna_serija(test_series)
                    else:
                        break

                if len(test_series) < 5:
                    continue

                model_test = ARIMA(series, order=(p_test, d_optimal, q_test))
                model_fit = model_test.fit()

                if model_fit.aic < best_aic:
                    best_aic = model_fit.aic
                    best_p = p_test
                    best_d = d_optimal
                    best_q = q_test

            except Exception as e:
                continue

            tested += 1

    return best_p, best_d, best_q, best_aic


def calculate_model_accuracy():
    global model, examples_X, examples_y

    if model is None or not examples_X or not examples_y:
        return 0, 0, 0, {}

    class_models = model['class_models']
    correct_predictions = 0
    total_predictions = len(examples_X)
    detailed_results = {}

    for i, (speed_sequence, true_label) in enumerate(zip(examples_X, examples_y)):
        class_errors = {}

        for cls, model_info in class_models.items():
            try:
                arima_model = model_info['model']
                history_forecast = arima_model.predict(start=0, end=len(speed_sequence) - 1)
                error = np.mean(np.abs(speed_sequence - history_forecast))
                class_errors[cls] = error
            except Exception as e:
                class_errors[cls] = float('inf')

        if class_errors:
            predicted_label = min(class_errors, key=class_errors.get)
        else:
            predicted_label = 0

        is_correct = (predicted_label == true_label)
        if is_correct:
            correct_predictions += 1

        detailed_results[i] = {
            'true_label': true_label,
            'predicted_label': predicted_label,
            'correct': is_correct,
            'errors': class_errors.copy()
        }

    accuracy_percentage = (correct_predictions / total_predictions) * 100 if total_predictions > 0 else 0

    return accuracy_percentage, correct_predictions, total_predictions, detailed_results


def add_example_manually():
    try:
        speed_str = speed_input.get("1.0", tk.END).strip()
        if not speed_str:
            raise ValueError("Prosim vnesite vrednosti hitrosti")

        speed_values = [float(x) for x in speed_str.replace(',', ' ').split()]

        if len(speed_values) < 5:
            raise ValueError("Prosim vnesite vsaj 5 vrednosti hitrosti")

        if len(speed_values) != sequence_length:
            speed_values = np.interp(
                np.linspace(0, 1, sequence_length),
                np.linspace(0, 1, len(speed_values)),
                speed_values
            )

        normalized_speed = normalize_data(speed_values)

        examples_X.append(normalized_speed)
        examples_y.append(int(economy_var.get()))

        status_label.config(text=f"Dodan primer #{len(examples_X)} (Ekonomičnost: {economy_var.get()})")

        speed_input.delete("1.0", tk.END)

        update_plot()

    except Exception as e:
        messagebox.showerror("Napaka", f"Neveljaven vnos: {e}")


def load_from_csv():
    try:
        filename = filedialog.askopenfilename(
            title="Izberite CSV datoteko",
            filetypes=(("CSV datoteke", "*.csv"), ("Vse datoteke", "*.*"))
        )

        if not filename:
            return

        loaded_count = 0

        with open(filename, 'r') as f:
            reader = csv.reader(f)
            header = next(reader)  # preskoči glavo

            for row in reader:
                try:
                    if len(row) < sequence_length + 1:
                        raise ValueError("Vrstica je prekratka")

                    # zadnji element je ocena ekonomičnosti
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
                    print(f"Napaka pri obdelavi vrstice: {e}")
                    continue

        status_label.config(text=f"Naloženih {loaded_count} primerov iz CSV datoteke")
        update_plot()

    except Exception as e:
        messagebox.showerror("Napaka", f"Nalaganje iz CSV ni uspelo: {e}")


def save_model():
    global model
    if model is None:
        messagebox.showinfo("Ni modela", "Prosim, najprej naučite model.")
        return

    try:
        filename = filedialog.asksaveasfilename(
            title="Shrani model",
            defaultextension=".joblib",
            filetypes=(("Joblib Model", "*.joblib"), ("Vse datoteke", "*.*"))
        )

        if filename:
            model_dir = os.path.splitext(filename)[0] + "_model_info"
            os.makedirs(model_dir, exist_ok=True)

            joblib.dump(model, filename)

            model_info = {
                "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "sequence_length": sequence_length,
                "num_examples": len(examples_X),
                "description": "ARIMA model za ekonomičnost vožnje z optimiziranimi parametri"
            }

            with open(os.path.join(model_dir, "model_info.txt"), 'w') as f:
                for key, value in model_info.items():
                    f.write(f"{key}: {value}\n")

            status_label.config(text=f"Model shranjen v {filename}")

    except Exception as e:
        messagebox.showerror("Napaka", f"Shranjevanje modela ni uspelo: {e}")


def load_model():
    global model

    try:
        filename = filedialog.askopenfilename(
            title="Naloži model",
            filetypes=(("Joblib Model", "*.joblib"), ("Vse datoteke", "*.*"))
        )

        if not filename:
            return

        model = joblib.load(filename)
        status_label.config(text=f"Model uspešno naložen iz {filename}")

    except Exception as e:
        messagebox.showerror("Napaka", f"Nalaganje modela ni uspelo: {str(e)}")
        print(f"Podrobna napaka: {repr(e)}")


def train_model():
    global model

    if len(examples_X) < 5:
        messagebox.showwarning("Premalo podatkov", "Prosim, dodajte vsaj 5 primerov.")
        return

    try:
        # Ustvari progress bar
        progress_window = tk.Toplevel(window)
        progress_window.title("Optimizacija ARIMA parametrov")
        progress_window.geometry("400x150")
        progress_window.transient(window)
        progress_window.grab_set()

        progress_label = tk.Label(progress_window, text="Iskanje optimalnih parametrov...")
        progress_label.pack(pady=10)

        progress_bar = ttk.Progressbar(progress_window, mode='determinate')
        progress_bar.pack(pady=10, padx=20, fill='x')

        status_text = tk.Label(progress_window, text="")
        status_text.pack(pady=5)

        class_models = {}
        unique_classes = list(set(examples_y))
        total_classes = len(unique_classes)

        progress_bar['maximum'] = total_classes

        for i, class_label in enumerate(unique_classes):
            progress_bar['value'] = i
            status_text.config(text=f"Optimiziram parametre za razred {class_label}...")
            progress_window.update()

            class_sequences = [x for j, x in enumerate(examples_X) if examples_y[j] == class_label]

            if len(class_sequences) == 0:
                continue

            avg_sequence = np.mean(class_sequences, axis=0)

            try:
                best_p, best_d, best_q, best_aic = find_best_arima_params(avg_sequence)

                model_fit = ARIMA(avg_sequence, order=(best_p, best_d, best_q)).fit()

                class_models[class_label] = {
                    'model': model_fit,
                    'order': (best_p, best_d, best_q),
                    'aic': best_aic
                }

            except Exception as e:
                print(f"Napaka pri optimizaciji za razred {class_label}: {e}")
                model_fit = ARIMA(avg_sequence, order=(1, 1, 1)).fit()
                class_models[class_label] = {
                    'model': model_fit,
                    'order': (1, 1, 1),
                    'aic': model_fit.aic
                }

        progress_bar['value'] = total_classes
        status_text.config(text="Optimizacija končana!")
        progress_window.update()

        model = {
            'class_models': class_models
        }

        result_text = "Optimizirani parametri za vsak razred:\n"
        for class_label, model_info in class_models.items():
            p_opt, d_opt, q_opt = model_info['order']
            aic_opt = model_info['aic']
            result_text += f"Razred {class_label}: p={p_opt}, d={d_opt}, q={q_opt} (AIC: {aic_opt:.2f})\n"

        progress_window.destroy()

        messagebox.showinfo("Optimizacija končana", result_text)

        accuracy, correct, total, _ = calculate_model_accuracy()
        status_text = f"Modeli ARIMA so optimizirani in pripravljeni | Natančnost: {accuracy:.2f}% ({correct}/{total})"
        status_label.config(text=status_text)

        clear_memory()

    except Exception as e:
        if 'progress_window' in locals():
            progress_window.destroy()
        messagebox.showerror("Napaka", f"Napaka med učenjem: {e}")
        import traceback
        traceback.print_exc()


def predict():
    global model

    if model is None:
        messagebox.showinfo("Ni modela", "Prosim, najprej naučite model.")
        return

    try:
        speed_str = predict_input.get("1.0", tk.END).strip()
        if not speed_str:
            raise ValueError("Prosim, vnesite vrednosti hitrosti")

        speed_values = [float(x) for x in speed_str.replace(',', ' ').split()]

        if len(speed_values) < 5:
            raise ValueError("Prosim, vnesite vsaj 5 vrednosti hitrosti")

        if len(speed_values) != sequence_length:
            speed_values = np.interp(
                np.linspace(0, 1, sequence_length),
                np.linspace(0, 1, len(speed_values)),
                speed_values
            )

        normalized_speed = normalize_data(speed_values)

        class_models = model['class_models']

        class_errors = {}
        for cls, model_info in class_models.items():
            try:
                arima_model = model_info['model']

                forecast = arima_model.forecast(steps=1)

                # obtojece s primerjavo
                history_forecast = arima_model.predict(start=0, end=len(normalized_speed) - 1)
                error = np.mean(np.abs(normalized_speed - history_forecast))

                class_errors[cls] = error
            except Exception as e:
                print(f"Napaka pri napovedi za razred {cls}: {e}")
                class_errors[cls] = float('inf')

        if class_errors:
            predicted_level = min(class_errors, key=class_errors.get)
        else:
            predicted_level = 1

        if sum(class_errors.values()) == float('inf') or all(err == float('inf') for err in class_errors.values()):
            probabilities = {cls: 1.0 / len(class_errors) for cls in class_errors.keys()}
        else:
            max_error = max(err for err in class_errors.values() if err < float('inf')) if any(
                err < float('inf') for err in class_errors.values()) else 1.0
            inv_errors = {cls: max(0.001, (max_error + 0.001 - err) / (max_error + 0.001)) for cls, err in
                          class_errors.items() if err < float('inf')}
            total_inv_error = sum(inv_errors.values())
            probabilities = {cls: err / total_inv_error for cls, err in
                             inv_errors.items()} if total_inv_error > 0 else {0: 1.0}

        descriptions = [
            "Zelo ekonomično (0)",
            "Ekonomično (1)",
            "Zmerno (2)",
            "Neekonomično (3)",
            "Vrlo neekonomično (4)"
        ]

        result_text = f"Napovedana ekonomičnost: {predicted_level}\n{descriptions[predicted_level]}"
        result_text += "\n\nVerjetnosti:"

        for i in range(5):
            prob = probabilities.get(i, 0)
            result_text += f"\nLevel {i}: {prob:.2%}"

        result_text += "\n\nNapake modelov:"
        for i in range(5):
            err = class_errors.get(i, float('inf'))
            result_text += f"\nLevel {i}: {err:.5f}" if err < float('inf') else f"\nLevel {i}: Ni na voljo"

        result_text += "\n\nOptimizirani parametri:"
        for cls, model_info in class_models.items():
            p_opt, d_opt, q_opt = model_info['order']
            result_text += f"\nLevel {cls}: p={p_opt}, d={d_opt}, q={q_opt}"

        result_label.config(text=result_text)

    except Exception as e:
        messagebox.showerror("Napaka", f"Napaka pri napovedi: {e}")
        import traceback
        traceback.print_exc()


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
    plt.title("Vzorci hitrosti")
    plt.xlabel("Časovni koraki")
    plt.ylabel("Normalizirana hitrost")

    if hasattr(update_plot, 'canvas'):
        update_plot.canvas.get_tk_widget().destroy()

    canvas = FigureCanvasTkAgg(plt.gcf(), master=plot_frame)
    canvas.draw()
    canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
    update_plot.canvas = canvas

    plt.close()


def plot_prediction(speed_sequence, predicted_level, forecast):
    plt.figure(figsize=(10, 6))

    plt.subplot(2, 1, 1)
    plt.plot(speed_sequence, 'b-', label='Vhodni podatki')

    forecast_x = np.arange(len(speed_sequence), len(speed_sequence) + len(forecast))
    plt.plot(forecast_x, forecast, 'r--', label='ARIMA napoved')

    plt.title(f"Napoved ekonomičnosti: Level {predicted_level}")
    plt.xlabel("Časovni korak")
    plt.ylabel("Normalizirana hitrost")
    plt.grid(True)
    plt.legend()

    plt.subplot(2, 1, 2)

    diff = diferencialna_serija(speed_sequence)
    plt.plot(diff, 'g-', label='Prva diferenca')

    is_stationary = check_stacionarnost(speed_sequence)
    plt.title(f"Analiza stacionarnosti (Stacionarno: {'Da' if is_stationary else 'Ne'})")
    plt.xlabel("Časovni korak")
    plt.ylabel("Sprememba hitrosti")
    plt.grid(True)
    plt.legend()

    plt.tight_layout()

    plt.savefig("prediction_analysis.png")
    plt.show()


def clear_data():
    global examples_X, examples_y

    if messagebox.askyesno("Potrditev", "Ali ste prepričani, da želite izbrisati vse podatke za učenje?"):
        examples_X = []
        examples_y = []
        status_label.config(text="Vsi podatki izbrisani")
        update_plot()
        clear_memory()


def clear_memory():
    gc.collect()
    status_label.config(text=status_label.cget("text") + " | Pomnilnik očiščen")


def export_data():
    if not examples_X or not examples_y:
        messagebox.showinfo("Ni podatkov", "Ni podatkov za izvoz.")
        return

    try:
        filename = filedialog.asksaveasfilename(
            title="Izvozi podatke",
            defaultextension=".csv",
            filetypes=(("CSV datoteke", "*.csv"), ("Vse datoteke", "*.*"))
        )

        if not filename:
            return

        with open(filename, 'w', newline='') as f:
            writer = csv.writer(f)

            header = [f"speed_{i}" for i in range(sequence_length)]
            header.append("economy_level")
            writer.writerow(header)

            for x, y in zip(examples_X, examples_y):
                row = list(x) + [y]
                writer.writerow(row)

        status_label.config(text=f"Podatki izvoženi v {filename}")

    except Exception as e:
        messagebox.showerror("Napaka", f"Izvoz podatkov ni uspel: {e}")


def exit_application():
    if messagebox.askyesno("Izhod", "Ali ste prepričani, da želite zapreti aplikacijo?"):
        clear_memory()
        window.destroy()


# GUI setup
window = tk.Tk()
window.title("ARIMA model za analizo ekonomičnosti vožnje - Optimiziran")
window.geometry("800x700")

notebook = ttk.Notebook(window)
notebook.grid(row=0, column=0, columnspan=2, sticky="nsew", padx=10, pady=10)

training_tab = ttk.Frame(notebook)
prediction_tab = ttk.Frame(notebook)
visualization_tab = ttk.Frame(notebook)
info_tab = ttk.Frame(notebook)

notebook.add(training_tab, text="Učenje")
notebook.add(prediction_tab, text="Napoved")
notebook.add(visualization_tab, text="Vizualizacija")
notebook.add(info_tab, text="O ARIMA")

examples_frame = ttk.LabelFrame(training_tab, text="Dodajanje učnih primerov")
examples_frame.pack(fill="both", expand=True, padx=10, pady=10)

tk.Label(examples_frame, text="Vnesite vrednosti hitrosti (ločene s presledkom ali vejico):").pack(anchor="w", padx=5,
                                                                                                   pady=5)
speed_input = tk.Text(examples_frame, height=5, width=60)
speed_input.pack(fill="both", expand=True, padx=5, pady=5)

economy_frame = ttk.Frame(examples_frame)
economy_frame.pack(fill="x", padx=5, pady=5)

tk.Label(economy_frame, text="Ekonomičnost:").pack(side="left")
economy_var = tk.StringVar(value="0")
for i, desc in enumerate(
        ["Zelo ekonomično (0)", "Ekonomično (1)", "Zmerno (2)", "Neekonomično (3)", "Zelo neekonomično (4)"]):
    tk.Radiobutton(economy_frame, text=desc, variable=economy_var, value=str(i)).pack(side="left", padx=5)

button_frame = ttk.Frame(examples_frame)
button_frame.pack(fill="x", padx=5, pady=10)

ttk.Button(button_frame, text="Dodaj primer", command=add_example_manually).pack(side="left", padx=5)
ttk.Button(button_frame, text="Naloži iz CSV", command=load_from_csv).pack(side="left", padx=5)
ttk.Button(button_frame, text="Izvozi podatke", command=export_data).pack(side="left", padx=5)
ttk.Button(button_frame, text="Izbriši podatke", command=clear_data).pack(side="left", padx=5)

model_frame = ttk.LabelFrame(training_tab, text="Učenje modela z optimizacijo")
model_frame.pack(fill="both", expand=True, padx=10, pady=10)

ttk.Button(model_frame, text="Nauči model (z optimizacijo)", command=train_model).pack(side="left", padx=10, pady=10)
ttk.Button(model_frame, text="Shrani model", command=save_model).pack(side="left", padx=10, pady=10)
ttk.Button(model_frame, text="Naloži model", command=load_model).pack(side="left", padx=10, pady=10)
ttk.Button(model_frame, text="Očisti pomnilnik", command=clear_memory).pack(side="left", padx=10, pady=10)

predict_frame = ttk.LabelFrame(prediction_tab, text="Napovej ekonomičnost")
predict_frame.pack(fill="both", expand=True, padx=10, pady=10)

tk.Label(predict_frame, text="Vnesite vrednosti hitrosti (ločene s presledkom ali vejico):").pack(anchor="w", padx=5,
                                                                                                  pady=5)
predict_input = tk.Text(predict_frame, height=5, width=60)
predict_input.pack(fill="both", expand=True, padx=5, pady=5)

ttk.Button(predict_frame, text="Napovej", command=predict).pack(padx=5, pady=10)

result_label = tk.Label(predict_frame, text="", font=("Arial", 12), justify="left")
result_label.pack(fill="both", expand=True, padx=5, pady=5)

plot_frame = ttk.LabelFrame(visualization_tab, text="Vzorci hitrosti")
plot_frame.pack(fill="both", expand=True, padx=10, pady=10)

info_frame = ttk.LabelFrame(info_tab, text="O ARIMA modelu")
info_frame.pack(fill="both", expand=True, padx=10, pady=10)

# Dodaj informacije o ARIMA modelu
info_text = """
ARIMA Model za Ekonomičnost Vožnje

ARIMA (AutoRegressive Integrated Moving Average) je statistični model za analizo časovnih vrst.

Parametri ARIMA modela:
• p (AutoRegressive): koliko prejšnjih vrednosti vpliva na trenutno
• d (Integrated): stopnja diferenciranja za doseganje stacionarnosti  
• q (Moving Average): koliko prejšnjih napak vpliva na trenutno

Kako deluje v tej aplikaciji:
1. Za vsak razred ekonomičnosti (0-4) se ustvari ločen ARIMA model
2. Model se optimizira z iskanjem najboljših parametrov (p,d,q)
3. Pri napovedi se izračuna napaka rekonstrukcije za vsak razred
4. Razred z najmanjšo napako je napovedani rezultat

Natančnost (Accuracy):
• Meri delež pravilnih napovedi: (pravilne napovedi / vse napovedi) × 100%
• Testira se na podatkih, ki so bili uporabljeni za učenje
• Višja vrednost pomeni boljši model
• Priporočena vrednost: >70% za dobro delovanje

Model je primeren za:
• Analizo vzorcev hitrosti
• Napoved ekonomičnosti vožnje
• Odkrivanje anomalij v voznih navadah
"""

info_label = tk.Label(info_frame, text=info_text, justify="left", font=("Arial", 10), wraplength=700)
info_label.pack(padx=10, pady=10, fill="both", expand=True)

status_label = tk.Label(window, text="Pripravljen", bd=1, relief=tk.SUNKEN, anchor=tk.W)
status_label.grid(row=10, column=0, columnspan=2, sticky="we")

ttk.Button(window, text="Izhod", command=exit_application).grid(row=11, column=0, columnspan=2, pady=5)

window.grid_rowconfigure(0, weight=1)
window.grid_columnconfigure(0, weight=1)

update_plot()

window.protocol("WM_DELETE_WINDOW", exit_application)

if __name__ == "__main__":
    window.mainloop()