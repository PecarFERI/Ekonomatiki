import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import os
import csv
from collections import Counter
import datetime
import joblib
from statsmodels.tsa.arima.model import ARIMA
from sklearn.ensemble import RandomForestClassifier

# --- Globalne spremenljivke ---
examples_X = []  # Shranjuje časovne vrste pospeškov
examples_y = []  # Shranjuje ekonomske ravni
model = None     # Naš klasifikacijski model
scaler = StandardScaler()  # Za konsistentno normalizacijo podatkov
sequence_length = 30
training_history = {
    'accuracy': [],
    'test_accuracy': [],
    'timestamps': []
}

# --- Funkcije za obdelavo podatkov ---
def normalize_data(data):
    """Normalizacija podatkov"""
    data_reshaped = np.array(data).reshape(-1, 1)
    if not hasattr(normalize_data, 'fitted'):
        scaler.fit(data_reshaped)
        normalize_data.fitted = True
    return scaler.transform(data_reshaped).flatten()

def extract_features(series):
    """Poenostavljena ekstrakcija le najpomembnejših značilnic iz časovne vrste"""
    # Ohranimo le najpomembnejše značilnice
    features = {
        'mean': np.mean(series),
        'std': np.std(series),
        'range': np.max(series) - np.min(series),
        'zero_crossings': np.sum(np.diff(np.signbit(series))),
        'energy': np.sum(series**2)
    }
    
    # Koeficienti ARIMA modela
    try:
        # Poskušamo prilagoditi ARIMA model časovni vrsti
        # Uporabljamo enostavne parametre (1,0,1)
        arima_model = ARIMA(series, order=(1, 0, 1))
        arima_result = arima_model.fit()
        
        # Dodajamo koeficiente modela kot značilnice
        features['ar_coef'] = arima_result.arparams[0] if len(arima_result.arparams) > 0 else 0
        features['ma_coef'] = arima_result.maparams[0] if len(arima_result.maparams) > 0 else 0
    except:
        # Če ARIMA ne deluje, uporabimo privzete vrednosti
        features['ar_coef'] = 0
        features['ma_coef'] = 0
        
    return list(features.values())

def get_class_distribution():
    """Porazdelitev razredov v učnih podatkih"""
    counts = Counter(examples_y)
    return [counts.get(i, 0) for i in range(5)]

def augment_data(X, y, augmentation_factor=2):
    """Preprosta augmentacija podatkov z dodajanjem manjšega šuma"""
    augmented_X = []
    augmented_y = []
    
    for series, label in zip(X, y):
        augmented_X.append(series)  # Originalne vrednosti
        augmented_y.append(label)
        
        # Dodamo nekaj primerov z dodanim šumom
        for _ in range(augmentation_factor - 1):
            # Dodamo manjši šum na originalno časovno vrsto
            noise = np.random.normal(0, 0.1, len(series))
            augmented_series = series + noise
            
            # Dodamo še shiftane vrednosti za rešitev problema preskakovanja
            shift = np.random.randint(1, 5)
            shifted_series = np.roll(augmented_series, shift)
            
            augmented_X.append(shifted_series)
            augmented_y.append(label)
    
    return augmented_X, augmented_y

# --- Funkcije za strojno učenje ---
def create_model():
    """Ustvari in vrne klasifikacijski model - uporabljamo RandomForest,
       saj ARIMA služi za ekstrakcijo značilnic, ne pa za samo klasifikacijo"""
    return RandomForestClassifier(
        n_estimators=50,  # Zmanjšano število dreves za hitrejše izvajanje
        max_depth=8,
        min_samples_split=5,
        random_state=42,
        class_weight='balanced'
    )

def add_example_manually():
    """Dodajanje ročno vnešenega primera"""
    try:
        # Preberemo vrednosti pospeškov
        accel_str = accel_input.get("1.0", tk.END).strip()
        if not accel_str:
            raise ValueError("Prosim vnesite vrednosti pospeškov")
        
        # Pretvorba v seznam števil
        accel_values = [float(x) for x in accel_str.replace(',', ' ').split()]
        
        if len(accel_values) < 5:
            raise ValueError("Prosim vnesite vsaj 5 vrednosti pospeškov")
            
        # Preberemo ekonomsko raven
        economy_level = int(economy_var.get())
        
        # Prilagoditev dolžine
        if len(accel_values) != sequence_length:
            accel_values = np.interp(
                np.linspace(0, 1, sequence_length),
                np.linspace(0, 1, len(accel_values)),
                accel_values
            )
        
        # Normalizacija podatkov
        normalized_accel = normalize_data(accel_values)
        
        # Shranjevanje primera
        examples_X.append(normalized_accel)
        examples_y.append(economy_level)
        
        # Posodobitev statusa
        status_label.config(text=f"Dodan primer #{len(examples_X)} (Ekonomska raven: {economy_level})")
        
        # Počistimo vnosno polje
        accel_input.delete("1.0", tk.END)
        
        # Posodobimo graf
        update_plot()
        
    except Exception as e:
        messagebox.showerror("Napaka", f"Neveljaven vnos: {e}")

def load_from_csv():
    """Nalaganje podatkov iz CSV datoteke"""
    try:
        filename = filedialog.askopenfilename(
            title="Izberite CSV datoteko",
            filetypes=(("CSV datoteke", "*.csv"), ("Vse datoteke", "*.*"))
        )
        
        if not filename:
            return
        
        loaded_count = 0
        
        with open(filename, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            header = next(reader)  # preskočimo glavo
            
            for row in reader:
                try:
                    if len(row) < sequence_length + 1:
                        continue

                    # Zadnji element je oznaka
                    label = int(row[-1])

                    # Ostali elementi so pospeški
                    accel_values = [float(x) for x in row[:-1]]

                    # Prilagoditev dolžine
                    if len(accel_values) != sequence_length:
                        accel_values = np.interp(
                            np.linspace(0, 1, sequence_length),
                            np.linspace(0, 1, len(accel_values)),
                            accel_values
                        )
                    
                    # Normalizacija
                    normalized_accel = normalize_data(accel_values)

                    # Shranjevanje
                    examples_X.append(normalized_accel)
                    examples_y.append(label)
                    loaded_count += 1
                    
                except Exception as e:
                    print(f"Napaka pri obdelavi vrstice: {e}")
                    continue
        
        status_label.config(text=f"Naloženih {loaded_count} primerov iz CSV")
        update_plot()
        
    except Exception as e:
        messagebox.showerror("Napaka", f"Napaka pri branju CSV: {e}")

def train_model():
    global model, training_history
    
    if len(examples_X) < 10:
        messagebox.showwarning("Premalo podatkov", "Prosim dodajte vsaj 10 primerov za osnovno učenje.")
        return

    try:
        # Augmentacija podatkov
        augmented_X, augmented_y = augment_data(examples_X, examples_y)
        
        # Priprava značilnic z uporabo ARIMA koeficientov
        X_features = np.array([extract_features(series) for series in augmented_X])
        y_labels = np.array(augmented_y)
        
        # Razdelimo podatke na učno in testno množico
        X_train, X_test, y_train, y_test = train_test_split(
            X_features, y_labels, test_size=0.2, random_state=42, stratify=y_labels
        )
        
        # Ustvarimo model
        model = create_model()

        # Učenje modela
        model.fit(X_train, y_train)
        
        # Ocenimo natančnost modela na učnih podatkih
        y_pred_train = model.predict(X_train)
        train_accuracy = accuracy_score(y_train, y_pred_train) * 100
        
        # Ocenimo natančnost modela na testnih podatkih
        y_pred_test = model.predict(X_test)
        test_accuracy = accuracy_score(y_test, y_pred_test) * 100
        
        # Izpišemo klasifikacijsko poročilo
        class_report = classification_report(y_test, y_pred_test)
        print("\nKlasifikacijsko poročilo:")
        print(class_report)
        
        # Beležimo zgodovino učenja
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        training_history['timestamps'].append(timestamp)
        training_history['accuracy'].append(train_accuracy)
        training_history['test_accuracy'].append(test_accuracy)
        
        status_label.config(text=f"✓ Model uspešno naučen. Točnost: {train_accuracy:.1f}% učni, {test_accuracy:.1f}% testni.")
        
        save_model()
        
    except Exception as e:
        messagebox.showerror("Napaka", f"Napaka pri učenju modela: {str(e)}")

def save_model():
    if model is None:
        return
        
    try:
        model_dir = "models"
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
            
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        model_path = os.path.join(model_dir, f"eco_model_{timestamp}.joblib")
        
        # Shrani model in zgodovino učenja
        data_to_save = {
            'model': model,
            'training_history': training_history
        }
        joblib.dump(data_to_save, model_path)
        print(f"Model shranjen v: {model_path}")
        
    except Exception as e:
        print(f"Napaka pri shranjevanju modela: {e}")

def load_model():
    global model, training_history
    
    try:
        filename = filedialog.askopenfilename(
            title="Izberite model",
            filetypes=(("Joblib datoteke", "*.joblib"), ("Vse datoteke", "*.*"))
        )
        
        if not filename:
            return
            
        data = joblib.load(filename)
        
        # Preverimo, če je v datoteki samo model ali tudi zgodovina učenja
        if isinstance(data, dict) and 'model' in data:
            model = data['model']
            if 'training_history' in data:
                training_history = data['training_history']
        else:
            model = data  # Starejši format, ki vsebuje samo model
            
        status_label.config(text=f"Model uspešno naložen iz: {os.path.basename(filename)}")
        
    except Exception as e:
        messagebox.showerror("Napaka", f"Napaka pri nalaganju modela: {e}")

def predict():
    """Napoved ekonomske ravni za nov vnos"""
    global model
    
    if model is None:
        messagebox.showinfo("Ni modela", "Prosim najprej naučite model.")
        return

    try:
        # Preberemo vrednosti pospeškov iz vnosnega polja
        accel_str = predict_input.get("1.0", tk.END).strip()
        if not accel_str:
            raise ValueError("Prosim vnesite vrednosti pospeškov")
            
        # Pretvorba v seznam števil
        accel_values = [float(x) for x in accel_str.replace(',', ' ').split()]
        
        if len(accel_values) < 5:
            raise ValueError("Prosim vnesite vsaj 5 vrednosti pospeškov")
            
        # Prilagoditev dolžine
        if len(accel_values) != sequence_length:
            accel_values = np.interp(
                np.linspace(0, 1, sequence_length),
                np.linspace(0, 1, len(accel_values)),
                accel_values
            )
        
        # Normalizacija
        normalized_accel = normalize_data(accel_values)
        
        # Ekstrakcija značilnic z upoštevanjem ARIMA modeliranja
        features = extract_features(normalized_accel)
        
        # Napoved z modelom (značilnice preoblikujemo v 2D obliko)
        features_2d = np.array(features).reshape(1, -1)
        pred_level = model.predict(features_2d)[0]
        
        # Če model podpira verjetnostne napovedi
        probabilities = None
        if hasattr(model, 'predict_proba'):
            probabilities = model.predict_proba(features_2d)[0]
        
        # Opisi ravni
        descriptions = [
            "0 - Zelo ekonomična vožnja",
            "1 - Ekonomična vožnja",
            "2 - Zmerna poraba", 
            "3 - Nekonomična vožnja",
            "4 - Zelo nekonomična vožnja"
        ]
        
        # Priprava rezultata
        result_text = f"Napovedana ekonomska raven: {pred_level}\n"
        result_text += f"{descriptions[pred_level]}\n\n"
        
        # Dodamo verjetnosti za vse razrede
        if probabilities is not None:
            result_text += "Verjetnosti po razredih:\n"
            for i, prob in enumerate(probabilities):
                result_text += f"Raven {i}: {prob*100:.1f}%\n"
            
        # Posodobimo prikaz rezultatov
        result_text_widget.delete(1.0, tk.END)
        result_text_widget.insert(tk.END, result_text)
        
    except Exception as e:
        messagebox.showerror("Napaka", f"Napaka pri napovedovanju: {e}")


# --- Vizualizacijske funkcije ---
def update_plot():
    """Posodobitev grafa s primeri"""
    if not examples_X or not examples_y:
        return
    
    plt.clf()  # Clear the current figure
    
    # Prikaz enega primera za vsako ekonomsko raven
    levels_shown = set()
    for i, (x, y) in enumerate(zip(examples_X, examples_y)):
        if y not in levels_shown and len(levels_shown) < 5:
            plt.plot(x, label=f"Raven {y}")
            levels_shown.add(y)
            
            if len(levels_shown) >= 5:
                break
    
    plt.legend()
    plt.title("Primeri vzorcev pospeševanja")
    plt.xlabel("Časovni koraki")
    plt.ylabel("Normaliziran pospešek")
    
    # Posodobitev platna
    if hasattr(update_plot, 'canvas'):
        update_plot.canvas.draw()
    else:
        canvas = FigureCanvasTkAgg(plt.gcf(), master=plot_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        update_plot.canvas = canvas

def show_class_distribution():
    """Prikaz porazdelitve razredov"""
    distribution = get_class_distribution()
    
    # Ustvarimo novo okno za prikaz grafa
    dist_window = tk.Toplevel(window)
    dist_window.title("Porazdelitev razredov")
    dist_window.geometry("400x300")
    
    # Ustvarimo graf
    fig, ax = plt.subplots(figsize=(8, 4))
    
    # Stolpični graf
    ax.bar(range(5), distribution, color='skyblue')
    ax.set_xticks(range(5))
    ax.set_xticklabels([f"Raven {i}" for i in range(5)])
    ax.set_title("Število primerov po razredih")
    ax.set_ylabel("Število primerov")
    
    plt.tight_layout()
    
    # Dodamo graf v okno
    canvas = FigureCanvasTkAgg(fig, master=dist_window)
    canvas.draw()
    canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

# --- GUI Nastavitev ---
window = tk.Tk()
window.title("ARIMA model za pospešek")
window.geometry("900x700")

# Ustvarimo zavihke
notebook = ttk.Notebook(window)
notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

# Zavihek za učenje modela
train_tab = ttk.Frame(notebook)
notebook.add(train_tab, text="Učenje modela")

# Zavihek za napoved
predict_tab = ttk.Frame(notebook)
notebook.add(predict_tab, text="Napoved")

# Zavihek za evalvacijo
eval_tab = ttk.Frame(notebook)
notebook.add(eval_tab, text="Modeli")

# Okvir za vnos podatkov
input_frame = tk.Frame(train_tab)
input_frame.pack(pady=10)

# Vnos pospeškov
tk.Label(input_frame, text="Vnesite vrednosti pospeškov (ločene s presledki):").grid(row=0, column=0, sticky="w")
accel_input = tk.Text(input_frame, height=5, width=50)
accel_input.grid(row=1, column=0, padx=5)

# Izbira ekonomske ravni
tk.Label(input_frame, text="Izberite ekonomsko raven:").grid(row=0, column=1, sticky="w")
economy_var = tk.StringVar(value="0")
economy_menu = tk.OptionMenu(input_frame, economy_var, "0", "1", "2", "3", "4")
economy_menu.grid(row=1, column=1, padx=5)

# Gumbi za upravljanje podatkov
button_frame = tk.Frame(train_tab)
button_frame.pack(pady=10)

add_button = tk.Button(button_frame, text="Dodaj primer", command=add_example_manually)
add_button.grid(row=0, column=0, padx=5)

load_button = tk.Button(button_frame, text="Naloži iz CSV", command=load_from_csv)
load_button.grid(row=0, column=1, padx=5)

train_button = tk.Button(button_frame, text="Nauči model", command=train_model)
train_button.grid(row=0, column=2, padx=5)

dist_button = tk.Button(button_frame, text="Porazdelitev", command=show_class_distribution)
dist_button.grid(row=0, column=3, padx=5)


# Okvir za graf
plot_frame = tk.Frame(train_tab)
plot_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

# --- Zavihek za napoved ---
# Vnos za napoved
predict_frame = tk.Frame(predict_tab)
predict_frame.pack(fill=tk.X, padx=10, pady=10)
tk.Label(predict_frame, text="Vnesite vrednosti pospeškov za napoved (ločene s presledki):").pack(anchor="w")
predict_input = tk.Text(predict_frame, height=5)
predict_input.pack(fill=tk.X)

predict_button = tk.Button(predict_tab, text="Napovej ekonomsko raven", command=predict)
predict_button.pack(pady=10)

# Prikaz rezultatov
result_frame = tk.Frame(predict_tab)
result_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

tk.Label(result_frame, text="Rezultat napovedi:").pack(anchor="w")
result_text_widget = tk.Text(result_frame, height=15, wrap=tk.WORD, font=("Arial", 10))
result_text_widget.pack(fill=tk.BOTH, expand=True)


eval_frame = tk.Frame(eval_tab)
eval_frame.pack(pady=20)

save_button = tk.Button(eval_frame, text="Shrani model", command=save_model)
save_button.grid(row=0, column=1, padx=10)

load_button = tk.Button(eval_frame, text="Naloži model", command=load_model)
load_button.grid(row=0, column=2, padx=10)

# Statusna vrstica
status_label = tk.Label(window, text="Pripravljen", bd=1, relief=tk.SUNKEN, anchor=tk.W)
status_label.pack(fill=tk.X)

# Zaženemo glavno zanko
if __name__ == "__main__":
    window.mainloop()