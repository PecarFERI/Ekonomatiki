import torch
import torch.nn as nn
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import numpy as np
import csv
import math
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import gc
import os
from sklearn.metrics import classification_report
import subprocess
import sys
import webbrowser
import gpxpy
import folium
from folium import PolyLine
import json


#==================definicije modelov
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

class DirectionLSTM(nn.Module):
    def __init__(self, input_size=1, hidden_size=128, output_size=3):
        super(DirectionLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.bidirectional = True
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True, bidirectional=self.bidirectional)
        self.fc = nn.Linear(hidden_size * 2 if self.bidirectional else hidden_size, output_size)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        out = self.fc(out)
        return out


def compute_bearing(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
    d_lon = lon2 - lon1
    x = math.sin(d_lon) * math.cos(lat2)
    y = math.cos(lat1) * math.sin(lat2) - math.sin(lat1) * math.cos(lat2) * math.cos(d_lon)
    bearing = math.atan2(x, y)
    return math.degrees(bearing)


def compute_bearing_sequence(coords):
    bearings = []
    for i in range(1, len(coords)):
        lat1, lon1 = coords[i - 1]
        lat2, lon2 = coords[i]
        angle = compute_bearing(lat1, lon1, lat2, lon2)
        bearings.append([angle])
    return bearings


class BearingPredictor:
    def __init__(self):
        self.model = None

    def load_model(self, path):
        self.model = DirectionLSTM()
        self.model.load_state_dict(torch.load(path, map_location=torch.device('cpu')))
        self.model.eval()

    def predict(self, coords):
        bearing_seq = compute_bearing_sequence(coords)
        if not bearing_seq:
            raise ValueError("Ni bilo mogoče izračunati bearing sekvence.")
        input_tensor = torch.tensor([bearing_seq], dtype=torch.float32)
        with torch.no_grad():
            output = self.model(input_tensor)
            probabilities = torch.softmax(output, dim=1)[0]
            pred = torch.argmax(probabilities).item()
            confidence = probabilities[pred].item()
            return pred, confidence


#==============skupni prediction
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
            except Exception as e:
                print(f"Napaka pri napovedi pospeška: {e}")

        if 'speed' in results and 'acceleration' in results:
            try:
                combined_probabilities = []
                for i in range(6):
                    weighted_prob = (results['speed']['probabilities'][i] * 0.6 +
                                   results['acceleration']['probabilities'][i] * 0.4)
                    combined_probabilities.append(weighted_prob)

                combined_prediction = combined_probabilities.index(max(combined_probabilities))
                results['combined'] = {
                    'prediction': combined_prediction,
                    'confidence': max(combined_probabilities),
                    'probabilities': combined_probabilities
                }
            except Exception as e:
                print(f"Napaka pri skupni napovedi: {e}")

        return results


# ===========izris zemljevida
def read_coordinates(gpx_file):
    with open(gpx_file, 'r') as f:
        gpx = gpxpy.parse(f)
    coordinates = []
    for track in gpx.tracks:
        for segment in track.segments:
            for point in segment.points:
                coordinates.append((point.latitude, point.longitude))
    return coordinates


def read_levels_from_matrix(levels_file):
    levels = []
    with open(levels_file, 'r') as f:
        for line in f:
            parts = line.strip().split(',')
            if len(parts) < 2:
                continue
            try:
                level = int(parts[-1])
                speeds = parts[:-1]
                valid_count = sum(1 for h in speeds if h.strip() != "0.0")
                levels.extend([level] * valid_count)
            except ValueError:
                print(f"Warning: Skipping malformed line: {line}")
    return levels


def add_legend(m):
    legend_html = """
     <div style='position: fixed; 
                 bottom: 50px; left: 50px; width: 180px; height: 220px; 
                 background-color: white; border:2px solid grey; z-index:9999; font-size:14px;
                 padding: 10px;'>
     <b>Legenda stopenj:</b><br>
     <i style='background: blue; width: 10px; height: 10px; display: inline-block;'></i> 0 – Mirovanje<br>
     <i style='background: lightgreen; width: 10px; height: 10px; display: inline-block;'></i> 1 - Zelo ekonomično<br>
     <i style='background: green; width: 10px; height: 10px; display: inline-block;'></i> 2 - Ekonomično<br>
     <i style='background: darkorange; width: 10px; height: 10px; display: inline-block;'></i> 3 - Zmerno ekonomično<br>
     <i style='background: red; width: 10px; height: 10px; display: inline-block;'></i> 4 - Neekonomično<br>
     <i style='background: darkred; width: 10px; height: 10px; display: inline-block;'></i> 5 - Zelo neekonomično<br> 
     </div>
    """
    m.get_root().html.add_child(folium.Element(legend_html))


def generate_animation_js(coordinates, levels):
    if len(levels) != len(coordinates) - 1:
        min_len = min(len(levels), len(coordinates) - 1)
        coordinates = coordinates[:min_len + 1]
        levels = levels[:min_len]

    colors = {
        0: 'blue',
        1: 'lightgreen',
        2: 'green',
        3: 'darkorange',
        4: 'red',
        5: 'darkred'
    }

    js_code = f"""
    <script>
        const ANIMATION_DURATION = 30000;
        const POINTS_PER_FRAME = 1;  //ena na enkrat
        const INITIAL_DELAY = 1000;
        const CAMERA_FOLLOW_ZOOM = 14;

        let coordinates = {json.dumps(coordinates)};
        let levels = {json.dumps(levels)};
        let colors = {json.dumps(colors)};

        let map;
        let currentIndex = 1;
        let animationId = null;
        let isAnimating = false;
        let polylines = [];

        function initializeAnimation() {{
            for (let key in window) {{
                if (key.startsWith('map_') && window[key] && typeof window[key].setView === 'function') {{
                    map = window[key];
                    break;
                }}
            }}

            if (!map) {{
                console.error('Map object not found');
                return;
            }}

            createStartButton();
        }}

        function createStartButton() {{
            let buttonHtml = `
                <div id="animationControls" style="position: fixed; top: 20px; right: 20px; z-index: 1000;">
                    <button id="startBtn" style="padding: 10px 15px; 
                           border: none; border-radius: 4px; cursor: pointer; font-weight: bold;">
                        ▶️ Začni animacijo
                    </button>
                </div>
            `;
            document.body.insertAdjacentHTML('beforeend', buttonHtml);
            document.getElementById('startBtn').addEventListener('click', function() {{
                if (!isAnimating) {{
                    startAnimation();
                }}
            }});
        }}

        function startAnimation() {{
            if (isAnimating) return;
            currentIndex = 1;
            isAnimating = true;

            polylines.forEach(p => map.removeLayer(p));
            polylines = [];

            map.setView(coordinates[0], CAMERA_FOLLOW_ZOOM);

            document.getElementById('startBtn').textContent = '⏳ Animacija poteka...';
            document.getElementById('startBtn').style.background = '#FF9800';
            document.getElementById('startBtn').disabled = true;

            animatePath();
        }}

        function animatePath() {{
            if (!isAnimating || currentIndex >= coordinates.length) {{
                animationComplete();
                return;
            }}

            let color = colors[levels[currentIndex - 1]] || 'gray';

            let segment = L.polyline([coordinates[currentIndex - 1], coordinates[currentIndex]], {{
                color: color,
                weight: 6,
                opacity: 0.9,
                lineCap: 'round',
                lineJoin: 'round'
            }}).addTo(map);
            polylines.push(segment);

            if (currentIndex % 10 === 0) {{
                map.panTo(coordinates[currentIndex], {{
                    animate: true,
                    duration: 0.5,
                    easeLinearity: 0.25
                }});
            }}

            currentIndex++;

            let delay = Math.max(10, ANIMATION_DURATION / coordinates.length);
            animationId = setTimeout(animatePath, delay);
        }}

        function animationComplete() {{
            isAnimating = false;
            document.getElementById('startBtn').textContent = '🔄 Ponovi animacijo';
            document.getElementById('startBtn').disabled = false;
            map.panTo(coordinates[coordinates.length - 1], {{
                animate: true,
                duration: 1
            }});
        }}

        document.addEventListener('DOMContentLoaded', function() {{
            setTimeout(initializeAnimation, INITIAL_DELAY);
        }});
    </script>

    <style>
        #animationControls {{
            font-family: Arial, sans-serif;
        }}
        #startBtn {{
            transition: all 0.3s ease;
        }}
        #startBtn:hover {{
            transform: scale(1.05);
            box-shadow: 0 2px 5px rgba(0,0,0,0.2);
        }}
    </style>
    """
    return js_code



def draw_route_on_map(gpx_path, levels_path, progressive=True):
    coordinates = read_coordinates(gpx_path)
    levels = read_levels_from_matrix(levels_path)

    if len(coordinates) < 2:
        print("Not enough coordinates to draw route.")
        return None

    if len(coordinates) - 1 > len(levels):
        print(f"Warning: More coordinates than levels. Trimming {len(coordinates) - 1 - len(levels)} excess points.")
        coordinates = coordinates[:len(levels) + 1]
    elif len(coordinates) - 1 < len(levels):
        print(f"Warning: More levels than coordinates. Trimming {len(levels) - (len(coordinates) - 1)} excess levels.")
        levels = levels[:len(coordinates) - 1]

    map_center = coordinates[0]
    m = folium.Map(location=map_center, zoom_start=14, tiles="CartoDB positron")

    base_name = os.path.splitext(os.path.basename(gpx_path))[0]

    if progressive:
        output_file = f"{base_name}_map.html"
        print("Creating animation map...")

        add_legend(m)

        folium.Marker(
            coordinates[0],
            popup="🏁 START",
            icon=folium.Icon(color='green', icon='play')
        ).add_to(m)

        folium.Marker(
            coordinates[-1],
            popup="🏁 CILJ",
            icon=folium.Icon(color='red', icon='stop')
        ).add_to(m)

        map_var_name = m.get_name()
        print(f"Map variable name: {map_var_name}")

        js_code = generate_animation_js(coordinates, levels)
        m.get_root().html.add_child(folium.Element(js_code))

    else:
        output_file = f"{base_name}_static_map.html"
        colors = {
            0: 'blue', 1: 'lightgreen', 2: 'green',
            3: 'darkorange', 4: 'red', 5: 'darkred'
        }

        folium.Marker(
            coordinates[0],
            popup="🏁 START",
            icon=folium.Icon(color='green', icon='play')
        ).add_to(m)

        folium.Marker(
            coordinates[-1],
            popup="🏁 CILJ",
            icon=folium.Icon(color='red', icon='stop')
        ).add_to(m)

        for i in range(len(levels)):
            segment = [coordinates[i], coordinates[i + 1]]
            level = levels[i]
            color = colors.get(level, 'gray')
            folium.PolyLine(
                segment,
                color=color,
                weight=5,
                opacity=0.8,
                popup=f"Segment {i + 1}: Stopnja {level}"
            ).add_to(m)

        add_legend(m)

    m.save(output_file)
    print(f"Map saved as '{output_file}'")
    print(f"Skupaj segmentov: {len(levels)}")
    print(f"Skupaj koordinat: {len(coordinates)}")

    return output_file


# ===============gui
def load_speed_model():
    filename = filedialog.askopenfilename(
        title="Izberi model za hitrost",
        filetypes=(("PyTorch Model", "*.pt"), ("All files", "*.*"))
    )
    if filename:
        if predictor.load_speed_model(filename):
            speed_model_label.config(
                text=f"✓ Model za hitrost: {os.path.basename(filename)}",
                fg=colors['success']
            )
            update_status_with_color("Model za hitrost uspešno naložen", 'success')
        else:
            speed_model_label.config(
                text=f"✗ Model za hitrost: napaka pri nalaganju",
                fg=colors['danger']
            )
            update_status_with_color("Napaka pri nalaganju modela za hitrost", 'error')

def load_acceleration_model():
    filename = filedialog.askopenfilename(
        title="Izberi model za pospešek",
        filetypes=(("PyTorch Model", "*.pt"), ("All files", "*.*"))
    )
    if filename:
        if predictor.load_acceleration_model(filename):
            accel_model_label.config(
                text=f"✓ Model za pospešek: {os.path.basename(filename)}",
                fg=colors['success']
            )
            update_status_with_color("Model za pospešek uspešno naložen", 'success')
        else:
            accel_model_label.config(
                text=f"✗ Model za pospešek: napaka pri nalaganju",
                fg=colors['danger']
            )
            update_status_with_color("Napaka pri nalaganju modela za pospešek", 'error')

def load_bearing_model():
    filename = filedialog.askopenfilename(
        title="Izberi bearing model",
        filetypes=(("PyTorch Model", "*.pt"), ("All files", "*.*"))
    )
    if filename:
        try:
            bearing_predictor.load_model(filename)
            move_model_label.config(
                text=f"✓ Bearing model: {os.path.basename(filename)}",
                fg=colors['success']
            )
            update_status_with_color("Bearing model uspešno naložen", 'success')
        except Exception as e:
            move_model_label.config(
                text=f"✗ Napaka pri nalaganju bearing modela",
                fg=colors['danger']
            )
            update_status_with_color(f"Napaka pri nalaganju bearing modela: {e}", 'error')

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
        update_status_with_color("Napoved uspešno generirana", 'success')

    except Exception as e:
        messagebox.showerror("Napaka", f"Napaka pri napovedovanju: {e}")
        update_status_with_color(f"Napaka pri napovedovanju: {e}", 'error')

def predict_bearing():
    try:
        input_str = move_input.get("1.0", tk.END).strip()
        if not input_str:
            raise ValueError("Vnesi vsaj 3 GPS točke (lon, lat)")

        values = list(map(float, input_str.replace(',', ' ').split()))
        if len(values) < 6 or len(values) % 2 != 0:
            raise ValueError("Potrebno je vsaj 3 GPS točke (lon,lat)*3")

        coords = [(values[i + 1], values[i]) for i in range(0, len(values), 2)]

        if bearing_predictor.model is None:
            raise ValueError("Model za smer ni naložen")

        pred, confidence = bearing_predictor.predict(coords)

        descriptions = {
            0: "Stabilna vožnja (ravna)",
            1: "Zmerna sprememba smeri",
            2: "Veliko sprememb smeri",
        }
        desc = descriptions.get(pred, "Neznano")

        result_text = f"🧭 BEARING MODEL (smer vožnje):\n"
        result_text += f"Napovedani razred: {pred} → {desc}\n"
        result_text += f"Zaupanje modela: {confidence:.2%}\n"

        result_display.config(state='normal')
        result_display.delete('1.0', tk.END)
        result_display.insert('1.0', result_text)
        result_display.config(state='disabled')

        update_status_with_color("Napoved za smer uspešno generirana", 'success')

    except Exception as e:
        messagebox.showerror("Napaka", f"Napaka pri napovedi smeri: {e}")
        update_status_with_color(f"Napaka pri napovedi smeri: {e}", 'error')

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
                out_header.append('predicted_label')
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
        update_status_with_color(f"CSV napoved končana: {processed_count} vrstic obdelanih, {error_count} napak", 'success')

    except Exception as e:
        messagebox.showerror("Napaka", f"Napaka pri obdelavi CSV: {e}")
        update_status_with_color(f"Napaka pri obdelavi CSV: {e}", 'error')

def draw_map():
    try:
        gpx_file = filedialog.askopenfilename(
            title="Izberi GPX datoteko",
            filetypes=(("GPX files", "*.gpx"), ("All files", "*.*"))
        )
        if not gpx_file:
            return

        levels_file = filedialog.askopenfilename(
            title="Izberi datoteko s stopnjami",
            filetypes=(("CSV files", "*.csv"), ("Text files", "*.txt"), ("All files", "*.*"))
        )
        if not levels_file:
            return

        update_status_with_color("Ustvarjam zemljevid...", 'info')
        window.update()

        output_file = draw_route_on_map(gpx_file, levels_file, progressive=True)
        if output_file:
            webbrowser.open('file://' + os.path.abspath(output_file))
            update_status_with_color("Zemljevid uspešno izrisan in odprt v brskalniku", 'success')
        else:
            update_status_with_color("Napaka pri ustvarjanju zemljevida", 'error')

    except Exception as e:
        messagebox.showerror("Napaka", f"Napaka pri izrisu zemljevida: {e}")
        update_status_with_color(f"Napaka pri izrisu zemljevida: {e}", 'error')

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
    update_status_with_color("Vnosi počiščeni", 'info')

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
        update_status_with_color("Test modelov uspešen", 'success')
    except Exception as e:
        result_text += f"Test neuspešen: {e}\n"
        update_status_with_color("Test modelov neuspešen", 'error')

    result_display.config(state='normal')
    result_display.delete('1.0', tk.END)
    result_display.insert('1.0', result_text)
    result_display.config(state='disabled')

def update_status_with_color(message, status_type):
    color_map = {
        'success': '#27ae60',
        'warning': '#f39c12',
        'error': '#e74c3c',
        'info': '#3498db'
    }
    status_label.config(
        text=f"{'✓' if status_type == 'success' else '⚠' if status_type == 'warning' else '✗' if status_type == 'error' else 'ℹ'} {message}",
        bg=colors['primary'],
        fg='white'
    )

def exit_application():
    if messagebox.askyesno("Izhod", "Ali si prepričan, da se želiš odjaviti?"):
        gc.collect()
        window.destroy()


#===============main gui
predictor = UnifiedEconomyPredictor()
bearing_predictor = BearingPredictor()

window = tk.Tk()
window.title("🚗 Ekonomičnost Vožnje - AI Analiza")
window.geometry("1400x1000")
window.minsize(1200, 800)

colors = {
    'primary': '#2c3e50',
    'secondary': '#3498db',
    'success': '#27ae60',
    'warning': '#f39c12',
    'danger': '#e74c3c',
    'light': '#ecf0f1',
    'dark': '#34495e',
    'white': '#ffffff',
    'accent': '#9b59b6'
}

style = ttk.Style()
style.theme_use('clam')

style.configure("Modern.TNotebook", 
               background=colors['light'],
               borderwidth=0,
               tabmargins=[5, 5, 5, 0])

style.configure("Modern.TNotebook.Tab", 
               padding=[25, 15],
               font=("Segoe UI", 12, "bold"),
               background=colors['primary'],
               foreground='white',
               borderwidth=1,
               focuscolor='none')

style.map("Modern.TNotebook.Tab",
         background=[('selected', colors['secondary']),
                    ('active', colors['accent'])],
         foreground=[('selected', 'white'),
                    ('active', 'white')])

style.configure("Modern.TFrame", 
               background=colors['light'],
               relief='flat',
               borderwidth=1)

style.configure("Modern.TLabelframe", 
               background=colors['light'],
               borderwidth=2,
               relief='solid',
               bordercolor=colors['primary'])

style.configure("Modern.TLabelframe.Label", 
               background=colors['light'],
               foreground=colors['primary'],
               font=("Segoe UI", 14, "bold"))

style.configure("Primary.TButton",
               font=("Segoe UI", 11, "bold"),
               padding=[20, 12],
               background=colors['primary'],
               foreground='white',
               borderwidth=0,
               focuscolor='none')

style.map("Primary.TButton",
         background=[('active', colors['secondary']),
                    ('pressed', colors['dark'])])

style.configure("Success.TButton",
               font=("Segoe UI", 11, "bold"),
               padding=[20, 12],
               background=colors['success'],
               foreground='white',
               borderwidth=0,
               focuscolor='none')

style.map("Success.TButton",
         background=[('active', '#2ecc71'),
                    ('pressed', '#229954')])

style.configure("Warning.TButton",
               font=("Segoe UI", 11, "bold"),
               padding=[20, 12],
               background=colors['warning'],
               foreground='white',
               borderwidth=0,
               focuscolor='none')

style.map("Warning.TButton",
         background=[('active', '#e67e22'),
                    ('pressed', '#d68910')])

style.configure("Danger.TButton",
               font=("Segoe UI", 11, "bold"),
               padding=[15, 10],
               background=colors['danger'],
               foreground='white',
               borderwidth=0,
               focuscolor='none')

style.map("Danger.TButton",
         background=[('active', '#ec7063'),
                    ('pressed', '#c0392b')])

notebook = ttk.Notebook(window, style='Modern.TNotebook')
notebook.pack(fill="both", expand=True, padx=15, pady=15)

# ============model tab
model_tab = ttk.Frame(notebook, style='Modern.TFrame')
notebook.add(model_tab, text="📊 Upravljanje Modelov")

model_frame = ttk.LabelFrame(model_tab, text="🤖 Nalaganje AI Modelov", style="Modern.TLabelframe")
model_frame.pack(fill="both", expand=True, padx=15, pady=15)

# Speed model section
speed_frame = tk.Frame(model_frame, bg=colors['light'])
speed_frame.pack(fill="x", pady=10, padx=10)

ttk.Button(speed_frame, text="📈 Naloži Model za Hitrost", 
          command=load_speed_model, style="Primary.TButton").pack(pady=10)
speed_model_label = tk.Label(speed_frame, text="Model za hitrost: ni naložen",
                            fg=colors['danger'], bg=colors['light'],
                            font=("Segoe UI", 11, "bold"))
speed_model_label.pack(pady=5)

# Acceleration model section
accel_frame = tk.Frame(model_frame, bg=colors['light'])
accel_frame.pack(fill="x", pady=10, padx=10)

ttk.Button(accel_frame, text="⚡ Naloži Model za Pospešek", 
          command=load_acceleration_model, style="Primary.TButton").pack(pady=10)
accel_model_label = tk.Label(accel_frame, text="Model za pospešek: ni naložen", 
                            fg=colors['danger'], bg=colors['light'],
                            font=("Segoe UI", 11, "bold"))
accel_model_label.pack(pady=5)

# Acceleration model section
move_frame = tk.Frame(model_frame, bg=colors['light'])
move_frame.pack(fill="x", pady=10, padx=10)

ttk.Button(move_frame, text="⚡ Naloži Model za Smer",
          command=load_bearing_model, style="Primary.TButton").pack(pady=10)
move_model_label = tk.Label(move_frame, text="Model za smer: ni naložen",
                            fg=colors['danger'], bg=colors['light'],
                            font=("Segoe UI", 11, "bold"))
move_model_label.pack(pady=5)

# Test button
test_frame = tk.Frame(model_frame, bg=colors['light'])
test_frame.pack(fill="x", pady=20, padx=10)
ttk.Button(test_frame, text="🔧 Testiraj Modele", 
          command=test_models, style="Warning.TButton").pack(pady=10)

# =========== prediction tab
predict_tab = ttk.Frame(notebook, style='Modern.TFrame')
notebook.add(predict_tab, text="🎯 Napovedi")

input_frame = ttk.LabelFrame(predict_tab, text="📝 Vhodni Podatki", style="Modern.TLabelframe")
input_frame.pack(fill="both", expand=True, padx=15, pady=15)

# Speed input
speed_label = tk.Label(input_frame, 
                      text="🚗 Hitrost (20 vrednosti, ločenih s presledki ali vejicami):",
                      bg=colors['light'], fg=colors['primary'],
                      font=("Segoe UI", 12, "bold"))
speed_label.pack(anchor="w", padx=10, pady=(10, 5))

speed_input = tk.Text(input_frame, height=3, width=80,
                     font=("Consolas", 11),
                     bg='white', fg=colors['dark'],
                     borderwidth=2, relief='solid',
                     highlightthickness=1, highlightcolor=colors['secondary'],
                     highlightbackground="#d1d5db")
speed_input.pack(fill="x", padx=10, pady=5)

# Acceleration input
accel_label = tk.Label(input_frame, 
                      text="⚡ Pospešek (20 vrednosti, ločenih s presledki ali vejicami):",
                      bg=colors['light'], fg=colors['primary'],
                      font=("Segoe UI", 12, "bold"))
accel_label.pack(anchor="w", padx=10, pady=(15, 5))

accel_input = tk.Text(input_frame, height=3, width=80,
                     font=("Consolas", 11),
                     bg='white', fg=colors['dark'],
                     borderwidth=2, relief='solid',
                     highlightthickness=1, highlightcolor=colors['secondary'],
                     highlightbackground="#d1d5db")
accel_input.pack(fill="x", padx=10, pady=5)

#button frame
button_frame = tk.Frame(input_frame, bg=colors['light'])
button_frame.pack(fill="x", padx=10, pady=15)


ttk.Button(button_frame, text="🎯 Napovej", 
          command=predict_manual, style="Success.TButton").pack(side="left", padx=10)
ttk.Button(button_frame, text="🗑️ Počisti", 
          command=clear_inputs, style="Warning.TButton").pack(side="left", padx=10)



#prediction za smer
input2_frame = ttk.LabelFrame(predict_tab, text="📝 Vhodni Podatki za smerni model", style="Modern.TLabelframe")
input2_frame.pack(fill="both", expand=True, padx=15, pady=15)

move_label = tk.Label(input2_frame,
                      text="⚡ Koordinate (20 vrednosti, ločenih s presledki ali vejicami):",
                      bg=colors['light'], fg=colors['primary'],
                      font=("Segoe UI", 12, "bold"))
move_label.pack(anchor="w", padx=10, pady=(15, 5))

move_input = tk.Text(input2_frame, height=3, width=80,
                     font=("Consolas", 11),
                     bg='white', fg=colors['dark'],
                     borderwidth=2, relief='solid',
                     highlightthickness=1, highlightcolor=colors['secondary'],
                     highlightbackground="#d1d5db")
move_input.pack(fill="x", padx=10, pady=5)

button2_frame = tk.Frame(input2_frame, bg=colors['light'])
button2_frame.pack(fill="x", padx=10, pady=15)

ttk.Button(button2_frame, text="🎯 Napovej",
          command=predict_bearing, style="Success.TButton").pack(side="left", padx=10)
ttk.Button(button2_frame, text="🗑️ Počisti",
          command=clear_inputs, style="Warning.TButton").pack(side="left", padx=10)

#results frame
result_frame = ttk.LabelFrame(predict_tab, text="📊 Rezultati Analize", style="Modern.TLabelframe")
result_frame.pack(fill="both", expand=True, padx=15, pady=15)

result_display = tk.Text(result_frame, height=15, width=80, state='disabled',
                        font=("Segoe UI", 11),
                        bg='white', fg=colors['dark'],
                        borderwidth=2, relief='solid',
                        highlightthickness=0)
result_scroll = ttk.Scrollbar(result_frame, orient="vertical", command=result_display.yview)
result_display.configure(yscrollcommand=result_scroll.set)
result_scroll.pack(side="right", fill="y")
result_display.pack(fill="both", expand=True, padx=10, pady=10)

#=========csv tab
csv_tab = ttk.Frame(notebook, style='Modern.TFrame')
notebook.add(csv_tab, text="📄 CSV Obdelava")

csv_frame = ttk.LabelFrame(csv_tab, text="📈 Batch Analiza", style="Modern.TLabelframe")
csv_frame.pack(fill="both", expand=True, padx=15, pady=15)

info_frame = tk.Frame(csv_frame, bg='white', relief='solid', borderwidth=2)
info_frame.pack(fill="x", padx=10, pady=10)

csv_info = tk.Label(info_frame,
                   text="📋 Format CSV datoteke:\n" +
                        "• Če imaš oba modela: prvih 20 stolpcev = hitrost, naslednjih 20 = pospešek\n" +
                        "• Če imaš samo model za hitrost: prvih 20 stolpcev = hitrost\n" +
                        "• Če imaš samo model za pospešek: prvih 20 stolpcev = pospešek\n" +
                        "• Output datoteka: ime_output.csv",
                   justify="left", bg='white', fg=colors['dark'],
                   font=("Segoe UI", 11), padx=15, pady=15)
csv_info.pack(anchor="w")

#buttons
csv_btn_frame = tk.Frame(csv_frame, bg=colors['light'])
csv_btn_frame.pack(pady=20)

ttk.Button(csv_btn_frame, text="📊 Obdelaj CSV Datoteko", 
          command=predict_csv, style="Primary.TButton").pack(pady=10)
ttk.Button(csv_btn_frame, text="🗺️ Izris Zemljevida", 
          command=draw_map, style="Success.TButton").pack(pady=10)


stats_frame = ttk.LabelFrame(csv_frame, text="📈 Statistika Zadnje Obdelave", style="Modern.TLabelframe")
stats_frame.pack(fill="x", padx=10, pady=10)

stats_label = tk.Label(stats_frame, text="Še ni bilo obdelave CSV datoteke",
                      justify="left", anchor="w", 
                      bg='white', fg=colors['dark'],
                      relief="solid", borderwidth=1,
                      font=("Segoe UI", 11),
                      padx=15, pady=15)
stats_label.pack(fill="x", padx=5, pady=5)

#status bar===============
status_frame = tk.Frame(window, bg=colors['primary'], height=40)
status_frame.pack(side="bottom", fill="x")
status_frame.pack_propagate(False)

status_label = tk.Label(status_frame, 
                       text="🚀 Pripravljeno - naloži modele za začetek",
                       bg=colors['primary'], fg='white',
                       font=("Segoe UI", 11, "bold"),
                       anchor=tk.W, padx=15)
status_label.pack(fill="both", expand=True)

#exit
exit_frame = tk.Frame(window, bg=colors['light'])
exit_frame.pack(side="bottom", fill="x", pady=5)

ttk.Button(exit_frame, text="🚪 Izhod", 
          command=exit_application, style="Danger.TButton").pack(pady=10)

window.protocol("WM_DELETE_WINDOW", exit_application)

if __name__ == "__main__":
    window.mainloop()