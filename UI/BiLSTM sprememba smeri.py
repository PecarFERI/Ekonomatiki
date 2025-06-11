import torch
import torch.nn as nn
import torch.optim as optim
import tkinter as tk
from tkinter import filedialog, messagebox, simpledialog
import math


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


class LSTMApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Klasifikacija stabilnosti vožnje - BiLSTM")

        self.model = DirectionLSTM()
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.01)

        self.X = None
        self.y = None

        self.build_gui()

    def build_gui(self):
        self.load_button = tk.Button(self.root, text="Naloži podatke", command=self.load_data)
        self.load_button.pack(pady=5)

        self.train_button = tk.Button(self.root, text="Treniraj model", command=self.train_model)
        self.train_button.pack(pady=5)

        self.predict_button = tk.Button(self.root, text="Napovej iz GPS podatkov", command=self.predict_sample)
        self.predict_button.pack(pady=5)

        self.log_box = tk.Text(self.root, height=20, width=100)
        self.log_box.pack(pady=10)

    def log(self, message):
        self.log_box.insert(tk.END, message + "\n")
        self.log_box.see(tk.END)

    def load_data(self):
        path = filedialog.askopenfilename(filetypes=[("Text/CSV Files", "*.txt *.csv")])
        if not path:
            return

        X_data = []
        y_data = []

        with open(path, 'r') as file:
            for line in file:
                # CSV datoteke lahko vsebujejo presledke ali ločila
                line = line.strip().replace(';', ',')
                if not line:
                    continue

                try:
                    values = list(map(float, line.strip().split(',')))
                    if len(values) < 6 or len(values) % 2 == 0:
                        continue  # najmanj 3 GPS točke + oznaka

                    *coord_vals, label = values
                    coords = [(coord_vals[i + 1], coord_vals[i]) for i in range(0, len(coord_vals), 2)]  # (lat, lon)
                    bearing_seq = compute_bearing_sequence(coords)

                    if len(bearing_seq) < 1:
                        continue

                    label = int(label) - 1
                    if label < 0 or label > 4:
                        continue  # ignoriraj neveljavne razrede

                    X_data.append(bearing_seq)
                    y_data.append(label)

                except Exception as e:
                    self.log(f"⚠️ Napaka pri vrstici: {line} → {e}")
                    continue

        if not X_data:
            messagebox.showerror("Napaka", "Ni bilo mogoče naložiti veljavnih podatkov.")
            return

        self.X = torch.tensor(X_data, dtype=torch.float32)
        self.y = torch.tensor(y_data, dtype=torch.long)
        self.log(f"📂 Naloženi podatki: {len(self.X)} primerov, dolžina sekvence: {len(X_data[0])}.")

    def train_model(self):
        if self.X is None or self.y is None:
            messagebox.showwarning("Napaka", "Najprej naloži podatke.")
            return

        self.model.train()
        for epoch in range(300):
            outputs = self.model(self.X)
            loss = self.criterion(outputs, self.y)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            if epoch % 20 == 0 or epoch == 299:
                with torch.no_grad():
                    preds = torch.argmax(outputs, dim=1)
                    acc = (preds == self.y).float().mean().item() * 100
                self.log(f"Epoch [{epoch}/300], Loss: {loss.item():.4f}, Accuracy: {acc:.2f}%")

        self.log("✅ Trening zaključen.")
        self.evaluate_model()

    def evaluate_model(self):
        descriptions = {
            0: "Stabilna vožnja (ravna).",
            1: "Zmerna sprememba smeri.",
            2: "Veliko sprememb smeri (ovinkasta vožnja)."
        }

        self.model.eval()
        with torch.no_grad():
            predictions = self.model(self.X)
            predicted_classes = torch.argmax(predictions, dim=1)

            self.log("\n== REZULTATI ==")
            for i in range(len(self.X)):
                angles = [round(val[0].item(), 1) for val in self.X[i]]
                pred = predicted_classes[i].item()
                desc = descriptions.get(pred, "Neznano")
                self.log(f"Smeri: {angles} → Napoved: {pred} → {desc}")

            acc = (predicted_classes == self.y).float().mean().item()
            self.log(f"\n🎯 Točnost modela: {acc * 100:.2f} %")

    def predict_sample(self):
        input_str = simpledialog.askstring("GPS vhod", "Vnesi GPS točke (lon,lat,...), brez oznake stabilnosti:")
        if not input_str:
            return

        try:
            values = list(map(float, input_str.strip().split(',')))
            if len(values) < 6 or len(values) % 2 != 0:
                raise ValueError("Potrebno je vsaj 3 točke: (lon,lat)*3")

            coords = [(values[i + 1], values[i]) for i in range(0, len(values), 2)]  # (lat, lon)
            bearing_seq = compute_bearing_sequence(coords)

            input_tensor = torch.tensor([bearing_seq], dtype=torch.float32)
            self.model.eval()
            with torch.no_grad():
                output = self.model(input_tensor)
                probabilities = torch.softmax(output, dim=1)[0]
                pred = torch.argmax(probabilities).item()
                confidence = probabilities[pred].item() * 100

            descriptions = {
                0: "Stabilna vožnja (ravna).",
                1: "Zmerna sprememba smeri.",
                2: "Veliko sprememb smeri (ovinkasta vožnja).",
                3: "Zelo nenavadna gibanja.",
                4: "Neznana / nenormalna pot."
            }
            desc = descriptions.get(pred, "Neznano")
            self.log(f"\n>> Vhodne koordinate: {coords}")
            self.log(f"Napovedani razred: {pred} → {desc}")
            self.log(f"🔍 Zaupanje modela: {confidence:.2f} %")

        except Exception as e:
            messagebox.showerror("Napaka", f"Napaka pri obdelavi: {e}")


if __name__ == "__main__":
    root = tk.Tk()
    app = LSTMApp(root)
    root.mainloop()
