import csv
import os
from tkinter import Tk
from tkinter.filedialog import askopenfilename
from typing import List, Tuple

class HybridPreprocessor:
    def __init__(self, sequence_length: int = 20, padding_value: float = 0.0):
        self.sequence_length = sequence_length
        self.padding_value = padding_value
    
    def calculate_acceleration(self, speeds: List[float], time_interval: float = 1.0) -> List[float]:
        """Izračuna pospeške iz hitrosti"""
        if len(speeds) < 2:
            return [0.0] * len(speeds)
        
        accelerations = []
        
        for i in range(1, len(speeds)):
            speed_curr_ms = speeds[i] * (1000/3600)  # km/h to m/s
            speed_prev_ms = speeds[i-1] * (1000/3600)  # km/h to m/s
            acceleration = (speed_curr_ms - speed_prev_ms) / time_interval
            accelerations.append(acceleration)
        
        return [0.0] + accelerations  # Prvi pospešek je 0
    
    def process_chunk(self, speeds: List[float], zone: int) -> List[List[float]]:
        """Obdelava posameznega kosa podatkov (hitrosti + pospeški)"""
        # Izračunaj pospeške
        accelerations = self.calculate_acceleration(speeds)
        
        # Ustvari kombinirane vrstice (prvih 20 hitrost, naslednjih 20 pospeškov)
        combined_rows = []
        for i in range(0, len(speeds), self.sequence_length):
            speed_chunk = speeds[i:i+self.sequence_length]
            accel_chunk = accelerations[i:i+self.sequence_length]
            
            # Dodaj padding če je potrebno
            while len(speed_chunk) < self.sequence_length:
                speed_chunk.append(self.padding_value)
            while len(accel_chunk) < self.sequence_length:
                accel_chunk.append(self.padding_value)
            
            # Kombiniraj hitrosti in pospeške
            combined_row = speed_chunk + accel_chunk + [zone]
            combined_rows.append(combined_row)
        
        return combined_rows
    
    def process_file(self, input_filename: str):
        """Glavna funkcija za procesiranje podatkov"""
        with open(input_filename, 'r') as file:
            reader = csv.DictReader(file)
            rows = list(reader)

        # Preberemo podatke
        data = []
        for row in rows:
            try:
                second = int(row['second'])
                speed = float(row['speed'])
                zone = int(row['zone'])
                data.append((second, speed, zone))
            except (ValueError, KeyError) as e:
                print(f"Napaka pri branju vrstice: {e}")
                continue

        if not data:
            print("Napaka: Ni podatkov za obdelavo.")
            return

        # Razvrsti po času
        data.sort(key=lambda x: x[0])

        output_rows = []
        current_zone = data[0][2]
        current_speeds = []

        for _, speed, zone in data:
            if zone != current_zone:
                output_rows.extend(self.process_chunk(current_speeds, current_zone))
                current_speeds = []
                current_zone = zone
            current_speeds.append(speed)

        # Obdelaj še zadnjo cono
        if current_speeds:
            output_rows.extend(self.process_chunk(current_speeds, current_zone))

        # Ustvari header (prvih 20 hitrosti, naslednjih 20 pospeškov)
        header = [f"speed_{i+1}" for i in range(self.sequence_length)] + \
                 [f"acc_{i+1}" for i in range(self.sequence_length)] + \
                 ["zone"]

        # Shrani procesirane podatke z oznakami
        base, ext = os.path.splitext(input_filename)
        output_filename = f"{base}_hybrid_predprocesirano{ext}"

        with open(output_filename, 'w', newline='') as outfile:
            writer = csv.writer(outfile)
            writer.writerow(header)
            writer.writerows(output_rows)

        print(f"Procesirana datoteka shranjena: {output_filename}")

        # Shrani datoteko za napovedovanje (brez oznak)
        predict_filename = f"{base}_hybrid_predict{ext}"
        with open(predict_filename, 'w', newline='') as predict_file:
            writer = csv.writer(predict_file)
            writer.writerow(header[:-1])  # Brez zone
            for row in output_rows:
                writer.writerow(row[:-1])  # Brez zone

        print(f"Datoteka za napovedovanje shranjena: {predict_filename}")


def main():
    Tk().withdraw()
    filepath = askopenfilename(
        title="Izberi CSV datoteko",
        filetypes=[("CSV datoteke", "*.csv")]
    )
    
    if not filepath:
        print("Datoteka ni bila izbrana.")
        return
    
    preprocessor = HybridPreprocessor(sequence_length=20, padding_value=0.0)
    preprocessor.process_file(filepath)


if __name__ == "__main__":
    main()