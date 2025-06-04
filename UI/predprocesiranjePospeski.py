import csv
import os
from tkinter import Tk
from tkinter.filedialog import askopenfilename
from typing import List, Tuple

class AccelerationPreprocessor:
    def __init__(self, sequence_length: int = 20, padding_value: float = 0.0):
        self.sequence_length = sequence_length
        self.padding_value = padding_value
    
    def calculate_acceleration(self, speeds: List[float], time_interval: float = 1.0) -> List[float]:
        if len(speeds) < 2:
            return [0.0]
        
        accelerations = []
        
        for i in range(1, len(speeds)):
            speed_curr_ms = speeds[i] * (1000/3600)  # km/h to m/s
            speed_prev_ms = speeds[i-1] * (1000/3600)  # km/h to m/s
            acceleration = (speed_curr_ms - speed_prev_ms) / time_interval
            accelerations.append(acceleration)
        
        return [0.0] + accelerations
    
    def padding(self, accelerations: List[float], zone: int) -> List[List[float]]:
        """Razdeli pospeške na sekvence z dolžino sequence_length in doda padding"""
        result = []
        for i in range(0, len(accelerations), self.sequence_length):
            chunk = accelerations[i:i+self.sequence_length]
            while len(chunk) < self.sequence_length:
                chunk.append(self.padding_value)
            chunk.append(zone)
            result.append(chunk)
        return result
    
    def process_file(self, input_filename: str):
        """Glavna funkcija za procesiranje pospeškov iz CSV datoteke"""
        with open(input_filename, 'r') as file:
            reader = csv.DictReader(file)
            rows = list(reader)

        # Preberemo podatke in pretvorimo hitrosti v števila
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

        # Razvrščanje po času
        data.sort(key=lambda x: x[0])

        output_rows = []
        current_zone = data[0][2]
        current_speeds = []

        for _, speed, zone in data:
            if zone != current_zone:
                accelerations = self.calculate_acceleration(current_speeds)
                output_rows.extend(self.padding(accelerations, current_zone))
                current_speeds = []
                current_zone = zone
            current_speeds.append(speed)

        if current_speeds:
            accelerations = self.calculate_acceleration(current_speeds)
            output_rows.extend(self.padding(accelerations, current_zone))

        header = [f"acc_{i+1}" for i in range(self.sequence_length)] + ["zone"]

        base, ext = os.path.splitext(input_filename)
        output_filename = f"{base}_pospeski_predprocesirano{ext}"

        with open(output_filename, 'w', newline='') as outfile:
            writer = csv.writer(outfile)
            writer.writerow(header)
            writer.writerows(output_rows)

        print(f"Procesirana datoteka shranjena: {output_filename}")

        predict_filename = f"{base}_pospeski_predict{ext}"
        with open(predict_filename, 'w', newline='') as predict_file:
            writer = csv.writer(predict_file)
            writer.writerow(header[:-1])  
            for row in output_rows:
                writer.writerow(row[:-1])  

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
    
    preprocessor = AccelerationPreprocessor(sequence_length=20, padding_value=0.0)
    preprocessor.process_file(filepath)


if __name__ == "__main__":
    main()