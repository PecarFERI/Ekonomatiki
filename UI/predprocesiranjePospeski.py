import pandas as pd
import numpy as np
import os
import glob
from typing import List, Tuple
import csv

class GPSDataPreprocessor:
    def __init__(self, sequence_length: int = 20, padding_value: float = 0.0):
        """
        Inicializacija preprocessorja za GPS podatke
        
        Args:
            sequence_length: Dolžina časovnih sekvenc (privzeto 20)
            padding_value: Vrednost za padding (privzeto 0.0)
        """
        self.sequence_length = sequence_length
        self.padding_value = padding_value
    
    def calculate_acceleration(self, speeds: List[float], time_interval: float = 1.0) -> List[float]:
        """
        Izračuna pospeške iz hitrosti
        
        Args:
            speeds: Seznam hitrosti v km/h
            time_interval: Časovni interval med meritvami v sekundah
            
        Returns:
            Seznam pospeškov v m/s²
        """
        if len(speeds) < 2:
            return [0.0]
        
        accelerations = []
        
        for i in range(1, len(speeds)):
            # Pretvorimo km/h v m/s
            speed_curr_ms = speeds[i] * (1000/3600)  # km/h to m/s
            speed_prev_ms = speeds[i-1] * (1000/3600)  # km/h to m/s
            
            # Izračunamo pospešek
            acceleration = (speed_curr_ms - speed_prev_ms) / time_interval
            accelerations.append(acceleration)
        
        # Prvi element je vedno 0 (ni prejšnje hitrosti)
        return [0.0] + accelerations
    
    def create_sequences(self, accelerations: List[float], zones: List[int]) -> List[Tuple[List[float], int]]:
        """
        Ustvari časovne sekvence z dolžino sequence_length
        
        Args:
            accelerations: Seznam pospeškov
            zones: Seznam con ekonomičnosti
            
        Returns:
            Seznam tuple-ov (sekvenca_pospeškov, zona)
        """
        if len(accelerations) != len(zones):
            min_len = min(len(accelerations), len(zones))
            accelerations = accelerations[:min_len]
            zones = zones[:min_len]
        
        sequences = []
        
        for i in range(len(accelerations) - self.sequence_length + 1):
            sequence = accelerations[i:i + self.sequence_length]
            # Vzamemo cono na koncu sekvence kot label
            zone = zones[i + self.sequence_length - 1]
            sequences.append((sequence, zone))
        
        return sequences
    
    def pad_sequence(self, sequence: List[float]) -> List[float]:
        """
        Doda padding sekvenci, če je krajša od sequence_length
        
        Args:
            sequence: Sekvenca za padding
            
        Returns:
            Sekvenca z dodanim paddingom
        """
        if len(sequence) >= self.sequence_length:
            return sequence[:self.sequence_length]
        else:
            padding_needed = self.sequence_length - len(sequence)
            return [self.padding_value] * padding_needed + sequence
    
    def process_csv_file(self, csv_file_path: str) -> List[Tuple[List[float], int]]:
        """
        Procesira posamezen CSV file
        
        Args:
            csv_file_path: Pot do CSV datoteke
            
        Returns:
            Seznam sekvenc s pripadajočimi conami
        """
        try:
            # Preberemo CSV datoteko
            df = pd.read_csv(csv_file_path)
            
            # Preverimo, če imamo potrebne stolpce
            required_columns = ['second', 'speed', 'zone']
            if not all(col in df.columns for col in required_columns):
                print(f"Napaka: CSV datoteka {csv_file_path} nima vseh potrebnih stolpcev: {required_columns}")
                return []
            
            # Sortiramo po času
            df = df.sort_values('second').reset_index(drop=True)
            
            speeds = df['speed'].tolist()
            zones = df['zone'].tolist()
            
            # Izračunamo pospeške
            accelerations = self.calculate_acceleration(speeds)
            
            # Ustvarimo sekvence
            sequences = self.create_sequences(accelerations, zones)
            
            # Dodamo padding, če je potreben
            padded_sequences = []
            for seq, zone in sequences:
                padded_seq = self.pad_sequence(seq)
                padded_sequences.append((padded_seq, zone))
            
            return padded_sequences
            
        except Exception as e:
            print(f"Napaka pri procesiranju datoteke {csv_file_path}: {str(e)}")
            return []
    
    def process_directory(self, input_dir: str, output_dir: str):
        """
        Procesira vse CSV datoteke v direktoriju in ustvari ločene izhodne datoteke
        
        Args:
            input_dir: Direktorij z Analysis CSV datotekami
            output_dir: Direktorij za izhodne datoteke
        """
        # Poiščemo vse CSV datoteke v direktoriju
        csv_pattern = os.path.join(input_dir, "*.csv")
        csv_files = glob.glob(csv_pattern)
        
        if not csv_files:
            print(f"Ni najdenih CSV datotek v direktoriju: {input_dir}")
            return
        
        # Ustvarimo izhodni direktorij, če ne obstaja
        os.makedirs(output_dir, exist_ok=True)
        
        print(f"Procesiranje {len(csv_files)} CSV datotek...")
        
        processed_files = []
        total_sequences = 0
        
        for csv_file in csv_files:
            print(f"\nProcesiranje: {os.path.basename(csv_file)}")
            sequences = self.process_csv_file(csv_file)
            
            if sequences:
                # Ustvarimo ime izhodne datoteke na osnovi vhodne datoteke
                base_name = os.path.splitext(os.path.basename(csv_file))[0]
                output_file = os.path.join(output_dir, f"processed_{base_name}.csv")
                
                # Shranimo procesirane podatke
                self.save_processed_data(sequences, output_file)
                
                print(f"  - Dodanih {len(sequences)} sekvenc")
                print(f"  - Shranjeno v: {os.path.basename(output_file)}")
                
                processed_files.append({
                    'input_file': csv_file,
                    'output_file': output_file,
                    'sequences_count': len(sequences)
                })
                
                total_sequences += len(sequences)
                
                # Prikažemo statistike za to datoteko
                self.print_file_statistics(sequences, os.path.basename(csv_file))
            else:
                print(f"  - Ni bilo mogoče procesirati datoteke")
        
        # Prikažemo skupne statistike
        print(f"\n{'='*60}")
        print(f"POVZETEK PROCESIRANJA")
        print(f"{'='*60}")
        print(f"Skupno procesiranih datotek: {len(processed_files)}")
        print(f"Skupno ustvarjenih sekvenc: {total_sequences}")
        print(f"\nProcesirane datoteke:")
        for file_info in processed_files:
            input_name = os.path.basename(file_info['input_file'])
            output_name = os.path.basename(file_info['output_file'])
            print(f"  {input_name} -> {output_name} ({file_info['sequences_count']} sekvenc)")
    
    def save_processed_data(self, sequences: List[Tuple[List[float], int]], output_file: str):
        """
        Shrani procesirane podatke v CSV datoteko
        
        Args:
            sequences: Seznam sekvenc s conami
            output_file: Pot do izhodne datoteke
        """
        try:
            with open(output_file, 'w', newline='') as csvfile:
                writer = csv.writer(csvfile)
                
                # Zapišemo header
                header = [f'acc_{i+1}' for i in range(self.sequence_length)] + ['zone']
                writer.writerow(header)
                
                # Zapišemo podatke
                for sequence, zone in sequences:
                    row = sequence + [zone]
                    writer.writerow(row)
                    
        except Exception as e:
            print(f"Napaka pri shranjevanju: {str(e)}")
    
    def print_file_statistics(self, sequences: List[Tuple[List[float], int]], filename: str):
        """
        Izpiše statistike za posamezno datoteko
        """
        if not sequences:
            return
        
        zone_counts = {}
        for seq, zone in sequences:
            zone_counts[zone] = zone_counts.get(zone, 0) + 1
        
        print(f"    Porazdelitev con:")
        for zone in sorted(zone_counts.keys()):
            percentage = (zone_counts[zone] / len(sequences)) * 100
            print(f"      Cona {zone}: {zone_counts[zone]} sekvenc ({percentage:.1f}%)")
    
    def print_statistics(self, sequences: List[Tuple[List[float], int]]):
        """
        Izpiše statistike o procesiranih podatkih
        """
        if not sequences:
            print("Ni podatkov za analizo.")
            return
        
        zone_counts = {}
        all_accelerations = []
        
        for seq, zone in sequences:
            zone_counts[zone] = zone_counts.get(zone, 0) + 1
            all_accelerations.extend(seq)
        
        print("\n=== STATISTIKE ===")
        print(f"Skupno sekvenc: {len(sequences)}")
        print(f"Dolžina sekvence: {self.sequence_length}")
        print(f"Padding vrednost: {self.padding_value}")
        
        print("\nPorazdelitev con:")
        for zone in sorted(zone_counts.keys()):
            percentage = (zone_counts[zone] / len(sequences)) * 100
            print(f"  Cona {zone}: {zone_counts[zone]} sekvenc ({percentage:.1f}%)")
        
        # Filtriramo padding vrednosti za statistike
        real_accelerations = [acc for acc in all_accelerations if acc != self.padding_value]
        
        if real_accelerations:
            print(f"\nStatistike pospeškov (brez padding):")
            print(f"  Min: {min(real_accelerations):.3f} m/s²")
            print(f"  Max: {max(real_accelerations):.3f} m/s²")
            print(f"  Povprečje: {np.mean(real_accelerations):.3f} m/s²")
            print(f"  Standardni odklon: {np.std(real_accelerations):.3f} m/s²")


def main():
    """
    Glavna funkcija za procesiranje GPS podatkov
    """
    # Nastavitve
    SEQUENCE_LENGTH = 20
    PADDING_VALUE = 0.0
    
    # Poti
    current_dir = os.path.dirname(os.path.abspath(__file__))
    analysis_dir = os.path.join(current_dir, "Analysis")
    output_dir = os.path.join(current_dir, "Processed")
    
    # Inicializiramo preprocessor
    preprocessor = GPSDataPreprocessor(
        sequence_length=SEQUENCE_LENGTH,
        padding_value=PADDING_VALUE
    )
    
    # Preverimo, če obstaja Analysis direktorij
    if not os.path.exists(analysis_dir):
        print(f"Napaka: Direktorij {analysis_dir} ne obstaja!")
        print("Najprej poženite GPS analizo, da boste ustvarili potrebne CSV datoteke.")
        return
    
    # Procesiramo podatke
    print("=== GPS DATA PREPROCESSOR ===")
    print(f"Vhodni direktorij: {analysis_dir}")
    print(f"Izhodni direktorij: {output_dir}")
    print(f"Dolžina sekvence: {SEQUENCE_LENGTH}")
    print(f"Padding vrednost: {PADDING_VALUE}")
    print("-" * 50)
    
    preprocessor.process_directory(analysis_dir, output_dir)


if __name__ == "__main__":
    main()