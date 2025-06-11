import csv
import os
from tkinter import Tk
from tkinter.filedialog import askopenfilename

def predprocesiraj_csv(filename):
    with open(filename, 'r') as file:
        reader = csv.DictReader(file)
        rows = list(reader)

    data = [(int(row['second']), row['speed'], int(row['zone'])) for row in rows]

    output_rows = []
    current_zone = data[0][2]
    current_speeds = []

    for _, speed_str, zone in data:
        if zone != current_zone:
            output_rows.extend(padding(current_speeds, current_zone))
            current_speeds = []
            current_zone = zone
        speed = int(float(speed_str)) if float(speed_str).is_integer() else float(speed_str)
        current_speeds.append(speed)

    if current_speeds:
        output_rows.extend(padding(current_speeds, current_zone))

    header = [f"speed_{i+1}" for i in range(20)] + ["zone"]

    base, ext = os.path.splitext(filename)
    output_filename = f"{base}_predprocesirana{ext}"

    with open(output_filename, 'w', newline='') as outfile:
        writer = csv.writer(outfile)
        writer.writerow(header)
        writer.writerows(output_rows)

    print(f"Izvoz dokončan: {output_filename}")

    naredi_predict_csv(output_rows, base, ext)

def padding(speeds, zone):
    result = []
    for i in range(0, len(speeds), 20):
        chunk = speeds[i:i+20]
        while len(chunk) < 20:
            chunk.append(0.0)
        chunk.append(zone)
        result.append(chunk)
    return result

def naredi_predict_csv(output_rows, base, ext):
    predict_header = [f"speed_{i+1}" for i in range(20)]
    predict_filename = f"{base}_predict{ext}"

    with open(predict_filename, 'w', newline='') as predict_file:
        writer = csv.writer(predict_file)
        writer.writerow(predict_header)
        for row in output_rows:
            writer.writerow(row[:-1])

    print(f"Prediktorska datoteka ustvarjenaaaa: {predict_filename}")


if __name__ == "__main__":
    Tk().withdraw()
    filepath = askopenfilename(
        title="Izberi CSV datoteko",
        filetypes=[("CSV datoteke", "*.csv")]
    )
    if filepath:
        predprocesiraj_csv(filepath)
    else:
        print("Datoteka ni bila izbrana.")
