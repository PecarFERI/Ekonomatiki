import gpxpy
import math
import csv


def compute_bearing(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
    d_lon = lon2 - lon1
    x = math.sin(d_lon) * math.cos(lat2)
    y = math.cos(lat1) * math.sin(lat2) - math.sin(lat1) * math.cos(lat2) * math.cos(d_lon)
    bearing = math.atan2(x, y)
    return math.degrees(bearing)


def classify_by_bearing(bearings):
    if len(bearings) < 2:
        return None
    deltas = [abs(bearings[i + 1] - bearings[i]) for i in range(len(bearings) - 1)]
    deltas = [delta if delta <= 180 else 360 - delta for delta in deltas]
    avg_delta = sum(deltas) / len(deltas)
    if avg_delta < 15:
        return 1
    elif avg_delta < 45:
        return 2
    else:
        return 3


def parse_gpx(file_path):
    with open(file_path, 'r') as gpx_file:
        gpx = gpxpy.parse(gpx_file)

    coords = []
    for track in gpx.tracks:
        for segment in track.segments:
            for point in segment.points:
                coords.append((point.latitude, point.longitude))
    return coords


def process_gpx_to_csv(gpx_file, output_csv, block_size=20):
    coords = parse_gpx(gpx_file)

    with open(output_csv, 'w', newline='') as f:
        writer = csv.writer(f)

        # glava (neobvezno)
        header = []
        for i in range(block_size):
            header.extend([f"lon{i + 1}", f"lat{i + 1}"])
        header.append("label")
        writer.writerow(header)

        for i in range(0, len(coords), block_size):
            block = coords[i:i + block_size]
            if len(block) < block_size:
                break  # zadnji del, če ni dovolj točk, preskoči

            bearings = [compute_bearing(block[j][0], block[j][1], block[j + 1][0], block[j + 1][1]) for j in
                        range(block_size - 1)]
            label = classify_by_bearing(bearings)

            row = []
            for lat, lon in block:
                row.extend([lon, lat])
            row.append(label)
            writer.writerow(row)

    print(f"📁 Obdelava zaključena, shranjeno v {output_csv}")


if __name__ == "__main__":
    import sys

    if len(sys.argv) != 3:
        print("Uporaba: python skripta.py vhodna_datoteka.gpx izhodna_datoteka.csv")
    else:
        gpx_path = sys.argv[1]
        csv_path = sys.argv[2]
        process_gpx_to_csv(gpx_path, csv_path)
