from math import e
import customtkinter as ctk
from tkinter import filedialog, messagebox
import gpxpy
import geopy
import folium
import webbrowser
import os
from geopy.distance import geodesic
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import random
from PIL import Image


class GPSMappingApp:
    def __init__(self, root):
        self.root = root
        self.root.title("GPS Tool")
        self.screen_width = root.winfo_screenwidth()
        self.screen_height = root.winfo_screenheight()
        self.screen_height -= 80
        self.root.geometry(f"{self.screen_width}x{self.screen_height}+0+0")
        ctk.set_appearance_mode("Light")
        self.main_color = "#FFFFFF"
        self.text_color = "black"
        self.backup_color = "#FAFAFA"
        self.current_theme = "light"
        self.create_widgets()
        self.project_root = os.path.dirname(os.path.abspath(__file__))

    def create_widgets(self):
        ctk.set_appearance_mode("light")

        self.colors = [
            "#007F00",
            "#C62828",
            "#F9A825",
            "#AD1457",
            "#1565C0",
            "#6A1B9A",
            "#EF6C00",
            "#00838F",
            "#689F38",
            "#D84315",
            "#BDBDBD",
            "#B71C1C",
            "#0277BD",
            "#558B2F",
            "#FBC02D"
        ]

        sun_icon_path = r"C:\FERI\2. letnik\4. semester\Avtonomna voznja\Private\Assests\light_mode_icon.png"
        moon_icon_path = r"C:\FERI\2. letnik\4. semester\Avtonomna voznja\Private\Assests\dark_mode_icon.png"

        self.sun_icon = ctk.CTkImage(
            light_image=Image.open(sun_icon_path),
            dark_image=Image.open(sun_icon_path),
            size=(24, 24)
        )
        self.moon_icon = ctk.CTkImage(
            light_image=Image.open(moon_icon_path),
            dark_image=Image.open(moon_icon_path),
            size=(24, 24)
        )

        self.main_frame = ctk.CTkFrame(self.root, corner_radius=10, fg_color=self.main_color)
        self.main_frame.pack(fill=ctk.BOTH, expand=True, padx=20, pady=20)

        self.main_frame.grid_rowconfigure(0, weight=0)
        self.main_frame.grid_rowconfigure(1, weight=0)
        self.main_frame.grid_rowconfigure(2, weight=0)
        self.main_frame.grid_rowconfigure(3, weight=0)
        self.main_frame.grid_rowconfigure(4, weight=0)
        self.main_frame.grid_rowconfigure(5, weight=0)
        self.main_frame.grid_rowconfigure(6, weight=0)
        self.main_frame.grid_rowconfigure(7, weight=0)
        self.main_frame.grid_rowconfigure(8, weight=1)

        self.main_frame.grid_columnconfigure(0, weight=1)
        self.main_frame.grid_columnconfigure(1, weight=1)
        self.main_frame.grid_columnconfigure(2, weight=1)
        self.main_frame.grid_columnconfigure(3, weight=1)
        self.main_frame.grid_columnconfigure(4, weight=1)
        self.main_frame.grid_columnconfigure(5, weight=1)
        self.main_frame.grid_columnconfigure(6, weight=1)

        self.title_label = ctk.CTkLabel(
            self.main_frame,
            text="GPS Tool",
            font=("Segoe UI", 28, "bold"),
            text_color=self.text_color
        )
        self.title_label.grid(row=0, column=0, columnspan=7, pady=(10, 20), padx=10, sticky="ew")

        self.toggle_button = ctk.CTkButton(
            self.main_frame,
            text="",
            image=self.moon_icon,
            command=self.toggle_mode,
            width=40,
            height=40,
            corner_radius=20,
            fg_color=self.backup_color,
            hover_color=self.main_color
        )
        self.toggle_button.grid(row=0, column=6, padx=10, pady=10, sticky="ne")

        self.input_file_label = ctk.CTkLabel(self.main_frame, text="Input GPS File:", width=200,
                                             font=("Segoe UI", 14, "bold"), text_color="#32CD32")
        self.input_file_label.grid(row=1, column=0, columnspan=3)
        self.input_file_entry = ctk.CTkEntry(self.main_frame, width=150, font=("Segoe UI", 10, "bold"),
                                             fg_color=self.backup_color, text_color="#32CD32")
        self.input_file_entry.grid(row=1, column=3, pady=5, columnspan=1)
        self.input_file_button = ctk.CTkButton(self.main_frame, text="Choose File", font=("Segoe UI", 12, "bold"),
                                               command=self.choose_input_file, fg_color=self.backup_color,
                                               hover_color="#81C784", text_color=self.text_color)
        self.input_file_button.grid(row=1, column=4, columnspan=3)

        self.output_file_label = ctk.CTkLabel(self.main_frame, text="Output Map Name:", font=("Segoe UI", 14, "bold"),
                                              width=200, text_color="#EF5350")
        self.output_file_label.grid(row=2, column=0, pady=10, columnspan=3)
        self.output_file_entry = ctk.CTkEntry(self.main_frame, width=150, font=("Segoe UI", 12, "bold"),
                                              fg_color=self.backup_color, text_color="#EF5350")
        self.output_file_entry.grid(row=2, column=3, pady=5)

        self.input_folder_label = ctk.CTkLabel(self.main_frame, text="Input GPS Folder:", width=200,
                                               font=("Segoe UI", 14, "bold"), text_color="#4FC3F7")
        self.input_folder_label.grid(row=3, column=0, columnspan=3)
        self.input_folder_entry = ctk.CTkEntry(self.main_frame, width=150, font=("Segoe UI", 10, "bold"),
                                               fg_color=self.backup_color, text_color="#4FC3F7")
        self.input_folder_entry.grid(row=3, column=3, pady=5, columnspan=1)
        self.input_folder_button = ctk.CTkButton(self.main_frame, text="Choose Folder", font=("Segoe UI", 12, "bold"),
                                                 command=self.choose_input_folder, fg_color=self.backup_color,
                                                 hover_color="#4FC3F7", text_color=self.text_color)
        self.input_folder_button.grid(row=3, column=4, columnspan=3)

        self.generate_button = ctk.CTkButton(self.main_frame, text="Generate Map", font=("Segoe UI", 12, "bold"),
                                             command=self.run_generation, fg_color=self.backup_color,
                                             hover_color="#4FC3F7", text_color=self.text_color)
        self.generate_button.grid(row=4, column=0, columnspan=7, pady=20)

        self.stats_frame = ctk.CTkFrame(self.main_frame, corner_radius=10, fg_color=self.backup_color, width=450,
                                        height=100)
        self.stats_frame.grid(row=5, column=0, columnspan=7, pady=5, padx=20, sticky="nsew")
        self.stats_frame.grid(row=6, column=0, columnspan=7, pady=5, padx=20, sticky="nsew")

        self.stats_frame.grid_rowconfigure(0, weight=1)
        self.stats_frame.grid_rowconfigure(1, weight=1)
        self.stats_frame.grid_columnconfigure(0, weight=1)
        self.stats_frame.grid_columnconfigure(1, weight=1)
        self.stats_frame.grid_columnconfigure(2, weight=1)
        self.stats_frame.grid_columnconfigure(3, weight=1)
        self.stats_frame.grid_columnconfigure(4, weight=1)
        self.stats_frame.grid_columnconfigure(5, weight=1)
        self.stats_frame.grid_columnconfigure(6, weight=1)

        self.total_distance_label = ctk.CTkLabel(self.stats_frame, text="Total Distance: -- km",
                                                 font=("Segoe UI", 12, "bold"), text_color="#00B0FF")
        self.total_distance_label.grid(row=0, column=1, pady=5)

        self.average_speed_label = ctk.CTkLabel(self.stats_frame, text="Average Speed: -- km/h",
                                                font=("Segoe UI", 12, "bold"), text_color="#7CB518")
        self.average_speed_label.grid(row=1, column=1, pady=5)

        self.max_speed_label = ctk.CTkLabel(self.stats_frame, text="Highest Speed: -- km/h",
                                            font=("Segoe UI", 12, "bold"), text_color="#FFC107")
        self.max_speed_label.grid(row=0, column=3, pady=5)

        self.elevation_change_label = ctk.CTkLabel(self.stats_frame, text="Elevation Change: -- m",
                                                   font=("Segoe UI", 12, "bold"), text_color="#FF2D95")
        self.elevation_change_label.grid(row=1, column=3, pady=5)

        self.total_time_label = ctk.CTkLabel(self.stats_frame, text="Total time: -- min", font=("Segoe UI", 12, "bold"),
                                             text_color="#FF00FF")
        self.total_time_label.grid(row=0, column=5, pady=5)

        self.made_by_label = ctk.CTkLabel(self.stats_frame, text="Made by: Luka Pecar", font=("Segoe UI", 12, "bold"),
                                          text_color="#FF5F00")
        self.made_by_label.grid(row=1, column=5, pady=5)

        self.map_saved_label = ctk.CTkLabel(self.main_frame, text="Map saved to: --", font=("Segoe UI", 12, "bold"),
                                            text_color=self.text_color)
        self.map_saved_label.grid(row=7, column=0, columnspan=7, pady=10)

        self.graph_frame = ctk.CTkFrame(self.main_frame, corner_radius=10, fg_color=self.backup_color)
        self.graph_frame.grid(row=8, column=0, columnspan=3, pady=10, sticky="nsew")

        self.acceleration_graph_frame = ctk.CTkFrame(self.main_frame, corner_radius=10, fg_color=self.backup_color)
        self.acceleration_graph_frame.grid(row=8, column=4, columnspan=3, pady=10, sticky="nsew")

        self.graph_frame.grid_rowconfigure(0, weight=1)
        self.graph_frame.grid_columnconfigure(0, weight=1)
        self.graph_frame.grid_columnconfigure(1, weight=1)
        self.graph_frame.grid_columnconfigure(2, weight=1)

        self.acceleration_graph_frame.grid_rowconfigure(0, weight=1)
        self.acceleration_graph_frame.grid_columnconfigure(0, weight=1)
        self.acceleration_graph_frame.grid_columnconfigure(1, weight=1)
        self.acceleration_graph_frame.grid_columnconfigure(2, weight=1)

    def toggle_mode(self):
        if self.current_theme == "dark":
            ctk.set_appearance_mode("Light")
            self.main_color = "#DADDE1"
            self.text_color = "#2C2C2C"
            self.backup_color = "#E3E6E9"
            self.toggle_button.configure(image=self.sun_icon)
            self.current_theme = "light"
        else:
            ctk.set_appearance_mode("Dark")
            self.main_color = "#1E1E1E"
            self.text_color = "#FFFFFF"
            self.backup_color = "#333333"
            self.toggle_button.configure(image=self.moon_icon)
            self.current_theme = "dark"

        self.main_frame.configure(fg_color=self.main_color)

        self.title_label.configure(text_color=self.text_color)
        self.toggle_button.configure(fg_color=self.backup_color, hover_color=self.main_color)

        self.root.configure(bg=self.main_color)

        self.input_file_button.configure(fg_color=self.backup_color, text_color=self.text_color)
        self.input_folder_button.configure(fg_color=self.backup_color, text_color=self.text_color)
        self.generate_button.configure(fg_color=self.backup_color, text_color=self.text_color)

        self.stats_frame.configure(fg_color=self.backup_color)
        self.graph_frame.configure(fg_color=self.backup_color)
        self.acceleration_graph_frame.configure(fg_color=self.backup_color)

        self.map_saved_label.configure(text_color=self.text_color)

        self.input_file_entry.configure(fg_color=self.backup_color)
        self.output_file_entry.configure(fg_color=self.backup_color)
        self.input_folder_entry.configure(fg_color=self.backup_color)

    def run_generation(self):
        if self.input_folder_entry.get():
            self.process_gpx_folder()
        else:
            input_file = self.input_file_entry.get()
            output_file_name = self.output_file_entry.get()
            self.generate_map(input_file, output_file_name)

    def choose_input_file(self):
        file_path = filedialog.askopenfilename(title="Choose Input GPS File", filetypes=[("GPX Files", "*.gpx")])
        if file_path:
            self.input_file_entry.delete(0, ctk.END)
            self.input_file_entry.insert(0, file_path)

    def choose_input_folder(self):
        folder_path = filedialog.askdirectory(title="Select Folder Containing GPX Files")
        if folder_path:
            self.input_folder_entry.delete(0, ctk.END)
            self.input_folder_entry.insert(0, folder_path)

    def process_gpx_folder(self):
        folder_path = self.input_folder_entry.get()
        if not folder_path:
            print("No folder selected. Exiting...")
            return

        gpx_files = [f for f in os.listdir(folder_path) if f.endswith('.gpx')]

        if not gpx_files:
            print("No .gpx files found in the selected folder.")
            return

        for gpx_file in gpx_files:
            file_path = os.path.join(folder_path, gpx_file)
            output_file_name = gpx_file.replace(".gpx", "")
            self.generate_map(file_path, output_file_name)

    def generate_map(self, input_file, output_file_name):
        if not input_file or not output_file_name:
            messagebox.showerror("Error", "Please choose both input file and provide output file name.")
            return

        try:
            gps_data, total_distance, average_speed, max_speed, elevation_change, total_time = self.process_gps_data(
                input_file)

            gps_map_dir = os.path.dirname(os.path.abspath(__file__))


            gps_maps_output_path = os.path.join(gps_map_dir, 'Maps')

            gps_maps_output_path = os.path.normpath(gps_maps_output_path)

            output_directory = gps_maps_output_path
            if not os.path.exists(output_directory):
                os.makedirs(output_directory)
            output_path = os.path.join(output_directory, f"{output_file_name}.html")

            self.create_map(gps_data, output_path)

            self.total_distance_label.configure(text=f"Total Distance: {total_distance:.2f} km")
            self.average_speed_label.configure(text=f"Average Speed: {average_speed:.2f} km/h")
            self.max_speed_label.configure(text=f"Highest Speed: {max_speed:.2f} km/h")
            self.elevation_change_label.configure(text=f"Elevation Change: {elevation_change:.2f} m")
            self.total_time_label.configure(text=f"Total time: {total_time:.2f} min")

            self.map_saved_label.configure(text=f"Map saved to: {output_path}")

        except Exception as e:
            messagebox.showerror("Error", f"An error occurred: {str(e)}")

    def remove_outliers(self, speeds, timestamps, z_threshold=2.0):
        if not speeds:
            return speeds, timestamps

        speeds_array = np.array(speeds)
        mean_speed = np.mean(speeds_array)
        std_dev = np.std(speeds_array)

        filtered_indices = [
            i for i, s in enumerate(speeds) if abs((s - mean_speed) / std_dev) <= z_threshold
        ]

        filtered_speeds = [speeds[i] for i in filtered_indices]
        filtered_timestamps = [timestamps[i] for i in filtered_indices]

        return filtered_speeds, filtered_timestamps

    def process_gps_data(self, gpx_file):
        for widget in self.graph_frame.winfo_children():
            widget.destroy()
        for widget in self.acceleration_graph_frame.winfo_children():
            widget.destroy()

        with open(gpx_file, "r", encoding="utf-8") as f:
            gpx = gpxpy.parse(f)

        coordinates = []
        timestamps = []
        elevations = []
        distances = []
        speeds = [0]

        for track in gpx.tracks:
            for segment in track.segments:
                for i, point in enumerate(segment.points):
                    coordinates.append((point.latitude, point.longitude))
                    timestamps.append(point.time)
                    elevations.append(point.elevation)

                    if i > 0:
                        dist = geodesic(coordinates[i - 1], coordinates[i]).meters
                        time_diff = (timestamps[i] - timestamps[i - 1]).total_seconds()

                        if time_diff > 0:
                            speed = (dist / time_diff) * 3.6
                            speeds.append(speed)
                            distances.append(dist)

        total_distance = sum(distances) / 1000
        total_time = (timestamps[-1] - timestamps[0]).total_seconds() / 3600
        average_speed = (total_distance / total_time) if total_time > 0 else 0
        filtered_speeds, filtered_timestamps = self.remove_outliers(speeds, timestamps)
        max_speed = max(filtered_speeds, default=0)
        elevation_change = max(elevations) - min(elevations)

        difference_amount = 0

        if (len(filtered_speeds) < len(filtered_timestamps)):
            difference_amount = len(filtered_timestamps) - len(filtered_speeds)
        elif (len(filtered_speeds) > len(filtered_timestamps)):
            difference_amount = len(filtered_speeds) - len(filtered_timestamps)

        fig, ax = plt.subplots(figsize=(10, 5))

        fig.patch.set_facecolor(self.main_color)
        ax.set_facecolor(self.main_color)

        start_time = filtered_timestamps[0]
        timestamps_for_plot = [(timestamp - start_time).total_seconds() for timestamp in
                               filtered_timestamps[difference_amount:]]
        plt.plot(timestamps_for_plot, filtered_speeds, label="Speed at Timestamp", color=self.text_color)

        ax.set_xlabel('Timestamp', color='#32CD32', font='Segoe UI', fontweight='bold')
        ax.set_ylabel('Speed (km/h)', color='#EF5350', font='Segoe UI', fontweight='bold')
        ax.set_title('Speed vs Time', color=self.text_color, font='Segoe UI', fontweight='bold', fontsize=14)
        ax.tick_params(axis='x', colors='#32CD32')
        ax.tick_params(axis='y', colors='#EF5350')

        ax.grid(True, color=self.text_color, linestyle='--', linewidth=0.5)

        ax.legend(frameon=False, loc='best', fontsize=10, facecolor=self.main_color, edgecolor=self.text_color,
                  labelcolor=self.text_color)

        canvas = FigureCanvasTkAgg(fig, master=self.graph_frame)
        canvas.get_tk_widget().pack(fill=ctk.BOTH, expand=True)
        canvas.draw()

        smoothed_speeds = moving_average(filtered_speeds, window_size=12)

        zone_ratings = []
        start_time = filtered_timestamps[0]

        counter_zones = 0

        for i in range(1, len(smoothed_speeds)):
            prev_speed = smoothed_speeds[i - 1]
            curr_speed = smoothed_speeds[i]
            timestamp = filtered_timestamps[difference_amount + i]

            if curr_speed < 1 and prev_speed < 1:
                zone = 0
            elif prev_speed == 0:
                zone = 5 if curr_speed > 5 else 1
            else:
                percent_change = abs((curr_speed - prev_speed) / prev_speed) * 100

                if percent_change <= 3:
                    zone = 1
                elif percent_change <= 7:
                    zone = 2
                elif percent_change <= 15:
                    zone = 3
                elif percent_change <= 25:
                    zone = 4
                else:
                    zone = 5

            timestamp_relative = (timestamp - start_time).total_seconds()
            zone_ratings.append((timestamp_relative, zone))

        min_duration = 5
        stable_zones = []

        fig2, ax2 = plt.subplots(figsize=(10, 5))

        fig2.patch.set_facecolor(self.main_color)
        ax2.set_facecolor(self.main_color)

        for start_t, end_t, zone in stable_zones:
            ax2.hlines(y=zone, xmin=start_t, xmax=end_t, colors=self.text_color, linewidth=3)

        ax2.set_ylim(-0.5, 5.5)
        ax2.set_yticks(range(6))
        ax2.set_xlabel('Time (s)', color='#00B0FF', font='Segoe UI', fontweight='bold')
        ax2.set_ylabel('Zone', color='#FF00FF', font='Segoe UI', fontweight='bold')
        ax2.set_title('Driving Zones vs Time', color=self.text_color, font='Segoe UI', fontweight='bold',
                      fontsize=14)
        ax2.tick_params(axis='x', colors='#00B0FF')
        ax2.tick_params(axis='y', colors='#FF00FF')

        ax2.grid(True, color=self.text_color, linestyle='--', linewidth=0.5)

        canvas2 = FigureCanvasTkAgg(fig2, master=self.acceleration_graph_frame)
        canvas2.get_tk_widget().pack(fill=ctk.BOTH, expand=True)
        canvas2.draw()

        current_dir = os.path.dirname(os.path.abspath(__file__))

        analysis_dir = os.path.join(current_dir, "Analysis")

        os.makedirs(analysis_dir, exist_ok=True)

        gpx_file_name = os.path.basename(gpx_file)
        output_file_name = gpx_file_name.replace(".gpx", ".txt")

        output_file_path = os.path.join(analysis_dir, output_file_name)

        formatted_lines = []
        for start_t, end_t, zone in stable_zones:
            start = int(start_t)
            end = int(end_t)
            formatted_lines.append(f"{start}-{end}: {zone}")

        with open(output_file_path, "w") as f:
            f.write(", ".join(formatted_lines))

        print(f"Zones written to {output_file_path}")

        total_time_minutes = total_time * 60
        return coordinates, total_distance, average_speed, max_speed, elevation_change, total_time_minutes

    def create_map(self, gps_data, output_file):
        random_index = random.randint(0, 14)
        random_color = self.colors[random_index]
        if gps_data:
            m = folium.Map(location=gps_data[0], zoom_start=12, tiles="CartoDB Voyager")
            folium.PolyLine(gps_data, color=random_color, weight=7, opacity=1).add_to(m)

            m.save(output_file)


def moving_average(data, window_size=10):
    return np.convolve(data, np.ones(window_size)/window_size, mode='same')


if __name__ == "__main__":
    root = ctk.CTk()
    app = GPSMappingApp(root)
    root.mainloop()
