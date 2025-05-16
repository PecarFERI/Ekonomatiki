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
        self.root.title("Program za obdelavo podatkov")
        self.screen_width = root.winfo_screenwidth()
        self.screen_height = root.winfo_screenheight()
        self.screen_height -= 80
        self.root.geometry(f"{self.screen_width}x{self.screen_height}+0+0")
        ctk.set_appearance_mode("Light")
        self.main_color = "#DADDE1"
        self.text_color = "#2C2C2C"
        self.backup_color = "#E3E6E9"
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
            "#AFB42B",
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
                                             font=("Segoe UI", 14, "bold"), text_color="#00FF00")
        self.input_file_label.grid(row=1, column=0, columnspan=3)
        self.input_file_entry = ctk.CTkEntry(self.main_frame, width=150, font=("Segoe UI", 10, "bold"),
                                             fg_color=self.backup_color, text_color="#00FF00")
        self.input_file_entry.grid(row=1, column=3, pady=5, columnspan=1)
        self.input_file_button = ctk.CTkButton(self.main_frame, text="Choose File", font=("Segoe UI", 12, "bold"),
                                               command=self.choose_input_file, fg_color=self.backup_color,
                                               hover_color="#4CAF50", text_color=self.text_color)
        self.input_file_button.grid(row=1, column=4, columnspan=3)

        self.output_file_label = ctk.CTkLabel(self.main_frame, text="Output Map Name:", font=("Segoe UI", 14, "bold"),
                                              width=200, text_color="#E53935")
        self.output_file_label.grid(row=2, column=0, pady=10, columnspan=3)
        self.output_file_entry = ctk.CTkEntry(self.main_frame, width=150, font=("Segoe UI", 12, "bold"),
                                              fg_color=self.backup_color, text_color="#E53935")
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
                                                font=("Segoe UI", 12, "bold"), text_color="#A4FF00")
        self.average_speed_label.grid(row=1, column=1, pady=5)

        self.max_speed_label = ctk.CTkLabel(self.stats_frame, text="Highest Speed: -- km/h",
                                            font=("Segoe UI", 12, "bold"), text_color="#FFEB3B")
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

    def create_map(self, gps_data, output_file):
        random_index = random.randint(0, 14)
        random_color = self.colors[random_index]
        if gps_data:
            m = folium.Map(location=gps_data[0], zoom_start=12, tiles="CartoDB Voyager")
            folium.PolyLine(gps_data, color=random_color, weight=7, opacity=1).add_to(m)

            m.save(output_file)


if __name__ == "__main__":
    root = ctk.CTk()
    app = GPSMappingApp(root)
    root.mainloop()