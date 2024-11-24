import tkinter as tk
from tkinter import filedialog
import matplotlib.pyplot as plt
import pandas as pd
from tkinter import ttk
from tkcalendar import DateEntry
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.dates import DateFormatter
from datetime import datetime
import numpy as np
import os

# Global DataFrames to store loaded data
gas_flow_rate_df = pd.DataFrame()
substrate_feeding_df = pd.DataFrame()

# Tkinter main window
root = tk.Tk()
root.title("Data Preprocessing and Visualization")
root.geometry('1200x800')

# Notebook for tabs
notebook = ttk.Notebook(root)
notebook.pack(expand=True, fill='both')

# Tabs
load_tab = tk.Frame(notebook, bg="#F0F0F0")
preprocess_tab = tk.Frame(notebook, bg="#FFFFFF")
notebook.add(load_tab, text="Load Data")
notebook.add(preprocess_tab, text="Preprocessing")

### Load Tab ###
def load_data():
    global gas_flow_rate_df, substrate_feeding_df

    folder_path = filedialog.askdirectory(title="Select Folder Containing Data")
    if folder_path:
        gas_flow_rate_files = []
        substrate_feeding_files = []

        # Walk through the selected folder
        for root_dir, dirs, files in os.walk(folder_path):
            for file in files:
                if file == "Gas flow rate Fermenter - current.csv":
                    gas_flow_rate_files.append(os.path.join(root_dir, file))
                elif file == "Substrate feeding Fermenter - today.csv":
                    substrate_feeding_files.append(os.path.join(root_dir, file))

        # Load files
        try:
            for file in gas_flow_rate_files:
                df = pd.read_csv(file, parse_dates=['TimeStamp'], dayfirst=True)
                gas_flow_rate_df = pd.concat([gas_flow_rate_df, df], ignore_index=True)

            for file in substrate_feeding_files:
                df = pd.read_csv(file, parse_dates=['TimeStamp'], dayfirst=True)
                substrate_feeding_df = pd.concat([substrate_feeding_df, df], ignore_index=True)

            # Ensure TimeStamp is in datetime format
            gas_flow_rate_df['TimeStamp'] = pd.to_datetime(gas_flow_rate_df['TimeStamp'], errors='coerce')
            substrate_feeding_df['TimeStamp'] = pd.to_datetime(substrate_feeding_df['TimeStamp'], errors='coerce')

            # Set default dates in calendars
            min_date = gas_flow_rate_df['TimeStamp'].min().date()
            max_date = gas_flow_rate_df['TimeStamp'].max().date()

            start_date_down.set_date(min_date)
            end_date_down.set_date(max_date)
            start_date_up.set_date(min_date)
            end_date_up.set_date(max_date)

            success_label.config(text="Data loaded successfully!", fg="green")
        except Exception as e:
            success_label.config(text=f"Error: {e}", fg="red")
    else:
        success_label.config(text="No folder selected!", fg="red")


def filter_data_by_date(data, start_date, end_date):
    """Filter data based on start and end dates."""
    return data[(data['TimeStamp'] >= start_date) & (data['TimeStamp'] <= end_date)]


def preprocess_data(data, smoothing_window=30, median_window=15, z_score_threshold=2):
    """Apply preprocessing steps: smoothing, noise cancellation."""
    # Apply rolling median for spike removal
    data['MedianFiltered'] = data['ValueNum'].rolling(window=median_window, center=True).median()

    # Apply rolling mean for smoothing
    data['Smoothed'] = data['MedianFiltered'].rolling(window=smoothing_window, center=True).mean()

    # Noise Cancellation (z-score filtering)
    data['ZScore'] = np.abs((data['MedianFiltered'] - data['MedianFiltered'].mean()) / data['MedianFiltered'].std())
    data = data[data['ZScore'] < z_score_threshold]

    return data


def plot_preprocessed_graph(start_date, end_date, canvas, fig, ax1, ax2, title, upwards=False):
    try:
        # Convert dates from DateEntry widgets
        start_date = pd.Timestamp(start_date.get_date())
        end_date = pd.Timestamp(end_date.get_date())

        # Filter data by date
        data_gas = filter_data_by_date(gas_flow_rate_df, start_date, end_date)
        data_substrate = filter_data_by_date(substrate_feeding_df, start_date, end_date)

        if data_gas.empty or data_substrate.empty:
            raise ValueError("No data available for the selected date range.")

        # Preprocess the data
        data_gas = preprocess_data(data_gas, smoothing_window=50, median_window=25)
        data_substrate = preprocess_data(data_substrate, smoothing_window=30, median_window=15)

        # Plot preprocessed data
        ax1.clear()
        ax2.clear()

        # Plot Gas Production Rate
        ax1.step(data_gas['TimeStamp'], data_gas['Smoothed'], where='pre' if upwards else 'post',
                 label="Gas production flow rate after preprocessing", color='blue', linestyle='--')
        ax1.set_xlabel("Time", fontsize=10)
        ax1.set_ylabel("Biogas production rate [m³/h]", color='blue', fontsize=10)
        ax1.tick_params(axis="y", labelcolor='blue')
        ax1.grid(True)

        # Plot Substrate Feeding Rate
        ax2.step(data_substrate['TimeStamp'], data_substrate['Smoothed'], where='pre' if upwards else 'post',
                 label="Substrate feeding rate after preprocessing", color='red', linestyle='-')
        ax2.set_ylabel("Substrate feeding rate [t]", color='red', fontsize=10)
        ax2.tick_params(axis="y", labelcolor='red')

        # Add Title and Legends
        ax1.set_title(title, fontsize=12)
        ax1.legend(loc='upper left', fontsize=9)
        ax2.legend(loc='upper right', fontsize=9)

        # Draw the plot
        canvas.draw()
    except ValueError as e:
        print(f"Error: {e}")
        success_label.config(text=f"Error: {e}", fg="red")


# Event Binding for Automatic Plot Updates
def on_tab_changed(event):
    """Handle tab change events and automatically display graphs in the Preprocess tab."""
    selected_tab = notebook.index("current")
    if notebook.tab(selected_tab, "text") == "Preprocessing":
        # Automatically plot graphs when switching to the Preprocess tab
        plot_preprocessed_graph(
            start_date_down, end_date_down, canvas_down, fig_down, ax_down1, ax_down2,
            title="Preprocessed data for step downwards", upwards=False
        )
        plot_preprocessed_graph(
            start_date_up, end_date_up, canvas_up, fig_up, ax_up1, ax_up2,
            title="Preprocessed data for step upwards", upwards=True
        )

# Bind the event to the notebook
notebook.bind("<<NotebookTabChanged>>", on_tab_changed)

# Load Tab UI
load_label = tk.Label(load_tab, text="Press the button to load data", font=("Arial", 12), bg="#F0F0F0")
load_label.pack(pady=10)
load_button = tk.Button(load_tab, text="Load Data", command=load_data, font=("Arial", 12))
load_button.pack(pady=10)
success_label = tk.Label(load_tab, text="", font=("Arial", 10), bg="#F0F0F0")
success_label.pack()

# Date Entry Widgets
date_label = tk.Label(load_tab, text="Select Start and End Dates for Preprocessing:", font=("Arial", 12), bg="#F0F0F0")
date_label.pack(pady=10)

start_date_down = DateEntry(load_tab, width=12, background='darkblue', foreground='white', borderwidth=2)
start_date_down.pack()
end_date_down = DateEntry(load_tab, width=12, background='darkblue', foreground='white', borderwidth=2)
end_date_down.pack()

start_date_up = DateEntry(load_tab, width=12, background='darkblue', foreground='white', borderwidth=2)
start_date_up.pack()
end_date_up = DateEntry(load_tab, width=12, background='darkblue', foreground='white', borderwidth=2)
end_date_up.pack()

# Preprocess Tab Graphs
fig_down, ax_down1 = plt.subplots()
ax_down2 = ax_down1.twinx()
canvas_down = FigureCanvasTkAgg(fig_down, master=preprocess_tab)
canvas_down.get_tk_widget().grid(row=1, column=0, padx=10, pady=10)

fig_up, ax_up1 = plt.subplots()
ax_up2 = ax_up1.twinx()
canvas_up = FigureCanvasTkAgg(fig_up, master=preprocess_tab)
canvas_up.get_tk_widget().grid(row=1, column=1, padx=10, pady=10)

# Configure Layout
preprocess_tab.grid_columnconfigure(0, weight=1)
preprocess_tab.grid_columnconfigure(1, weight=1)

# Run the application
root.mainloop()
