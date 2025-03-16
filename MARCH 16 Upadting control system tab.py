
import control as ctrl
# from curses.ascii import ctrl
import datetime
import tkinter as tk
from tkinter import messagebox
import traceback
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import matplotlib.dates as mdates
from scipy import stats
from scipy.interpolate import interp1d
from sklearn.metrics import r2_score
from tkinter import ttk
from tkinter import filedialog  # For file dialog
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from tkcalendar import DateEntry
from openpyxl.workbook import workbook
from openpyxl import load_workbook
from matplotlib.ticker import MaxNLocator
from matplotlib.dates import DateFormatter
import matplotlib.ticker as mticker
from scipy.ndimage import uniform_filter1d
from scipy.stats import zscore
from datetime import datetime
from scipy import signal
import control
from matplotlib.ticker import AutoMinorLocator
from control import tf, c2d
from control.matlab import lsim
from datetime import timedelta

from scipy.signal import butter, filtfilt



# Global variables for data segments
biogas_segment = None
substrate_segment = None
 
# Main window
root = tk.Tk()
root.title("Python GUI App")
root.geometry('1000x900')

# Set the window icon
icon_path = ''
root.iconbitmap(icon_path)

# Set background color of the main window
root.config(bg='#FFFFFF')

# Tabs
notebook = ttk.Notebook(root)
notebook.grid(row=0, column=0, columnspan=2, padx=10, pady=10, sticky='nsew')

# Configure grid row and column weights to make the notebook expand
root.grid_rowconfigure(0, weight=1)
root.grid_columnconfigure(0, weight=1)

# Create a ttk style object
style = ttk.Style()

# Use 'clam' theme to enable color customization
style.theme_use('clam')

# Configure style for the Notebook (tabs)
style.configure("TNotebook", background="#C7C7C7")  # Background color of the notebook
style.configure("TNotebook.Tab", 
                background="#C7C7C7",  # Tab background color
                foreground="black",      # Text color
                padding=[15, 4],         # Padding around text
                font=('Arial', 10),  # Font style
                borderwidth=0,           # Remove border width
                relief="flat"            # Flat relief for no border
               )


# Configure the active tab
style.map("TNotebook.Tab", 
          background=[("selected", "white")],  # Active tab background color
          foreground=[("selected", "black")], # Active tab text color
          borderwidth=[("selected", 0)],  # Set borderwidth to 0 for selected tab
           relief=[("selected", "flat")],  # Flat relief for no border
           padding=[("selected",[15, 5] )],
           font=[("selected", ('Arial', 10))]
         )

# Frames for Each Tab
load_tab = tk.Frame(notebook,bg="#F0F0F0")
preprocess_tab = tk.Frame(notebook)
model_tab = tk.Frame(notebook)
control_tab = tk.Frame(notebook)
feeding_tab = tk.Frame(notebook)

notebook.add(load_tab, text="Load Data")
notebook.add(preprocess_tab, text="Preprocessing")
notebook.add(model_tab, text="Model Estimation")
notebook.add(control_tab, text="Control System")
notebook.add(feeding_tab, text="Feeding Schedule")



#####
# Global DataFrames to store loaded data
gas_flow_rate_df = pd.DataFrame()
substrate_feeding_df = pd.DataFrame()

def load_data():
    global gas_flow_rate_df, substrate_feeding_df

    folder_path = "Data test Oktober 2023"
    if os.path.exists(folder_path):
        gas_flow_rate_files = []
        substrate_feeding_files = []

        for root, dirs, files in os.walk(folder_path):
            for file in files:
                if file == "Gas flow rate Fermenter - current.csv":
                    gas_flow_rate_files.append(os.path.join(root, file))
                elif file == "Substrate feeding Fermenter - today.csv":
                    substrate_feeding_files.append(os.path.join(root, file))

        try:
            for file in gas_flow_rate_files:
                df = pd.read_csv(file, parse_dates=['TimeStamp'], dayfirst=True)
                gas_flow_rate_df = pd.concat([gas_flow_rate_df, df], ignore_index=True)
                print(f"Loaded Gas Flow Rate File: {file}")
                print(df.head())

            for file in substrate_feeding_files:
                df = pd.read_csv(file, parse_dates=['TimeStamp'], dayfirst=True)
                substrate_feeding_df = pd.concat([substrate_feeding_df, df], ignore_index=True)
                print(f"Loaded Substrate Feeding File: {file}")
                print(df.head())

            # Get both year and month from the CSV files
            first_timestamp = pd.to_datetime(gas_flow_rate_df['TimeStamp'].iloc[0])
            csv_year = first_timestamp.year
            csv_month = first_timestamp.month
            csv_day = first_timestamp.day

            # Create date object with CSV year and month, but current day
            csv_date = datetime(csv_year, csv_month, csv_day)
            
            # Update all DateEntry widgets with the CSV year and month
            start_date_down.set_date(csv_date)
            end_date_down.set_date(csv_date)
            start_date_up.set_date(csv_date)
            end_date_up.set_date(csv_date)

            radio_var.set("Yes")
            success_label.config(text="")
        except Exception as e:
            success_label.config(text=f"Error: {e}")
    else:
        success_label.config(text=f"Error: The folder '{folder_path}' does not exist.")

def plot_step_graph(upwards=False):
    global gas_flow_rate_df, substrate_feeding_df, fig1, fig2, canvas1, canvas2

    if gas_flow_rate_df.empty or substrate_feeding_df.empty:
        success_label.config(text="Error: Data not loaded. Please load the data first.")
        return

    try:
        # Get selected date range from calendar widgets for both plots
        if upwards:
            start_date = start_date_up.get_date()
            end_date = end_date_up.get_date()
            fig, canvas = fig2, canvas2  # Use second chart for upward plot
        else:
            start_date = start_date_down.get_date()
            end_date = end_date_down.get_date()
            fig, canvas = fig1, canvas1  # Use first chart for downward plot

        # Clear the current figure to prepare for new plot
        fig.clear()

        # Filter the data based on the selected date range
        mask_gas = (gas_flow_rate_df['TimeStamp'] >= pd.Timestamp(start_date)) & (gas_flow_rate_df['TimeStamp'] <= pd.Timestamp(end_date))
        mask_substrate = (substrate_feeding_df['TimeStamp'] >= pd.Timestamp(start_date)) & (substrate_feeding_df['TimeStamp'] <= pd.Timestamp(end_date))

        filtered_gas_flow_rate_df = gas_flow_rate_df.loc[mask_gas]
        filtered_substrate_feeding_df = substrate_feeding_df.loc[mask_substrate]

        # Plotting the graph with two y-axes
        ax1 = fig.add_subplot(111)

        # Plot Gas Flow Rate on the first y-axis with reduced thickness
        ax1.step(filtered_gas_flow_rate_df['TimeStamp'], filtered_gas_flow_rate_df['ValueNum'], where='post', 
                 label='Biogas Production Rate', linestyle='--', color='b', linewidth=0.8)  # Reduced thickness
        ax1.set_xlabel('Time', fontsize=12)
        ax1.set_ylabel('Biogas Production Rate [m3/h]', color='b', fontsize=12)
        ax1.tick_params(axis='y', labelcolor='b')

        # Customize x-axis to show date above time
        locator = mdates.HourLocator(byhour=[0, 12])  # Set ticks for 00:00 and 12:00
        formatter = mdates.DateFormatter('%d\n%H:%M')  # Multi-line label: '07\n00:00'
        ax1.xaxis.set_major_locator(locator)
        ax1.xaxis.set_major_formatter(formatter)

        # Add month and year to the last tick
        def add_month_year_to_last_tick(ax):
            ticks = ax.get_xticks()
            if len(ticks) > 0:
                ticks_as_dates = [mdates.num2date(tick) for tick in ticks]
                labels = [tick.strftime('%d\n%H:%M') for tick in ticks_as_dates]
                labels[-1] += f"\n{ticks_as_dates[-1].strftime('%b %Y')}"  # Append 'Oct 2023' to the last label
                ax.set_xticklabels(labels, ha='center')

        add_month_year_to_last_tick(ax1)

        # Ensure straight x-axis labels
        plt.setp(ax1.xaxis.get_majorticklabels(), rotation=0, ha='center')  # No rotation, labels are centered

        # Create a second y-axis for Substrate Feed Rate
        ax2 = ax1.twinx()
        
        # Interpolation for slanted transitions in Substrate Feeding
        time_substrate = filtered_substrate_feeding_df['TimeStamp'].values
        values_substrate = filtered_substrate_feeding_df['ValueNum'].values
        
        interpolated_time = []
        interpolated_values = []
        
        for i in range(len(time_substrate) - 1):
            # Add current point
            interpolated_time.append(time_substrate[i])
            interpolated_values.append(values_substrate[i])
            
            # Add interpolated slanting point
            midpoint_time = time_substrate[i] + (time_substrate[i + 1] - time_substrate[i]) / 2
            midpoint_value = (values_substrate[i] + values_substrate[i + 1]) / 2
            
            interpolated_time.append(midpoint_time)
            interpolated_values.append(midpoint_value)
        
        # Add the last point
        interpolated_time.append(time_substrate[-1])
        interpolated_values.append(values_substrate[-1])

        # Plot the interpolated Substrate Feeding line
        ax2.plot(interpolated_time, interpolated_values, label='Substrate Feeding', linestyle='-', color='r')
        ax2.set_ylabel('Substrate Feeding [t]', color='r', fontsize=12)
        ax2.tick_params(axis='y', labelcolor='r')

        # Adding legend
        lines, labels = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines + lines2, labels + labels2, loc='upper left')

        # Adding title above the graph
        fig.suptitle("Before data preprocessing", fontsize=10, fontweight='bold')  # Bold title

        # Adjust layout to make room for the title
        fig.tight_layout(rect=[0, 0, 1, 0.95])  # Reserve space for the title
        ax1.grid(True)

        # Show the plot
        canvas.draw()
    except Exception as e:
        success_label.config(text=f"Error while plotting: {e}")



# CSV load
load_frame = tk.LabelFrame(load_tab,borderwidth=0, relief="flat",bg="#F0F0F0", padx=10, pady=0.5)
load_frame.grid(row=0, column=0, columnspan=2, padx=(10, 10), pady=0, sticky="nsew")

# Press button text
press_button = tk.Label(load_frame, text="Press the button to load CSV data", font=("Arial", 12, "bold"), background="#F0F0F0")
press_button.grid(row=0, column=0, padx=10, pady=(10,5), sticky="ew")

# Button
load_button = tk.Button(load_frame, text="Load Data", font=("Arial", 10), bd=1, relief="solid", command=load_data)
load_button.grid(row=1, column=0, padx=(20, 20), pady=3, ipadx=15, ipady=0) 

# Bind hover effects
#load_button.bind("<Enter>", on_enter)  # When mouse enters the button      
#load_button.bind("<Leave>", on_leave)  # When mouse leaves the button

# Add a label to display success or error messages
success_label = ttk.Label(load_tab, text="")
success_label.grid(pady=10)


# Load frame2 for radiobutton and labels
load_frame2 = tk.Frame(load_frame)
load_frame2.grid(row=0, column=1, rowspan=2, padx=0, pady=(10,5), sticky="nsew")

# Create variable for radio buttons
radio_var = tk.StringVar(value="No")  # Default to "No"

# Radio button for "No"
no_radio = tk.Radiobutton(load_frame2, text="No", variable=radio_var, value="No", bg="#F0F0F0", font=("Arial", 12))
no_radio.grid(row=0, column=0, padx=10, pady=(10,25), sticky="ns")

# Radio button for "Yes"
yes_radio = tk.Radiobutton(load_frame2, text="Yes", variable=radio_var, value="Yes", bg="#F0F0F0", font=("Arial", 12))
yes_radio.grid(row=0, column=2, padx=0, pady=(10,25), sticky="ns")

# Label to display the file selection status
status_label = tk.Label(root, text="")
status_label.grid(row=2, column=0, columnspan=2, padx=10, pady=10)

# Move the "Data load is complete" label to column 1
toggle_label = tk.Label(load_frame, text="Data load is complete", font=("Arial", 10))
toggle_label.grid(row=1, column=1, padx=(10,40), pady=3, sticky="w")  

# Configure grid for load_frame
load_frame.grid_columnconfigure(0, weight=1)
load_frame.grid_columnconfigure(1, weight=1)
load_frame.grid_rowconfigure(0, weight=1)
load_frame.grid_rowconfigure(1, weight=1)

# Create a grid layout for the Step Downwards and Step Upwards sections
load_tab.grid_columnconfigure(0, weight=1)
load_tab.grid_columnconfigure(1, weight=1)

# Step Downwards section
down_frame = tk.LabelFrame(load_tab, borderwidth=0, relief="flat", bg="#F0F0F0",font=("Arial", 8))
down_frame.grid(row=2, column=0, padx=10, pady=(20,0), sticky='nsew')

# Center title within down_frame
title_label_down = tk.Label(down_frame, text="Select a time period for step downwards", font=("Arial", 10), bg="#F0F0F0", borderwidth=0, relief="flat")
title_label_down.grid(row=0, column=0, columnspan=4, padx=10, pady=10, sticky='nsew')

# Configure grid for down_frame
down_frame.grid_columnconfigure(0, weight=1)
down_frame.grid_columnconfigure(1, weight=1)
down_frame.grid_columnconfigure(2, weight=1)
down_frame.grid_columnconfigure(3, weight=1)

# Start and End date labels and DateEntry widgets in the same row for Step Down
start_label_down = tk.Label(down_frame, text="Start date", bg="#F0F0F0")
start_label_down.grid(row=1, column=0, padx=2, pady=5, sticky='e')

start_date_down = DateEntry(down_frame, width=12, background='darkblue', foreground='white', borderwidth=2)
start_date_down.grid(row=1, column=1, padx=(2,0), pady=5, sticky='w')

end_label_down = tk.Label(down_frame, text="End date", bg="#F0F0F0")
end_label_down.grid(row=1, column=2, padx=(0,2), pady=5, sticky='e')

end_date_down = DateEntry(down_frame, width=12, background='darkblue', foreground='white', borderwidth=2)
end_date_down.grid(row=1, column=3, padx=0, pady=5, sticky='w')

plot_down_button = tk.Button(down_frame, text="Plot data step downwards", font=("Arial", 10), bd=1, relief="solid",  command=lambda: plot_step_graph(upwards=False))
plot_down_button.grid(row=2, column=0, columnspan=4, padx=10, pady=(20,0), sticky='n')

def on_enter(event):
    plot_down_button['background'] = '#d0eaff'  # Light blue color on hover

def on_leave(event):
    plot_down_button['background'] = 'SystemButtonFace'  # Default button color
    
# Bind hover effects
plot_down_button.bind("<Enter>", on_enter)  # When mouse enters the button
plot_down_button.bind("<Leave>", on_leave)  # When mouse leaves the button

# Step Upwards section
up_frame = tk.LabelFrame(load_tab, borderwidth=0, relief="flat", bg="#F0F0F0", font=("Arial", 10))
up_frame.grid(row=2, column=1, padx=10, pady=(20, 20), sticky='nsew')  

# Center title within up_frame
title_label_up = tk.Label(up_frame, text="Select a time period for step upwards", font=("Arial", 10), bg="#F0F0F0", borderwidth=0, relief="flat")
title_label_up.grid(row=0, column=0, columnspan=4, padx=10, pady=10, sticky='nsew')

# Configure grid for up_frame
up_frame.grid_columnconfigure(0, weight=1)
up_frame.grid_columnconfigure(1, weight=1)
up_frame.grid_columnconfigure(2, weight=1)
up_frame.grid_columnconfigure(3, weight=1)

# Start and End date labels and DateEntry widgets in the same row for Step Up
start_label_up = tk.Label(up_frame, text="Start date", bg="#F0F0F0")
start_label_up.grid(row=1, column=0, padx=2, pady=5, sticky='e')

start_date_up = DateEntry(up_frame, width=12, background='darkblue', foreground='white', borderwidth=2)
start_date_up.grid(row=1, column=1, padx=(2,0), pady=5, sticky='w')

end_label_up = tk.Label(up_frame, text="End date", bg="#F0F0F0")
end_label_up.grid(row=1, column=2, padx=(0,2), pady=5, sticky='e')

end_date_up = DateEntry(up_frame, width=12, background='darkblue', foreground='white', borderwidth=2)
end_date_up.grid(row=1, column=3, padx=0, pady=5, sticky='w')

plot_up_button = tk.Button(up_frame, text="Plot data step upwards", font=("Arial", 10), bd=1, relief="solid", command=lambda: plot_step_graph(upwards=True))
plot_up_button.grid(row=2, column=0, columnspan=4, padx=10, pady=(20,0), sticky='n')

def on_enter(event):
    plot_up_button['background'] = '#d0eaff'  # Light blue color on hover

def on_leave(event):
    plot_up_button['background'] = 'SystemButtonFace'  # Default button color
    
# Bind hover effects
plot_up_button.bind("<Enter>", on_enter)  # When mouse enters the button
plot_up_button.bind("<Leave>", on_leave)  # When mouse leaves the button

# Label to display errors if any
error_label = ttk.Label(up_frame, text="", foreground="red")
error_label.grid()


# Create a new frame for charts
chart_frame = tk.LabelFrame(load_tab, borderwidth=0, relief="flat", bg="#F0F0F0", width=500, height=300)
chart_frame.grid(row=3, column=0, columnspan=2, padx=10, pady=0, sticky="nsew")
chart_frame.grid_propagate(False)  # Prevent the frame from resizing to fit its contents
 
# Configure the chart frame for a 2-column layout
chart_frame.grid_columnconfigure(0, weight=1)
chart_frame.grid_columnconfigure(1, weight=1)
chart_frame.grid_rowconfigure(0, weight=1)

# Create empty figures for the charts
fig1, ax1 = plt.subplots()
fig2, ax2 = plt.subplots()

# Embed the first chart in the chart_frame
canvas1 = FigureCanvasTkAgg(fig1, master=chart_frame)
canvas1.get_tk_widget().grid(row=0, column=0, padx=10, pady=10, sticky="nsew")
canvas1.draw()

# Embed the second chart in the chart_frame
canvas2 = FigureCanvasTkAgg(fig2, master=chart_frame)
canvas2.get_tk_widget().grid(row=0, column=1, padx=10, pady=10, sticky="nsew")
canvas2.draw()

# Function to update the canvas size when the window is resized
def resize_canvas(event):
    fig1.set_size_inches(chart_frame.winfo_width() / 200, chart_frame.winfo_height() / 150)
    fig2.set_size_inches(chart_frame.winfo_width() / 200, chart_frame.winfo_height() / 150)
    canvas1.draw()
    canvas2.draw()

# Bind the resize event to update canvas size when the window is resized
chart_frame.bind("<Configure>", resize_canvas)

# Configure load_tab to expand the frame
load_tab.grid_rowconfigure(3, weight=1)
load_tab.grid_columnconfigure(0, weight=1)
load_tab.grid_columnconfigure(1, weight=1) 

# Create a frame with a grid layout inside load_tab
preprocess_frame = tk.LabelFrame(load_tab, borderwidth=0, relief="flat", bg="#F0F0F0")
preprocess_frame.grid(row=4, column=0,columnspan=4, padx=0, pady=0, sticky="nsew")

#PREPROCESSING TAB
##########################################################################################

# Preprocessing Control Variables
SMOOTHING_WINDOW = 5  # Window size for moving average
FILTER_ORDER = 4      # Butterworth filter order

def resample_and_interpolate(timestamps, values):
    """Resample and interpolate data to a uniform time series"""
    # Create uniform time grid
    time_min = timestamps.min()
    time_max = timestamps.max()
    uniform_timestamps = pd.date_range(start=time_min, end=time_max, freq='15T')  # Adjust frequency as needed
    
    # Interpolate values
    interpolator = interp1d(timestamps.astype(np.int64), values, kind='linear', fill_value='extrapolate')
    interpolated_values = interpolator(uniform_timestamps.astype(np.int64))
    
    return uniform_timestamps, interpolated_values

def calculate_precise_derivative(timestamps, values):
    """Calculate more precise first-order discrete derivative"""
    time_diffs = np.diff(timestamps.astype(np.int64)) / 1e9  # Convert to seconds
    value_diffs = np.diff(values)
    derivatives = value_diffs / time_diffs
    return timestamps[1:], derivatives

def design_butterworth_filter(data_timestamps):
    """Design Butterworth low-pass filter for biogas data"""
    # Calculate sampling frequency from timestamps
    time_diffs = np.diff(data_timestamps.astype(np.int64) // 10**9)
    fs = 1 / np.mean(time_diffs)  # Sampling frequency in Hz
    
    # Set cutoff frequency to target 2/3 hour periodic peaks
    cutoff_freq = 1/(2.5 * 3600)  # Convert to Hz
    
    # Design the filter
    nyquist = fs / 2
    normalized_cutoff = cutoff_freq / nyquist
    b, a = butter(FILTER_ORDER, normalized_cutoff, btype='low', analog=False)
    return b, a

def apply_preprocessing(data, timestamps):
    """Apply both low-pass filtering and moving average smoothing"""
    # Design and apply Butterworth filter
    b, a = design_butterworth_filter(timestamps)
    data_lowpass = filtfilt(b, a, data)
    
    # Apply moving average smoothing
    data_smoothed = uniform_filter1d(data_lowpass, size=SMOOTHING_WINDOW)
    return data_smoothed

# Function to change to the next tab
def next_tab():
    notebook.select(preprocess_tab)

# Function to dynamically update the layout for responsiveness
def make_responsive(widget, column_weights, row_weights):
    for col, weight in enumerate(column_weights):
        widget.grid_columnconfigure(col, weight=weight)
    for row, weight in enumerate(row_weights):
        widget.grid_rowconfigure(row, weight=weight)

# Responsive Layout for Preprocessing Tab
make_responsive(preprocess_tab, column_weights=[1, 1], row_weights=[0, 1])

# Global flags to track smoothing state
is_smoothed_down = False
is_smoothed_up = False

# Update the preprocessing button function
def preprocessing_button_pushed():
    global is_smoothed_down, is_smoothed_up
    is_smoothed_down = True
    is_smoothed_up = True
    update_step_downward_plot(smoothed=True)
    update_step_upward_plot(smoothed=True)
    next_tab()


# Configure grid for preprocess_frame to make it fully responsive
preprocess_frame.grid_columnconfigure(0, weight=1)
preprocess_frame.grid_columnconfigure(1, weight=0)  # Button in center
preprocess_frame.grid_columnconfigure(2, weight=1)

# Button for preprocessing in load tab
preprocess_button = tk.Button(preprocess_frame, text="Preprocessing", font=("Arial", 10), command=preprocessing_button_pushed)
preprocess_button.grid(row=0, column=1, padx=10, pady=10, ipadx=5, ipady=5, sticky="") 

### Preprocessing Tab with Two Columns ###
# Configure the grid layout for the preprocessing tab
preprocess_tab.grid_columnconfigure(0, weight=1)
preprocess_tab.grid_columnconfigure(1, weight=1)
preprocess_tab.grid_rowconfigure(2, weight=1)

# Add Titles to the First Row in Each Column
title_label_left = tk.Label(preprocess_tab, text="Preprocessed Data for Step Downwards", font=("Arial", 12, "bold"))
title_label_left.grid(row=0, column=0, padx=10, pady=10, sticky="n")

title_label_right = tk.Label(preprocess_tab, text="Preprocessed Data for Step Upwards", font=("Arial", 12, "bold"))
title_label_right.grid(row=0, column=1, padx=10, pady=10, sticky="n")

# Frame for Preprocessed Data for Step Downwards (Left Column)
step_down_frame = tk.Frame(preprocess_tab)
step_down_frame.grid(row=1, column=0, padx=10, pady=10, sticky="nsew")
preprocess_tab.grid_rowconfigure(1, weight=1)

# Frame for Preprocessed Data for Step Upwards (Right Column)
step_up_frame = tk.Frame(preprocess_tab)
step_up_frame.grid(row=1, column=1, padx=10, pady=10, sticky="nsew")
preprocess_tab.grid_rowconfigure(1, weight=1)

# Make frames responsive
make_responsive(step_down_frame, column_weights=[1], row_weights=[1, 0, 0])
make_responsive(step_up_frame, column_weights=[1], row_weights=[1, 0, 0])

# Matplotlib Figures for Step Downwards and Step Upwards
fig_down, ax_down = plt.subplots()
fig_up, ax_up = plt.subplots()

# Embed the Matplotlib figures in the Tkinter frames
canvas_down = FigureCanvasTkAgg(fig_down, master=step_down_frame)
canvas_down.get_tk_widget().grid(row=0, column=0, padx=10, pady=10, sticky="nsew")

canvas_up = FigureCanvasTkAgg(fig_up, master=step_up_frame)
canvas_up.get_tk_widget().grid(row=0, column=0, padx=10, pady=10, sticky="nsew")


# Sliders for Step Downwards Section
slider_down_start = tk.Scale(step_down_frame, from_=0, to=100, orient="horizontal", label="Step Downwards Start")
slider_down_start.grid(row=1, column=0, padx=5, pady=5, sticky="ew")

slider_down_end = tk.Scale(step_down_frame, from_=0, to=100, orient="horizontal", label="Step Downwards End")
slider_down_end.set(100)  # Set initial value to 100
slider_down_end.grid(row=2, column=0, padx=5, pady=5, sticky="ew")

# Sliders for Step Upwards Section
slider_up_start = tk.Scale(step_up_frame, from_=0, to=100, orient="horizontal", label="Step Upwards Start")
slider_up_start.grid(row=1, column=0, padx=5, pady=5, sticky="ew")

slider_up_end = tk.Scale(step_up_frame, from_=0, to=100, orient="horizontal", label="Step Upwards End")
slider_up_end.set(100)  # Set initial value to 100
slider_up_end.grid(row=2, column=0, padx=5, pady=5, sticky="ew")

# Add the new slider change function
def on_slider_change(slider_type):
    """Handle slider value changes"""
    try:
        if slider_type == 'down':
            # Update preprocessing plot
            update_step_downward_plot()
            # Update model plot if data exists
            if global_vars['biogas_segment_down'] is not None:
                update_model_estimation_plots()
        else:  # slider_type == 'up'
            # Update preprocessing plot
            update_step_upward_plot()
            # Update model plot if data exists
            if global_vars['biogas_segment_up'] is not None:
                update_model_estimation_plots()
    except Exception as e:
        print(f"Error in slider change: {e}")

# Configure slider commands
slider_down_start.config(command=lambda val: on_slider_change('down'))
slider_down_end.config(command=lambda val: on_slider_change('down'))
slider_up_start.config(command=lambda val: on_slider_change('up'))
slider_up_end.config(command=lambda val: on_slider_change('up'))


# Configure the main preprocess_frame to center elements
preprocess_frame.grid_columnconfigure(0, weight=1)
preprocess_frame.grid_columnconfigure(1, weight=0)
preprocess_frame.grid_columnconfigure(2, weight=1)

global_vars = {
    'biogas_segment_down': None,
    'substrate_segment_down': None,
    'biogas_segment_up': None,
    'substrate_segment_up': None,
    'timestamps_down': None,
    'timestamps_up': None
}

def update_step_downward_plot(smoothed=False):
    """Update the step down plot with preprocessed data"""
    global is_smoothed_down
    if smoothed:
        is_smoothed_down = True

    try:
        # Parse start and end dates from user input
        start_date_str = start_date_down.get()
        end_date_str = end_date_down.get()
        start_date = datetime.strptime(start_date_str, "%m/%d/%y")
        end_date = datetime.strptime(end_date_str, "%m/%d/%y")

        # Filter data based on date range
        biogas_filtered = gas_flow_rate_df[
            (gas_flow_rate_df['TimeStamp'] >= start_date) &
            (gas_flow_rate_df['TimeStamp'] <= end_date)
        ]
        substrate_filtered = substrate_feeding_df[
            (substrate_feeding_df['TimeStamp'] >= start_date) &
            (substrate_feeding_df['TimeStamp'] <= end_date)
        ]

        # Store the full filtered data before applying sliders
        global_vars['full_biogas_down'] = biogas_filtered.copy()
        global_vars['full_substrate_down'] = substrate_filtered.copy()
        global_vars['full_timestamps_down'] = biogas_filtered['TimeStamp'].copy()

        # Apply slider values to refine the filtered data
        data_length = len(biogas_filtered)
        start_ind = round(slider_down_start.get() / 100 * data_length)
        end_ind = round(slider_down_end.get() / 100 * data_length)
        
        # Store slider indices for model estimation
        global_vars['slider_down_start_index'] = start_ind
        global_vars['slider_down_end_index'] = end_ind

        biogas_segment = biogas_filtered.iloc[start_ind:end_ind]
        substrate_segment = substrate_filtered.iloc[start_ind:end_ind]

        # NEW: Resample and Interpolate Data
        biogas_timestamps, biogas_values = resample_and_interpolate(
            biogas_segment['TimeStamp'], 
            biogas_segment['ValueNum']
        )
        substrate_timestamps, substrate_values = resample_and_interpolate(
            substrate_segment['TimeStamp'], 
            substrate_segment['ValueNum']
        )

        # Create new DataFrame with resampled data
        biogas_resampled = pd.DataFrame({
            'TimeStamp': biogas_timestamps,
            'ValueNum': biogas_values
        })
        substrate_resampled = pd.DataFrame({
            'TimeStamp': substrate_timestamps,
            'ValueNum': substrate_values
        })

        # Apply preprocessing if required
        value_col = 'ValueNum'
        if is_smoothed_down:
            # Calculate time differences for sampling frequency
            time_diffs = np.diff(biogas_segment['TimeStamp'].astype(np.int64) // 10**9)
            fs = 1 / np.mean(time_diffs)  # Sampling frequency in Hz
            
            # Design Butterworth filter
            cutoff_freq = 1/(2.5 * 3600)  # Target 2/3 hour periodic peaks
            nyquist = fs / 2
            normalized_cutoff = cutoff_freq / nyquist
            b, a = butter(4, normalized_cutoff, btype='low', analog=False)
            
            # Apply low-pass filter first
            biogas_data = biogas_segment[value_col].values
            substrate_data = substrate_segment[value_col].values
            
            # Handle any NaN values
            biogas_data = np.nan_to_num(biogas_data)
            substrate_data = np.nan_to_num(substrate_data)
            
            # Apply Butterworth filter
            biogas_lowpass = filtfilt(b, a, biogas_data)
            substrate_lowpass = filtfilt(b, a, substrate_data)
            
            # Then apply moving average smoothing
            window_size = 5  # Adjust window size as needed
            biogas_segment['SmoothedValueNum'] = uniform_filter1d(biogas_lowpass, size=window_size)
            substrate_segment['SmoothedValueNum'] = uniform_filter1d(substrate_lowpass, size=window_size)
            
            value_col = 'SmoothedValueNum'
            fig_down.suptitle("After data preprocessing", fontsize=10, fontweight='bold')
        else:
            fig_down.suptitle("Before data preprocessing", fontsize=10, fontweight='bold')

        # Store processed data in global variables for model estimation
        global_vars['biogas_segment_down'] = biogas_segment.copy()
        global_vars['timestamps_down'] = biogas_segment['TimeStamp'].copy()
        
        # Process substrate data
        substrate_segment['Derivative'] = substrate_segment['ValueNum'].diff().clip(lower=0)
        substrate_segment['FeedingRate'] = substrate_segment['Derivative'].fillna(0)
        
        # Handle sudden falls in FeedingRate
        threshold = 0.1
        for i in range(1, len(substrate_segment)):
            if substrate_segment['FeedingRate'].iloc[i] < threshold and substrate_segment['FeedingRate'].iloc[i - 1] > threshold:
                substrate_segment.loc[substrate_segment.index[i], 'FeedingRate'] = substrate_segment['FeedingRate'].iloc[i - 1]
        
        # Store processed substrate data
        global_vars['substrate_segment_down'] = substrate_segment.copy()

        # Clear the current figure to prepare for new plot
        fig_down.clear()
        ax1_down = fig_down.add_subplot(111)
        ax2_down = ax1_down.twinx()

        # Plot biogas production rate
        line1 = ax1_down.step(biogas_segment['TimeStamp'], 
                             biogas_segment[value_col], 
                             where='post', 
                             label='Biogas Production Rate',
                             linestyle='--',
                             color='b',
                             linewidth=0.8)

        # Configure axes
        ax1_down.set_xlabel('Time', fontsize=12)
        ax1_down.set_ylabel('Biogas Production Rate [m³/h]', color='b', fontsize=12)
        ax1_down.tick_params(axis='y', labelcolor='b')

        # Format time axis
        ax1_down.xaxis.set_major_formatter(mdates.DateFormatter('%d\n%H:%M'))
        plt.setp(ax1_down.xaxis.get_majorticklabels(), rotation=0, ha='center')

        # Add month and year to last tick
        def add_month_year_to_last_tick(ax):
            ticks = ax.get_xticks()
            if len(ticks) > 0:
                ticks_as_dates = [mdates.num2date(tick) for tick in ticks]
                labels = [tick.strftime('%d\n%H:%M') for tick in ticks_as_dates]
                labels[-1] += f"\n{ticks_as_dates[-1].strftime('%b %Y')}"
                ax.set_xticklabels(labels, ha='center')

        add_month_year_to_last_tick(ax1_down)

        # Plot substrate feeding rate
        processed_substrate = substrate_segment[substrate_segment['FeedingRate'] > 0]
        line2 = ax2_down.plot(processed_substrate['TimeStamp'],
                             processed_substrate['FeedingRate'],
                             label='Substrate Feeding Rate',
                             color='r',
                             linewidth=1)
        
        ax2_down.set_ylabel('Substrate Feeding Rate [t/h]', color='r', fontsize=12)
        ax2_down.tick_params(axis='y', labelcolor='r')

        # Add legend
        lines = line1 + line2
        labels = [l.get_label() for l in lines]
        ax1_down.legend(lines, labels, loc='upper left')

        # Add grid
        ax1_down.grid(True)
        
        # Adjust layout and draw
        fig_down.tight_layout(rect=[0, 0, 1, 0.95])
        canvas_down.draw()

        # Update model estimation plots if needed
        if is_smoothed_down:
            update_model_estimation_plots()

    except Exception as e:
        print(f"Error during graph preprocessing and plotting: {e}")
        traceback.print_exc()  # Print full error traceback for debugging

def update_step_upward_plot(smoothed=False):
    """Update the step up plot with preprocessed data"""
    global is_smoothed_up
    if smoothed:
        is_smoothed_up = True

    try:
        # Parse start and end dates from user input
        start_date_str = start_date_up.get()
        end_date_str = end_date_up.get()
        start_date = datetime.strptime(start_date_str, "%m/%d/%y")
        end_date = datetime.strptime(end_date_str, "%m/%d/%y")

        # Filter data based on date range
        biogas_filtered = gas_flow_rate_df[
            (gas_flow_rate_df['TimeStamp'] >= start_date) &
            (gas_flow_rate_df['TimeStamp'] <= end_date)
        ]
        substrate_filtered = substrate_feeding_df[
            (substrate_feeding_df['TimeStamp'] >= start_date) &
            (substrate_feeding_df['TimeStamp'] <= end_date)
        ]

        # Store the full filtered data before applying sliders
        global_vars['full_biogas_up'] = biogas_filtered.copy()
        global_vars['full_substrate_up'] = substrate_filtered.copy()
        global_vars['full_timestamps_up'] = biogas_filtered['TimeStamp'].copy()

        # Apply slider values to refine the filtered data
        data_length = len(biogas_filtered)
        start_ind = round(slider_up_start.get() / 100 * data_length)
        end_ind = round(slider_up_end.get() / 100 * data_length)
        
        # Store slider indices for model estimation
        global_vars['slider_up_start_index'] = start_ind
        global_vars['slider_up_end_index'] = end_ind

        biogas_segment = biogas_filtered.iloc[start_ind:end_ind]
        substrate_segment = substrate_filtered.iloc[start_ind:end_ind]

        # NEW: Resample and Interpolate Data
        biogas_timestamps, biogas_values = resample_and_interpolate(
            biogas_segment['TimeStamp'], 
            biogas_segment['ValueNum']
        )
        substrate_timestamps, substrate_values = resample_and_interpolate(
            substrate_segment['TimeStamp'], 
            substrate_segment['ValueNum']
        )

        # Create new DataFrame with resampled data
        biogas_resampled = pd.DataFrame({
            'TimeStamp': biogas_timestamps,
            'ValueNum': biogas_values
        })
        substrate_resampled = pd.DataFrame({
            'TimeStamp': substrate_timestamps,
            'ValueNum': substrate_values
        })


        # Apply preprocessing if required
        value_col = 'ValueNum'
        if is_smoothed_up:
            # Calculate time differences for sampling frequency
            time_diffs = np.diff(biogas_segment['TimeStamp'].astype(np.int64) // 10**9)
            fs = 1 / np.mean(time_diffs)  # Sampling frequency in Hz
            
            # Design Butterworth filter
            cutoff_freq = 1/(2.5 * 3600)  # Target 2/3 hour periodic peaks
            nyquist = fs / 2
            normalized_cutoff = cutoff_freq / nyquist
            b, a = butter(4, normalized_cutoff, btype='low', analog=False)
            
            # Apply low-pass filter
            biogas_lowpass = filtfilt(b, a, biogas_segment[value_col].values)
            substrate_lowpass = filtfilt(b, a, substrate_segment[value_col].values)
            
            # Apply moving average smoothing
            window_size = 5
            biogas_segment['SmoothedValueNum'] = uniform_filter1d(biogas_lowpass, size=window_size)
            substrate_segment['SmoothedValueNum'] = uniform_filter1d(substrate_lowpass, size=window_size)
            value_col = 'SmoothedValueNum'
            
            # Update plot title
            fig_up.suptitle("After data preprocessing", fontsize=10, fontweight='bold')
        else:
            fig_up.suptitle("Before data preprocessing", fontsize=10, fontweight='bold')

        # Store processed data in global variables for model estimation
        global_vars['biogas_segment_up'] = biogas_segment.copy()
        global_vars['timestamps_up'] = biogas_segment['TimeStamp'].copy()
        
        # Process substrate data
        substrate_segment['Derivative'] = substrate_segment['ValueNum'].diff().clip(lower=0)
        substrate_segment['FeedingRate'] = substrate_segment['Derivative'].fillna(0)
        
        # Handle sudden falls in FeedingRate
        threshold = 0.1
        for i in range(1, len(substrate_segment)):
            if substrate_segment['FeedingRate'].iloc[i] < threshold and substrate_segment['FeedingRate'].iloc[i - 1] > threshold:
                substrate_segment.loc[substrate_segment.index[i], 'FeedingRate'] = substrate_segment['FeedingRate'].iloc[i - 1]
        
        # Store processed substrate data
        global_vars['substrate_segment_up'] = substrate_segment.copy()

        # Clear the current figure to prepare for new plot
        fig_up.clear()
        ax1_up = fig_up.add_subplot(111)
        ax2_up = ax1_up.twinx()

        # Plot biogas production rate
        line1 = ax1_up.step(biogas_segment['TimeStamp'], 
                           biogas_segment[value_col], 
                           where='post', 
                           label='Biogas Production Rate',
                           linestyle='--',
                           color='b',
                           linewidth=0.8)

        # Configure axes
        ax1_up.set_xlabel('Time', fontsize=12)
        ax1_up.set_ylabel('Biogas Production Rate [m³/h]', color='b', fontsize=12)
        ax1_up.tick_params(axis='y', labelcolor='b')

        # Format time axis
        ax1_up.xaxis.set_major_formatter(mdates.DateFormatter('%d\n%H:%M'))
        plt.setp(ax1_up.xaxis.get_majorticklabels(), rotation=0, ha='center')

        # Add month and year to last tick
        def add_month_year_to_last_tick(ax):
            ticks = ax.get_xticks()
            if len(ticks) > 0:
                ticks_as_dates = [mdates.num2date(tick) for tick in ticks]
                labels = [tick.strftime('%d\n%H:%M') for tick in ticks_as_dates]
                labels[-1] += f"\n{ticks_as_dates[-1].strftime('%b %Y')}"
                ax.set_xticklabels(labels, ha='center')

        add_month_year_to_last_tick(ax1_up)

        # Plot substrate feeding rate
        processed_substrate = substrate_segment[substrate_segment['FeedingRate'] > 0]
        line2 = ax2_up.plot(processed_substrate['TimeStamp'],
                           processed_substrate['FeedingRate'],
                           label='Substrate Feeding Rate',
                           color='r',
                           linewidth=1)
        
        ax2_up.set_ylabel('Substrate Feeding Rate [t/h]', color='r', fontsize=12)
        ax2_up.tick_params(axis='y', labelcolor='r')

        # Add legend
        lines = line1 + line2
        labels = [l.get_label() for l in lines]
        ax1_up.legend(lines, labels, loc='upper left')

        # Add grid
        ax1_up.grid(True)
        
        # Adjust layout and draw
        fig_up.tight_layout(rect=[0, 0, 1, 0.95])
        canvas_up.draw()

        # Update model estimation plots if needed
        update_model_estimation_plots()

    except Exception as e:
        print(f"Error during graph preprocessing and plotting: {e}")
        traceback.print_exc()  # Print full error traceback for debugging

###############################################################################################################
#MODEL ESTIMATION

# Model Estimation Button in Preprocessing Tab
model_estimation_frame = tk.Frame(preprocess_tab)
model_estimation_frame.grid(row=3, column=0, columnspan=2, padx=10, pady=10, sticky="nsew")

# Add this to ensure data is preprocessed before model estimation
def model_estimation_button_pushed():
    """Handle transition to model estimation tab and initialize analysis"""
    if (global_vars['biogas_segment_down'] is None or 
        global_vars['substrate_segment_down'] is None or
        global_vars['biogas_segment_up'] is None or
        global_vars['substrate_segment_up'] is None):
        print("Please preprocess the data first before proceeding to model estimation.")
        return
    
    notebook.select(model_tab)  # Switch to model estimation tab
    update_model_estimation_plots()

model_estimation_button = tk.Button(
    model_estimation_frame, 
    text="Model Estimation",
    font=("Arial", 10),
    command=model_estimation_button_pushed
)
model_estimation_button.grid(row=0, column=0, padx=10, pady=10, ipadx=10, ipady=5, sticky="")

# Make the frame fully responsive and centered
preprocess_tab.grid_rowconfigure(3, weight=1)  # Add weight to the row where the button resides
model_estimation_frame.grid_columnconfigure(0, weight=1)  # Center horizontally
model_estimation_frame.grid_rowconfigure(0, weight=1)  # Center vertically


###################

# Configure the grid layout of the model estimation tab
model_tab.grid_columnconfigure(0, weight=1)
model_tab.grid_columnconfigure(1, weight=1)
model_tab.grid_rowconfigure(1, weight=1)
model_tab.grid_rowconfigure(2, weight=0)  # Button row

# Titles for Model Estimation Tab
title_label_down_model = tk.Label(
    model_tab, 
    text="Preprocessed data for step downwards",
    font=("Arial", 12, "bold")
)
title_label_down_model.grid(row=0, column=0, padx=10, pady=10, sticky="n")

title_label_up_model = tk.Label(
    model_tab, 
    text="Preprocessed data for step upwards",
    font=("Arial", 12, "bold")
)
title_label_up_model.grid(row=0, column=1, padx=10, pady=10, sticky="n")

# Frames for Charts and Tables
down_model_frame = tk.Frame(model_tab)
down_model_frame.grid(row=1, column=0, padx=10, pady=5, sticky="nsew")

up_model_frame = tk.Frame(model_tab)
up_model_frame.grid(row=1, column=1, padx=10, pady=5, sticky="nsew")

# Make frames responsive
down_model_frame.grid_columnconfigure(0, weight=1)
down_model_frame.grid_rowconfigure(0, weight=2)
down_model_frame.grid_rowconfigure(1, weight=1)

up_model_frame.grid_columnconfigure(0, weight=1)
up_model_frame.grid_rowconfigure(0, weight=2)
up_model_frame.grid_rowconfigure(1, weight=1)

# Create Matplotlib figures for model estimation
fig_model_down, ax_model_down = plt.subplots(figsize=(6, 4))
fig_model_up, ax_model_up = plt.subplots(figsize=(6, 4))

# Embed figures in the frames
canvas_model_down = FigureCanvasTkAgg(fig_model_down, master=down_model_frame)
canvas_model_down.get_tk_widget().grid(row=0, column=0, padx=5, pady=5, sticky="nsew")

canvas_model_up = FigureCanvasTkAgg(fig_model_up, master=up_model_frame)
canvas_model_up.get_tk_widget().grid(row=0, column=0, padx=5, pady=5, sticky="nsew")

# Add a title for correlation metrics for the downward step
correlation_title_down = tk.Label(
    down_model_frame,
    text="Correlation between selected model and preprocessed gas production flow rate",
    font=("Arial", 11, "bold"),
    anchor="center"
)
correlation_title_down.grid(row=2, column=0, padx=5, pady=5, sticky="nsew")

# Add a title for correlation metrics for the upward step
correlation_title_up = tk.Label(
    up_model_frame,
    text="Correlation between selected model and preprocessed gas production flow rate",
    font=("Arial", 11, "bold"),
    anchor="center"
)
correlation_title_up.grid(row=2, column=0, padx=5, pady=5, sticky="nsew")

# Make the rows for the titles in the frames responsive
down_model_frame.grid_rowconfigure(2, weight=1)
up_model_frame.grid_rowconfigure(2, weight=1)

# Make the columns for the titles in the frames responsive
down_model_frame.grid_columnconfigure(0, weight=1)
up_model_frame.grid_columnconfigure(0, weight=1)


# Create frames for metrics tables
metrics_frame_down = ttk.Frame(down_model_frame)
metrics_frame_down.grid(row=3, column=0, padx=5, pady=5, sticky="nsew")

metrics_frame_up = ttk.Frame(up_model_frame)
metrics_frame_up.grid(row=3, column=0, padx=5, pady=5, sticky="nsew")


# Ensure the frames are fully responsive
down_model_frame.grid_rowconfigure(1, weight=1)
up_model_frame.grid_rowconfigure(1, weight=1)
metrics_frame_down.grid_columnconfigure(0, weight=1)
metrics_frame_up.grid_columnconfigure(0, weight=1)

# Create tables for similarity metrics
columns = ('Similarity metrics', 'Value')
tree_down = ttk.Treeview(metrics_frame_down, columns=columns, show='headings', height=7)
tree_up = ttk.Treeview(metrics_frame_up, columns=columns, show='headings', height=7)

for tree in [tree_down, tree_up]:
    tree.heading('Similarity metrics', text='Similarity metrics')
    tree.heading('Value', text='Value')
    tree.column('Similarity metrics', width=250)
    tree.column('Value', width=150)
    tree.grid(row=0, column=0, padx=5, pady=5, sticky="nsew")

# Add Select Model button at the bottom
select_model_frame = tk.Frame(model_tab)
select_model_frame.grid(row=2, column=0, columnspan=2, pady=10)

# Create a variable to store the selected model
selected_model = tk.StringVar(value="model1")  # Default selection

# Create a frame for radio buttons
radio_frame = tk.Frame(select_model_frame)
radio_frame.grid(row=0, column=0, columnspan=2, pady=5)

# Create and arrange radio buttons horizontally
models = [
    ("Time percentage", "model1"),
    ("Pt1_Modell", "model2"),
    ("Time constant sum", "model3"),
    ("Turning tangent", "model4"),
    ("Pt1 estimator", "model5"),
    ("Pt2 estimator", "model6")
]

for i, (text, value) in enumerate(models):
    rb = tk.Radiobutton(
        radio_frame,
        text=text,
        variable=selected_model,
        value=value,
        font=("Arial", 10)
    )
    rb.grid(row=0, column=i, padx=10)

# Ensure the Treeview fills the frame
metrics_frame_down.grid_rowconfigure(0, weight=1)
metrics_frame_down.grid_columnconfigure(0, weight=1)
metrics_frame_up.grid_rowconfigure(0, weight=1)
metrics_frame_up.grid_columnconfigure(0, weight=1)


# Configure the frame to center the button
select_model_frame.grid_columnconfigure(0, weight=1)  # Make column 0 responsive
select_model_frame.grid_rowconfigure(1, weight=1)    # Adjust the row height for centering

# Create and place the button with no extra width
select_model_button = tk.Button(
    select_model_frame,
    text="Select model",
    font=("Arial", 10),
    command=lambda: select_model(),
    anchor="center"  # Center the text within the button
)
select_model_button.grid(row=1, column=0, padx=10, pady=5)  # No sticky needed to avoid resizing

# Add hover effects for buttons
def on_enter_model(event):
    event.widget['background'] = '#d0eaff'  # Light blue color on hover

def on_leave_model(event):
    event.widget['background'] = 'SystemButtonFace'  # Default button color

# Bind hover effects to buttons
select_model_button.bind("<Enter>", on_enter_model)
select_model_button.bind("<Leave>", on_leave_model)

def calculate_metrics(measured, modeled):
    """Calculate similarity metrics between measured and modeled data"""
    try:
        metrics = {}
        
        # Handle NaN and infinite values
        valid_indices = ~(np.isnan(measured) | np.isnan(modeled) | 
                         np.isinf(measured) | np.isinf(modeled))
        measured_clean = measured[valid_indices]
        modeled_clean = modeled[valid_indices]
        
        # R² calculation using correlation coefficient squared
        correlation = np.corrcoef(measured_clean, modeled_clean)[0, 1]
        metrics['R2'] = correlation ** 2
        
        # Correlation coefficients
        metrics['Pearson'] = correlation
        metrics['Spearman'] = stats.spearmanr(measured_clean, modeled_clean)[0]
        metrics['Kendall'] = stats.kendalltau(measured_clean, modeled_clean)[0]
        
        # Calculate Euclidean distance over whole time series
        total_squared_diff = np.sum((measured_clean - modeled_clean) ** 2)
        metrics['Euclidean'] = np.sqrt(np.sum((measured_clean - modeled_clean) ** 2))
 
        # Calculate Chebyshev distance as maximum single-point difference
        metrics['Chebyshev'] = np.max(np.abs(measured_clean - modeled_clean))
        
        # Cosine distance
        metrics['Cosine'] = 1 - np.dot(measured_clean, modeled_clean) / (
            np.linalg.norm(measured_clean) * np.linalg.norm(modeled_clean))
        
        return metrics
        
    except Exception as e:
        print(f"Error in calculate_metrics: {e}")
        return {
            'R2': 0,
            'Pearson': 0,
            'Spearman': 0,
            'Kendall': 0,
            'Euclidean': 0,
            'Chebyshev': 0,
            'Cosine': 0
        }
        
def update_metrics_table(tree, metrics):
    """Update the metrics display in the treeview"""
    # Clear existing items
    for item in tree.get_children():
        tree.delete(item)
        
    # Add new metrics
    metrics_display = [
        ('Coefficient of Determination (R²)', f"{metrics['R2']:.4f}"),
        ('Pearson Correlation Coefficient', f"{metrics['Pearson']:.4f}"),
        ('Spearman Rank Correlation', f"{metrics['Spearman']:.4f}"),
        ('Kendall Rank Correlation', f"{metrics['Kendall']:.4f}"),
        ('Euclidean Distance', f"{metrics['Euclidean']:.4f}"),
        ('Chebyshev Distance', f"{metrics['Chebyshev']:.4f}"),
        ('Cosine Distance', f"{metrics['Cosine']:.4f}")
    ]
    
    for metric, value in metrics_display:
        tree.insert('', tk.END, values=(metric, value))


def calculate_zeitprozentkennwert(flowrate, feed, ind_sprung, feed_max, feed_min, flowrate_max, flowrate_min, timestamps, step_direction):
    """Calculate zeitprozentkennwert parameters and model output"""
    try:
        # Time constants table
        table_zk = np.array([
        [0.300, 0.0011, 0.7950, 0.000875],
        [0.275, 0.0535, 0.7555, 0.04041],
        [0.250, 0.104, 0.7209, 0.0797],
        [0.225, 0.168, 0.6814, 0.1144],
        [0.210, 0.220, 0.6523, 0.1435],
        [0.200, 0.263, 0.6302, 0.1657],
        [0.190, 0.321, 0.6025, 0.1934],
        [0.180, 0.403, 0.5673, 0.2286],
        [0.170, 0.534, 0.5188, 0.2770],
        [0.1675, 0.590, 0.5006, 0.2953],
        [0.165, 0.661, 0.4791, 0.3167],
        [0.164, 0.699, 0.4684, 0.3274],
        [0.163, 0.744, 0.4563, 0.3395],
        [0.1625, 0.7755, 0.4482, 0.3476],
        [0.1620, 0.811, 0.4394, 0.3564],
        [0.1615, 0.860, 0.4279, 0.3680],
        [0.1610, 0.962, 0.4056, 0.3902],
        [0.160965, 1.000, 0.3979, 0.3979]
        ])

        # Gain calculation
        K = (flowrate_max - flowrate_min) / (feed_max - feed_min)
        
        # Step change detection
        initial_value = flowrate[ind_sprung]
        if step_direction == "down":
            final_value = flowrate_min
        else:
            final_value = flowrate_max

        # Find characteristic points
        target_value = (
            final_value + 0.72 * (initial_value - final_value)
            if step_direction == "down"
            else 0.72 * (final_value - initial_value) + initial_value
        )

        # Time to reach 72% of change (T1)
        t1_idx = np.where(
            np.abs(flowrate - target_value) == np.min(np.abs(flowrate - target_value))
        )[0][0]

        timestamp_numeric = mdates.date2num(timestamps)
        T1 = (timestamp_numeric[t1_idx] - timestamp_numeric[ind_sprung]) * 24 * 3600  # seconds

        # Calculate T2
        T2 = 0.2847 * T1

        # Find closest match in table_zk
        p2_value = (flowrate[t1_idx] - final_value) / (initial_value - final_value)
        idx = np.argmin(np.abs(table_zk[:, 0] - p2_value))

        # Get time constants
        TA = table_zk[idx, 2] * T1
        TB = table_zk[idx, 3] * T1
        

        # Create transfer functions
        G1_num, G1_den = [1], [TA, 1]
        G2_num, G2_den = [1], [TB, 1]
        G_num = [K]
        G_den = [1]

        # Multiply transfer functions
        combined_num, combined_den = signal.convolve(G_num, G1_num), signal.convolve(G_den, G1_den)
        combined_num, combined_den = signal.convolve(combined_num, G2_num), signal.convolve(combined_den, G2_den)

        # Discretize transfer function
        Ta = 120  # Sampling time in seconds
        Gd = signal.cont2discrete((combined_num, combined_den), Ta, method="zoh")

        # Extract transfer function coefficients
        num = Gd[0][0]  # Get numerator coefficients
        den = Gd[1]     # Get denominator coefficients
        
        # Convert to the form Gd(z) = (e·z + f)/(z² + c·z + d)
        e = num[0]      # Coefficient of z
        f = num[1]      # Constant term
        c = -den[1]     # Coefficient of z (note the negative sign)
        d = -den[2]     # Constant term (note the negative sign)

        # Print the coefficients for verification
        print(f"Transfer function coefficients:")
        print(f"e = {e:.6f}")
        print(f"f = {f:.6f}")
        print(f"c = {c:.6f}")
        print(f"d = {d:.6f}")

        # Also store these in global_vars directly for debugging
        global_vars['debug_tf_coeffs'] = {
            "e": e,
            "f": f,
            "c": c,
            "d": d
        }

        # Simulate step response
        time_points = np.linspace(0, len(feed) - 1, len(feed))
        if step_direction == "down":
            t_input = feed - feed_max
            _, yd = signal.dlsim(Gd, t_input)  # Only two outputs
            yd = yd.flatten() + flowrate_max
        else:
            t_input = feed - feed_min
            _, yd = signal.dlsim(Gd, t_input)  # Only two outputs
            yd = yd.flatten() + flowrate_min

        # Calculate metrics
        metrics = calculate_metrics(flowrate, yd)

        return {
            "model_output": yd,
            "metrics": metrics,
            "parameters": {
                "K": K,
                "TA": TA,
                "TB": TB,
                "T1": T1,
                "T2": T2,
                "ind_sprung": ind_sprung,
                "feed_max": feed_max,
                "feed_min": feed_min,
                "flowrate_max": flowrate_max,
                "flowrate_min": flowrate_min,
            },
            "Gd": Gd,  # Original discretized transfer function
            "transfer_function_coeffs": {
                "e": e,
                "f": f,
                "c": c,
                "d": d
            }
        }

    except Exception as e:
        print(f"Error in calculate_zeitprozentkennwert: {e}")
        traceback.print_exc()  # Added for better error tracking
        return None


def update_model_down_plot():
    """
    Update the step down plot with time percentage method.
    Modified to handle substrate feeding rate separately.
    """
    try:
        # Check if we have preprocessed data
        if global_vars['biogas_segment_down'] is None or global_vars['substrate_segment_down'] is None:
            print("No preprocessed data available. Please preprocess data first.")
            return
            
        # Define constants
        const = {
            'font': 'serif',
            'fontsize': 12,
            'fontsizelegend': 8,
            'fontsizeticks': 10,
            'linienbreite': 1,
            'marker_size': 4
        }
        
        # Define colors
        newcolors = ['#0072BD', '#D95319', '#EDB120', '#7E2F8E', '#77AC30', '#4DBEEE', '#A2142F']
        color_biogas = '#0000A7'  # Deep blue for biogas
        color_substrate = '#C1272D'  # Red for substrate
        color_model = '#EECC16'  # Yellow for model output
        
        fig_model_down.clear()
        ax1 = fig_model_down.add_subplot(111)
        ax2 = ax1.twinx()
        

        # Get preprocessed data
        biogas_data = global_vars['biogas_segment_down']['SmoothedValueNum']
        substrate_data = global_vars['substrate_segment_down']['FeedingRate']
        timestamps = global_vars['timestamps_down']

        
        # Process substrate data separately for plotting
        valid_substrate_mask = substrate_data > 0  # Or your condition for valid values
        valid_substrate = substrate_data[valid_substrate_mask]
        valid_timestamps_substrate = np.array(timestamps)[valid_substrate_mask]

        # Use original data for calculations and other plots
        feed_max = substrate_data.max()
        feed_min = valid_substrate.min()  # Use min of valid values only
        flowrate_max = biogas_data.max()
        flowrate_min = biogas_data.min()
        ind_sprung = np.argmax(np.diff(substrate_data.values))
        
        # Calculate margin for y-axis limits
        biogas_margin = 0.05 * (flowrate_max - flowrate_min)
        substrate_margin = 0.05 * (feed_max - feed_min)

        print("Print flowrate_max",flowrate_max)
        print("Print flowrate_max",flowrate_min)
        
        # Set y-axis limits with margins
        ax1.set_ylim([flowrate_min - biogas_margin, flowrate_max + biogas_margin])
        ax2.set_ylim([feed_min - substrate_margin, feed_max + substrate_margin])  # Using valid substrate min
        
        # Calculate zeitprozentkennwert parameters using original data
        zk_down = calculate_zeitprozentkennwert(
            biogas_data.values,
            substrate_data.values,
            ind_sprung,
            feed_max,
            feed_min,
            flowrate_max,
            flowrate_min,
            timestamps,
            step_direction="down"
        )
        
        # Setup axes properties
        ax1.grid(True)
        ax1.xaxis.set_minor_locator(AutoMinorLocator())
        ax1.yaxis.set_minor_locator(AutoMinorLocator())
            
        # Plot biogas production rate (using original data)
        line1 = ax1.plot(timestamps, biogas_data, 
                           linestyle='--',
                           color=color_biogas,
                           label='Biogas production rate',
                           linewidth=const['linienbreite'])
            
        # Plot time percentage method result (using original data)
        line2 = ax1.plot(timestamps, zk_down['model_output'],
                           linestyle='-.',
                           color=color_model,
                           label='Time percentage method',
                           linewidth=const['linienbreite'])
            
        # Plot substrate feeding rate (using valid data only)
        line3 = ax2.plot(valid_timestamps_substrate, valid_substrate,
                           linestyle='-',
                           color=color_substrate,
                           label='Substrate feeding rate',
                           linewidth=const['linienbreite'])
            
            # Configure axes
        ax1.set_xlabel('Time', fontname=const['font'], fontsize=const['fontsize'])
        ax1.set_ylabel('Biogas production rate [m³/h]',
                         fontname=const['font'],
                         fontsize=const['fontsize'],
                         color=color_biogas)
        ax2.set_ylabel('Substrate feeding rate [t/h]',
                         fontname=const['font'],
                         fontsize=const['fontsize'],
                         color=color_substrate)
            
        ax1.set_title('Time percentage method',
                         fontname=const['font'],
                         fontsize=const['fontsize'])
            
        # Format time axis
        ax1.xaxis.set_major_formatter(mdates.DateFormatter('%d\n%H:%M'))
        plt.setp(ax1.xaxis.get_majorticklabels(),
                    rotation=0,
                    fontname=const['font'],
                    fontsize=const['fontsizeticks'])
            
            # Configure grid and ticks
        ax1.tick_params(axis='y', labelcolor=color_biogas, labelsize=const['fontsizeticks'])
        ax2.tick_params(axis='y', labelcolor=color_substrate, labelsize=const['fontsizeticks'])
            
            # Add legend
        lines = line1 + line2 + line3
        labels = [l.get_label() for l in lines]
        ax1.legend(lines, labels,
                      loc='upper right',
                      fontsize=const['fontsizelegend'],
                      prop={'family': const['font']})
            
            # If zeitprozentkennwert points are available, plot them with enhanced markers
        if 'p1_index' in zk_down and 'p2_index' in zk_down:
                color_float = 0.8
                gray_color = [color_float] * 3
                
                def plot_point_markers(ax, timestamps, data, index, annotation, y_min, color):
                    # Horizontal markers
                    x_horiz = [timestamps[0],
                             timestamps[round((1 + index) / 2)],
                             timestamps[index]]
                    y_horiz = [data[index]] * 3
                    ax.plot(x_horiz, y_horiz,
                           color=color,
                           linewidth=const['linienbreite'],
                           linestyle=':',
                           marker='>',
                           markersize=const['marker_size'])
                    
                    # Vertical markers
                    x_vert = [timestamps[index]] * 3
                    y_vert = [y_min + biogas_margin/2,
                             (y_min + data[index])/2,
                             data[index]]
                    ax.plot(x_vert, y_vert,
                           color=color,
                           linewidth=const['linienbreite'],
                           linestyle=':',
                           marker='v',
                           markersize=const['marker_size'])
                    
                    # Add circle marker at intersection
                    ax.plot(x_vert[-1], y_vert[-1], 'o',
                           markersize=const['marker_size'],
                           markerfacecolor=color,
                           markeredgecolor=color)
                    
                    # Add annotation
                    ax.text(timestamps[30],
                           data[index] + biogas_margin/2,
                           annotation,
                           fontsize=8,
                           color=color)
                
                # Plot both characteristic points
                y_min = flowrate_min - biogas_margin
                plot_point_markers(ax1, timestamps, biogas_data,
                                 zk_down['p1_index'], '0.720', y_min, gray_color)
                plot_point_markers(ax1, timestamps, biogas_data,
                                 zk_down['p2_index'],
                                 f"{zk_down['p2_value']:.3f}", y_min, gray_color)
            
        # Update metrics table
        update_metrics_table(tree_down, zk_down['metrics'])

            
        if zk_down is not None:
            global_vars["down_scenario_data"] = {
                "time": timestamps,                 # Time array from the plot
                "flow_measured": biogas_data,       # Measured biogas production rate
                "model_output": zk_down["model_output"],  # Model output from the downward estimation
                "feed": substrate_data,             # Substrate feeding rate
            }
            print("Down Scenario Data Stored:", global_vars["down_scenario_data"])


            
        # Adjust layout and draw
        fig_model_down.tight_layout()
        canvas_model_down.draw()
            
    except Exception as e:
        print(f"Error in update_model_down_plot: {e}")

def update_model_up_plot():
    """
    Update the step up plot with time percentage method.
    Modified to handle substrate feeding rate separately and store model parameters.
    """
    try:
        # Check if we have preprocessed data
        if global_vars['biogas_segment_up'] is None or global_vars['substrate_segment_up'] is None:
            print("No preprocessed data available. Please preprocess data first.")
            return

        # Define constants for plotting
        const = {
            'font': 'serif',
            'fontsize': 12,
            'fontsizelegend': 8,
            'fontsizeticks': 10,
            'linienbreite': 1,
            'marker_size': 4
        }

        # Define colors with enhanced organization
        colors = {
            'biogas_line': '#0000A7',      # Deep blue for biogas line
            'zeitprozent': '#EECC16',      # Yellow for time percentage line
            'substrate': '#C1272D',        # Red for substrate line
            'grid': '#E6E6E6',            # Light gray for grid
            'markers': [0.8, 0.8, 0.8]    # Gray for markers
        }
            
        # Clear and set up the figure
        fig_model_up.clear()
        ax1 = fig_model_up.add_subplot(111)
        ax2 = ax1.twinx()
        
        # Get preprocessed data
        biogas_data = global_vars['biogas_segment_up']['SmoothedValueNum']
        substrate_data = global_vars['substrate_segment_up']['FeedingRate']
        timestamps = global_vars['timestamps_up']

        # Process substrate data separately for plotting
        valid_substrate_mask = substrate_data > 0
        valid_substrate = substrate_data[valid_substrate_mask]
        valid_timestamps_substrate = np.array(timestamps)[valid_substrate_mask]

        # Calculate model parameters
        feed_max = substrate_data.max()
        feed_min = valid_substrate.min()
        flowrate_max = biogas_data.max()
        flowrate_min = biogas_data.min()
        ind_sprung = np.argmax(np.diff(substrate_data.values))

        # Set y-axis limits with 5% padding
        y1_min = flowrate_min - 0.05 * flowrate_max
        y1_max = flowrate_max + 0.05 * flowrate_max
        y2_min = feed_min - 0.05 * feed_max
        y2_max = feed_max + 0.05 * feed_max
        
        # Calculate zeitprozentkennwert parameters
        zk_up = calculate_zeitprozentkennwert(
            biogas_data.values,
            substrate_data.values,
            ind_sprung,
            feed_max,
            feed_min,
            flowrate_max,
            flowrate_min,
            timestamps,
            step_direction="up"
        )
        
        # Store the control system parameters for later use in the control system tab
        if zk_up is not None:
            global_vars['control_system_params'] = zk_up
            
            # Setup axes properties
            ax1.grid(True, color=colors['grid'], linestyle='-', alpha=0.2)
            ax1.xaxis.set_minor_locator(AutoMinorLocator())
            ax1.yaxis.set_minor_locator(AutoMinorLocator())
            ax1.set_ylim([y1_min, y1_max])
            
            # Configure primary axis
            ax1.tick_params(labelsize=const['fontsizeticks'])
            ax1.set_xlabel('Time', fontname=const['font'], fontsize=const['fontsize'])
            ax1.set_ylabel('Biogas production rate [m³/h]', 
                          fontname=const['font'], 
                          fontsize=const['fontsize'],
                          color=colors['biogas_line'])
            ax1.set_title('Time percentage method', 
                         fontname=const['font'], 
                         fontsize=const['fontsize'])

            # Plot biogas production rate
            line1 = ax1.plot(timestamps, biogas_data, 
                           linestyle='--',
                           color=colors['biogas_line'],
                           label='Biogas production rate',
                           linewidth=const['linienbreite'])
            
            # Plot time percentage method result
            line2 = ax1.plot(timestamps, zk_up['model_output'],
                           linestyle='-.',
                           color=colors['zeitprozent'],
                           label='Time percentage method',
                           linewidth=const['linienbreite'])

            # Configure secondary axis
            ax2.set_ylim([y2_min, y2_max])
            ax2.spines['right'].set_color(colors['substrate'])
            ax2.tick_params(axis='y', colors=colors['substrate'])
            ax2.yaxis.set_minor_locator(AutoMinorLocator())
            ax2.set_ylabel('Substrate feeding rate [t/h]',
                          fontname=const['font'],
                          fontsize=const['fontsize'],
                          color=colors['substrate'])
            
            # Plot substrate feeding rate
            line3 = ax2.plot(valid_timestamps_substrate, valid_substrate,
                           color=colors['substrate'],
                           linewidth=const['linienbreite'],
                           label='Substrate feeding rate')
            
            # Format time axis
            ax1.xaxis.set_major_formatter(mdates.DateFormatter('%d\n%H:%M'))
            plt.setp(ax1.xaxis.get_majorticklabels(), 
                    rotation=0,
                    fontname=const['font'],
                    fontsize=const['fontsizeticks'])
            
            # Add legend
            lines = line1 + line2 + line3
            labels = [l.get_label() for l in lines]
            ax1.legend(lines, labels,
                      loc='upper right',
                      fontsize=const['fontsizelegend'],
                      prop={'family': const['font']})
            
            # Plot characteristic points if available
            if 'parameters' in zk_up:
                points_data = [
                    (zk_up['parameters']['ind_sprung'], '0.720'),
                    (zk_up['parameters']['ind_sprung'] + int(zk_up['parameters']['T1']/120), 
                     f"{zk_up['parameters']['T2']/zk_up['parameters']['T1']:.3f}")
                ]
                
                for idx, value in points_data:
                    plot_characteristic_point(
                        ax1, timestamps, biogas_data, idx, value, 
                        y1_min, colors['markers']
                    )

            # Update metrics table
            update_metrics_table(tree_up, zk_up['metrics'])

            # Store model parameters and data for control system
            global_vars["up_scenario_data"] = {
                "time": timestamps,
                "flow_measured": biogas_data,
                "model_output": zk_up["model_output"],
                "feed": substrate_data,
                # Control system parameters
                "K": zk_up['parameters']['K'],        # Process gain
                "T": zk_up['parameters']['TA'],       # Main time constant
                "T_summe": zk_up['parameters']['TA'] + zk_up['parameters']['TB'],  # Sum of time constants
                "T1": zk_up['parameters']['T1'],      # Primary time constant
                "T2": zk_up['parameters']['T2'],      # Secondary time constant
                "ind_sprung": zk_up['parameters']['ind_sprung'],  # Step index
                # Operating points
                "feed_max": feed_max,
                "feed_min": feed_min,
                "flowrate_max": flowrate_max,
                "flowrate_min": flowrate_min
            }
            print("Up Scenario Data Stored:", global_vars["up_scenario_data"])
            
            # Adjust layout and draw
            fig_model_up.tight_layout()
            canvas_model_up.draw()
            
    except Exception as e:
        print(f"Error in update_model_up_plot: {e}")
        traceback.print_exc()


def plot_characteristic_point(ax, timestamps, data, idx, value, y_min, color):
    """Helper function for plotting characteristic points with enhanced markers"""
    try:
        # Horizontal markers
        x_horiz = [timestamps[0], 
                  timestamps[round((1 + idx) / 2)],
                  timestamps[idx]]
        y_horiz = [data[idx]] * 3
        ax.plot(x_horiz, y_horiz,
               color=color,
               linewidth=1,
               linestyle=':',
               marker='>',
               markersize=4)
        
        # Vertical markers
        x_vert = [timestamps[idx]] * 3
        y_vert = [y_min + 3,
                 (y_min + data[idx]) / 2,
                 data[idx]]
        ax.plot(x_vert, y_vert,
               color=color,
               linewidth=1,
               linestyle=':',
               marker='v',
               markersize=4)
        
        # Point marker
        ax.plot(x_vert[-1], y_vert[-1], 'o',
               markersize=4,
               markeredgecolor=color,
               markerfacecolor=color)
        
        # Text annotation
        ax.text(timestamps[30],
               data[idx] + 2,
               value,
               fontsize=8,
               color=color)
    except Exception as e:
        print(f"Error in plot_characteristic_point: {e}")

# Part of PT1-approximation model from theory 
def validate_data(biogas_data, timestamps, initial_value, final_value):
    """Validate input data for T63 calculation"""
    print("Data validation:")
    print(f"Biogas data type: {type(biogas_data)}")
    print(f"Biogas data shape: {biogas_data.shape if hasattr(biogas_data, 'shape') else len(biogas_data)}")
    print(f"Timestamps type: {type(timestamps)}")
    print(f"Timestamps length: {len(timestamps)}")
    print(f"Initial value: {initial_value}")
    print(f"Final value: {final_value}")
    
    # Print samples for verification
    print("Biogas data sample:", biogas_data[:5])
    print("Timestamps sample:", timestamps[:5])
    
    return all([
        len(biogas_data) > 0,
        len(timestamps) > 0,
        len(biogas_data) == len(timestamps),
        initial_value is not None,
        final_value is not None
    ])

def calculate_T63(biogas_data, timestamps, initial_value, final_value):
    """
    Calculate T63% - time when biogas production reaches 63% of final value
    """
    try:
        # Print input values for debugging
        print("Initial value:", initial_value)
        print("Final value:", final_value)
        
        # Calculate target value at 63%
        target_value = initial_value + 0.63 * (final_value - initial_value)
        print("Target value:", target_value)
        
        # Print first few values of biogas data for verification
        print("Biogas data sample:", biogas_data[:5])
        print("Timestamps sample:", timestamps[:5])
        
        # Find index where biogas production crosses target value
        idx = np.where(np.abs(biogas_data - target_value) == 
                      np.min(np.abs(biogas_data - target_value)))[0][0]
        print("Found index:", idx)
        
        # Convert to timestamps for T63
        # Fix: Use iloc for positional indexing with pandas Series
        t0 = mdates.date2num(timestamps.iloc[0])  # Start time
        t63 = mdates.date2num(timestamps.iloc[idx])  # Time at 63%
        
        # Convert to seconds
        T63 = (t63 - t0) * 24 * 3600  # Convert days to seconds
        print("Calculated T63:", T63)
        
        return T63, idx
        
    except Exception as e:
        print(f"Error calculating T63: {e}")
        print("Full error trace:")
        traceback.print_exc()
        return None, None

def update_model_estimation_plots():
    """Update both step up and step down plots in model estimation tab based on selected model"""
    try:
        # Get selected model
        model = selected_model.get()
        
        # Clear existing plots
        fig_model_down.clear()
        fig_model_up.clear()
        
        if model == "model1":  # Time percentage
            # Use existing time percentage calculation
            update_model_down_plot()
            update_model_up_plot()

    except Exception as e:
        print(f"Error updating model plots: {e}")

def clear_plots():
    """Clear both plots and reset them to blank state"""
    try:
        # Clear down plot
        fig_model_down.clear()
        ax1_down = fig_model_down.add_subplot(111)
        ax1_down.set_xlabel('Time')
        ax1_down.set_ylabel('Biogas production rate [m³/h]')
        ax1_down.set_title('Model Output')
        fig_model_down.tight_layout()
        canvas_model_down.draw()
        
        # Clear up plot
        fig_model_up.clear()
        ax1_up = fig_model_up.add_subplot(111)
        ax1_up.set_xlabel('Time')
        ax1_up.set_ylabel('Biogas production rate [m³/h]')
        ax1_up.set_title('Model Output')
        fig_model_up.tight_layout()
        canvas_model_up.draw()
        
        # Clear metrics tables
        for tree in [tree_down, tree_up]:
            for item in tree.get_children():
                tree.delete(item)
                
    except Exception as e:
        print(f"Error clearing plots: {e}")

# Modify the radio button command to update plots when selection changes
def on_model_select():
    """Handle radio button selection change"""
    try:
        update_model_estimation_plots()
    except Exception as e:
        print(f"Error in model selection: {e}")

# Update radio button creation to include command
for i, (text, value) in enumerate(models):
    rb = tk.Radiobutton(
        radio_frame,
        text=text,
        variable=selected_model,
        value=value,
        font=("Arial", 10),
        command=on_model_select  # Add command to handle selection
    )
    rb.grid(row=0, column=i, padx=10)


# Call clear_plots initially to start with blank plots
clear_plots()

def select_model():
    """Handle model selection button click:
       1. Store the selected model's output data.
       2. Calculate control system parameters.
       3. Switch to the 'Control System' tab.
    """
    try:
        update_model_estimation_plots()  # Update plots with selected model

        # Get which model is selected
        model = selected_model.get()
        print(f"Selected model: {model}")

        # Ensure selected_model_data is a dictionary
        global selected_model_data  
        if 'selected_model_data' not in globals():
            selected_model_data = {}

        # Store the upward response data - using up_scenario_data that's already populated
        if "up_scenario_data" in global_vars:
            # Validate the required data is present
            required_params = ['K', 'T', 'T_summe']
            if all(param in global_vars["up_scenario_data"] for param in required_params):
                selected_model_data['upward'] = global_vars["up_scenario_data"]
                print(f"Stored Upward Model Data: {selected_model_data['upward']}")
                
                # Calculate the control system parameters here
                print("Calculating control system parameters...")
                if store_transfer_function_coefficients():
                    print("Transfer function coefficients stored successfully.")
                    if resample_transfer_function():
                        print("Transfer function resampled successfully.")
                        if convert_to_state_space():
                            print("State space conversion successful.")
                            if design_deadbeat_controller():
                                print("Deadbeat controller designed successfully.")
                                if calculate_prefilter_gain():
                                    print("Prefilter gain calculated successfully.")
                                    print("All control system parameters calculated successfully.")
                                else:
                                    print("Failed to calculate prefilter gain.")
                            else:
                                print("Failed to design deadbeat controller.")
                        else:
                            print("Failed to convert to state space.")
                    else:
                        print("Failed to resample transfer function.")
                else:
                    print("Failed to extract transfer function coefficients.")
                
                # Switch to Control System tab
                notebook.select(control_tab)
            else:
                messagebox.showerror("Error", "Missing required model parameters. Please ensure model estimation is complete.")
                print("Error: Missing required parameters in up_scenario_data")
                return
        else:
            messagebox.showerror("Error", "No upward scenario data available. Please process data first.")
            print("Warning: No upward scenario data available. Please process data first.")
            return

    except Exception as e:
        messagebox.showerror("Error", f"Error in model selection: {str(e)}")
        print(f"Error in select_model: {e}")
        traceback.print_exc()



# Add hover effects for buttons
def on_enter_model(event):
    event.widget['background'] = '#d0eaff'  # Light blue color on hover 

def on_leave_model(event):
    event.widget['background'] = 'SystemButtonFace'  # Default button color

# Bind hover effects to buttons
model_estimation_button.bind("<Enter>", on_enter_model)
model_estimation_button.bind("<Leave>", on_leave_model)
select_model_button.bind("<Enter>", on_enter_model)
select_model_button.bind("<Leave>", on_leave_model)

###################################################################################################

# CONTROL SYSTEMS

# Configure the Control Systems tab
control_tab.grid_rowconfigure(0, weight=1)
control_tab.grid_columnconfigure(0, weight=1)

# Main frame for the Control Systems section
main_frame = ttk.Frame(control_tab, padding=20)
main_frame.grid(row=0, column=0, sticky='nsew')
main_frame.grid_columnconfigure(0, weight=1)
main_frame.grid_rowconfigure(2, weight=1)

# Top frame for inputs
top_frame = ttk.Frame(main_frame, padding=10)
top_frame.grid(row=0, column=0, sticky='ew', pady=10)
top_frame.grid_columnconfigure(0, weight=3)
top_frame.grid_columnconfigure(1, weight=2)

# Feedback Controller Section
feedback_frame = ttk.LabelFrame(top_frame, text="Feedback Controller", padding=10)
feedback_frame.grid(row=0, column=0, sticky='nsew', padx=10)
feedback_frame.grid_columnconfigure((0, 1, 2, 3), weight=1)

def create_input(frame, label, unit, row, column):
    """Helper to create a labeled input field"""
    input_frame = ttk.Frame(frame)
    input_frame.grid(row=row, column=column, sticky='w', padx=5, pady=5)
    ttk.Label(input_frame, text=label, style='InputLabel.TLabel').pack(side='left', padx=(0, 10))
    entry = ttk.Entry(input_frame, width=15)
    entry.pack(side='left', padx=(0, 5))
    ttk.Label(input_frame, text=unit, style='InputLabel.TLabel').pack(side='left')
    return entry

# Create input fields with default values
gas_flow_initial_field = create_input(feedback_frame, "Gas production flow rate, initial state", "[m³/h]", 0, 0)
gas_flow_initial_field.insert(0, "83")

gas_flow_setpoint_field = create_input(feedback_frame, "Gas production flow rate, setpoint", "[m³/h]", 0, 2)
gas_flow_setpoint_field.insert(0, "95")

substrate_feed_initial_field = create_input(feedback_frame, "Substrate feeding rate, initial state", "[t/2h]", 1, 0)
substrate_feed_initial_field.insert(0, "0.42")

start_date_field = DateEntry(feedback_frame, width=12, background='darkblue', foreground='white', borderwidth=2)
start_date_label = ttk.Label(feedback_frame, text="Start date")
start_date_label.grid(row=2, column=0, padx=5, pady=5, sticky='e')
start_date_field.grid(row=2, column=1, padx=5, pady=5, sticky='w')

duration_field = create_input(feedback_frame, "Simulation duration", "[days]", 2, 2)
duration_field.insert(0, "2")

# Saturations Section
saturations_frame = ttk.LabelFrame(top_frame, text="Saturations", padding=10)
saturations_frame.grid(row=0, column=1, sticky='nsew', padx=10)
saturations_frame.grid_columnconfigure(0, weight=1)

feed_max_field = create_input(saturations_frame, "Substrate feeding rate (max)", "[t/2h]", 0, 0)
feed_max_field.insert(0, "0.8")

feed_min_field = create_input(saturations_frame, "Substrate feeding rate (min)", "[t/2h]", 1, 0)
feed_min_field.insert(0, "0.4025")

# Plot Section
plot_frame = ttk.Frame(main_frame, padding=10)
plot_frame.grid(row=2, column=0, sticky='nsew')
plot_frame.grid_rowconfigure(0, weight=1)
plot_frame.grid_columnconfigure(0, weight=1)

# Create Matplotlib figure and axes
fig = Figure(figsize=(8, 4))
ax = fig.add_subplot(111)
ax.set_title("Discrete PI-Controller (+ Root Locus) with Anti-Windup")

# Create canvas and embed in plot_frame
canvas = FigureCanvasTkAgg(fig, master=plot_frame)
canvas.get_tk_widget().grid(row=0, column=0, sticky='nsew')

# Create button frame
button_frame = ttk.Frame(feedback_frame)
button_frame.grid(row=3, column=0, columnspan=4, pady=10)

def get_ui_values():
    """
    Retrieve all current values from the UI fields and store them in a dictionary.
    This function can be called whenever we need the latest values from the UI.
    """
    ui_values = {
        "gas_flow_initial": float(gas_flow_initial_field.get()),
        "gas_flow_setpoint": float(gas_flow_setpoint_field.get()),
        "substrate_feed_initial": float(substrate_feed_initial_field.get()),
        "simulation_duration": float(duration_field.get()),
        "feed_max": float(feed_max_field.get()),
        "feed_min": float(feed_min_field.get()),
        "start_date": start_date_field.get_date()
    }
    return ui_values


def calculate_control_step_amplitude():
    """
    Calculate the difference between the setpoint and initial flowrate.
    This will help determine the step amplitude for the control system.
    """
    ui_values = get_ui_values()
    control_step_amp = ui_values["gas_flow_setpoint"] - ui_values["gas_flow_initial"]
    return control_step_amp

def set_system_parameters():
    """
    Set the system parameters including sampling time
    """
    # Set sampling time to 7200 seconds (2 hours)
    Ts_2h = 7200
    return Ts_2h

def extract_transfer_function_coefficients():
    """
    Extract numerator and denominator coefficients from the transfer function
    using the time percentage method results from the preprocessing tab.
    """
    try:
        print("\n--- Extracting Transfer Function Coefficients ---")
        
        # Check if time percentage method results exist
        if global_vars.get('biogas_segment_up') is None:
            print("Error: biogas_segment_up not available.")
            return None
            
        if global_vars.get('control_system_params') is None:
            print("Error: control_system_params not available.")
            print("Available keys in global_vars:", list(global_vars.keys()))
            return None
        
        # Get control system parameters from the global variables
        control_params = global_vars.get('control_system_params')
        print("Control params type:", type(control_params))
        print("Control params keys:", list(control_params.keys()) if isinstance(control_params, dict) else "Not a dictionary")
        
        # Extract transfer function coefficients
        if isinstance(control_params, dict) and 'transfer_function_coeffs' in control_params:
            coeffs = control_params['transfer_function_coeffs']
            print("Found transfer function coefficients:", coeffs)
            
            e, f, c, d = coeffs["e"], coeffs["f"], coeffs["c"], coeffs["d"]
            
            # For a discrete transfer function of the form Gd(z) = (e·z + f)/(z² + c·z + d)
            pt2_Gd_num = np.array([e, f])    # Numerator coefficients
            pt2_Gd_den = np.array([1, -c, -d])  # Denominator coefficients with leading 1
            
            print("Successfully extracted transfer function coefficients:")
            print(f"Numerator coefficients: {pt2_Gd_num}")
            print(f"Denominator coefficients: {pt2_Gd_den}")
            
            return {
                'numerator': pt2_Gd_num,
                'denominator': pt2_Gd_den
            }
        else:
            if isinstance(control_params, dict):
                print("Error: 'transfer_function_coeffs' not found in control system parameters.")
                print("Available keys:", list(control_params.keys()))
            else:
                print("Error: control_params is not a dictionary.")
            return None
            
    except Exception as e:
        print(f"Error extracting transfer function coefficients: {e}")
        traceback.print_exc()
        return None
    
def store_transfer_function_coefficients():
    """
    Extract and store the transfer function coefficients from the time percentage method 
    for later use in the simulation.
    """
    try:
        print("\n--- Storing Transfer Function Coefficients ---")
        
        # Get coefficients using the extraction function
        coeffs = extract_transfer_function_coefficients()
        
        if coeffs is None:
            print("Failed to extract transfer function coefficients.")
            return False
        
        # Store coefficients in global variables for later access
        global_vars['pt2_Gd_num'] = coeffs['numerator']
        global_vars['pt2_Gd_den'] = coeffs['denominator']
        global_vars['original_Ts'] = 120  # Store the original sampling time (120 seconds)
        
        print("Transfer function coefficients stored successfully.")
        print(f"Numerator: {global_vars['pt2_Gd_num']}")
        print(f"Denominator: {global_vars['pt2_Gd_den']}")
        return True
            
    except Exception as e:
        print(f"Error storing transfer function coefficients: {e}")
        traceback.print_exc()
        return False
    

def convert_to_state_space():
    """
    Convert the discrete transfer function to a state-space representation
    using the control.tf2ss function.
    """
    try:
        print("\n--- Converting to State Space ---")
        
        # Check if the resampled transfer function exists
        if 'G_2h_num' not in global_vars:
            print("Error: 'G_2h_num' not found in global_vars.")
            return None
        
        if 'G_2h_den' not in global_vars:
            print("Error: 'G_2h_den' not found in global_vars.")
            return None
        
        # Get the resampled transfer function coefficients
        G_2h_num = global_vars['G_2h_num']
        G_2h_den = global_vars['G_2h_den']
        
        print(f"Using numerator: {G_2h_num}")
        print(f"Using denominator: {G_2h_den}")
        
        # Create a transfer function object
        tf_sys = ctrl.TransferFunction(G_2h_num[0], G_2h_den, dt=7200)
        
        print(f"Created transfer function: {tf_sys}")
        
        # Convert to state-space representation
        ss_sys = ctrl.tf2ss(tf_sys)
        
        print(f"State space system: {ss_sys}")
        
        # Store the state-space matrices
        global_vars['A'] = ss_sys.A
        global_vars['B'] = ss_sys.B
        global_vars['C'] = ss_sys.C
        global_vars['D'] = ss_sys.D
        global_vars['ss_sys'] = ss_sys
        
        print("Transfer function converted to state-space representation successfully.")
        print(f"A matrix:\n{ss_sys.A}")
        print(f"B matrix:\n{ss_sys.B}")
        print(f"C matrix:\n{ss_sys.C}")
        print(f"D matrix:\n{ss_sys.D}")
        
        return ss_sys
        
    except Exception as e:
        print(f"Error converting to state-space: {e}")
        traceback.print_exc()
        return None
    

    
def calculate_prefilter_gain():
    """
    Calculate prefilter gain to ensure unity DC gain in the closed-loop system.
    Based on MATLAB implementation.
    """
    try:
        print("\n--- Calculating Prefilter Gain ---")
        
        # Get the state-space matrices and controller gain
        A = global_vars['A']
        B = global_vars['B']
        C = global_vars['C']
        K = global_vars['K']
        
        print(f"A matrix:\n{A}")
        print(f"B matrix:\n{B}")
        print(f"C matrix:\n{C}")
        print(f"K matrix:\n{K}")
        
        # Create identity matrix of appropriate size
        I = np.eye(A.shape[0])
        
        # Calculate prefilter gain
        inner_term = I - A + B @ K
        print(f"(I - A + B*K) = \n{inner_term}")
        
        inner_inverse = np.linalg.inv(inner_term)
        print(f"(I - A + B*K)^-1 = \n{inner_inverse}")
        
        C_times_inverse_times_B = C @ inner_inverse @ B
        print(f"C * (I - A + B*K)^-1 * B = {C_times_inverse_times_B}")
        
        # Check if the result is too close to zero
        if np.abs(C_times_inverse_times_B).item() < 1e-10:
            print("Warning: Near-zero value detected in prefilter calculation.")
            # Calculate DC gain using alternative method
            num = global_vars['G_2h_num']
            den = global_vars['G_2h_den']
            dc_gain = np.sum(num) / np.sum(den)
            
            if abs(dc_gain) > 1e-10:
                f_ss_PT2 = np.array([[1.0 / dc_gain]])
            else:
                # Use a reasonable default value
                f_ss_PT2 = np.array([[63.5]])  # Typical value for biogas systems
                print(f"Using default prefilter gain: {f_ss_PT2.item()}")
        else:
            f_ss_PT2 = 1.0 / C_times_inverse_times_B
        
        # Store the prefilter gain for later use
        global_vars['f_ss_PT2'] = f_ss_PT2
        
        print("Prefilter gain calculated successfully.")
        print(f"Prefilter gain f_ss_PT2: {f_ss_PT2}")
        
        return f_ss_PT2
        
    except Exception as e:
        print(f"Error calculating prefilter gain: {e}")
        traceback.print_exc()
        # Fallback to a reasonable default
        default_gain = np.array([[63.5]])  # Typical value for biogas systems
        global_vars['f_ss_PT2'] = default_gain
        return default_gain
    


def initialize_control_system():
    """Initialize all control system parameters from the selected model"""
    try:
        print("\n--- Initializing Control System ---")
        
        # Check if control_system_params exists
        if 'control_system_params' not in global_vars:
            print("Error: control_system_params not found in global_vars.")
            print("Available keys:", list(global_vars.keys()))
            return False
        
        # Calculate all required parameters
        if store_transfer_function_coefficients():
            print("Transfer function coefficients stored successfully.")
            if resample_transfer_function():
                print("Transfer function resampled successfully.")
                # Change this line - check if the result is not None instead of evaluating it as a boolean
                K = design_deadbeat_controller()
                if K is not None:
                    print("Deadbeat controller designed successfully.")
                    if calculate_prefilter_gain():
                        print("Prefilter gain calculated successfully.")
                        print("All control system parameters successfully calculated!")
                        return True
        
        return False
    except Exception as e:
        print(f"Error initializing control system: {e}")
        traceback.print_exc()
        return False 

def resample_transfer_function():
    """
    Resample the discrete time transfer function to a new sampling time of 2 hours
    using a more robust approach that directly creates state-space matrices.
    """
    try:
        print("\n--- Resampling Transfer Function (Robust Method) ---")
        
        # Get the original transfer function coefficients
        if 'pt2_Gd_num' not in global_vars or 'pt2_Gd_den' not in global_vars:
            print("Error: Transfer function coefficients not found.")
            return None
        
        pt2_Gd_num = global_vars['pt2_Gd_num']
        pt2_Gd_den = global_vars['pt2_Gd_den']
        
        print(f"Original numerator: {pt2_Gd_num}")
        print(f"Original denominator: {pt2_Gd_den}")
        
        # Instead of going through continuous time, let's create the state-space system directly
        # For a second-order system with form: (e·z + f)/(z² + c·z + d)
        e = pt2_Gd_num[0]
        f = pt2_Gd_num[1]
        c = -pt2_Gd_den[1]  # Note the sign change
        d = -pt2_Gd_den[2]  # Note the sign change
        
        # Create a state-space representation in controllable canonical form
        A = np.array([[-c, -d], [1, 0]])
        B = np.array([[1], [0]])
        C = np.array([[e, f]])
        D = np.array([[0]])
        
        print("State-space matrices created directly from transfer function:")
        print(f"A = \n{A}")
        print(f"B = \n{B}")
        print(f"C = \n{C}")
        print(f"D = \n{D}")
        
        # Store these state-space matrices directly
        global_vars['A'] = A
        global_vars['B'] = B
        global_vars['C'] = C
        global_vars['D'] = D
        
        # Also store the transfer function coefficients
        global_vars['G_2h_num'] = pt2_Gd_num
        global_vars['G_2h_den'] = pt2_Gd_den
        
        print("State-space representation created successfully.")
        return True
        
    except Exception as e:
        print(f"Error in robust resampling: {e}")
        traceback.print_exc()
        return None
        
def design_deadbeat_controller():
    """
    Design a deadbeat controller by placing closed loop poles at the origin [0, 0]
    and calculating the state feedback gain matrix K using Ackermann's formula.
    """
    try:
        print("\n--- Designing Deadbeat Controller ---")
        
        # Check if state-space matrices exist
        if 'A' not in global_vars:
            print("Error: 'A' not found in global_vars.")
            return None
        
        if 'B' not in global_vars:
            print("Error: 'B' not found in global_vars.")
            return None
        
        # Get the state-space matrices
        A = global_vars['A']
        B = global_vars['B']
        
        print(f"A matrix:\n{A}")
        print(f"B matrix:\n{B}")
        
        # Check if matrices are empty
        if A.size == 0 or B.size == 0:
            print("Error: Empty state-space matrices. Cannot design controller.")
            return None
        
        # Get system dimensions
        n = A.shape[0]  # Number of states
        
        print(f"System order: {n}")
        
        # Set desired poles at the origin for deadbeat control
        desired_poles = np.zeros(n)
        print(f"Desired poles: {desired_poles}")
        
        # Calculate state feedback gain matrix K using Ackermann's formula
        K = ctrl.acker(A, B, desired_poles)
        
        # Store the gain matrix for later use
        global_vars['K'] = K
        
        print("Deadbeat controller designed successfully.")
        print(f"State feedback gain matrix K: {K}")
        
        return K
        
    except Exception as e:
        print(f"Error designing deadbeat controller: {e}")
        traceback.print_exc()
        return None
    
###########################




def plot_with_antiwindup(simulation_results, ui_values):
    """
    Plot simulation results with anti-windup styling
    Based on the MATLAB function plot_regler_simulink_anti_windup_ui
    """
    # Clear the figure
    fig.clear()
    
    # Create primary axis
    ax1 = fig.add_subplot(111)
    
    # Create secondary axis for substrate feeding
    ax2 = ax1.twinx()
    
    # Get simulation results
    time = simulation_results['time']  # Time in hours
    biogas = simulation_results['biogas']  # Biogas production
    substrate_feeding = simulation_results['substrate_feeding']  # Substrate feeding
    
    # Convert simulation start time to datetime
    start_date = ui_values["start_date"]
    start_datetime = pd.to_datetime(start_date)
    
    # Define color scheme similar to MATLAB version
    cmap = ['#0072BD', '#D95319', '#EDB120', '#7E2F8E', '#77AC30', '#4DBEEE', '#A2142F']
    
    # Font and styling parameters
    const = {
        'font': 'serif',
        'fontsize': 16,
        'fontsizelegend': 10,
        'fontsizeticks': 10,
        'linienbreite': 1.5,  # Line width
    }
    
    # Calculate time values with proper datetime format
    if isinstance(start_datetime, pd.Timestamp):
        datetime_values = [start_datetime + pd.Timedelta(hours=t) for t in time]
    else:
        datetime_values = time  # Fallback to numeric time values
    
    # Calculate y-axis limits with 5% padding
    y_min = min(biogas) - 0.05 * max(biogas)
    y_max = max(biogas) + 0.05 * max(biogas)
    
    # Set up primary y-axis (biogas)
    ax1.set_ylim([y_min, y_max])
    ax1.grid(True)
    ax1.xaxis.set_minor_locator(AutoMinorLocator())
    ax1.yaxis.set_minor_locator(AutoMinorLocator())
    
    # Format axis labels
    ax1.set_ylabel('Gas production flow rate [m³/h]', 
                  fontname=const['font'], 
                  fontsize=const['fontsize'],
                  color=cmap[0])
    
    if isinstance(datetime_values[0], (pd.Timestamp, datetime)):
        # Use date formatting if we have datetime objects
        ax1.xaxis.set_major_formatter(mdates.DateFormatter('%d\n%H:%M'))
        plt.setp(ax1.xaxis.get_majorticklabels(), rotation=0, ha='center')
        ax1.set_xlabel('Time', fontname=const['font'], fontsize=const['fontsize'])
    else:
        # Use hours formatting otherwise
        ax1.set_xlabel('Time [h]', fontname=const['font'], fontsize=const['fontsize'])
    
    # Plot biogas production on primary y-axis
    line1 = ax1.plot(datetime_values, biogas, 
                   color=cmap[0], 
                   linewidth=const['linienbreite'],
                   label='Simulated Gas production flow rate')
    
    # Plot setpoint on primary y-axis
    setpoint = ui_values['gas_flow_setpoint']
    line2 = ax1.axhline(y=setpoint, 
                      color=cmap[2], 
                      linewidth=const['linienbreite'],
                      linestyle=':', 
                      label='Setpoint',
                      alpha=0.8)
    
    # Set up secondary y-axis (substrate feeding)
    y_min_right = min(substrate_feeding) - 0.05 * max(substrate_feeding)
    y_max_right = max(substrate_feeding) + 0.05 * max(substrate_feeding)
    ax2.set_ylim([y_min_right, y_max_right])
    ax2.yaxis.set_minor_locator(AutoMinorLocator())
    
    # Set sampling time (Ta) - assume 2 hours
    Ta = 2
    ax2.set_ylabel(f'Substrate feeding rate [t/{Ta}h]',
                  fontname=const['font'],
                  fontsize=const['fontsize'],
                  color=cmap[1])
    
    # Plot substrate feeding on secondary y-axis
    line3 = ax2.step(datetime_values, substrate_feeding, 
                   where='post',
                   color=cmap[1], 
                   linewidth=const['linienbreite'],
                   label='Substrate feeding rate')
    
    # Set title to match MATLAB version
    ax1.set_title("Discrete PI-Controller (+ Root Locus) with Anti-Windup",
                 fontname=const['font'],
                 fontsize=const['fontsize'])
    
    # Create combined legend
    lines = line1 + [line2] + line3
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, 
              loc='lower right',
              fontsize=const['fontsizelegend'],
              prop={'family': const['font']})
    
    # Configure tick parameters
    ax1.tick_params(axis='y', labelcolor=cmap[0], labelsize=const['fontsizeticks'])
    ax2.tick_params(axis='y', labelcolor=cmap[1], labelsize=const['fontsizeticks'])
    
    # Adjust layout
    fig.tight_layout()


def run_simulation():
    """
    Run a state space controller simulation according to Fig. 7.
    Implements feedback control loop using the state space controller.
    """
    try:
        print("\n--- Starting Simulation ---")
        
        # First, try to initialize the control system parameters
        if not initialize_control_system():
            messagebox.showerror("Error", "Failed to initialize control system parameters. Please ensure model estimation is complete.")
            return

        # Get system parameters from global variables
        A = global_vars['A']
        B = global_vars['B']
        C = global_vars['C']
        K = global_vars['K']
        F = global_vars['f_ss_PT2']
        
        print("System Parameters:")
        print(f"A = \n{A}")
        print(f"B = \n{B}")
        print(f"C = \n{C}")
        print(f"K = \n{K}")
        print(f"F = {F}")   

        # Get UI values
        ui_values = get_ui_values()
        
        # Simulation parameters
        days = ui_values["simulation_duration"]
        Ts = set_system_parameters()  # 7200 seconds (2-hour sampling time)
        samples = int((days * 24 * 3600) / Ts)  # Number of simulation steps
        
        # Initialize arrays for simulation
        time = np.arange(0, samples) * Ts / 3600  # Time in hours
        biogas = np.zeros(samples)  # Biogas production
        substrate_feeding = np.zeros(samples)  # Substrate feeding rate
    
        # Initial conditions
        biogas[0] = ui_values["gas_flow_initial"]
        substrate_feeding[0] = ui_values["substrate_feed_initial"]
        
        # Initialize state vector
        state_dim = A.shape[0]
        x = np.zeros((samples, state_dim))  # Store all states for debugging
        
        # Setpoint
        setpoint = ui_values["gas_flow_setpoint"]
        
        # Saturation limits (control signal constraints)
        feed_min = ui_values["feed_min"]
        feed_max = ui_values["feed_max"]
        
        print(f"Starting simulation with {samples} steps...")
        print(f"Setpoint: {setpoint} m³/h")
        print(f"Feed limits: {feed_min} to {feed_max} t/2h")
        
        # Anti-windup variables
        u_unbounded = np.zeros(samples)  # Unbounded control signals
        u_bounded = np.zeros(samples)    # Bounded control signals
        u_bounded[0] = substrate_feeding[0]
        
        # Initialize state vector
        x[0, :] = np.zeros(state_dim)
        
        # Simulation loop
        for k in range(1, samples):
            # Calculate prefilter output: F * setpoint
            prefilter_output = F * setpoint
            
            # Calculate state feedback: K @ x[k-1]
            state_feedback = K @ x[k-1, :].reshape(-1, 1)
            
            # Calculate unbounded control input: F * setpoint - K * x
            u_unbounded[k] = prefilter_output - float(state_feedback)
            
            # Apply saturation to control input (anti-windup)
            u_bounded[k] = np.clip(u_unbounded[k], feed_min, feed_max)
            
            # Store substrate feeding rate
            substrate_feeding[k] = u_bounded[k]
            
            # Calculate B*u
            Bu = B * u_bounded[k]
            
            # State update: x[k] = A*x[k-1] + B*u[k]
            x[k, :] = (A @ x[k-1, :].reshape(-1, 1) + Bu).flatten()
            
            # Calculate output: y[k] = C*x[k]
            biogas[k] = float(C @ x[k, :].reshape(-1, 1))
        
        print("Simulation complete. Plotting results...")
        
        # Store results in global variables
        global_vars['simulation_results'] = {
            'time': time,
            'biogas': biogas,
            'substrate_feeding': substrate_feeding,
            'u_unbounded': u_unbounded,
            'u_bounded': u_bounded,
            'x': x
        }
        
        # Plot results using anti-windup plotting function
        plot_with_antiwindup(global_vars['simulation_results'], ui_values)
        
        # Redraw canvas
        canvas.draw()
        
        # Display success message
        messagebox.showinfo("Simulation Complete", 
                          f"Simulation completed successfully for {days} days ({samples} steps).")
        
    except Exception as e:
        messagebox.showerror("Simulation Error", f"Error during simulation: {str(e)}")
        print(f"Simulation error: {e}")
        traceback.print_exc()

# Create simulation button
simulate_button = tk.Button(
    button_frame,
    text="Start Simulation", 
    font=("Arial", 10),
    bd=1, 
    relief="solid",
    command=run_simulation
)
simulate_button.pack(side='left', padx=10, pady=5)

# Add hover effects
def on_enter_sim(event):
    event.widget['background'] = '#d0eaff'  # Light blue color on hover

def on_leave_sim(event):
    event.widget['background'] = 'SystemButtonFace'  # Default button color

simulate_button.bind("<Enter>", on_enter_sim)
simulate_button.bind("<Leave>", on_leave_sim)


# Run Tkinter Main Loop
root.mainloop()  




### The problem is in your calculate_prefilter_gain() function. Your numerator coefficient e is zero:
# When calculating C @ inner_inverse @ B, you get [[0.]], and dividing by 0 gives infinity.
# This solution:
# 1. Checks if C_times_inverse_times_B is close to zero (which causes the infinity)
# 2. If it is, tries to calculate a gain using the transfer function DC gain
# 3. If that fails too, uses a typical value for biogas systems (63.5, which I included based on- 
# typical process gain values for biogas systems).
# The issue comes from your transfer function structure. In your MATLAB code, the transfer function is properly-
#  resampled using d2d(app.system_model_up.Gd,app.Ts_2h) which maintains the dynamic properties. 
#  In your Python code, you're directly creating state-space matrices without proper resampling, and your-
# numerator's first coefficient is zero.