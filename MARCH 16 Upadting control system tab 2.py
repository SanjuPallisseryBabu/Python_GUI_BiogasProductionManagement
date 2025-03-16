
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
from scipy.interpolate import interp1d
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
        # Time constants table - matches the table in the paper
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
        
        # Determine initial and final values based on step direction
        initial_value = flowrate[ind_sprung]
        if step_direction == "down":
            final_value = flowrate_min
        else:
            final_value = flowrate_max
            
        # Calculate limit value (y∞)
        y_inf = final_value
        
        # Calculate the target value for T72% (based on Algorithm 2, line 2)
        # For T72%, we need to find when the response reaches 72% of the final value
        if step_direction == "down":
            # For step down, we're looking for when the value has decreased to 72% of the total change
            target_value = initial_value - 0.72 * (initial_value - final_value)
        else:
            # For step up, we're looking for when the value has increased to 72% of the total change
            target_value = initial_value + 0.72 * (final_value - initial_value)

        # Find the index where the response is closest to the 72% target
        t1_idx = np.argmin(np.abs(flowrate - target_value))
        
        # Convert to time in seconds (T1)
        timestamp_numeric = mdates.date2num(timestamps)
        T1 = (timestamp_numeric[t1_idx] - timestamp_numeric[ind_sprung]) * 24 * 3600  # seconds
        
        # Calculate T2 (based on Algorithm 2, line 6)
        T2 = 0.2847 * T1
        
        # Find the index closest to T2 seconds after the step
        t2_time = timestamp_numeric[ind_sprung] + (T2 / (24 * 3600))
        t2_idx = np.argmin(np.abs(timestamp_numeric - t2_time))
        
        # Get the y2 value at time T2
        y2 = flowrate[t2_idx]
        
        # Calculate the ratio y2/y∞ (to match the table lookup in the paper)
        if step_direction == "down":
            # For step down, we need to normalize the ratio correctly
            y2_ratio = (y2 - final_value) / (initial_value - final_value)
        else:
            # For step up
            y2_ratio = (y2 - initial_value) / (final_value - initial_value)
        
        # Find closest match in table_zk (Column 0 contains y2/y∞ values)
        idx = np.argmin(np.abs(table_zk[:, 0] - y2_ratio))
        
        # Get time constants from the table (TA/T1 and TB/T1)
        TA_ratio = table_zk[idx, 2]  # Column 2 is TA/T1
        TB_ratio = table_zk[idx, 3]  # Column 3 is TB/T1
        
        # Calculate actual time constants
        TA = TA_ratio * T1
        TB = TB_ratio * T1
        
        print(f"Time constants calculation:")
        print(f"T1 (T72%): {T1:.2f} seconds")
        print(f"T2: {T2:.2f} seconds")
        print(f"y2/y∞ ratio: {y2_ratio:.6f}")
        print(f"TA/T1: {TA_ratio:.6f}")
        print(f"TB/T1: {TB_ratio:.6f}")
        print(f"TA: {TA:.6f} seconds")
        print(f"TB: {TB:.6f} seconds")

        # Create transfer functions as in the paper: G(s) = K/((TA·s+1)·(TB·s+1))
        G1_num, G1_den = [1], [TA, 1]
        G2_num, G2_den = [1], [TB, 1]
        G_num = [K]
        G_den = [1]

        # Multiply transfer functions
        combined_num, combined_den = signal.convolve(G_num, G1_num), signal.convolve(G_den, G1_den)
        combined_num, combined_den = signal.convolve(combined_num, G2_num), signal.convolve(combined_den, G2_den)

        # Discretize transfer function (as in Algorithm 2, line 10)
        Ta = 120  # Sampling time in seconds
        Gd = signal.cont2discrete((combined_num, combined_den), Ta, method="zoh")

        # Extract transfer function coefficients for discrete form
        num = Gd[0][0]  # Get numerator coefficients
        den = Gd[1]     # Get denominator coefficients
        
        # Convert to the form Gd(z) = (e·z + f)/(z² + c·z + d) as shown in Algorithm 2, line 10
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
                "y2_ratio": y2_ratio,
                "y2": y2,
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
    Modified to handle substrate feeding rate separately and show T1 and T2 markers.
    Fixed to handle numpy int64 indices correctly.
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
        color_marker = '#AAAAAA'  # Gray for T1 and T2 markers
        
        fig_model_down.clear()
        ax1 = fig_model_down.add_subplot(111)
        ax2 = ax1.twinx()
        
        # Get preprocessed data
        biogas_data = global_vars['biogas_segment_down']['SmoothedValueNum']
        substrate_data = global_vars['substrate_segment_down']['FeedingRate']
        timestamps = global_vars['timestamps_down']
        
        # Convert timestamps to a list to ensure integer indexing works
        # This allows us to use integer indices (Python int) with timestamps
        timestamps_list = timestamps.tolist() if hasattr(timestamps, 'tolist') else list(timestamps)
        
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

        print("Print flowrate_max", flowrate_max)
        print("Print flowrate_min", flowrate_min)
        
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
        
        # Get T1 and T2 values from zk_down
        if zk_down is not None and 'parameters' in zk_down:
            # Get the step time (start time)
            timestamp_numeric = mdates.date2num(timestamps)
            step_time = timestamp_numeric[ind_sprung]
            
            # Calculate time points for T1 and T2
            T1 = zk_down['parameters']['T1']  # T1 in seconds
            T2 = zk_down['parameters']['T2']  # T2 in seconds
            
            # Convert to datetime indices
            t1_time = step_time + (T1 / (24 * 3600))  # Convert seconds to days
            t2_time = step_time + (T2 / (24 * 3600))  # Convert seconds to days
            
            # Find closest indices and convert to Python int
            t1_idx = int(np.argmin(np.abs(timestamp_numeric - t1_time)))
            t2_idx = int(np.argmin(np.abs(timestamp_numeric - t2_time)))
            
            # Ensure indices are within valid range
            t1_idx = min(t1_idx, len(timestamps) - 1)
            t2_idx = min(t2_idx, len(timestamps) - 1)
            
            print(f"T1 index: {t1_idx}, max index: {len(timestamps)-1}")
            print(f"T2 index: {t2_idx}, max index: {len(timestamps)-1}")
            
            # Get y-values at T1 and T2
            y_t1 = biogas_data.iloc[t1_idx] if hasattr(biogas_data, 'iloc') else biogas_data[t1_idx]
            y_t2 = biogas_data.iloc[t2_idx] if hasattr(biogas_data, 'iloc') else biogas_data[t2_idx]
            
            # Get y range for plotting vertical lines
            y_min = flowrate_min - biogas_margin
            y_max = flowrate_max + biogas_margin
            
            # Plot T1 marker (72% mark)
            # Vertical line - use timestamps_list with Python int indices
            ax1.plot([timestamps_list[t1_idx], timestamps_list[t1_idx]], [y_min, y_t1], 
                   linestyle=':', color=color_marker, linewidth=1.0)
            # Horizontal line
            ax1.plot([timestamps_list[0], timestamps_list[t1_idx]], [y_t1, y_t1], 
                   linestyle=':', color=color_marker, linewidth=1.0)
            # Add T1 annotation
            ax1.text(timestamps_list[min(t1_idx+5, len(timestamps_list)-1)], y_min + biogas_margin, 
                   f"T1 (0.72)", fontsize=8, color=color_marker,
                   horizontalalignment='right', verticalalignment='bottom')
            
            # Plot T2 marker
            # Vertical line
            ax1.plot([timestamps_list[t2_idx], timestamps_list[t2_idx]], [y_min, y_t2], 
                   linestyle=':', color=color_marker, linewidth=1.0)
            # Horizontal line
            ax1.plot([timestamps_list[0], timestamps_list[t2_idx]], [y_t2, y_t2], 
                   linestyle=':', color=color_marker, linewidth=1.0)
            # Add T2 annotation with the y2_ratio value from the calculation
            ax1.text(timestamps_list[min(t2_idx+5, len(timestamps_list)-1)], y_min + biogas_margin, 
                   f"T2 ({zk_down['parameters'].get('y2_ratio', 0):.3f})", 
                   fontsize=8, color=color_marker,
                   horizontalalignment='right', verticalalignment='bottom')
            
            # Mark the actual points with circles
            ax1.plot(timestamps_list[t1_idx], y_t1, 'o', 
                   markersize=4, markerfacecolor=color_marker, markeredgecolor='none')
            ax1.plot(timestamps_list[t2_idx], y_t2, 'o', 
                   markersize=4, markerfacecolor=color_marker, markeredgecolor='none')
            
        # Update metrics table
        update_metrics_table(tree_down, zk_down['metrics'])

        # Store data for later use if needed
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
        import traceback
        traceback.print_exc()  # Print detailed error information

def update_model_up_plot():
    """
    Update the step up plot with time percentage method.
    Modified to handle substrate feeding rate separately and display T1 and T2 markers.
    Fixed to handle numpy int64 indices correctly.
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
            'grid': '#E6E6E6',             # Light gray for grid
            'markers': '#AAAAAA'           # Gray for T1 and T2 markers
        }
            
        # Clear and set up the figure
        fig_model_up.clear()
        ax1 = fig_model_up.add_subplot(111)
        ax2 = ax1.twinx()
        
        # Get preprocessed data
        biogas_data = global_vars['biogas_segment_up']['SmoothedValueNum']
        substrate_data = global_vars['substrate_segment_up']['FeedingRate']
        timestamps = global_vars['timestamps_up']
        
        # Convert timestamps to a list to ensure integer indexing works
        timestamps_list = timestamps.tolist() if hasattr(timestamps, 'tolist') else list(timestamps)

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

        # Calculate margins for y-axis limits
        biogas_margin = 0.05 * (flowrate_max - flowrate_min)
        substrate_margin = 0.05 * (feed_max - feed_min)
        
        # Set y-axis limits with margins
        ax1.set_ylim([flowrate_min - biogas_margin, flowrate_max + biogas_margin])
        ax2.set_ylim([feed_min - substrate_margin, feed_max + substrate_margin])
        
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
        
        if zk_up is not None:
            # Setup axes properties
            ax1.grid(True, color=colors['grid'], linestyle='-', alpha=0.2)
            ax1.xaxis.set_minor_locator(AutoMinorLocator())
            ax1.yaxis.set_minor_locator(AutoMinorLocator())
            
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

            # Configure primary axis
            ax1.tick_params(axis='y', labelcolor=colors['biogas_line'], labelsize=const['fontsizeticks'])
            ax1.set_xlabel('Time', fontname=const['font'], fontsize=const['fontsize'])
            ax1.set_ylabel('Biogas production rate [m³/h]', 
                          fontname=const['font'], 
                          fontsize=const['fontsize'],
                          color=colors['biogas_line'])
            ax1.set_title('Time percentage method', 
                         fontname=const['font'], 
                         fontsize=const['fontsize'])

            # Configure secondary axis
            ax2.spines['right'].set_color(colors['substrate'])
            ax2.tick_params(axis='y', colors=colors['substrate'], labelsize=const['fontsizeticks'])
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
            
            # Add T1 and T2 markers
            if 'parameters' in zk_up:
                # Get the step time (start time)
                timestamp_numeric = mdates.date2num(timestamps)
                step_time = timestamp_numeric[ind_sprung]
                
                # Calculate time points for T1 and T2
                T1 = zk_up['parameters']['T1']  # T1 in seconds
                T2 = zk_up['parameters']['T2']  # T2 in seconds
                
                # Convert to datetime indices
                t1_time = step_time + (T1 / (24 * 3600))  # Convert seconds to days
                t2_time = step_time + (T2 / (24 * 3600))  # Convert seconds to days
                
                # Find closest indices and convert to Python int
                t1_idx = int(np.argmin(np.abs(timestamp_numeric - t1_time)))
                t2_idx = int(np.argmin(np.abs(timestamp_numeric - t2_time)))
                
                # Ensure indices are within valid range
                t1_idx = min(t1_idx, len(timestamps) - 1)
                t2_idx = min(t2_idx, len(timestamps) - 1)
                
                print(f"T1 index: {t1_idx}, max index: {len(timestamps)-1}")
                print(f"T2 index: {t2_idx}, max index: {len(timestamps)-1}")
                
                # Get y-values at T1 and T2
                y_t1 = biogas_data.iloc[t1_idx] if hasattr(biogas_data, 'iloc') else biogas_data[t1_idx]
                y_t2 = biogas_data.iloc[t2_idx] if hasattr(biogas_data, 'iloc') else biogas_data[t2_idx]
                
                # Get y range for plotting vertical lines
                y_min = flowrate_min - biogas_margin
                y_max = flowrate_max + biogas_margin
                
                # Plot T1 marker (72% mark)
                # Vertical line - use timestamps_list with Python int indices
                ax1.plot([timestamps_list[t1_idx], timestamps_list[t1_idx]], [y_min, y_t1], 
                       linestyle=':', color=colors['markers'], linewidth=1.0)
                # Horizontal line
                ax1.plot([timestamps_list[0], timestamps_list[t1_idx]], [y_t1, y_t1], 
                       linestyle=':', color=colors['markers'], linewidth=1.0)
                # Add T1 annotation
                ax1.text(timestamps_list[min(t1_idx+5, len(timestamps_list)-1)], y_min + biogas_margin, 
                       f"T1 (0.72)", fontsize=8, color=colors['markers'],
                       horizontalalignment='right', verticalalignment='bottom')
                
                # Plot T2 marker
                # Vertical line
                ax1.plot([timestamps_list[t2_idx], timestamps_list[t2_idx]], [y_min, y_t2], 
                       linestyle=':', color=colors['markers'], linewidth=1.0)
                # Horizontal line
                ax1.plot([timestamps_list[0], timestamps_list[t2_idx]], [y_t2, y_t2], 
                       linestyle=':', color=colors['markers'], linewidth=1.0)
                # Add T2 annotation with the y2_ratio value
                ax1.text(timestamps_list[min(t2_idx+5, len(timestamps_list)-1)], y_min + biogas_margin, 
                       f"T2 ({zk_up['parameters'].get('y2_ratio', 0):.3f})", 
                       fontsize=8, color=colors['markers'],
                       horizontalalignment='right', verticalalignment='bottom')
                
                # Mark the actual points with circles
                ax1.plot(timestamps_list[t1_idx], y_t1, 'o', 
                       markersize=4, markerfacecolor=colors['markers'], markeredgecolor='none')
                ax1.plot(timestamps_list[t2_idx], y_t2, 'o', 
                       markersize=4, markerfacecolor=colors['markers'], markeredgecolor='none')

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
                "flowrate_min": flowrate_min,
                # Additional parameters
                "y2_ratio": zk_up['parameters'].get('y2_ratio', 0)  # Add y2_ratio for reference
            }
            print("Up Scenario Data Stored:", global_vars["up_scenario_data"])
            
            # Adjust layout and draw
            fig_model_up.tight_layout()
            canvas_model_up.draw()
            
    except Exception as e:
        print(f"Error in update_model_up_plot: {e}")
        import traceback
        traceback.print_exc()  # Print detailed error information

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
    Calculate the time constant T63 where the response reaches 63% of its final value
    """
    try:
        # Calculate the total change in value
        total_change = final_value - initial_value
        
        # Calculate the target value (63% of the change)
        target_value = initial_value + 0.63 * total_change
        
        print(f"Initial value: {initial_value}")
        print(f"Final value: {final_value}")
        print(f"Target value: {target_value}")
        print(f"Biogas data sample: {biogas_data[:10]}")
        
        # Find the index where the biogas data first crosses the target value
        if final_value > initial_value:  # Upward step
            # Find first index where biogas value is >= target value
            mask = biogas_data >= target_value
            if not any(mask):
                return None, None
            idx_63 = np.where(mask)[0][0]
        else:  # Downward step
            # Find first index where biogas value is <= target value
            mask = biogas_data <= target_value
            if not any(mask):
                return None, None
            idx_63 = np.where(mask)[0][0]
        
        # Get timestamp at this index
        t63_timestamp = timestamps[idx_63]
        t0_timestamp = timestamps[0]  # Assuming the step starts at the first timestamp
        
        # Calculate time difference in seconds
        if isinstance(t63_timestamp, pd.Timestamp) and isinstance(t0_timestamp, pd.Timestamp):
            T63 = (t63_timestamp - t0_timestamp).total_seconds()
        else:
            # If using numpy datetime64
            T63 = (t63_timestamp - t0_timestamp) / np.timedelta64(1, 's')
        
        print(f"Found index: {idx_63}")
        print(f"Calculated T63: {T63}")
        
        return T63, idx_63
    
    except Exception as e:
        print(f"Error in calculate_T63: {e}")
        return None, None


def pt1_model_modified(flowrate, feed, step_index, feed_max, feed_min, flowrate_max, flowrate_min, timestamps, step_dir):
    """
    Modified PT1 model implementation to work with pandas Series data
    """
    try:
        # Create dictionary to store PT1 model parameters
        pt1 = {}
        
        # Gain calculation
        pt1['K'] = (flowrate_max - flowrate_min) / (feed_max - feed_min)
        
        # Starting point of the step
        pt1['step_index'] = step_index
        
        # Calculate critical value (63% of response)
        pt1['critical_value'] = 0.63
        
        # Calculate the y-value at 63% (the value we're looking for in the response)
        if 'down' in step_dir:
            pt1['y_value_at_critical'] = flowrate_max - (flowrate_max - flowrate_min) * pt1['critical_value']
        else:
            pt1['y_value_at_critical'] = flowrate_min + (flowrate_max - flowrate_min) * pt1['critical_value']
                
        T63, idx_63 = calculate_T63(
            biogas_data=flowrate.values,
            timestamps=timestamps,
            initial_value=flowrate.values[step_index],
            final_value=flowrate_min if 'down' in step_dir else flowrate_max
        )
        
        if T63 is not None:
            pt1['T'] = T63  # Time constant
            pt1['index_at_critical'] = idx_63  # Store index for plotting
        else:
            # Fallback to original calculation if T63 calculation fails
            if 'down' in step_dir:
                try:
                    pt1['index_at_critical'] = np.where(flowrate.values < pt1['y_value_at_critical'])[0][0]
                except IndexError:
                    # If no value is below critical, use the last index
                    pt1['index_at_critical'] = len(flowrate) - 1
            else:
                try:
                    pt1['index_at_critical'] = np.where(flowrate.values > pt1['y_value_at_critical'])[0][0]
                except IndexError:
                    # If no value is above critical, use the last index
                    pt1['index_at_critical'] = len(flowrate) - 1
            
            # Calculate time constant T (in seconds)
            timestamp_numeric = mdates.date2num(timestamps)
            pt1['T'] = (timestamp_numeric[pt1['index_at_critical']] - timestamp_numeric[pt1['step_index']]) * 24 * 3600

        # Make sure the index is valid for timestamps array
        if pt1['index_at_critical'] >= len(timestamps):
            pt1['index_at_critical'] = len(timestamps) - 1
            
        # Store t63 timestamp for plotting
        if isinstance(timestamps, pd.DatetimeIndex) or isinstance(timestamps, pd.Series):
            try:
                pt1['t63_timestamp'] = timestamps.iloc[pt1['index_at_critical']]
            except (IndexError, KeyError):
                # If index is out of bounds, use the last timestamp
                pt1['t63_timestamp'] = timestamps.iloc[-1]
        else:
            try:
                pt1['t63_timestamp'] = timestamps[pt1['index_at_critical']]
            except (IndexError, KeyError):
                # If index is out of bounds, use the last timestamp
                pt1['t63_timestamp'] = timestamps[-1]
        
        # Rest of your existing code...
        # Continuous transfer function coefficients
        num = [pt1['K']]  # Numerator coefficients
        den = [pt1['T'], 1]  # Denominator coefficients
        
        # Continuous to discrete conversion
        pt1['Ta'] = 120  # Sampling time in seconds
        sys_discrete = signal.cont2discrete((num, den), pt1['Ta'], method='zoh')
        
        # Extract discrete transfer function coefficients
        num_d = sys_discrete[0].flatten()  # Numerator coefficients
        den_d = sys_discrete[1]  # Denominator coefficients
        
        # Create input signal
        if 'down' in step_dir:
            t_input = feed.values - feed_max
        else:
            t_input = feed.values - feed_min
            
        # Simulate the system
        t = np.arange(len(feed)) * pt1['Ta']
        output = signal.dlsim((num_d, den_d, pt1['Ta']), t_input, t)
        yout = output[1]  # Extract just the output values
        
        # Add offset back to output
        if 'down' in step_dir:
            pt1['yd'] = yout + flowrate_max
        else:
            pt1['yd'] = yout + flowrate_min
        
        return pt1
        
    except Exception as e:
        print(f"Error in pt1_model_modified: {e}")
        traceback.print_exc()  # This will print the full error trace
        return None


def update_model_down_plot_pt1():
    """Update the PT1 model down plot to match the slider-selected date range.
    Modified to handle numpy int64 indices, fix matplotlib legend warning,
    and use vertical T63 marker instead of full horizontal line."""
    try:
        # Check if preprocessed data is available
        if global_vars['biogas_segment_down'] is None or global_vars['substrate_segment_down'] is None:
            print("No preprocessed data available. Please preprocess data first.")
            return

        # Use the exact same indices from preprocessing tab
        start_ind = global_vars['slider_down_start_index']
        end_ind = global_vars['slider_down_end_index']
        
        # Get the biogas data, substrate data and timestamps using the same indices
        biogas_data = global_vars['biogas_segment_down']['SmoothedValueNum']
        substrate_data = global_vars['substrate_segment_down']['FeedingRate']
        timestamps = global_vars['timestamps_down']
        
        # Convert timestamps to a list for safer indexing
        timestamps_list = timestamps.tolist() if hasattr(timestamps, 'tolist') else list(timestamps)

        # Process substrate data for valid values
        valid_substrate_mask = substrate_data > 0
        valid_substrate = substrate_data[valid_substrate_mask]
        
        # Handle indexing on timestamps using numpy array conversion to avoid pandas index alignment issues
        if isinstance(valid_substrate_mask, pd.Series):
            # Convert to numpy array to avoid pandas index alignment issues
            valid_mask_array = valid_substrate_mask.values
        else:
            valid_mask_array = valid_substrate_mask
            
        # Get valid timestamps based on the data type
        if isinstance(timestamps, pd.DatetimeIndex):
            valid_timestamps_substrate = timestamps[valid_mask_array]
        elif isinstance(timestamps, pd.Series):
            # For pandas Series, use .iloc with numpy boolean array
            valid_timestamps_substrate = timestamps.iloc[valid_mask_array]
        else:
            # For list or numpy array of timestamps
            valid_indices = [i for i, is_valid in enumerate(valid_mask_array) if is_valid]
            valid_timestamps_substrate = [timestamps_list[i] for i in valid_indices]

        # Run the PT1 model calculation
        step_index = int(np.argmax(np.diff(substrate_data.values))) if len(substrate_data) > 1 else 0
        pt1_down = pt1_model_modified(
            flowrate=biogas_data,
            feed=substrate_data,
            step_index=step_index,
            feed_max=substrate_data.max(),
            feed_min=valid_substrate.min() if not valid_substrate.empty else 0,
            flowrate_max=biogas_data.max(),
            flowrate_min=biogas_data.min(),
            timestamps=timestamps,
            step_dir='down'
        )

        if pt1_down is not None:
            # Clear and set up the plot
            fig_model_down.clear()
            ax1 = fig_model_down.add_subplot(111)
            ax2 = ax1.twinx()

            # Calculate y-axis limits with margins
            y_min = biogas_data.min() - 0.05 * (biogas_data.max() - biogas_data.min())
            y_max = biogas_data.max() + 0.05 * (biogas_data.max() - biogas_data.min())
            
            # Set y-axis limits
            ax1.set_ylim([y_min, y_max])
            
            # Plot biogas production rate
            ax1.plot(
                timestamps, biogas_data, '--',
                color='#0000A7', label='Biogas production rate', linewidth=1
            )

            # Plot PT1 model output
            ax1.plot(
                timestamps, pt1_down['yd'], '-.',
                color='#EECC16', label='PT1 model', linewidth=1
            )

            # Plot substrate feeding rate
            ax2.plot(
                valid_timestamps_substrate, valid_substrate, '-',
                color='#C1272D', label='Substrate feeding rate', linewidth=1
            )
            
            # Variable to store T63 line for legend
            t63_line = None
            
            # Add the T63% marking
            if 'y_value_at_critical' in pt1_down and 't63_timestamp' in pt1_down:
                y_t63 = pt1_down['y_value_at_critical']
                t63_timestamp = pt1_down['t63_timestamp']
                
                # Add a short horizontal dotted line at T63% level from the left edge to the T63 point
                t63_line = ax1.plot(
                    [timestamps_list[0], t63_timestamp], 
                    [y_t63, y_t63],
                    ':',
                    color='grey',
                    linewidth=1.0,
                    alpha=0.7,
                    label="T63% level"
                )[0]
                
                # Add vertical dotted line from T63 point down to the bottom
                ax1.plot(
                    [t63_timestamp, t63_timestamp], 
                    [y_min, y_t63],
                    ':',
                    color='grey',
                    linewidth=1.0,
                    alpha=0.7,
                    label='_T63_vertical'  # Underscore prefix to exclude from legend
                )
                
                # Mark the T63 point with a circle
                ax1.plot(
                    t63_timestamp,
                    y_t63,
                    'o',
                    color='grey',
                    markersize=5,
                    label='_t63_point'  # Underscore prefix to exclude from legend
                )
                        
            # Ensure valid_substrate is not empty before calculating limits
            if not valid_substrate.empty:
                ax2.set_ylim([
                    valid_substrate.min() - 0.05 * (valid_substrate.max() - valid_substrate.min()),
                    valid_substrate.max() + 0.05 * (valid_substrate.max() - valid_substrate.min())
                ])

            # Set axis properties
            ax1.grid(True, which='both')
            
            ax1.set_title("PT1 Model Down Plot", fontsize=12)
            ax1.set_xlabel("Time", fontsize=10)
            ax1.set_ylabel("Biogas production rate [m³/h]", fontsize=10, color='#0000A7')
            ax2.set_ylabel("Substrate feeding rate [t/h]", fontsize=10, color='#C1272D')

            # Format time axis
            ax1.xaxis.set_major_formatter(mdates.DateFormatter('%d\n%H:%M'))
            plt.setp(ax1.xaxis.get_majorticklabels(), rotation=0)
 
            # Add legend with filtering for underscore labels
            lines = ax1.get_lines() + ax2.get_lines()
            labels = [line.get_label() for line in lines]
            
            # Filter out artists with labels starting with underscore
            filtered_lines = [line for line in lines if not line.get_label().startswith('_')]
            filtered_labels = [label for label in labels if not label.startswith('_')]
            
            ax1.legend(filtered_lines, filtered_labels, loc='upper right', fontsize=8)

            # Calculate metrics and update table
            metrics = calculate_metrics(biogas_data.values, pt1_down['yd'].flatten())
            update_metrics_table(tree_down, metrics)

            # Adjust layout and redraw
            fig_model_down.tight_layout()
            canvas_model_down.draw()

    except Exception as e:
        print(f"Error in update_model_down_plot_pt1: {e}")
        import traceback
        traceback.print_exc()

def update_model_up_plot_pt1():
    """Update the PT1 model up plot to match the slider-selected date range.
    Modified to handle numpy int64 indices, fix matplotlib legend warning,
    and use vertical T63 marker instead of full horizontal line."""
    try:
        # Check if preprocessed data is available
        if global_vars['biogas_segment_up'] is None or global_vars['substrate_segment_up'] is None:
            print("No preprocessed data available. Please preprocess data first.")
            return

        # Use the exact same indices from preprocessing tab
        start_ind = global_vars['slider_up_start_index']
        end_ind = global_vars['slider_up_end_index']
        
        # Get the biogas data, substrate data and timestamps using the same indices
        biogas_data = global_vars['biogas_segment_up']['SmoothedValueNum']
        substrate_data = global_vars['substrate_segment_up']['FeedingRate']
        timestamps = global_vars['timestamps_up']
        
        # Convert timestamps to a list for safer indexing
        timestamps_list = timestamps.tolist() if hasattr(timestamps, 'tolist') else list(timestamps)

        # Process substrate data for valid values
        valid_substrate_mask = substrate_data > 0
        valid_substrate = substrate_data[valid_substrate_mask]
        
        # Handle indexing on timestamps using numpy array conversion to avoid pandas index alignment issues
        if isinstance(valid_substrate_mask, pd.Series):
            # Convert to numpy array to avoid pandas index alignment issues
            valid_mask_array = valid_substrate_mask.values
        else:
            valid_mask_array = valid_substrate_mask
            
        # Get valid timestamps based on the data type
        if isinstance(timestamps, pd.DatetimeIndex):
            valid_timestamps_substrate = timestamps[valid_mask_array]
        elif isinstance(timestamps, pd.Series):
            # For pandas Series, use .iloc with numpy boolean array
            valid_timestamps_substrate = timestamps.iloc[valid_mask_array]
        else:
            # For list or numpy array of timestamps
            valid_indices = [i for i, is_valid in enumerate(valid_mask_array) if is_valid]
            valid_timestamps_substrate = [timestamps_list[i] for i in valid_indices]

        # Run the PT1 model calculation with explicit int conversion
        step_index = int(np.argmax(np.diff(substrate_data.values))) if len(substrate_data) > 1 else 0
        pt1_up = pt1_model_modified(
            flowrate=biogas_data,
            feed=substrate_data,
            step_index=step_index,
            feed_max=substrate_data.max(),
            feed_min=valid_substrate.min() if not valid_substrate.empty else 0,
            flowrate_max=biogas_data.max(),
            flowrate_min=biogas_data.min(),
            timestamps=timestamps,
            step_dir='up'
        )

        if pt1_up is not None:
            # Clear and set up the plot
            fig_model_up.clear()
            ax1 = fig_model_up.add_subplot(111)
            ax2 = ax1.twinx()

            # Calculate y-axis limits with margins
            y_min = biogas_data.min() - 0.05 * (biogas_data.max() - biogas_data.min())
            y_max = biogas_data.max() + 0.05 * (biogas_data.max() - biogas_data.min())
            
            # Set y-axis limits
            ax1.set_ylim([y_min, y_max])

            # Plot biogas production rate
            ax1.plot(
                timestamps, biogas_data, '--',
                color='#0000A7', label='Biogas production rate', linewidth=1
            )

            # Plot PT1 model output
            ax1.plot(
                timestamps, pt1_up['yd'], '-.',
                color='#EECC16', label='PT1 model', linewidth=1
            )

            # Plot substrate feeding rate
            ax2.plot(
                valid_timestamps_substrate, valid_substrate, '-',
                color='#C1272D', label='Substrate feeding rate', linewidth=1
            )
            
            # Variable to store T63 line for legend
            t63_line = None
            
            # Add the T63% marking
            if 'y_value_at_critical' in pt1_up and 't63_timestamp' in pt1_up:
                y_t63 = pt1_up['y_value_at_critical']
                t63_timestamp = pt1_up['t63_timestamp']
                
                # Add a short horizontal dotted line at T63% level from the left edge to the T63 point
                t63_line = ax1.plot(
                    [timestamps_list[0], t63_timestamp], 
                    [y_t63, y_t63],
                    ':',
                    color='grey',
                    linewidth=1.0,
                    alpha=0.7,
                    label="T63% level"
                )[0]
                
                # Add vertical dotted line from T63 point down to the bottom
                ax1.plot(
                    [t63_timestamp, t63_timestamp], 
                    [y_min, y_t63],
                    ':',
                    color='grey',
                    linewidth=1.0,
                    alpha=0.7,
                    label='_T63_vertical'  # Underscore prefix to exclude from legend
                )
                
                # Mark the T63 point with a circle
                ax1.plot(
                    t63_timestamp,
                    y_t63,
                    'o',
                    color='grey',
                    markersize=5,
                    label='_t63_point'  # Underscore prefix to exclude from legend
                )
                        
            # Ensure valid_substrate is not empty before calculating limits
            if not valid_substrate.empty:
                ax2.set_ylim([
                    valid_substrate.min() - 0.05 * (valid_substrate.max() - valid_substrate.min()),
                    valid_substrate.max() + 0.05 * (valid_substrate.max() - valid_substrate.min())
                ])

            # Set axis properties
            ax1.grid(True, which='both')
            
            ax1.set_title("PT1 Model Up Plot", fontsize=12)
            ax1.set_xlabel("Time", fontsize=10)
            ax1.set_ylabel("Biogas production rate [m³/h]", fontsize=10, color='#0000A7')
            ax2.set_ylabel("Substrate feeding rate [t/h]", fontsize=10, color='#C1272D')

            # Format time axis
            ax1.xaxis.set_major_formatter(mdates.DateFormatter('%d\n%H:%M'))
            plt.setp(ax1.xaxis.get_majorticklabels(), rotation=0)
 
            # Add legend with filtering for underscore labels
            lines = ax1.get_lines() + ax2.get_lines()
            labels = [line.get_label() for line in lines]
            
            # Filter out artists with labels starting with underscore
            filtered_lines = [line for line in lines if not line.get_label().startswith('_')]
            filtered_labels = [label for label in labels if not label.startswith('_')]
            
            ax1.legend(filtered_lines, filtered_labels, loc='upper right', fontsize=8)

            # Calculate metrics and update table
            metrics = calculate_metrics(biogas_data.values, pt1_up['yd'].flatten())
            update_metrics_table(tree_up, metrics)

            # Adjust layout and redraw
            fig_model_up.tight_layout()
            canvas_model_up.draw()

    except Exception as e:
        print(f"Error in update_model_up_plot_pt1: {e}")
        import traceback
        traceback.print_exc()

 

import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
from matplotlib.dates import date2num
import control
import pandas as pd

def validate_inputs(flowrate, feed, ind_sprung):
    """Validate input data for zeitkonstantesumme calculation"""
    if flowrate is None or feed is None:
        raise ValueError("Flowrate and feed data cannot be None")
    
    if ind_sprung < 0 or ind_sprung >= len(flowrate):
        raise ValueError("Invalid step index")
    
    return True

def find_inflection_point(flowrate_values, timestamps):
    """Find inflection point using second derivative"""
    try:
        # Calculate first derivative
        first_deriv = np.gradient(flowrate_values)
        # Calculate second derivative
        second_deriv = np.gradient(first_deriv)
        # Find inflection point (where second derivative crosses zero)
        inflection_idx = np.where(np.diff(np.signbit(second_deriv)))[0][0]
        # Calculate slope at inflection point
        slope = first_deriv[inflection_idx]
        
        return inflection_idx, slope
    except Exception as e:
        print(f"Error finding inflection point: {e}")
        return None, None
    

def convert_timedelta_to_seconds(td):
    """
    Convert timedelta to seconds, exactly matching MATLAB's seconds() function behavior
    
    Parameters:
    -----------
    td : pandas.Timedelta
        Time difference to convert to seconds
    
    Returns:
    --------
    float
        Total seconds in the time difference
    """
    if pd.isna(td):
        return np.nan
    
    try:
        # Convert to total seconds with high precision
        total_seconds = td.total_seconds()
        return total_seconds
    except AttributeError:
        # If input is not a timedelta, attempt to convert
        try:
            converted_td = pd.Timedelta(td)
            return converted_td.total_seconds()
        except Exception as e:
            raise ValueError(f"Cannot convert to seconds: {e}")

def zeitkonstantesumme(flowrate, feed, ind_sprung, feed_max, feed_min, 
                       flowrate_max, flowrate_min, ind_wp, steigung_wp, step_dir):

    # Initialize results dictionary
    zt = {}
    
    # Extract values
    flowrate_values = flowrate['interp_y_smooth'].values
    timestamps = flowrate['interp_dt'].values
    feed_values = feed['interp_diff_substrate_feed'].values
    
    # Gain calculation
    zt['K'] = (flowrate_max - flowrate_min) / (feed_max - feed_min)
    
    # Step change information
    zt['ind_sprung'] = ind_sprung
    zt['zeit_sprung'] = timestamps[ind_sprung]
    
    # Inflection point calculation
    zt['index'] = ind_wp
    zt['steigung'] = steigung_wp
    zt['wert'] = flowrate_values[ind_wp]
    zt['offset'] = zt['wert'] - zt['steigung'] * zt['index']
    
    # Create time array and linear approximation
    zt['t'] = np.arange(len(timestamps))
    zt['gleichung'] = zt['steigung'] * zt['t'] + zt['offset']
    
    # Determine Tu and Tg indices based on step direction
    if 'down' in step_dir:
        tu_indices = np.where(zt['gleichung'] > flowrate_max)[0]
        tg_indices = np.where(zt['gleichung'] < flowrate_min)[0]
        
        if len(tu_indices) > 0 and len(tg_indices) > 0:
            zt['ind_Tu'] = tu_indices[-1]
            zt['ind_Tg'] = tg_indices[0]
        else:
            # Fallback if no crossing found
            zt['ind_Tu'] = ind_sprung
            zt['ind_Tg'] = ind_sprung + 1
    else:  # 'up' direction
        tu_indices = np.where(zt['gleichung'] < flowrate_min)[0]
        tg_indices = np.where(zt['gleichung'] > flowrate_max)[0]
        
        if len(tu_indices) > 0 and len(tg_indices) > 0:
            zt['ind_Tu'] = tu_indices[-1]
            zt['ind_Tg'] = tg_indices[0]
        else:
            # Fallback if no crossing found
            zt['ind_Tu'] = ind_sprung
            zt['ind_Tg'] = ind_sprung + 1
    
    # Calculate Tu and Tg
    zt['Tu'] = convert_timedelta_to_seconds(
        timestamps[zt['ind_Tu']] - timestamps[zt['ind_sprung']]
    )
    zt['Tg'] = convert_timedelta_to_seconds(
        timestamps[zt['ind_Tg']] - timestamps[zt['ind_Tu']]
    )
    
    # Predefined value for system response
    zt['wert_fuer_n_2'] = 0.575
    
    # Calculate value at time constant sum
    if 'down' in step_dir:
        zt['y_wert_bei_T_summe'] = flowrate_max - (flowrate_max - flowrate_min) * zt['wert_fuer_n_2']
        cross_indices = np.where(flowrate_values < zt['y_wert_bei_T_summe'])[0]
    else:
        zt['y_wert_bei_T_summe'] = flowrate_min + (flowrate_max - flowrate_min) * zt['wert_fuer_n_2']
        cross_indices = np.where(flowrate_values > zt['y_wert_bei_T_summe'])[0]
    
    # Determine index at time constant sum
    zt['index_bei_T_summe'] = cross_indices[0] if len(cross_indices) > 0 else ind_sprung
    
    # Calculate total time constant sum
    zt['T_summe'] = convert_timedelta_to_seconds(
        timestamps[zt['index_bei_T_summe']] - timestamps[zt['ind_sprung']]
    )
    
    # Calculate system order
    zt['n'] = round(zt['Tu'] / zt['Tg'] * 10 + 1)
    
    # Calculate time constant
    zt['T'] = zt['T_summe'] / zt['n']
    
    # Create transfer function
    # Initial transfer function with gain
    zt['G'] = control.TransferFunction([zt['K']], [zt['T'], 1])
    
    # Add additional identical transfer functions in series
    for _ in range(zt['n'] - 1):
        zt['G'] = control.series(zt['G'], control.TransferFunction([1], [zt['T'], 1]))
    
    # Discretization parameters
    zt['Ta'] = 120  # Sampling time
    zt['test_time'] = np.arange(0, zt['Ta'] * len(timestamps), zt['Ta'])
    
    # Discretize transfer function
    zt['Gd'] = control.sample_system(zt['G'], zt['Ta'])
    
    # Simulate step response
    if 'down' in step_dir:
        input_signal = feed_values - feed_max
        _, yd = control.forced_response(zt['Gd'], T=zt['test_time'], U=input_signal)
        zt['yd'] = yd + flowrate_max
    else:
        input_signal = feed_values - feed_min
        _, yd = control.forced_response(zt['Gd'], T=zt['test_time'], U=input_signal)
        zt['yd'] = yd + flowrate_min
    
    return zt



def plot_zeitkonstantesumme_ui(i, ui_axes, flowrate, feed, zt):
    colors = ['#0072BD', '#D95319', '#EDB120', '#7E2F8E', '#77AC30', '#4DBEEE', '#A2142F']
    const = {
        'font': 'serif',
        'fontsize': 12,
        'fontsizelegend': 8,
        'fontsizeticks': 10,
        'linienbreite': 1,
        'y_min': min(flowrate['interp_y_smooth']) - 0.05 * max(flowrate['interp_y_smooth']),
        'y_max': max(flowrate['interp_y_smooth']) + 0.05 * max(flowrate['interp_y_smooth'])
    }
    color_float = 0.8
    gray_color = [color_float] * 3

    ui_axes.clear()
    ui_axes.grid(True, which='both')
    ui_axes.tick_params(which='both', labelsize=const['fontsizeticks'])
    ui_axes.set_ylabel('Gas production flow rate [m³/h]', 
                      fontfamily=const['font'], 
                      fontsize=const['fontsize'])
    ui_axes.set_xlabel('Time', 
                      fontfamily=const['font'], 
                      fontsize=const['fontsize'])
    ui_axes.set_title('Sum of time constants method', 
                     fontfamily=const['font'], 
                     fontsize=const['fontsize'])
    ui_axes.set_ylim([const['y_min'], const['y_max']])

    # Main plots
    line1 = ui_axes.plot(flowrate['interp_dt'], flowrate['interp_y_smooth'],
                        linestyle='--', 
                        color=colors[0], linewidth=const['linienbreite'], 
                        label='Gas production flow rate after preprocessing')
    
    ui_axes.scatter(flowrate['interp_dt'].iloc[int(zt['index'])], 
                   flowrate['interp_y_smooth'].iloc[int(zt['index'])], 
                   color=gray_color, s=100, marker='o', 
                   label='Turning point')

    line7 = ui_axes.plot(flowrate['interp_dt'], zt['yd'],
                        linestyle='-.', 
                        color=colors[2], linewidth=const['linienbreite'], 
                        label='Sum of time constants method')

    # Secondary y-axis for substrate feeding rate
    ax2 = ui_axes.twinx()
    
    # Filter substrate data for values > 0
    feed_df = pd.DataFrame({
        'time': flowrate['interp_dt'],
        'feed': feed['interp_diff_substrate_feed']
    })
    feed_filtered = feed_df[feed_df['feed'] > 0]
    
    y_min_right = feed_filtered['feed'].min() - 0.05 * feed_filtered['feed'].max()
    y_max_right = feed_filtered['feed'].max() + 0.05 * feed_filtered['feed'].max()
    ax2.set_ylim([y_min_right, y_max_right])
    ax2.set_ylabel('Substrate feeding rate [t/3h]', 
                  fontfamily=const['font'], 
                  fontsize=const['fontsize'],
                  color=colors[1])
    
    line3 = ax2.plot(feed_filtered['time'], feed_filtered['feed'],
                     linestyle='-',
                     color=colors[1], linewidth=const['linienbreite'],
                     label='Substrate feeding rate')

    # Legend
    lines = line1 + line7 + line3
    labels = [l.get_label() for l in lines]
    ui_axes.legend(lines, labels, loc='best', 
                  fontsize=const['fontsizelegend'], 
                  prop={'family': const['font']})
    
import numpy as np
import logging
import traceback

def validate_input_data(substrate_data, biogas_data):
    """Validate input data before processing"""
    if len(substrate_data) == 0 or len(biogas_data) == 0:
        raise ValueError("Empty input data")
    
    if not np.all(np.isfinite(substrate_data)) or not np.all(np.isfinite(biogas_data)):
        raise ValueError("Input data contains non-finite values")

def find_inflection_point(y_smooth):
    """Efficiently calculate inflection point"""
    dy = np.gradient(y_smooth)
    d2y = np.gradient(dy)
    return np.argmin(np.abs(d2y)), dy[np.argmin(np.abs(d2y))]

def update_model_down_plot_zeitkonstantesumme():
    """
    Update and plot downward model for biogas process
    """
    try:
        # Comprehensive data validation
        if not all([
            global_vars.get('biogas_segment_down') is not None,
            global_vars.get('substrate_segment_down') is not None,
            global_vars.get('timestamps_down') is not None
        ]):
            logging.warning("Insufficient data for model calculation")
            return

        # Clear previous figure
        fig_model_down.clear()
        ax1 = fig_model_down.add_subplot(111)

        # Extract data
        biogas_data = global_vars['biogas_segment_down']['SmoothedValueNum']
        substrate_data = global_vars['substrate_segment_down']['FeedingRate']
        timestamps = global_vars['timestamps_down']

        # Validate input data
        validate_input_data(substrate_data, biogas_data)

        # Calculate key parameters
        feed_max = substrate_data.max()
        feed_min = substrate_data[substrate_data > 0].min()
        flowrate_max = biogas_data.max()
        flowrate_min = biogas_data.min()
        
        # Find step change index
        ind_sprung = np.argmax(np.diff(substrate_data.values))

        # Find inflection point
        ind_wp, steigung_wp = find_inflection_point(biogas_data.values)

        # Prepare data dictionaries
        flowrate_data = {
            'interp_y_smooth': biogas_data,
            'interp_dt': timestamps,
            'sel_time': timestamps
        }
        
        feed_data = {
            'interp_diff_substrate_feed': substrate_data,
            'interp_dt': timestamps
        }

        # Calculate zeitkonstantesumme
        zt_result = zeitkonstantesumme(
            flowrate_data,
            feed_data,
            ind_sprung,
            feed_max,
            feed_min, 
            flowrate_max,
            flowrate_min,
            ind_wp,
            steigung_wp,
            'down'
        )

        # Process and visualize results
        if zt_result is not None:
            plot_zeitkonstantesumme_ui(0, ax1, flowrate_data, feed_data, zt_result)
            
            # Calculate and update metrics
            metrics = calculate_metrics(biogas_data.values, zt_result['yd'])
            update_metrics_table(tree_down, metrics)

            # Store scenario data
            global_vars["down_scenario_data"] = {
                "time": timestamps,
                "flow_measured": biogas_data,
                "model_output": zt_result['yd'],
                "feed": substrate_data,
                "Tu": zt_result['Tu'],
                "Tg": zt_result['Tg'],
                "T_summe": zt_result['T_summe'],
                "n": zt_result['n'],
                "K": zt_result['K']
            }

            # Update plot
            fig_model_down.tight_layout()
            canvas_model_down.draw()

    except Exception as e:
        logging.error(f"Error in downward plot zeitkonstantesumme: {e}")
        logging.exception("Detailed traceback")
        traceback.print_exc()



def update_model_up_plot_zeitkonstantesumme():
    try:
        if global_vars['biogas_segment_up'] is None:
            return

        fig_model_up.clear()
        ax1 = fig_model_up.add_subplot(111)

        biogas_data = global_vars['biogas_segment_up']['SmoothedValueNum']
        substrate_data = global_vars['substrate_segment_up']['FeedingRate']
        timestamps = global_vars['timestamps_up']

        feed_max = substrate_data.max()
        feed_min = substrate_data[substrate_data > 0].min()
        flowrate_max = biogas_data.max()
        flowrate_min = biogas_data.min()
        ind_sprung = np.argmax(np.diff(substrate_data.values))

        y_smooth = biogas_data.values
        dy = np.gradient(y_smooth)
        d2y = np.gradient(dy)
        ind_wp = np.argmin(np.abs(d2y))
        steigung_wp = dy[ind_wp]

        flowrate_data = {
            'interp_y_smooth': biogas_data,
            'interp_dt': timestamps,
            'sel_time': timestamps
        }
        
        feed_data = {
            'interp_diff_substrate_feed': substrate_data,
            'interp_dt': timestamps
        }

        zt_result = zeitkonstantesumme(
            flowrate_data,
            feed_data,
            ind_sprung,
            feed_max,
            feed_min, 
            flowrate_max,
            flowrate_min,
            ind_wp,
            steigung_wp,
            'up'
        )

        if zt_result is not None:
            plot_zeitkonstantesumme_ui(0, ax1, flowrate_data, feed_data, zt_result)
            metrics = calculate_metrics(biogas_data.values, zt_result['yd'])
            update_metrics_table(tree_up, metrics)

            global_vars["up_scenario_data"] = {
                "time": timestamps,
                "flow_measured": biogas_data,
                "model_output": zt_result['yd'],
                "feed": substrate_data,
                "Tu": zt_result['Tu'],
                "Tg": zt_result['Tg'],
                "T_summe": zt_result['T_summe'],
                "n": zt_result['n'],
                "K": zt_result['K']
            }

            fig_model_up.tight_layout()
            canvas_model_up.draw()

    except Exception as e:
        print(f"Error in upward plot zeitkonstantesumme: {e}")
        traceback.print_exc()


import numpy as np
import pandas as pd
from scipy import signal
import traceback

def wendetangente(flowrate, feed, ind_sprung, feed_max, feed_min, flowrate_max, flowrate_min, ind_wp, steigung_wp, step_dir):

    try:
        wt = {}
        
        # Calculate gain
        wt['K'] = (flowrate_max - flowrate_min) / (feed_max - feed_min)
        
        # Step start point
        wt['ind_sprung'] = ind_sprung
        wt['zeit_sprung'] = flowrate['interp_dt'].iloc[wt['ind_sprung']]
        
        # Turning tangent equation
        wt['index'] = ind_wp
        wt['steigung'] = steigung_wp
        wt['wert'] = flowrate['interp_y_smooth'].iloc[wt['index']]
        wt['offset'] = wt['wert'] - wt['steigung'] * wt['index']
        
        # Match MATLAB's 1-indexed array creation
        wt['t'] = np.arange(1, len(flowrate['interp_dt']) + 1)
        wt['gleichung'] = wt['steigung'] * wt['t'] + wt['offset']
        
        # Determine Tu and Tg indices - more closely matching MATLAB's find function
        if 'down' in step_dir:
            # Find last index where gleichung > flowrate_max
            indices = np.where(wt['gleichung'] > flowrate_max)[0]
            wt['ind_Tu'] = indices[-1] if len(indices) > 0 else ind_sprung
            
            # Find first index where gleichung < flowrate_min
            indices = np.where(wt['gleichung'] < flowrate_min)[0]
            wt['ind_Tg'] = indices[0] if len(indices) > 0 else ind_wp
        elif 'up' in step_dir:
            # Find last index where gleichung < flowrate_min
            indices = np.where(wt['gleichung'] < flowrate_min)[0]
            wt['ind_Tu'] = indices[-1] if len(indices) > 0 else ind_sprung
            
            # Find first index where gleichung > flowrate_max
            indices = np.where(wt['gleichung'] > flowrate_max)[0]
            wt['ind_Tg'] = indices[0] if len(indices) > 0 else ind_wp
        
        # Calculate time differences (matching seconds() function)
        def seconds_diff(t1, t2):
            return (pd.Timestamp(t1) - pd.Timestamp(t2)).total_seconds()
        
        wt['Tu'] = seconds_diff(flowrate['interp_dt'].iloc[wt['ind_Tu']], 
                               flowrate['interp_dt'].iloc[wt['ind_sprung']])
        wt['Tg'] = seconds_diff(flowrate['interp_dt'].iloc[wt['ind_Tg']], 
                               flowrate['interp_dt'].iloc[wt['ind_Tu']])
        wt['Twp'] = seconds_diff(flowrate['interp_dt'].iloc[wt['index']], 
                                flowrate['interp_dt'].iloc[wt['ind_sprung']])
        
        # Calculate time constants
        wt['T_Twp'] = wt['Twp'] / 1
        wt['T_Tu'] = wt['Tu'] / 0.282
        wt['T_Tg'] = wt['Tg'] / 2.718
        wt['T_mean'] = (wt['T_Twp'] + wt['T_Tu'] + wt['T_Tg']) / 3
        
        # Create transfer functions
        wt['Ta'] = 120
        wt['test_time'] = np.arange(0, wt['Ta'] * len(flowrate['interp_dt']), wt['Ta'])
        
        # Create transfer functions directly matching MATLAB pattern
        def create_tf(T):
            # G(s) = K / (Ts+1)^2
            num = [wt['K']]
            den = np.convolve([T, 1], [T, 1])  # (Ts+1)(Ts+1)
            return num, den
        
        # Create all transfer functions
        wt['G_Twp_num'], wt['G_Twp_den'] = create_tf(wt['T_Twp'])
        wt['G_Tu_num'], wt['G_Tu_den'] = create_tf(wt['T_Tu'])
        wt['G_Tg_num'], wt['G_Tg_den'] = create_tf(wt['T_Tg'])
        wt['G_Tmean_num'], wt['G_Tmean_den'] = create_tf(wt['T_mean'])
        
        # Convert to discrete-time systems
        wt['Gd_Twp'] = signal.cont2discrete((wt['G_Twp_num'], wt['G_Twp_den']), wt['Ta'], method='zoh')
        wt['Gd_Tu'] = signal.cont2discrete((wt['G_Tu_num'], wt['G_Tu_den']), wt['Ta'], method='zoh')
        wt['Gd_Tg'] = signal.cont2discrete((wt['G_Tg_num'], wt['G_Tg_den']), wt['Ta'], method='zoh')
        wt['Gd_Tmean'] = signal.cont2discrete((wt['G_Tmean_num'], wt['G_Tmean_den']), wt['Ta'], method='zoh')
        
        # Simulate systems
        if 'down' in step_dir:
            u = feed['interp_diff_substrate_feed'].values - feed_max
            offset = flowrate_max
        elif 'up' in step_dir:
            u = feed['interp_diff_substrate_feed'].values - feed_min
            offset = flowrate_min
        else:
            raise ValueError(f"Invalid step direction: {step_dir}. Must contain 'up' or 'down'.")
        
        # Simulate responses with discrete systems
        _, wt['yd_Twp'] = signal.dlsim(wt['Gd_Twp'], u, t=wt['test_time'])
        _, wt['yd_Tu'] = signal.dlsim(wt['Gd_Tu'], u, t=wt['test_time'])
        _, wt['yd_Tg'] = signal.dlsim(wt['Gd_Tg'], u, t=wt['test_time'])
        _, wt['yd_Tmean'] = signal.dlsim(wt['Gd_Tmean'], u, t=wt['test_time'])
        
        # Add offset to match original levels and convert to 1D arrays
        wt['yd_Twp'] = wt['yd_Twp'].flatten() + offset
        wt['yd_Tu'] = wt['yd_Tu'].flatten() + offset
        wt['yd_Tg'] = wt['yd_Tg'].flatten() + offset
        wt['yd_Tmean'] = wt['yd_Tmean'].flatten() + offset
        
        # Use Tg response as final output
        wt['yd'] = wt['yd_Tg']
        
        return wt
        
    except Exception as e:
        print(f"Error in wendetangente: {str(e)}")
        traceback.print_exc()
        return None



def plot_wendetangente_ui(i, ui_axes, flowrate, feed, wt):
    try:
        # Define color schemes
        colors = ['#0072BD', '#D95319', '#EDB120', '#7E2F8E', '#77AC30', '#4DBEEE', '#A2142F']
        gray_color = 0.8 * np.array([1, 1, 1])

        # Define plotting constants
        const = {
            'font': 'serif',
            'fontsize': 12,
            'fontsizelegend': 8,
            'fontsizeticks': 10,
            'linienbreite': 1
        }

        # Calculate data ranges
        flowrate_min = min(flowrate['interp_y_smooth'])
        flowrate_max = max(flowrate['interp_y_smooth'])
        feed_min = min(feed['interp_diff_substrate_feed'][feed['interp_diff_substrate_feed'] > 0])
        feed_max = max(feed['interp_diff_substrate_feed'])

        # Set axis limits with 5% padding
        y1_min = flowrate_min - 0.05 * flowrate_max
        y1_max = flowrate_max + 0.05 * flowrate_max
        y2_min = feed_min - 0.05 * feed_max
        y2_max = feed_max + 0.05 * feed_max

        # Configure primary axis
        ui_axes.set_ylim([y1_min, y1_max])
        ui_axes.xaxis.set_major_formatter(mdates.DateFormatter('%d\n%H:%M'))
        ui_axes.xaxis.set_major_locator(mdates.HourLocator(interval=12))
        ui_axes.grid(True, which='both')
        ui_axes.tick_params(labelsize=const['fontsizeticks'])
        
        # Set labels and title
        ui_axes.set_ylabel('Biogas production rate [m³/h]', fontsize=const['fontsize'])
        ui_axes.set_xlabel('Time', fontsize=const['fontsize'])
        ui_axes.set_title('Turning tangent method', fontsize=const['fontsize'])
        
        # Plot biogas production rate
        line1 = ui_axes.plot(flowrate['interp_dt'], flowrate['interp_y_smooth'], 
                           color=colors[0], linewidth=const['linienbreite'],
                           linestyle='--',
                           label='Gas production flow rate after preprocessing')
        
        # Plot turning tangent
        line2 = ui_axes.plot(flowrate['interp_dt'], wt['gleichung'],
                          color=gray_color, linewidth=const['linienbreite'],
                          linestyle='-.', 
                          label='Turning tangent')

        # Plot turning point
        line4 = ui_axes.plot(flowrate['interp_dt'].iloc[wt['index']], 
                           flowrate['interp_y_smooth'].iloc[wt['index']], 'o',
                           markersize=10, markerfacecolor=gray_color, 
                           markeredgecolor=gray_color,
                           label='Turning point')

        # Plot model output
        line7 = ui_axes.plot(flowrate['interp_dt'], wt['yd_Tg'],
                           color=colors[2], linewidth=const['linienbreite'],
                           linestyle='-.',
                           label='Turning tangent method')

        # Configure secondary axis
        ax2 = ui_axes.twinx()
        ax2.set_ylabel('Substrate feeding rate [t/3h]', fontsize=const['fontsize'])
        ax2.tick_params(axis='y', colors=colors[1])
        ax2.set_ylim([y2_min, y2_max])

        # Plot substrate feeding rate (only non-zero values)
        valid_feed_mask = feed['interp_diff_substrate_feed'].values > 0
        valid_feed = feed['interp_diff_substrate_feed'].values[valid_feed_mask]
        valid_time = np.array(flowrate['interp_dt'])[valid_feed_mask]

        line3 = ax2.plot(valid_time, valid_feed,
                        color=colors[1], linewidth=const['linienbreite'],
                        label='Substrate feeding rate')

        # Create legend
        lines = line1 + line2 + [line4[0]] + line7 + line3
        labels = [l.get_label() for l in lines]
        legend = ui_axes.legend(lines, labels, loc='lower left', fontsize=const['fontsizelegend'])
        legend.get_frame().set_facecolor('white')

        return {'primary_axis': ui_axes, 'secondary_axis': ax2}

    except Exception as e:
        print(f"Error in plot_wendetangente_ui: {e}")
        traceback.print_exc()
        return None

def update_model_down_plot_wendetangente():
  try:
      if global_vars['biogas_segment_down'] is None:
          return

      fig_model_down.clear()
      ax1 = fig_model_down.add_subplot(111)

      biogas_data = global_vars['biogas_segment_down']['SmoothedValueNum']
      substrate_data = global_vars['substrate_segment_down']['FeedingRate']
      timestamps = global_vars['timestamps_down']

      feed_max = substrate_data.max()
      feed_min = substrate_data[substrate_data > 0].min()
      flowrate_max = biogas_data.max()
      flowrate_min = biogas_data.min()
      ind_sprung = np.argmax(np.diff(substrate_data.values))

      y_smooth = biogas_data.values
      dy = np.gradient(y_smooth)
      d2y = np.gradient(dy)
      ind_wp = np.argmin(np.abs(d2y))
      steigung_wp = dy[ind_wp]

      data = {
          'interp_y_smooth': biogas_data,
          'interp_dt': timestamps,
          'sel_time': timestamps
      }
      
      valid_substrate_mask = substrate_data > 0
      feed_data = {
          'interp_diff_substrate_feed': substrate_data,
          'interp_dt': timestamps
      }

      wt_result = wendetangente(
          data, feed_data,
          ind_sprung, feed_max, feed_min, flowrate_max, flowrate_min,
          ind_wp, steigung_wp, 'down'
      )

      if wt_result is not None:
          plot_wendetangente_ui(0, ax1, data, feed_data, wt_result)
          metrics = calculate_metrics(biogas_data.values, wt_result['yd'])
          update_metrics_table(tree_down, metrics) 

          global_vars["down_scenario_data"] = {
              "time": timestamps,
              "flow_measured": biogas_data,
              "model_output": wt_result['yd'],
              "feed": substrate_data
          }

          fig_model_down.tight_layout()
          canvas_model_down.draw()

  except Exception as e:
      print(f"Error in downward plot: {e}")
      traceback.print_exc()

def update_model_up_plot_wendetangente():
  try:
      if global_vars['biogas_segment_up'] is None:
          return

      fig_model_up.clear()
      ax1 = fig_model_up.add_subplot(111)

      biogas_data = global_vars['biogas_segment_up']['SmoothedValueNum']
      substrate_data = global_vars['substrate_segment_up']['FeedingRate']
      timestamps = global_vars['timestamps_up']

      feed_max = substrate_data.max()
      feed_min = substrate_data[substrate_data > 0].min()
      flowrate_max = biogas_data.max()
      flowrate_min = biogas_data.min()
      ind_sprung = np.argmax(np.diff(substrate_data.values))

      y_smooth = biogas_data.values
      dy = np.gradient(y_smooth)
      d2y = np.gradient(dy)
      ind_wp = np.argmin(np.abs(d2y))
      steigung_wp = dy[ind_wp]

      data = {
          'interp_y_smooth': biogas_data,
          'interp_dt': timestamps,
          'sel_time': timestamps
      }
      
      feed_data = {
          'interp_diff_substrate_feed': substrate_data,
          'interp_dt': timestamps
      }

      wt_result = wendetangente(
          data, feed_data,
          ind_sprung, feed_max, feed_min, flowrate_max, flowrate_min,
          ind_wp, steigung_wp, 'up'
      )

      if wt_result is not None:
          plot_wendetangente_ui(0, ax1, data, feed_data, wt_result)
          metrics = calculate_metrics(biogas_data.values, wt_result['yd'])
          update_metrics_table(tree_up, metrics)

          global_vars["up_scenario_data"] = {
              "time": timestamps,
              "flow_measured": biogas_data,
              "model_output": wt_result['yd'],
              "feed": substrate_data
          }

          fig_model_up.tight_layout()
          canvas_model_up.draw()

  except Exception as e:
      print(f"Error in upward plot: {e}")
      traceback.print_exc()


import numpy as np
from scipy.sparse.linalg import lsqr
from scipy.signal import lsim, TransferFunction

def schaetzer_pt1(flowrate, feed, feed_max, feed_min, flowrate_max, flowrate_min, step_dir):
    try:
        sch_pt1 = {}
        
        print("Input shapes:")
        print(f"flowrate shape: {flowrate['interp_y_smooth'].shape}")
        print(f"feed shape: {feed['interp_diff_substrate_feed'].shape}")
        
        # Matrix setup
        A = np.column_stack((flowrate['interp_y_smooth'][:-1], 
                           feed['interp_diff_substrate_feed'][:-1]))
        b = flowrate['interp_y_smooth'][1:]
        
        print(f"A shape: {A.shape}")
        print(f"b shape: {b.shape}")
        
        x = lsqr(A, b)[0]
        print(f"x values: {x}")
        
        # Discrete domain
        Ta = 120
        test_time = np.arange(0, Ta * len(flowrate['interp_dt']), Ta)
        
        # Transfer function
        num = [0, x[1]]
        den = [1, -x[0]]
        print(f"num: {num}, den: {den}")
        
        # Input calculation
        if 'down' in step_dir:
            u = feed['interp_diff_substrate_feed'] - feed_max
        else:
            u = feed['interp_diff_substrate_feed'] - feed_min
            
        print(f"u shape: {u.shape}")
        
        # System response
        yout = signal.dlsim((num, den, Ta), u)[1]
        print(f"yout shape: {yout.shape}")
        
        # Final output
        yd = yout.flatten() + (flowrate_max if 'down' in step_dir else flowrate_min)
        print(f"yd shape: {yd.shape}")
        
        sch_pt1['yd'] = yd
        return sch_pt1
        
    except Exception as e:
        print(f"Detailed error in schaetzer_pt1: {str(e)}")
        print(f"At line: {traceback.format_exc()}")
        return None

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime
import numpy as np

def plot_schaetzer_ui(i, ui_axes, flowrate, feed, sch_pt1, sch_name):
    """
    Plot PT1-Modell Verfahren
    
    Parameters:
    -----------
    i : int
        Index or identifier
    ui_axes : matplotlib.axes.Axes
        The axes object to plot on
    flowrate : object
        Object containing flow rate data
    feed : object
        Object containing feed data
    sch_pt1 : object
        Object containing PT1 model data
    sch_name : str
        Name of the scheme/model
    """
    
    # Define colors (equivalent to MATLAB's newcolors)
    colors = ['#0072BD', '#D95319', '#EDB120', '#7E2F8E', 
             '#77AC30', '#4DBEEE', '#A2142F']
    
    # Constants
    const = {
        'font': 'serif',
        'fontsize': 16,
        'fontsizelegend': 10,
        'fontsizeticks': 10,
        'linienbreite': 2,  # line width
        'y_min': min(flowrate.interp_y_smooth) - 0.05 * max(flowrate.interp_y_smooth),
        'y_max': max(flowrate.interp_y_smooth) + 0.05 * max(flowrate.interp_y_smooth),
        'start_date': flowrate.sel_time[0],
        'end_date': flowrate.sel_time[-1]
    }
    
    # Configure primary y-axis
    ui_axes.grid(True, which='both')
    ui_axes.set_xlim([mdates.date2num(const['start_date']), 
                      mdates.date2num(const['end_date'])])
    ui_axes.set_ylim([const['y_min'], const['y_max']])
    
    # Set font properties
    ui_axes.tick_params(axis='both', which='both', labelsize=const['fontsizeticks'])
    for label in ui_axes.get_xticklabels() + ui_axes.get_yticklabels():
        label.set_fontname(const['font'])
    
    # Labels and title
    ui_axes.set_ylabel('Gas production flow rate [m³/h]', 
                      fontname=const['font'], 
                      fontsize=const['fontsize'])
    ui_axes.set_xlabel('Time', 
                      fontname=const['font'], 
                      fontsize=const['fontsize'])
    ui_axes.set_title(sch_name, 
                     fontname=const['font'], 
                     fontsize=const['fontsize'])
    
    # Primary axis plots
    line1 = ui_axes.plot(flowrate.interp_dt, flowrate.interp_y_smooth, 
                        color=colors[0], 
                        linewidth=const['linienbreite'], 
                        label='Gas production flow rate after preprocessing')
    
    line2 = ui_axes.plot(flowrate.interp_dt, sch_pt1.yd, 
                        color=colors[2], 
                        linewidth=const['linienbreite'], 
                        label=sch_name)
    
    # Create secondary y-axis
    ax2 = ui_axes.twinx()
    
    # Configure secondary y-axis limits
    y_min_right = min(feed.interp_diff_substrate_feed) - 0.05 * max(feed.interp_diff_substrate_feed)
    y_max_right = max(feed.interp_diff_substrate_feed) + 0.05 * max(feed.interp_diff_substrate_feed)
    ax2.set_ylim([y_min_right, y_max_right])
    
    # Secondary axis color and label
    ax2.spines['right'].set_color(colors[1])
    ax2.tick_params(axis='y', colors=colors[1])
    ax2.set_ylabel('Substrate feeding rate [t/3h]', 
                  fontname=const['font'], 
                  fontsize=const['fontsize'], 
                  color=colors[1])
    
    # Secondary axis plot
    line3 = ax2.plot(flowrate.interp_dt, feed.interp_diff_substrate_feed, 
                    color=colors[1], 
                    linewidth=const['linienbreite'], 
                    label='Substrate feeding rate')
    
    # Create combined legend for both axes
    lines = line1 + line2
    labels = [line.get_label() for line in lines]
    legend = ui_axes.legend(lines, labels, 
                          loc='southwest', 
                          fontsize=const['fontsizelegend'])
    legend.get_frame().set_facecolor('white')
    for text in legend.get_texts():
        text.set_fontname(const['font'])
    
    # Return plot elements if needed
    return {
        'primary_axis': ui_axes,
        'secondary_axis': ax2,
        'lines': [line1[0], line2[0], line3[0]],
        'legend': legend
    }


def update_model_down_plot_schaetzer_pt1():
    try:
        if global_vars['biogas_segment_down'] is None or global_vars['substrate_segment_down'] is None:
            return
            
        fig_model_down.clear()
        ax1 = fig_model_down.add_subplot(111)
        ax2 = ax1.twinx()
        
        biogas_data = global_vars['biogas_segment_down']['SmoothedValueNum']
        substrate_data = global_vars['substrate_segment_down']['FeedingRate']
        timestamps = global_vars['timestamps_down']

        feed_max = substrate_data.max()
        feed_min = substrate_data[substrate_data > 0].min()
        flowrate_max = biogas_data.max()
        flowrate_min = biogas_data.min()

        # Calculate PT1 estimation
        pt1_result = schaetzer_pt1(
            {'interp_y_smooth': biogas_data.values, 'interp_dt': timestamps},
            {'interp_diff_substrate_feed': substrate_data.values},
            feed_max, feed_min, flowrate_max, flowrate_min, 'down'
        )
        
        if pt1_result is None:
            raise ValueError("PT1 estimation failed")

        # Plot biogas production rate
        line1 = ax1.plot(timestamps, biogas_data, 
                        linestyle='--', color='#0000A7',
                        label='Biogas production rate', linewidth=1)

        # Plot model output
        line2 = ax1.plot(timestamps, pt1_result['yd'],
                        linestyle='-.', color='#EECC16',
                        label='PT1 estimator', linewidth=1)

        # Plot substrate feeding rate
        valid_substrate = substrate_data[substrate_data > 0]
        valid_timestamps = np.array(timestamps)[substrate_data > 0]
        line3 = ax2.plot(valid_timestamps, valid_substrate,
                        color='#C1272D', linewidth=1,
                        label='Substrate feeding rate')

        # Configure axes
        ax1.set_xlabel('Time', fontsize=12)
        ax1.set_ylabel('Biogas production rate [m³/h]', color='#0000A7', fontsize=12)
        ax2.set_ylabel('Substrate feeding rate [t/h]', color='#C1272D', fontsize=12)
        
        ax1.grid(True)
        ax1.xaxis.set_major_formatter(mdates.DateFormatter('%d\n%H:%M'))
        
        # Legend
        lines = line1 + line2 + line3
        labels = [l.get_label() for l in lines]
        ax1.legend(lines, labels, loc='upper left')

        # Calculate and update metrics
        metrics = calculate_metrics(biogas_data.values, pt1_result['yd'])
        update_metrics_table(tree_down, metrics)

        # Store scenario data
        global_vars["down_scenario_data"] = {
            "time": timestamps,
            "flow_measured": biogas_data,
            "model_output": pt1_result['yd'],
            "feed": substrate_data
        }
            
        fig_model_down.tight_layout()
        canvas_model_down.draw()
            
    except Exception as e:
        print(f"Error in update_model_down_plot_pt1: {e}")

def update_model_up_plot_schaetzer_pt1():
    try:
        if global_vars['biogas_segment_up'] is None or global_vars['substrate_segment_up'] is None:
            return
            
        fig_model_up.clear()
        ax1 = fig_model_up.add_subplot(111)
        ax2 = ax1.twinx()
        
        biogas_data = global_vars['biogas_segment_up']['SmoothedValueNum']
        substrate_data = global_vars['substrate_segment_up']['FeedingRate']
        timestamps = global_vars['timestamps_up']

        feed_max = substrate_data.max()
        feed_min = substrate_data[substrate_data > 0].min()
        flowrate_max = biogas_data.max()
        flowrate_min = biogas_data.min()

        # Calculate PT1 estimation
        pt1_result = schaetzer_pt1(
            {'interp_y_smooth': biogas_data.values, 'interp_dt': timestamps},
            {'interp_diff_substrate_feed': substrate_data.values},
            feed_max, feed_min, flowrate_max, flowrate_min, 'up'
        )
        
        if pt1_result is None:
            raise ValueError("PT1 estimation failed")

        # Plot biogas production rate
        line1 = ax1.plot(timestamps, biogas_data, 
                        linestyle='--', color='#0000A7',
                        label='Biogas production rate', linewidth=1)

        # Plot model output
        line2 = ax1.plot(timestamps, pt1_result['yd'],
                        linestyle='-.', color='#EECC16',
                        label='PT1 estimator', linewidth=1)

        # Plot substrate feeding rate
        valid_substrate = substrate_data[substrate_data > 0]
        valid_timestamps = np.array(timestamps)[substrate_data > 0]
        line3 = ax2.plot(valid_timestamps, valid_substrate,
                        color='#C1272D', linewidth=1,
                        label='Substrate feeding rate')

        # Configure axes
        ax1.set_xlabel('Time', fontsize=12)
        ax1.set_ylabel('Biogas production rate [m³/h]', color='#0000A7', fontsize=12)
        ax2.set_ylabel('Substrate feeding rate [t/h]', color='#C1272D', fontsize=12)
        
        ax1.grid(True)
        ax1.xaxis.set_major_formatter(mdates.DateFormatter('%d\n%H:%M'))
        
        # Legend
        lines = line1 + line2 + line3
        labels = [l.get_label() for l in lines]
        ax1.legend(lines, labels, loc='upper left')

        # Calculate and update metrics
        metrics = calculate_metrics(biogas_data.values, pt1_result['yd'])
        update_metrics_table(tree_up, metrics)

        # Store scenario data
        global_vars["up_scenario_data"] = {
            "time": timestamps,
            "flow_measured": biogas_data,
            "model_output": pt1_result['yd'],
            "feed": substrate_data
        }
            
        fig_model_up.tight_layout()
        canvas_model_up.draw()
            
    except Exception as e:
        print(f"Error in update_model_up_plot_pt1: {e}")





def schaetzer_pt2(flowrate, feed, feed_max, feed_min, flowrate_max, flowrate_min, step_dir):
    try:
        sch_pt2 = {}
        
        # Matrix setup for PT2 model
        sch_pt2['A'] = np.column_stack((
            flowrate['interp_y_smooth'][1:-1],
            flowrate['interp_y_smooth'][:-2],
            feed['interp_diff_substrate_feed'][1:-1],
            feed['interp_diff_substrate_feed'][:-2]
        ))
        sch_pt2['b'] = flowrate['interp_y_smooth'][2:]
        sch_pt2['x'] = lsqr(sch_pt2['A'], sch_pt2['b'])[0]
        
        # Discrete domain information
        sch_pt2['Ta'] = 120
        sch_pt2['test_time'] = np.arange(0, 
                                        sch_pt2['Ta'] * len(flowrate['interp_dt']), 
                                        sch_pt2['Ta'])
        
        # Transfer function coefficients
        num = [0, sch_pt2['x'][2], sch_pt2['x'][3]]
        den = [1, -sch_pt2['x'][0], -sch_pt2['x'][1]]
        
        # Calculate step response
        if 'down' in step_dir:
            u = feed['interp_diff_substrate_feed'] - feed_max
            yout = signal.dlsim((num, den, sch_pt2['Ta']), u)[1]
            sch_pt2['yd'] = yout.flatten() + flowrate_max
        else:  # 'up' in step_dir
            u = feed['interp_diff_substrate_feed'] - feed_min
            yout = signal.dlsim((num, den, sch_pt2['Ta']), u)[1]
            sch_pt2['yd'] = yout.flatten() + flowrate_min
        
        return sch_pt2
        
    except Exception as e:
        print(f"Error in schaetzer_pt2: {str(e)}")
        return None

def update_model_down_plot_schaetzer_pt2():
    try:
        if global_vars['biogas_segment_down'] is None or global_vars['substrate_segment_down'] is None:
            return
            
        fig_model_down.clear()
        ax1 = fig_model_down.add_subplot(111)
        ax2 = ax1.twinx()
        
        biogas_data = global_vars['biogas_segment_down']['SmoothedValueNum']
        substrate_data = global_vars['substrate_segment_down']['FeedingRate']
        timestamps = global_vars['timestamps_down']

        feed_max = substrate_data.max()
        feed_min = substrate_data[substrate_data > 0].min()
        flowrate_max = biogas_data.max()
        flowrate_min = biogas_data.min()

        # Calculate PT2 estimation
        pt2_result = schaetzer_pt2(
            {'interp_y_smooth': biogas_data.values, 'interp_dt': timestamps},
            {'interp_diff_substrate_feed': substrate_data.values},
            feed_max, feed_min, flowrate_max, flowrate_min, 'down'
        )
        
        if pt2_result is None:
            raise ValueError("PT2 estimation failed")

        # Plot biogas production rate
        line1 = ax1.plot(timestamps, biogas_data, 
                        linestyle='--', color='#0000A7',
                        label='Biogas production rate', linewidth=1)

        # Plot model output
        line2 = ax1.plot(timestamps, pt2_result['yd'],
                        linestyle='-.', color='#EECC16',
                        label='PT2 estimator', linewidth=1)

        # Plot substrate feeding rate
        valid_substrate = substrate_data[substrate_data > 0]
        valid_timestamps = np.array(timestamps)[substrate_data > 0]
        line3 = ax2.plot(valid_timestamps, valid_substrate,
                        color='#C1272D', linewidth=1,
                        label='Substrate feeding rate')

        # Configure axes
        ax1.set_xlabel('Time', fontsize=12)
        ax1.set_ylabel('Biogas production rate [m³/h]', color='#0000A7', fontsize=12)
        ax2.set_ylabel('Substrate feeding rate [t/h]', color='#C1272D', fontsize=12)
        
        ax1.grid(True)
        ax1.xaxis.set_major_formatter(mdates.DateFormatter('%d\n%H:%M'))
        
        # Legend
        lines = line1 + line2 + line3
        labels = [l.get_label() for l in lines]
        ax1.legend(lines, labels, loc='upper left')

        # Calculate and update metrics
        metrics = calculate_metrics(biogas_data.values, pt2_result['yd'])
        update_metrics_table(tree_down, metrics)

        # Store scenario data
        global_vars["down_scenario_data"] = {
            "time": timestamps,
            "flow_measured": biogas_data,
            "model_output": pt2_result['yd'],
            "feed": substrate_data
        }
            
        fig_model_down.tight_layout()
        canvas_model_down.draw()
            
    except Exception as e:
        print(f"Error in update_model_down_plot_pt2: {e}")

def update_model_up_plot_schaetzer_pt2():
    try:
        if global_vars['biogas_segment_up'] is None or global_vars['substrate_segment_up'] is None:
            return
            
        fig_model_up.clear()
        ax1 = fig_model_up.add_subplot(111)
        ax2 = ax1.twinx()
        
        biogas_data = global_vars['biogas_segment_up']['SmoothedValueNum']
        substrate_data = global_vars['substrate_segment_up']['FeedingRate']
        timestamps = global_vars['timestamps_up']

        feed_max = substrate_data.max()
        feed_min = substrate_data[substrate_data > 0].min()
        flowrate_max = biogas_data.max()
        flowrate_min = biogas_data.min()

        # Calculate PT2 estimation
        pt2_result = schaetzer_pt2(
            {'interp_y_smooth': biogas_data.values, 'interp_dt': timestamps},
            {'interp_diff_substrate_feed': substrate_data.values},
            feed_max, feed_min, flowrate_max, flowrate_min, 'up'
        )
        
        if pt2_result is None:
            raise ValueError("PT2 estimation failed")

        # Plot biogas production rate
        line1 = ax1.plot(timestamps, biogas_data, 
                        linestyle='--', color='#0000A7',
                        label='Biogas production rate', linewidth=1)

        # Plot model output
        line2 = ax1.plot(timestamps, pt2_result['yd'],
                        linestyle='-.', color='#EECC16',
                        label='PT2 estimator', linewidth=1)

        # Plot substrate feeding rate
        valid_substrate = substrate_data[substrate_data > 0]
        valid_timestamps = np.array(timestamps)[substrate_data > 0]
        line3 = ax2.plot(valid_timestamps, valid_substrate,
                        color='#C1272D', linewidth=1,
                        label='Substrate feeding rate')

        # Configure axes
        ax1.set_xlabel('Time', fontsize=12)
        ax1.set_ylabel('Biogas production rate [m³/h]', color='#0000A7', fontsize=12)
        ax2.set_ylabel('Substrate feeding rate [t/h]', color='#C1272D', fontsize=12)
        
        ax1.grid(True)
        ax1.xaxis.set_major_formatter(mdates.DateFormatter('%d\n%H:%M'))
        
        # Legend
        lines = line1 + line2 + line3
        labels = [l.get_label() for l in lines]
        ax1.legend(lines, labels, loc='upper left')

        # Calculate and update metrics
        metrics = calculate_metrics(biogas_data.values, pt2_result['yd'])
        update_metrics_table(tree_up, metrics)

        # Store scenario data
        global_vars["up_scenario_data"] = {
            "time": timestamps,
            "flow_measured": biogas_data,
            "model_output": pt2_result['yd'],
            "feed": substrate_data
        }
            
        fig_model_up.tight_layout()
        canvas_model_up.draw()
            
    except Exception as e:
        print(f"Error in update_model_up_plot_pt2: {e}")







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
                
        elif model == "model2":  # Pt1_Modell   
            update_model_down_plot_pt1()
            update_model_up_plot_pt1()

        elif model == "model3":  # Time constant sum
            update_model_down_plot_zeitkonstantesumme()
            update_model_up_plot_zeitkonstantesumme()

        elif model == "model4":  # Turning tangent
            update_model_down_plot_wendetangente()
            update_model_up_plot_wendetangente()
            
        elif model == "model5":  # Pt1 estimator
            # Implement Pt1 estimator calculation
            update_model_down_plot_schaetzer_pt1()
            update_model_up_plot_schaetzer_pt1()
            
        elif model == "model6":  # Pt2 estimator
            update_model_down_plot_schaetzer_pt2()
            update_model_up_plot_schaetzer_pt2()

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
       2. Switch to the 'Control System' tab.
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

from scipy.signal import place_poles
import scipy.linalg
from control import acker
import control as ctrl
print("All libraries imported successfully!")
 

# === Simulation Setup ===
sampling_time = 7200  # Sampling time (2 hours = 7200 seconds)
simulation_duration = int(float(duration_field.get()) * 60 * 60 * 24)  # Convert days to seconds
time_steps = simulation_duration // sampling_time  # Number of discrete time steps

# Initialize State Vectors
x = np.zeros((2, time_steps))  # State vector
x[:, 0] = np.array([float(gas_flow_initial_field.get()), 0])  # Initial state

# Initialize Control Input
u = np.zeros(time_steps)
u_min = float(feed_min_field.get())  # Minimum substrate feeding rate
u_max = float(feed_max_field.get())  # Maximum substrate feeding rate
u[0] = np.clip(float(substrate_feed_initial_field.get()), u_min, u_max)  # Initial feeding rate

# Initialize Output and Reference Signals
y = np.zeros(time_steps)  # Output initialization
r = np.full(time_steps, float(gas_flow_setpoint_field.get()))  # Reference setpoint

# === Retrieve A, B, and C from the Upward Scenario Model ===
# Assume result is obtained from `calculate_zeitprozentkennwert()` for upward case
def initialize_control_system():
    """
    Initialize the control system components using the processed biogas data.
    Returns a dictionary containing all necessary control system parameters and models.
    """
    try:
        # Check if data is available
        if global_vars.get('biogas_segment_up') is None:
            print("No data available. Please load and process data first.")
            return None

        # Get the process data from global variables
        biogas_data = global_vars['biogas_segment_up']['SmoothedValueNum']
        substrate_data = global_vars['substrate_segment_up']['FeedingRate']
        timestamps = global_vars['timestamps_up']

        # Calculate step response parameters
        step_index = np.argmax(np.diff(substrate_data.values))
        feed_max = substrate_data.max()
        feed_min = substrate_data[substrate_data > 0].min()
        flowrate_max = biogas_data.max()
        flowrate_min = biogas_data.min()

        # Calculate model parameters using zeitprozentkennwert method
        result = calculate_zeitprozentkennwert(
            biogas_data.values,
            substrate_data.values,
            step_index,
            feed_max,
            feed_min,
            flowrate_max,
            flowrate_min,
            timestamps,
            step_direction="up"
        )

        if result is None:
            raise ValueError("Failed to calculate system parameters")

        # Extract transfer function coefficients
        coeffs = result["transfer_function_coeffs"]
        e, f, c, d = coeffs["e"], coeffs["f"], coeffs["c"], coeffs["d"]

        # System parameters
        Ts_2h = 7200  # Sampling time (2 hours in seconds)

        # Create continuous-time transfer function
        # Use T1 as the time constant since it's the primary time constant
        num = [result["parameters"]["K"]]  # Numerator coefficients
        den = [result["parameters"]["T1"]**2, 2*result["parameters"]["T1"], 1]  # Denominator coefficients
        
        # Create transfer function model
        system_model_up_Gd = ctrl.TransferFunction(num, den)
        
        # Discretize the system
        G_2h = ctrl.sample_system(system_model_up_Gd, Ts_2h, method='zoh')
        
        # Convert to state-space
        G_ss_PT2 = ctrl.ss(G_2h)

        # Define state-space matrices
        G_ss_PT2.A = np.array([
            [-c, -d],
            [1, 0]
        ])

        G_ss_PT2.B = np.array([
            [e],
            [f]
        ])

        G_ss_PT2.C = np.array([[1, 0]])
        G_ss_PT2.D = np.array([[0]])

        # Place poles exactly at origin
        # p_ss_PT2 = [0, 0] 
        # K_ss_PT2 = ctrl.acker(G_ss_PT2.A, G_ss_PT2.B, p_ss_PT2)

        p_ss_PT2 = [0.5, 0.6]  # Small positive values close to zero
        K_ss_PT2 = ctrl.acker(G_ss_PT2.A, G_ss_PT2.B, p_ss_PT2)

        # # Calculate pre-filter using a numerically stable approach
        # try:
        #     BK_minus_A = G_ss_PT2.B @ K_ss_PT2 - G_ss_PT2.A
        #     inverse_term = np.linalg.inv(BK_minus_A)
        #     f_ss_PT2 = np.linalg.inv(G_ss_PT2.C @ inverse_term @ G_ss_PT2.B)
        # except np.linalg.LinAlgError:
        #     # Fallback calculation if matrix is near-singular
        #     f_ss_PT2 = 1 / (G_ss_PT2.C @ np.linalg.inv(G_ss_PT2.A - G_ss_PT2.B @ K_ss_PT2) @ G_ss_PT2.B)
        # Calculate pre-filter using MATLAB's approach
        try:
            # Create identity matrix of appropriate size
            I = np.eye(len(K_ss_PT2))
            
            # Calculate (I - A + B*K) instead of (B*K - A)
            I_minus_A_plus_BK = I - G_ss_PT2.A + G_ss_PT2.B @ K_ss_PT2 
            
            # Calculate inverse
            inverse_term = np.linalg.inv(I_minus_A_plus_BK)
            
            # Calculate final pre-filter value
            f_ss_PT2 = np.linalg.inv(G_ss_PT2.C @ inverse_term @ G_ss_PT2.B)
            
        except np.linalg.LinAlgError:
            # Fallback calculation if matrix is near-singular
            # Also using MATLAB's formula approach
            f_ss_PT2 = 1 / (G_ss_PT2.C @ np.linalg.inv(np.eye(len(K_ss_PT2)) - 
                                                    G_ss_PT2.A + 
                                                    G_ss_PT2.B @ K_ss_PT2) @ G_ss_PT2.B)


        # Store operating points
        operating_points = {
            'feed_max': feed_max,
            'feed_min': feed_min,
            'flowrate_max': flowrate_max,
            'flowrate_min': flowrate_min
        }

        # Print system information
        print("\nControl System Parameters:")
        print(f"Sampling Time (Ts): {Ts_2h} seconds")
        print(f"State-Space A matrix:\n{G_ss_PT2.A}")
        print(f"State-Space B matrix:\n{G_ss_PT2.B}")
        print(f"State-Space C matrix:\n{G_ss_PT2.C}")
        print(f"Controller gain K:\n{K_ss_PT2}")
        print(f"Pre-filter gain f:\n{f_ss_PT2}")
        print(f"Operating points:\n{operating_points}")

        # Return all necessary components
        return {
            'G_ss_PT2': G_ss_PT2,
            'K_ss_PT2': K_ss_PT2,
            'f_ss_PT2': f_ss_PT2,
            'Ts': Ts_2h,
            'operating_points': operating_points,
            'model_parameters': result["parameters"]
        }

    except Exception as e:
        print(f"Error initializing control system: {e}")
        traceback.print_exc()
        return None

# === Simulation Function ===
def simulate_feedback_system():
    try:
        control_system = initialize_control_system()
        if control_system is None:
            messagebox.showerror("Error", "Please load and process data first")
            return None, None, None

        # Get initial conditions and setpoints
        initial_gas = float(gas_flow_initial_field.get())    # e.g. 83 m³/h
        setpoint = float(gas_flow_setpoint_field.get())      # e.g. 95 m³/h
        initial_feed = float(substrate_feed_initial_field.get())  # e.g. 0.42 t/2h
        feed_min = float(feed_min_field.get())   # e.g. 0.4025 t/2h            
        feed_max = float(feed_max_field.get())   # e.g. 0.8 t/2h            

        # Initialize state vector at initial operating point
        x = np.zeros((2, time_steps))
        x[0, 0] = initial_gas
        x[1, 0] = 0  # Second state starts at 0

        # Initialize output arrays at initial operating point
        y = np.zeros(time_steps)
        y[0] = initial_gas  # Start at initial gas flow
        u = np.zeros(time_steps)
        u[0] = initial_feed  # Start at initial feed rate
        
        # Setpoint trajectory starting at initial value
        r = np.full(time_steps, initial_gas)
        step_index = int(time_steps * 0.1)  # Step at 10% of simulation
        r[step_index:] = setpoint

        # Get system matrices
        A = control_system['G_ss_PT2'].A
        B = control_system['G_ss_PT2'].B.flatten()
        C = control_system['G_ss_PT2'].C
        K = control_system['K_ss_PT2']
        f = control_system['f_ss_PT2'].item()

        # Run simulation
        for k in range(time_steps - 1):
            # Calculate control input
            error = r[k] - y[k]  # Error signal
            state_feedback = np.dot(K, x[:, k]).item()
            u_unsat = state_feedback + f * error  # Positive feedback with error
            u[k] = np.clip(float(u_unsat), feed_min, feed_max)
            
            # Update state
            x_next = np.dot(A, x[:,k]) + B * u[k]
            x[:,k+1] = x_next
            
            # Calculate output
            y[k+1] = np.dot(C, x[:,k+1]).item()

            # Ensure output stays positive
            y[k+1] = max(0, y[k+1])

        return y, u, r

    except Exception as e:
        print(f"Error in simulation: {e}")
        traceback.print_exc()
        return None, None, None

def plot_regler_simulink_anti_windup_ui(i, ui_axes, start_date, out, t_stop, Ta):
    try:
        ui_axes.clear()
        
        # Time points for x-axis
        start_datetime = pd.to_datetime(start_date)
        time_points = pd.date_range(start=start_datetime, 
                                  periods=len(out['mit_antiwindup_y_modif']), 
                                  freq=f'{Ta}h')

        # Gas production y-axis (left)
        initial_gas = float(gas_flow_initial_field.get())
        setpoint = float(gas_flow_setpoint_field.get())
        y_min = min(initial_gas - 2, min(out['mit_antiwindup_y_modif']) - 2)
        y_max = max(setpoint + 2, max(out['mit_antiwindup_y_modif']) + 2)
        
        ui_axes.set_ylim([y_min, y_max])
        ui_axes.grid(True, linestyle='--', alpha=0.7)
        
        # Plot gas production and setpoint
        line1 = ui_axes.plot(time_points, out['mit_antiwindup_y_modif'],
                           color='#0072BD', linewidth=2,
                           label='Simulated Gas production flow rate')
        
        line2 = ui_axes.plot(time_points, out['y_soll'],
                           color='black', linewidth=1.5, linestyle=':',
                           label='Setpoint')
        
        ui_axes.set_ylabel('Gas production flow rate [m³/h]', color='black')
        ui_axes.tick_params(axis='y', labelcolor='black')

        # Substrate feeding y-axis (right)
        ax2 = ui_axes.twinx()
        feed_min = float(feed_min_field.get())
        feed_max = float(feed_max_field.get())
        ax2.set_ylim([feed_min - 0.05, feed_max + 0.05])
        
        line3 = ax2.plot(time_points, out['mit_antiwindup_u_modif'],
                        color='#D95319', linewidth=2,
                        label='Substrate feeding rate')
        
        ax2.set_ylabel(f'Substrate feeding rate [t/{Ta}h]', color='#D95319')
        ax2.tick_params(axis='y', labelcolor='#D95319')
        
        # X-axis formatting
        ui_axes.set_xlabel('Time')
        ui_axes.xaxis.set_major_formatter(mdates.DateFormatter('%b %d, %H:%M'))
        ui_axes.xaxis.set_major_locator(mdates.AutoDateLocator())
        plt.setp(ui_axes.xaxis.get_majorticklabels(), rotation=0)
        
        # Title and legend
        ui_axes.set_title('Discrete PI-Controller (+ Root Locus) with Anti-Windup', pad=20)
        lines = line1 + line2 + line3
        labels = [l.get_label() for l in lines]
        ui_axes.legend(lines, labels, loc='lower right', fontsize=9)
        
        plt.tight_layout()
        canvas.draw()
        
    except Exception as e:
        print(f"Error in plotting: {e}")
        traceback.print_exc()

# === Start Simulation ===
def start_simulation():
    try:
        y_out, u_out, r_out = simulate_feedback_system()
        
        if y_out is not None:
            out = {
                'mit_antiwindup_y_modif': y_out,
                'y_soll': r_out,
                'mit_antiwindup_u_modif': u_out
            }
            
            start_date = start_date_field.get_date()
            plot_regler_simulink_anti_windup_ui(1, ax, start_date, 
                                              out, simulation_duration, 2)
            
    except Exception as e:
        messagebox.showerror("Error", f"Simulation failed: {str(e)}")
        print(f"Error in start_simulation: {e}")
        traceback.print_exc()

# Connect Start Button to Simulation
start_button = ttk.Button(button_frame, text="Start Simulation", command=start_simulation)
start_button.pack(side='left', padx=10)

# Run Tkinter Main Loop
root.mainloop()  