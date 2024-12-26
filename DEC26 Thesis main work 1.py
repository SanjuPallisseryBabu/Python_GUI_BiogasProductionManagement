import datetime
import tkinter as tk
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

            # Get the year from the CSV files and update calendar widgets
            csv_year = pd.to_datetime(gas_flow_rate_df['TimeStamp'].iloc[0]).year
            today = datetime.now()
            csv_date = datetime(csv_year, today.month, today.day)
            
            # Update all DateEntry widgets with the CSV year
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

##########################################################################################
#PREPROCESSING TAB
##########################################################################################

# Smoothing Control Variables
SMOOTHING_LEVEL = 100  # Fixed smoothing level

# Function to change to the next tab (in this case, from Load to Preprocess)
def next_tab():
    notebook.select(preprocess_tab)

# Function to apply smoothing to the data
def apply_smoothing(data, level):
    # Apply uniform filter with a fixed window size of 100
    return uniform_filter1d(data, size=level)

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

# Update the preprocessing button function to handle smoothing immediately
def preprocessing_button_pushed():
    global is_smoothed_down, is_smoothed_up
    is_smoothed_down = True
    is_smoothed_up = True
    update_step_downward_plot(smoothed=True)
    update_step_upward_plot(smoothed=True)
    next_tab()  # Switch to Preprocess tab

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

###########################

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

    # Clear the existing plot
    fig_down.clear()

    # Create primary and secondary axes
    ax1_down = fig_down.add_subplot(111)
    ax2_down = ax1_down.twinx()

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

        # Apply slider values to refine the filtered data
        data_length = len(biogas_filtered)
        start_ind = round(slider_down_start.get() / 100 * data_length)
        end_ind = round(slider_down_end.get() / 100 * data_length)
        biogas_segment = biogas_filtered.iloc[start_ind:end_ind]
        substrate_segment = substrate_filtered.iloc[start_ind:end_ind]

        # Apply smoothing if required
        value_col = 'ValueNum'
        if is_smoothed_down:
            biogas_segment['SmoothedValueNum'] = apply_smoothing(biogas_segment[value_col], SMOOTHING_LEVEL)
            substrate_segment['SmoothedValueNum'] = apply_smoothing(substrate_segment[value_col], SMOOTHING_LEVEL)
            value_col = 'SmoothedValueNum'

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

        # Plot Biogas Production Rate
        biogas_plot = ax1_down.step(
            biogas_segment['TimeStamp'],
            biogas_segment[value_col],
            where="post",
            color='blue', linestyle='--', label='Biogas Production Rate'
        )
        ax1_down.set_xlabel("Time")
        ax1_down.set_ylabel("Biogas Production Rate [m³/h]", color='blue')
        ax1_down.tick_params(axis="y", labelcolor='blue')

        # Set y-axis intervals for Biogas Production Rate
        ax1_down.set_ylim(
            biogas_segment[value_col].min() - 5,
            biogas_segment[value_col].max() + 5
        )
        ax1_down.yaxis.set_major_locator(plt.MultipleLocator(5))
        ax1_down.yaxis.set_minor_locator(plt.MultipleLocator(1))
        ax1_down.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{int(x)}"))

        # Filter and plot substrate data
        processed_substrate = substrate_segment[substrate_segment['FeedingRate'] > 0]
        
        if not processed_substrate.empty:
            substrate_min = processed_substrate['FeedingRate'].min()
            substrate_max = processed_substrate['FeedingRate'].max()
            
            y_min = 0
            y_max = substrate_max + 0.1
            ax2_down.set_ylim(y_min, y_max)

            tick_interval = 1 if y_max - y_min > 1 else 0.5
            ax2_down.yaxis.set_major_locator(plt.MultipleLocator(tick_interval))
            ax2_down.yaxis.set_major_formatter(
                plt.FuncFormatter(lambda x, _: f"{x:.1f}" if tick_interval < 1 else f"{int(x)}")
            )
            
            substrate_plot = ax2_down.step(
                processed_substrate['TimeStamp'],
                processed_substrate['FeedingRate'],
                where="post",
                color='red',
                label='Substrate Feeding Rate'
            )
        else:
            ax2_down.set_ylim(0, 1)

        ax2_down.set_ylabel("Substrate Feeding Rate [t/h]", color='red')
        ax2_down.tick_params(axis="y", labelcolor='red', labelsize=8)

        # Format time axis
        ax1_down.xaxis.set_major_formatter(mdates.DateFormatter('%d\n%H:%M'))
        plt.setp(ax1_down.xaxis.get_majorticklabels(), rotation=0)

        # Add month and year to last tick
        def add_month_year_to_last_tick(ax):
            ticks = ax.get_xticks()
            if len(ticks) > 0:
                ticks_as_dates = [mdates.num2date(tick) for tick in ticks]
                labels = [tick.strftime('%d\n%H:%M') for tick in ticks_as_dates]
                labels[-1] += f"\n{ticks_as_dates[-1].strftime('%b %Y')}"
                ax.set_xticklabels(labels, ha='center')

        add_month_year_to_last_tick(ax1_down)

        # Adjust layout and add legend
        fig_down.subplots_adjust(left=0.1, right=0.85)
        if 'substrate_plot' in locals():
            ax1_down.legend(handles=[biogas_plot[0], substrate_plot[0]], loc='upper left', fontsize=9)
        else:
            ax1_down.legend(handles=[biogas_plot[0]], loc='upper left', fontsize=9)

        # Add grid and title
        ax1_down.grid(True)
        ax1_down.set_title("After Data Preprocessing", fontsize=10, fontweight='bold')
        fig_down.tight_layout(pad=2)
        canvas_down.draw()

    except Exception as e:
        print(f"Error during graph preprocessing and plotting: {e}")

def update_step_upward_plot(smoothed=False):
    """Update the step up plot with preprocessed data"""
    global is_smoothed_up
    if smoothed:
        is_smoothed_up = True

    # Clear the existing plot
    fig_up.clear()

    # Create primary and secondary axes
    ax1_up = fig_up.add_subplot(111)
    ax2_up = ax1_up.twinx()

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

        # Apply slider values to refine the filtered data
        data_length = len(biogas_filtered)
        start_ind = round(slider_up_start.get() / 100 * data_length)
        end_ind = round(slider_up_end.get() / 100 * data_length)
        biogas_segment = biogas_filtered.iloc[start_ind:end_ind]
        substrate_segment = substrate_filtered.iloc[start_ind:end_ind]

        # Apply smoothing if required
        value_col = 'ValueNum'
        if is_smoothed_up:
            biogas_segment['SmoothedValueNum'] = apply_smoothing(biogas_segment[value_col], SMOOTHING_LEVEL)
            substrate_segment['SmoothedValueNum'] = apply_smoothing(substrate_segment[value_col], SMOOTHING_LEVEL)
            value_col = 'SmoothedValueNum'

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
        
        # Plot Biogas Production Rate
        biogas_plot = ax1_up.step(
            biogas_segment['TimeStamp'],
            biogas_segment[value_col],
            where="post",
            color='blue', linestyle='--', label='Biogas Production Rate'
        )
        ax1_up.set_xlabel("Time")
        ax1_up.set_ylabel("Biogas Production Rate [m³/h]", color='blue')
        ax1_up.tick_params(axis="y", labelcolor='blue')

        # Set y-axis intervals
        ax1_up.set_ylim(
            biogas_segment[value_col].min() - 5,
            biogas_segment[value_col].max() + 5
        )
        ax1_up.yaxis.set_major_locator(plt.MultipleLocator(5))
        ax1_up.yaxis.set_minor_locator(plt.MultipleLocator(1))
        ax1_up.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{int(x)}"))

        # Filter and plot substrate data
        processed_substrate = substrate_segment[substrate_segment['FeedingRate'] > 0]
        
        if not processed_substrate.empty:
            substrate_min = processed_substrate['FeedingRate'].min()
            substrate_max = processed_substrate['FeedingRate'].max()
            
            y_min = 0
            y_max = substrate_max + 0.1
            ax2_up.set_ylim(y_min, y_max)

            tick_interval = 1 if y_max - y_min > 1 else 0.5
            ax2_up.yaxis.set_major_locator(plt.MultipleLocator(tick_interval))
            ax2_up.yaxis.set_major_formatter(
                plt.FuncFormatter(lambda x, _: f"{x:.1f}" if tick_interval < 1 else f"{int(x)}")
            )
            
            substrate_plot = ax2_up.step(
                processed_substrate['TimeStamp'],
                processed_substrate['FeedingRate'],
                where="post",
                color='red',
                label='Substrate Feeding Rate'
            )
        else:
            ax2_up.set_ylim(0, 1)

        ax2_up.set_ylabel("Substrate Feeding Rate [t/h]", color='red')
        ax2_up.tick_params(axis="y", labelcolor='red', labelsize=8)

        # Format time axis
        ax1_up.xaxis.set_major_formatter(mdates.DateFormatter('%d\n%H:%M'))
        plt.setp(ax1_up.xaxis.get_majorticklabels(), rotation=0)

        # Adjust layout and add legend
        fig_up.subplots_adjust(left=0.1, right=0.85)
        if 'substrate_plot' in locals():
            ax1_up.legend(handles=[biogas_plot[0], substrate_plot[0]], loc='upper left', fontsize=9)
        else:
            ax1_up.legend(handles=[biogas_plot[0]], loc='upper left', fontsize=9)

        # Add grid and title
        ax1_up.grid(True)
        ax1_up.set_title("After Data Preprocessing", fontsize=10, fontweight='bold')
        fig_up.tight_layout(pad=2)
        canvas_up.draw()

    except Exception as e:
        print(f"Error during graph preprocessing and plotting: {e}")

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


###################Model Estimation

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

# Create frames for metrics tables
metrics_frame_down = ttk.Frame(down_model_frame)
metrics_frame_down.grid(row=1, column=0, padx=5, pady=5, sticky="nsew")

metrics_frame_up = ttk.Frame(up_model_frame)
metrics_frame_up.grid(row=1, column=0, padx=5, pady=5, sticky="nsew")


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
    ("Pt1_Modell", "model1"),
    ("Time percentage", "model2"),
    ("Turning tangent", "model3"),
    ("Time constant sum", "model4"),
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



select_model_button = tk.Button(
    select_model_frame,
    text="Select model",
    font=("Arial", 10),
    command=lambda: select_model()
)
select_model_button.grid(row=1, column=0, padx=10, pady=5)



# Add hover effects for buttons
def on_enter_model(event):
    event.widget['background'] = '#d0eaff'  # Light blue color on hover

def on_leave_model(event):
    event.widget['background'] = 'SystemButtonFace'  # Default button color

# Bind hover effects to buttons
select_model_button.bind("<Enter>", on_enter_model)
select_model_button.bind("<Leave>", on_leave_model)

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
            else initial_value + 0.72 * (final_value - initial_value)
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
            "Gd": Gd,  # Discretized transfer function
        }

    except Exception as e:
        print(f"Error in calculate_zeitprozentkennwert: {e}")
        return None


def calculate_metrics(measured, modeled):
    """Calculate similarity metrics between measured and modeled data"""
    try:
        metrics = {}
        
        # R² value
        metrics['R2'] = r2_score(measured, modeled)
        
        # Correlation coefficients
        metrics['Pearson'] = np.corrcoef(measured, modeled)[0, 1]
        metrics['Spearman'] = stats.spearmanr(measured, modeled)[0]
        metrics['Kendall'] = stats.kendalltau(measured, modeled)[0]
        
        # Distance metrics
        metrics['Euclidean'] = np.sqrt(np.sum((measured - modeled) ** 2))
        metrics['Chebyshev'] = np.max(np.abs(measured - modeled))
        metrics['Cosine'] = 1 - np.dot(measured, modeled) / (np.linalg.norm(measured) * np.linalg.norm(modeled))
        
        return metrics
        
    except Exception as e:
        print(f"Error in calculate_metrics: {e}")
        return None

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

def update_model_down_plot():
    """
    Update the step down plot with time percentage method.
    Includes enhanced point plotting and markers from the original implementation.
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
        feed_max = substrate_data.max()
        feed_min = substrate_data.min()
        flowrate_max = biogas_data.max()
        flowrate_min = biogas_data.min()
        ind_sprung = np.argmax(np.diff(substrate_data.values))
        
        # Calculate margin for y-axis limits
        biogas_margin = 0.05 * (flowrate_max - flowrate_min)
        substrate_margin = 0.05 * (feed_max - feed_min)
        
        # Set y-axis limits with margins
        ax1.set_ylim([flowrate_min - biogas_margin, flowrate_max + biogas_margin])
        ax2.set_ylim([feed_min - substrate_margin, feed_max + substrate_margin])
        
        # Calculate zeitprozentkennwert parameters
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
        
        if zk_down is not None:
            # Setup axes properties
            ax1.grid(True)
            ax1.xaxis.set_minor_locator(AutoMinorLocator())
            ax1.yaxis.set_minor_locator(AutoMinorLocator())
            
            # Plot biogas production rate
            line1 = ax1.plot(timestamps, biogas_data, 
                           linestyle='--',
                           color=color_biogas,
                           label='Biogas production rate',
                           linewidth=const['linienbreite'])
            
            # Plot time percentage method result
            line2 = ax1.plot(timestamps, zk_down['model_output'],
                           linestyle='-.',
                           color=color_model,
                           label='Time percentage method',
                           linewidth=const['linienbreite'])
            
            # Plot substrate feeding rate
            line3 = ax2.plot(timestamps, substrate_data,
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
            
            # Adjust layout and draw
            fig_model_down.tight_layout()
            canvas_model_down.draw()
            
    except Exception as e:
        print(f"Error in update_model_down_plot: {e}")


def update_model_up_plot():
    """
    Update the step up plot with time percentage method.
    Includes enhanced point plotting and markers from the original implementation.
    """
    try:
        # Check if we have preprocessed data
        if global_vars['biogas_segment_up'] is None or global_vars['substrate_segment_up'] is None:
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

        # Define colors with enhanced organization
        colors = {
            'biogas_line': '#0000A7',      # Deep blue for biogas line
            'zeitprozent': '#EECC16',      # Yellow for time percentage line
            'substrate': '#C1272D',        # Red for substrate line
            'grid': '#E6E6E6',            # Light gray for grid
            'markers': [0.8, 0.8, 0.8]    # Gray for markers
        }
            
        fig_model_up.clear()
        ax1 = fig_model_up.add_subplot(111)
        ax2 = ax1.twinx()
        
        # Get preprocessed data
        biogas_data = global_vars['biogas_segment_up']['SmoothedValueNum']
        substrate_data = global_vars['substrate_segment_up']['FeedingRate']
        timestamps = global_vars['timestamps_up']
        feed_max = substrate_data.max()
        feed_min = substrate_data.min()
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
        
        if zk_up is not None:
            # Setup enhanced axes properties
            ax1.grid(True, color=colors['grid'], linestyle='-', alpha=0.2)
            ax1.xaxis.set_minor_locator(AutoMinorLocator())
            ax1.yaxis.set_minor_locator(AutoMinorLocator())
            ax1.set_ylim([y1_min, y1_max])
            
            # Configure primary axis (ax1)
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

            # Configure secondary axis (ax2) with enhanced styling
            ax2.set_ylim([y2_min, y2_max])
            ax2.spines['right'].set_color(colors['substrate'])
            ax2.tick_params(axis='y', colors=colors['substrate'])
            ax2.yaxis.set_minor_locator(AutoMinorLocator())
            ax2.set_ylabel('Substrate feeding rate [t/h]',
                          fontname=const['font'],
                          fontsize=const['fontsize'],
                          color=colors['substrate'])
            
            # Plot substrate feeding rate
            line3 = ax2.plot(timestamps, substrate_data,
                           color=colors['substrate'],
                           linewidth=const['linienbreite'],
                           label='Substrate feeding rate')
            
            # Enhanced time axis formatting
            ax1.xaxis.set_major_formatter(mdates.DateFormatter('%d\n%H:%M'))
            plt.setp(ax1.xaxis.get_majorticklabels(), 
                    rotation=0,
                    fontname=const['font'],
                    fontsize=const['fontsizeticks'])
            
            # Add legend with enhanced styling
            lines = line1 + line2 + line3
            labels = [l.get_label() for l in lines]
            ax1.legend(lines, labels,
                      loc='upper right',
                      fontsize=const['fontsizelegend'],
                      prop={'family': const['font']})
            
            def plot_characteristic_point(ax, timestamps, data, idx, value, y_min, color):
                """Helper function for plotting characteristic points with enhanced markers"""
                # Horizontal markers
                x_horiz = [timestamps[0], 
                          timestamps[round((1 + idx) / 2)],
                          timestamps[idx]]
                y_horiz = [data[idx]] * 3
                ax.plot(x_horiz, y_horiz,
                       color=color,
                       linewidth=const['linienbreite'],
                       linestyle=':',
                       marker='>',
                       markersize=const['marker_size'])
                
                # Vertical markers with enhanced styling
                x_vert = [timestamps[idx]] * 3
                y_vert = [y_min + 3,
                         (y_min + data[idx]) / 2,
                         data[idx]]
                ax.plot(x_vert, y_vert,
                       color=color,
                       linewidth=const['linienbreite'],
                       linestyle=':',
                       marker='v',
                       markersize=const['marker_size'])
                
                # Enhanced point marker
                ax.plot(x_vert[-1], y_vert[-1], 'o',
                       markersize=const['marker_size'],
                       markeredgecolor=color,
                       markerfacecolor=color)
                
                # Enhanced text annotation with better positioning
                ax.text(timestamps[30],
                       data[idx] + 2,
                       value,
                       fontsize=8,
                       color=color)
            
            # Plot characteristic points if available
            if 'index_bei_p_1' in zk_up and 'index_bei_p_2' in zk_up:
                points_data = [
                    (zk_up['index_bei_p_1'], '0.720'),
                    (zk_up['index_bei_p_2'], f"{zk_up['wert_fuer_p_2']:.3f}")
                ]
                
                for idx, value in points_data:
                    plot_characteristic_point(
                        ax1, timestamps, biogas_data, idx, value, 
                        y1_min, colors['markers']
                    )

            # Update metrics table
            update_metrics_table(tree_up, zk_up['metrics'])
            
            # Adjust layout and draw
            fig_model_up.tight_layout()
            canvas_model_up.draw()
            
    except Exception as e:
        print(f"Error in update_model_up_plot: {e}")



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
        
        # Time constant calculation
        pt1['critical_value'] = 0.63
        
        if 'down' in step_dir:
            pt1['y_value_at_critical'] = flowrate_max - (flowrate_max - flowrate_min) * pt1['critical_value']
            pt1['index_at_critical'] = np.where(flowrate.values < pt1['y_value_at_critical'])[0][0]
        
        if 'up' in step_dir:
            pt1['y_value_at_critical'] = flowrate_min + (flowrate_max - flowrate_min) * pt1['critical_value']
            pt1['index_at_critical'] = np.where(flowrate.values > pt1['y_value_at_critical'])[0][0]
        
        # Calculate time constant T (in seconds)
        timestamp_numeric = mdates.date2num(timestamps)
        pt1['T'] = (timestamp_numeric[pt1['index_at_critical']] - timestamp_numeric[pt1['step_index']]) * 24 * 3600
        
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
    """Update the step down plot with PT1 model with enhanced styling"""
    try:
        # Check for preprocessed data
        if global_vars['biogas_segment_down'] is None or global_vars['substrate_segment_down'] is None:
            print("No preprocessed data available. Please preprocess data first.")
            return
        

        # Get the current slider values and apply them to the data
        data_length = len(global_vars['biogas_segment_down'])
        start_ind = round(slider_down_start.get() / 100 * data_length)
        end_ind = round(slider_down_end.get() / 100 * data_length)
        
        # Apply slider range to data
        biogas_data = global_vars['biogas_segment_down']['SmoothedValueNum'].iloc[start_ind:end_ind]
        substrate_data = global_vars['substrate_segment_down']['FeedingRate'].iloc[start_ind:end_ind]
        timestamps = global_vars['timestamps_down'].iloc[start_ind:end_ind]
        
        
        # Define colors and constants (keep your existing definitions)
        new_colors = {
            'biogas': '#0000A7',      # Dark blue for biogas
            'pt1': '#EECC16',         # Yellow for PT1 model
            'substrate': '#C1272D',    # Red for substrate
            'markers': '#CCCCCC'       # Gray for markers
        }
        
        const = {
            'font': 'serif',
            'fontsize': 12,
            'fontsizelegend': 8,
            'fontsizeticks': 10,
            'line_width': 1,
            'marker_size': 4
        }

        # Calculate parameters based on filtered data
        feed_max = substrate_data.max()
        feed_min = substrate_data.min()
        flowrate_max = biogas_data.max()
        flowrate_min = biogas_data.min()
        ind_sprung = np.argmax(np.diff(substrate_data.values))

        # Calculate y-axis limits with 5% padding
        y1_min = flowrate_min - 0.05 * flowrate_max
        y1_max = flowrate_max + 0.05 * flowrate_max
        y2_min = feed_min - 0.05 * feed_max
        y2_max = feed_max + 0.05 * feed_max

        # Clear and setup figure
        fig_model_down.clear()
        ax1 = fig_model_down.add_subplot(111)
        ax2 = ax1.twinx()
        
        # Calculate PT1 model parameters
        pt1_down = pt1_model_modified(
            flowrate=biogas_data,
            feed=substrate_data,
            step_index=ind_sprung,
            feed_max=feed_max,
            feed_min=feed_min,
            flowrate_max=flowrate_max,
            flowrate_min=flowrate_min,
            timestamps=timestamps,
            step_dir='down'
        )
        
        if pt1_down is not None:
            # Configure primary y-axis (left)
            ax1.set_ylim([y1_min, y1_max])
            ax1.grid(True, which='both', linestyle=':')
            ax1.tick_params(axis='both', which='both', labelsize=const['fontsizeticks'])
            
            # Plot biogas production rate
            line1 = ax1.plot(timestamps, biogas_data, '--', 
                           color=new_colors['biogas'],
                           label='Biogas production rate', 
                           linewidth=const['line_width'])
            
            # Plot PT1 model result
            line2 = ax1.plot(timestamps, pt1_down['yd'], '-.', 
                           color=new_colors['pt1'],
                           label='PT1 model', 
                           linewidth=const['line_width'])
            
            # Configure secondary y-axis (right)
            ax2.set_ylim([y2_min, y2_max])
            ax2.spines['right'].set_color(new_colors['substrate'])
            ax2.tick_params(axis='y', colors=new_colors['substrate'])
            
            # Plot substrate feeding rate
            line3 = ax2.plot(timestamps, substrate_data, '-', 
                           color=new_colors['substrate'],
                           label='Substrate feeding rate',
                           linewidth=const['line_width'])
            
            # Set labels and title with consistent formatting
            ax1.set_xlabel('Time', fontname=const['font'], 
                         fontsize=const['fontsize'])
            ax1.set_ylabel('Biogas production rate [m³/h]', 
                         color=new_colors['biogas'],
                         fontname=const['font'], 
                         fontsize=const['fontsize'])
            ax2.set_ylabel('Substrate feeding rate [t/h]', 
                         color=new_colors['substrate'],
                         fontname=const['font'], 
                         fontsize=const['fontsize'])
            ax1.set_title('PT1 Model Approximation', 
                         fontname=const['font'], 
                         fontsize=const['fontsize'])
            
            # Format time axis
            ax1.xaxis.set_major_formatter(mdates.DateFormatter('%d\n%H:%M'))
            plt.setp(ax1.xaxis.get_majorticklabels(), 
                    rotation=0, 
                    fontname=const['font'], 
                    fontsize=const['fontsizeticks'])
            
            # Add critical point markers if available
            if 'index_bei_Tkrit' in pt1_down:
                tk_index = pt1_down['index_bei_Tkrit']
                tk_value = biogas_data.iloc[tk_index]
                
                # Add vertical and horizontal markers
                ax1.axhline(y=tk_value, 
                          color=new_colors['annotation'], 
                          linestyle=':', 
                          linewidth=const['line_width'])
                ax1.axvline(x=timestamps[tk_index], 
                          color=new_colors['annotation'], 
                          linestyle=':', 
                          linewidth=const['line_width'])
                
                # Add marker point
                ax1.plot(timestamps[tk_index], tk_value, 'o',
                        color=new_colors['annotation'],
                        markersize=const['marker_size'])
                
                # Add text annotation
                ax1.text(timestamps[30], tk_value + 2, '0.63',
                        fontsize=const['fontsizelegend'],
                        color=new_colors['annotation'])
            
            # Add legend with consistent formatting
            lines = line1 + line2 + line3
            labels = [l.get_label() for l in lines]
            ax1.legend(lines, labels, 
                      loc='upper right',
                      fontsize=const['fontsizelegend'],
                      prop={'family': const['font']})
            
            # Calculate metrics
            metrics = calculate_metrics(biogas_data.values, pt1_down['yd'].flatten())
            
        #     # Update metrics table and draw
        # if pt1_down is not None:
        #     metrics = calculate_metrics(biogas_data.values, pt1_down['yd'].flatten())
        #     update_metrics_table(tree_down, metrics)
            
            # Adjust layout and draw
            fig_model_down.tight_layout()
            canvas_model_down.draw()
            
    except Exception as e:
        print(f"Error in update_model_down_plot_pt1: {e}")
        traceback.print_exc()  # Print full traceback for debugging




def update_model_up_plot_pt1():
    """Update the step up plot with PT1 model with enhanced styling"""
    try:
        # Check for preprocessed data
        if global_vars['biogas_segment_up'] is None or global_vars['substrate_segment_up'] is None:
            print("No preprocessed data available. Please preprocess data first.")
            return
        
        # Get the current slider values
        data_length = len(global_vars['biogas_segment_up'])
        start_ind = round(slider_up_start.get() / 100 * data_length)
        end_ind = round(slider_up_end.get() / 100 * data_length)
        
        # Apply slider range to data
        biogas_data = global_vars['biogas_segment_up']['SmoothedValueNum'].iloc[start_ind:end_ind]
        substrate_data = global_vars['substrate_segment_up']['FeedingRate'].iloc[start_ind:end_ind]
        timestamps = global_vars['timestamps_up'].iloc[start_ind:end_ind]
        
        # Define colors
        newcolors = {
            'biogas': '#0000A7',      # Dark blue for biogas
            'pt1': '#EECC16',         # Yellow for PT1 model
            'substrate': '#C1272D',    # Red for substrate
            'markers': '#CCCCCC'       # Gray for markers
        }
        
        # Define constants
        const = {
            'font': 'serif',
            'fontsize': 12,
            'fontsizelegend': 8,
            'fontsizeticks': 10,
            'line_width': 1,
            'marker_size': 4,
            'marker_indices': 1
        }
        
        # Clear and setup figure
        fig_model_up.clear()
        ax1 = fig_model_up.add_subplot(111)
        ax2 = ax1.twinx()
        
        # Get preprocessed data
        biogas_data = global_vars['biogas_segment_up']['SmoothedValueNum']
        substrate_data = global_vars['substrate_segment_up']['FeedingRate']
        timestamps = global_vars['timestamps_up']
        feed_max = substrate_data.max()
        feed_min = substrate_data.min()
        flowrate_max = biogas_data.max()
        flowrate_min = biogas_data.min()
        ind_sprung = np.argmax(np.diff(substrate_data.values))
        
        # Calculate y-axis limits with 5% padding
        y1_min = flowrate_min - 0.05 * flowrate_max
        y1_max = flowrate_max + 0.05 * flowrate_max
        y2_min = feed_min - 0.05 * feed_max
        y2_max = feed_max + 0.05 * feed_max
        
        # Calculate PT1 model parameters
        pt1_up = pt1_model_modified(
            flowrate=biogas_data,
            feed=substrate_data,
            step_index=ind_sprung,
            feed_max=feed_max,
            feed_min=feed_min,
            flowrate_max=flowrate_max,
            flowrate_min=flowrate_min,
            timestamps=timestamps,
            step_dir='up'
        )
        
        if pt1_up is not None:
            # Configure primary axis (ax1)
            ax1.set_xlabel('Time', fontname=const['font'], fontsize=const['fontsize'])
            ax1.set_ylabel('Biogas production rate [m³/h]', 
                          fontname=const['font'], 
                          fontsize=const['fontsize'],
                          color=newcolors['biogas'])
            ax1.set_title('PT1-Model Approximation', 
                         fontname=const['font'], 
                         fontsize=const['fontsize'])
            
            # Set axis properties
            ax1.grid(True, which='both')
            ax1.set_ylim([y1_min, y1_max])
            ax1.tick_params(axis='both', which='both', 
                          labelsize=const['fontsizeticks'])
            
            # Plot biogas production rate
            line1 = ax1.plot(timestamps, biogas_data, '--', 
                            color=newcolors['biogas'],
                            linewidth=const['line_width'],
                            label='Biogas production rate')
            
            # Plot PT1 model result
            line2 = ax1.plot(timestamps, pt1_up['yd'], '-.', 
                            color=newcolors['pt1'],
                            linewidth=const['line_width'],
                            label='PT1-Model approximation')
            
            # Add critical point markers if available
            if 'index_bei_Tkrit' in pt1_up:
                # Horizontal marker
                x_horiz = [timestamps[0],
                          timestamps[round((1 + pt1_up['index_bei_Tkrit']) / 2)],
                          timestamps[pt1_up['index_bei_Tkrit']]]
                y_horiz = [biogas_data[pt1_up['index_bei_Tkrit']]] * 3
                ax1.plot(x_horiz, y_horiz, ':', 
                        color=newcolors['markers'],
                        linewidth=const['line_width'],
                        marker='>',
                        markersize=const['marker_size'])
                
                # Vertical marker
                x_vert = [timestamps[pt1_up['index_bei_Tkrit']]] * 3
                y_vert = [y1_min + 3,
                         (y1_min + biogas_data[pt1_up['index_bei_Tkrit']]) / 2,
                         biogas_data[pt1_up['index_bei_Tkrit']]]
                ax1.plot(x_vert, y_vert, ':', 
                        color=newcolors['markers'],
                        linewidth=const['line_width'],
                        marker='v',
                        markersize=const['marker_size'])
                
                # Add critical point and text
                ax1.plot(x_vert[-1], y_vert[-1], 'o',
                        color=newcolors['markers'],
                        markersize=const['marker_size'])
                ax1.text(timestamps[30],
                        biogas_data[pt1_up['index_bei_Tkrit']] + 2,
                        '0.63',
                        fontsize=8,
                        color=newcolors['markers'])
            
            # Configure secondary axis (ax2)
            ax2.set_ylabel('Substrate feeding rate [t/h]', 
                          fontname=const['font'],
                          fontsize=const['fontsize'],
                          color=newcolors['substrate'])
            ax2.set_ylim([y2_min, y2_max])
            
            # Plot substrate feeding rate
            line3 = ax2.plot(timestamps, substrate_data, '-',
                            color=newcolors['substrate'],
                            linewidth=const['line_width'],
                            label='Substrate feeding rate')
            
            # Format time axis
            ax1.xaxis.set_major_formatter(mdates.DateFormatter('%d\n%H:%M'))
            plt.setp(ax1.xaxis.get_majorticklabels(), rotation=0)
            
            # Configure axis colors
            ax1.tick_params(axis='y', labelcolor=newcolors['biogas'])
            ax2.tick_params(axis='y', labelcolor=newcolors['substrate'])
            
            # Add legend
            lines = line1 + line2 + line3
            labels = [l.get_label() for l in lines]
            ax1.legend(lines, labels, 
                      loc='upper right',
                      fontsize=const['fontsizelegend'],
                      frameon=True)
            
            # Calculate and update metrics
            metrics = calculate_metrics(biogas_data.values, pt1_up['yd'].flatten())
            update_metrics_table(tree_up, metrics)
            
            # Adjust layout and draw
            fig_model_up.tight_layout()
            canvas_model_up.draw()
            
    except Exception as e:
        print(f"Error in update_model_up_plot_pt1: {e}")
        raise  # Re-raise the exception for debugging



def update_model_estimation_plots():
    """Update both step up and step down plots in model estimation tab based on selected model"""
    try:
        # Get selected model
        model = selected_model.get()
        
        # Clear existing plots
        fig_model_down.clear()
        fig_model_up.clear()
        
        if model == "model1":  # Pt1_Modell
            update_model_down_plot_pt1()
            update_model_up_plot_pt1()
            
        elif model == "model2":  # Time percentage
            # Use existing time percentage calculation
            update_model_down_plot()
            update_model_up_plot()
            
        elif model == "model3":  # Turning tangent
            # Pass the required arguments to the Wendetangente functions
            print("Turning tangent selected - To be implemented")
            clear_plots()
            
            
        elif model == "model4":  # Time constant sum
            # Implement time constant sum calculation
            print("Time constant sum selected - To be implemented")
            clear_plots()
           
            
        elif model == "model5":  # Pt1 estimator
            # Implement Pt1 estimator calculation
            print("Pt1 estimator selected - To be implemented")
            clear_plots()
            
        elif model == "model6":  # Pt2 estimator
            # Implement Pt2 estimator calculation
            print("Pt2 estimator selected - To be implemented")
            clear_plots()

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

# Modify select_model function
def select_model():
    """Handle model selection button click"""
    try:
        model = selected_model.get()
        print(f"Selected model: {model}")
        update_model_estimation_plots()
    except Exception as e:
        print(f"Error in select_model: {e}")
        traceback.print_exc()

# Call clear_plots initially to start with blank plots
clear_plots()

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

def select_model():
    """Handle model selection button click"""
    try:
        # Update plots with current model parameters
        update_model_estimation_plots()
    except Exception as e:
        print(f"Error in select_model: {e}")

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

# Run Tkinter Main Loop
root.mainloop()