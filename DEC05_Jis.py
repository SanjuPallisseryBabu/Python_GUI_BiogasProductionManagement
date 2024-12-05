import datetime
import tkinter as tk
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import matplotlib.dates as mdates
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


# Assuming you have already set up your Tkinter window and imported necessary libraries...

# Smoothing Control Variables
SMOOTHING_LEVEL = 100  # Fixed smoothing level

# Function to change to the next tab (in this case, from Load to Preprocess)
def next_tab():
    notebook.select(preprocess_tab)

# Function to apply smoothing to the data
def apply_smoothing(data, level):
    # Apply uniform filter with a fixed window size of 100
    return uniform_filter1d(data, size=level)

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

# Matplotlib Figures for Step Downwards and Step Upwards
fig_down, ax_down = plt.subplots()
fig_up, ax_up = plt.subplots()

# Embed the Matplotlib figures in the Tkinter frames
canvas_down = FigureCanvasTkAgg(fig_down, master=step_down_frame)
canvas_down.get_tk_widget().grid(row=0, column=0, padx=10, pady=10, sticky="nsew")

canvas_up = FigureCanvasTkAgg(fig_up, master=step_up_frame)
canvas_up.get_tk_widget().grid(row=0, column=0, padx=10, pady=10, sticky="nsew")


# Sliders for Step Downwards Section
slider_down_start = tk.Scale(step_down_frame, from_=0, to=100, orient="horizontal", label="Step Downwards Start", command=lambda val: update_step_downward_plot())
slider_down_start.grid(row=1, column=0, padx=5, pady=5, sticky="ew")

slider_down_end = tk.Scale(step_down_frame, from_=0, to=100, orient="horizontal", label="Step Downwards End", command=lambda val: update_step_downward_plot())
slider_down_end.set(100)  # Set initial value to 100
slider_down_end.grid(row=2, column=0, padx=5, pady=5, sticky="ew")

# Sliders for Step Upwards Section
slider_up_start = tk.Scale(step_up_frame, from_=0, to=100, orient="horizontal", label="Step Upwards Start", command=lambda val: update_step_upward_plot())
slider_up_start.grid(row=1, column=0, padx=5, pady=5, sticky="ew")

slider_up_end = tk.Scale(step_up_frame, from_=0, to=100, orient="horizontal", label="Step Upwards End", command=lambda val: update_step_upward_plot())
slider_up_end.set(100)  # Set initial value to 100
slider_up_end.grid(row=2, column=0, padx=5, pady=5, sticky="ew")



###########################
###########################
def update_step_downward_plot(smoothed=False):
    global ax1_down, ax2_down, is_smoothed_down

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

        # Plot Biogas Production Rate on primary y-axis
        biogas_plot, = ax1_down.step(
            biogas_segment['TimeStamp'],
            biogas_segment[value_col],
            where="post",
            color='blue', linestyle='--', label='Biogas Production Rate (Step Downward)'
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

        # Substrate Feeding Rate preprocessing
        substrate_segment['Derivative'] = substrate_segment['ValueNum'].diff().clip(lower=0)
        substrate_segment['FeedingRate'] = substrate_segment['Derivative'].fillna(0)

        # Detect and handle sudden falls in FeedingRate
        threshold = 0.1  # Threshold below which we treat as "sudden fall"
        for i in range(1, len(substrate_segment)):
            if substrate_segment['FeedingRate'].iloc[i] < threshold and substrate_segment['FeedingRate'].iloc[i - 1] > threshold:
                substrate_segment.loc[substrate_segment.index[i], 'FeedingRate'] = substrate_segment['FeedingRate'].iloc[i - 1]

        # Filter out very small or zero values
        processed_substrate = substrate_segment[substrate_segment['FeedingRate'] > 0]

        # Dynamically adjust y-axis for FeedingRate with clean labels
        if not processed_substrate.empty:
            substrate_min = processed_substrate['FeedingRate'].min()
            substrate_max = processed_substrate['FeedingRate'].max()
            
            # Set y-axis range starting from 0 for consistency
            y_min = 0
            y_max = substrate_max + 0.1

            ax2_down.set_ylim(y_min, y_max)

            # Determine tick interval dynamically
            if y_max - y_min > 1:  # Integer-based ticks for larger ranges
                tick_interval = 1
            else:  # Decimal-based ticks for smaller ranges
                tick_interval = 0.5  # Ensure intervals like 0, 0.5, 1, 1.5 are displayed

            # Apply ticks and formatting
            ax2_down.yaxis.set_major_locator(plt.MultipleLocator(tick_interval))
            ax2_down.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.1f}" if tick_interval < 1 else f"{int(x)}"))
        else:
            print("No valid substrate feeding rate to plot.")
            ax2_down.set_ylim(0, 1)  # Default range if no data is available
            ax2_down.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.1f}"))

        # Plot Substrate Feeding Rate (Red Line)
        if not processed_substrate.empty:
            substrate_plot, = ax2_down.step(
                processed_substrate['TimeStamp'],
                processed_substrate['FeedingRate'],
                where="post",
                color='red', label='Substrate Feeding Rate (Step Downward)'
            )
        else:
            substrate_plot = None

        ax2_down.set_ylabel("Substrate Feeding Rate [t/h]", color='red')
        ax2_down.tick_params(axis="y", labelcolor='red', labelsize=8)

        # Format x-axis with custom tick labels
        ax1_down.xaxis.set_major_formatter(mdates.DateFormatter('%d\n%H:%M'))  # Date and time below
        ax1_down.tick_params(axis="x", rotation=0)  # Keep x-axis labels straight

        # Add month and year to the last tick
        def add_month_year_to_last_tick(ax):
            ticks = ax.get_xticks()
            if len(ticks) > 0:
                ticks_as_dates = [mdates.num2date(tick) for tick in ticks]
                labels = [tick.strftime('%d\n%H:%M') for tick in ticks_as_dates]  # Default format
                labels[-1] += f"\n{ticks_as_dates[-1].strftime('%b %Y')}"  # Add month and year to the last tick
                ax.set_xticklabels(labels, ha='center')

        add_month_year_to_last_tick(ax1_down)

        # Ensure enough spacing between axes to avoid overlap
        fig_down.subplots_adjust(left=0.1, right=0.85)  # Adjust margins to prevent overlapping labels

        # Add legends
        if substrate_plot:
            ax1_down.legend(handles=[biogas_plot, substrate_plot], loc='upper left', fontsize=9)
        else:
            ax1_down.legend(handles=[biogas_plot], loc='upper left', fontsize=9)

        # Add grid for better visualization
        ax1_down.grid(which='major', linestyle='-', linewidth=0.5)

        # Update title
        ax1_down.set_title("After Data Preprocessing", fontsize=10, fontweight='bold')

        # Use tight_layout to resolve overlaps and adjust spacing
        fig_down.tight_layout(pad=2)

        # Update canvas
        canvas_down.draw()

    except Exception as e:
        print(f"Error during graph preprocessing and plotting: {e}")


def update_step_upward_plot(smoothed=False):
    global ax1_up, ax2_up, is_smoothed_up

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

        # Plot Biogas Production Rate on primary y-axis
        biogas_plot, = ax1_up.step(
            biogas_segment['TimeStamp'],
            biogas_segment[value_col],
            where="post",
            color='blue', linestyle='--', label='Biogas Production Rate (Step Upward)'
        )
        ax1_up.set_xlabel("Time")
        ax1_up.set_ylabel("Biogas Production Rate [m³/h]", color='blue')
        ax1_up.tick_params(axis="y", labelcolor='blue')

        # Set y-axis intervals for Biogas Production Rate
        ax1_up.set_ylim(
            biogas_segment[value_col].min() - 5,
            biogas_segment[value_col].max() + 5
        )
        ax1_up.yaxis.set_major_locator(plt.MultipleLocator(5))
        ax1_up.yaxis.set_minor_locator(plt.MultipleLocator(1))
        ax1_up.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{int(x)}"))

        # Substrate Feeding Rate preprocessing
        substrate_segment['Derivative'] = substrate_segment['ValueNum'].diff().clip(lower=0)
        substrate_segment['FeedingRate'] = substrate_segment['Derivative'].fillna(0)

        # Detect and handle sudden falls in FeedingRate
        threshold = 0.1  # Threshold below which we treat as "sudden fall"
        for i in range(1, len(substrate_segment)):
            if substrate_segment['FeedingRate'].iloc[i] < threshold and substrate_segment['FeedingRate'].iloc[i - 1] > threshold:
                # Replace the sudden drop with the previous valid value
                substrate_segment.loc[substrate_segment.index[i], 'FeedingRate'] = substrate_segment['FeedingRate'].iloc[i - 1]

        # Filter out very small or zero values
        processed_substrate = substrate_segment[substrate_segment['FeedingRate'] > 0]

        # Dynamically adjust y-axis for FeedingRate with clean labels
        if not processed_substrate.empty:
            substrate_min = processed_substrate['FeedingRate'].min()
            substrate_max = processed_substrate['FeedingRate'].max()
            
            # Set y-axis range starting from 0 for consistency
            y_min = 0
            y_max = substrate_max + 0.1

            ax2_up.set_ylim(y_min, y_max)

            # Determine tick interval dynamically
            if y_max - y_min > 1:  # Integer-based ticks
                tick_interval = 1
                ax2_up.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{int(x)}"))
            else:  # Decimal-based ticks
                tick_interval = round((y_max - y_min) / 10, 1)
                ax2_up.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.1f}"))

            # Set ticks
            ax2_up.yaxis.set_major_locator(plt.MultipleLocator(tick_interval))
        else:
            print("No valid substrate feeding rate to plot.")
            ax2_up.set_ylim(0, 1)  # Default range if no data is available
            ax2_up.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.1f}"))

        # Plot Substrate Feeding Rate (Red Line)
        if not processed_substrate.empty:
            substrate_plot, = ax2_up.step(
                processed_substrate['TimeStamp'],
                processed_substrate['FeedingRate'],
                where="post",
                color='red', label='Substrate Feeding Rate (Step Upward)'
            )
        else:
            substrate_plot = None

        ax2_up.set_ylabel("Substrate Feeding Rate [t/h]", color='red')
        ax2_up.tick_params(axis="y", labelcolor='red', labelsize=8)

        # Format x-axis with custom tick labels
        ax1_up.xaxis.set_major_formatter(mdates.DateFormatter('%d\n%H:%M'))  # Date and time below
        ax1_up.tick_params(axis="x", rotation=0)  # Keep x-axis labels straight

        # Add month and year to the last tick
        def add_month_year_to_last_tick(ax):
            ticks = ax.get_xticks()
            if len(ticks) > 0:
                ticks_as_dates = [mdates.num2date(tick) for tick in ticks]
                labels = [tick.strftime('%d\n%H:%M') for tick in ticks_as_dates]  # Default format
                labels[-1] += f"\n{ticks_as_dates[-1].strftime('%b %Y')}"  # Add month and year to the last tick
                ax.set_xticklabels(labels, ha='center')

        add_month_year_to_last_tick(ax1_up)

        # Ensure enough spacing between axes to avoid overlap
        fig_up.subplots_adjust(left=0.1, right=0.85)  # Adjust margins to prevent overlapping labels

        # Add legends
        if substrate_plot:
            ax1_up.legend(handles=[biogas_plot, substrate_plot], loc='upper left', fontsize=9)
        else:
            ax1_up.legend(handles=[biogas_plot], loc='upper left', fontsize=9)

        # Add grid for better visualization
        ax1_up.grid(which='major', linestyle='-', linewidth=0.5)

        # Update title
        ax1_up.set_title("After Data Preprocessing", fontsize=10, fontweight='bold')

        # Use tight_layout to resolve overlaps and adjust spacing
        fig_up.tight_layout(pad=2)

        # Update canvas
        canvas_up.draw()

    except Exception as e:
        print(f"Error during graph preprocessing and plotting: {e}")



###########################
#Model Estimation Tab
###########################

def modelestimation_button_pushed():
   """Switches to Model Estimation Tab and plots graphs"""
   plot_model_estimation_graphs()
   notebook.select(model_tab)

def plot_model_estimation_graphs():
   """Plot both step downward and upward graphs"""
   plot_step_downward_model()
   plot_step_upward_model()

def detect_step_change(feed_data):
    """
    Detects the first significant step change in feeding data.
    
    Parameters:
        feed_data (pd.Series): Feeding rate data.

    Returns:
        int: Index of the detected step change.
    """
    # Use numpy to compute the derivative
    diff_feed = np.diff(feed_data)
    # Find the first significant positive change
    step_index = np.argmax(diff_feed > 0)  # Returns the first occurrence
    return step_index
   
def pt1_model(flowrate_data, substrate_data, step_index, feed_max, feed_min, flowrate_max, flowrate_min, direction):
    """
    Implements a PT1 model for step-down or step-up analysis.
    """
    # Calculate time difference in seconds
    time = (flowrate_data['TimeStamp'] - flowrate_data['TimeStamp'].iloc[step_index]).dt.total_seconds()
    time_constant = 7200  # Adjust based on dataset (e.g., 2 hours = 7200 seconds)

    # Apply PT1 model formula for step-down
    if direction == "down":
        response = flowrate_max + (flowrate_min - flowrate_max) * (1 - np.exp(-time / time_constant))
    else:  # For step-up (not used here)
        response = flowrate_min + (flowrate_max - flowrate_min) * (1 - np.exp(-time / time_constant))

    # Return response only for valid time indices (time >= 0)
    response[time < 0] = flowrate_max  # Keep initial value before step
    return response


def time_percentage_method(flowrate_data, feed_data, step_index, feed_max, feed_min, flowrate_max, flowrate_min):
    """
    Calculates the percentage of time spent within specific ranges.

    Parameters:
        flowrate_data (pd.DataFrame): Flowrate data with timestamps.
        feed_data (pd.Series): Feeding rate data.
        step_index (int): Index of the step change.
        feed_max (float): Maximum feeding rate.
        feed_min (float): Minimum feeding rate.
        flowrate_max (float): Maximum flowrate.
        flowrate_min (float): Minimum flowrate.

    Returns:
        dict: Time percentages for feed and flowrate ranges.
    """
    time_differences = flowrate_data['TimeStamp'].diff().dt.total_seconds().fillna(0)

    # Define ranges
    feed_range = (feed_min, feed_max)
    flowrate_range = (flowrate_min, flowrate_max)

    percentages = {}

    # Time spent in feeding rate range
    feed_mask = (feed_data >= feed_range[0]) & (feed_data < feed_range[1])
    time_in_feed_range = time_differences[feed_mask].sum()
    percentages["Feed Range"] = (time_in_feed_range / time_differences.sum()) * 100

    # Time spent in flowrate range
    flowrate_mask = (flowrate_data['ValueNum'] >= flowrate_range[0]) & (flowrate_data['ValueNum'] < flowrate_range[1])
    time_in_flowrate_range = time_differences[flowrate_mask].sum()
    percentages["Flowrate Range"] = (time_in_flowrate_range / time_differences.sum()) * 100

    return percentages


def plot_step_downward_model():
    """Plot step downward graph with Time Percentage Method (Yellow Line) added."""
    global ax1_down_model, ax2_down_model, fig_down_model

    # Clear the previous plot
    fig_down_model.clear()
    ax1_down_model = fig_down_model.add_subplot(111)
    ax2_down_model = ax1_down_model.twinx()

    try:
        # Filter data by date range
        start_date_str = start_date_down.get()
        end_date_str = end_date_down.get()
        start_date = datetime.strptime(start_date_str, "%m/%d/%y")
        end_date = datetime.strptime(end_date_str, "%m/%d/%y")

        biogas_filtered = gas_flow_rate_df[
            (gas_flow_rate_df['TimeStamp'] >= start_date) & (gas_flow_rate_df['TimeStamp'] <= end_date)
        ]
        substrate_filtered = substrate_feeding_df[
            (substrate_feeding_df['TimeStamp'] >= start_date) & (substrate_feeding_df['TimeStamp'] <= end_date)
        ]

        value_col = 'ValueNum'
        if is_smoothed_down:
            biogas_filtered['SmoothedValueNum'] = apply_smoothing(biogas_filtered[value_col], SMOOTHING_LEVEL)
            substrate_filtered['SmoothedValueNum'] = apply_smoothing(substrate_filtered[value_col], SMOOTHING_LEVEL)
            value_col = 'SmoothedValueNum'

        # Blue Line (Biogas Production Rate)
        biogas_plot, = ax1_down_model.step(
            biogas_filtered['TimeStamp'],
            biogas_filtered[value_col],
            where="post",
            color='blue', linestyle='--', label='Biogas Production Rate (Step Downward)'
        )
        ax1_down_model.set_xlabel("Time")
        ax1_down_model.set_ylabel("Biogas Production Rate [m³/h]", color='blue')
        ax1_down_model.tick_params(axis="y", labelcolor='blue')

        ax1_down_model.set_ylim(80, 125)
        ax1_down_model.yaxis.set_major_locator(plt.MultipleLocator(5))
        ax1_down_model.yaxis.set_minor_locator(plt.MultipleLocator(1))
        ax1_down_model.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{int(x)}"))

        # Red Line (Substrate Feeding Rate)
        substrate_filtered['Derivative'] = substrate_filtered['ValueNum'].diff().clip(lower=0)
        substrate_filtered['FeedingRate'] = substrate_filtered['Derivative'].fillna(0)

        threshold = 0.1
        for i in range(1, len(substrate_filtered)):
            if substrate_filtered['FeedingRate'].iloc[i] < threshold and substrate_filtered['FeedingRate'].iloc[i - 1] > threshold:
                substrate_filtered.loc[substrate_filtered.index[i], 'FeedingRate'] = substrate_filtered['FeedingRate'].iloc[i - 1]

        processed_substrate = substrate_filtered[substrate_filtered['FeedingRate'] > 0]

        if not processed_substrate.empty:
            substrate_plot, = ax2_down_model.step(
                processed_substrate['TimeStamp'],
                processed_substrate['FeedingRate'],
                where="post",
                color='red', label='Substrate Feeding Rate (Step Downward)'
            )
        else:
            substrate_plot = None

        ax2_down_model.set_ylabel("Substrate Feeding Rate [t/h]", color='red')
        ax2_down_model.tick_params(axis="y", labelcolor='red', labelsize=8)
        ax2_down_model.set_ylim(0.4, 0.7)
        ax2_down_model.yaxis.set_major_locator(plt.MultipleLocator(0.05))
        ax2_down_model.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.2f}"))

        # Add Yellow Line (Time Percentage Method)
        step_index = detect_step_change(substrate_filtered['FeedingRate'])
        feed_max = substrate_filtered['FeedingRate'][:step_index].mean()
        feed_min = substrate_filtered['FeedingRate'][step_index:].mean()
        flowrate_max = biogas_filtered[value_col][:step_index].mean()
        flowrate_min = biogas_filtered[value_col][step_index:].mean()

        pt1_response = pt1_model(
            biogas_filtered, substrate_filtered, step_index, feed_max, feed_min, flowrate_max, flowrate_min, "down"
        )

        # Plot Yellow Line
        pt1_plot, = ax1_down_model.plot(
            biogas_filtered['TimeStamp'],
            pt1_response,
            linestyle='-', color='yellow', linewidth=2, label='Time Percentage Method'
        )

        # Format x-axis
        ax1_down_model.xaxis.set_major_formatter(mdates.DateFormatter('%d\n%H:%M'))
        ax1_down_model.tick_params(axis="x", rotation=0)

        def add_month_year_to_last_tick(ax):
            ticks = ax.get_xticks()
            if len(ticks) > 0:
                ticks_as_dates = [mdates.num2date(tick) for tick in ticks]
                labels = [tick.strftime('%d\n%H:%M') for tick in ticks_as_dates]
                labels[-1] += f"\n{ticks_as_dates[-1].strftime('%b %Y')}"
                ax.set_xticklabels(labels, ha='center')

        add_month_year_to_last_tick(ax1_down_model)

        # Add legends
        handles = [biogas_plot, pt1_plot]
        if substrate_plot:
            handles.append(substrate_plot)
        ax1_down_model.legend(handles=handles, loc='upper left', fontsize=9)

        # Add grid, title, and adjust layout
        ax1_down_model.grid(which='major', linestyle='-', linewidth=0.5)
        ax1_down_model.set_title("Time Percentage Method", fontsize=10)

        fig_down_model.set_size_inches(10, 6)
        plt.margins(x=0.02)
        plt.tight_layout(pad=2.0, w_pad=0.5, h_pad=1.0)

        canvas_down_model.draw()

    except Exception as e:
        print(f"Error during graph preprocessing and plotting: {e}")



def plot_step_upward_model():
   """Plot step upward graph"""
   global ax1_up_model, ax2_up_model, fig_up_model

   fig_up_model.clear()
   ax1_up_model = fig_up_model.add_subplot(111)
   ax2_up_model = ax1_up_model.twinx()

   try:
       start_date_str = start_date_up.get()
       end_date_str = end_date_up.get()
       start_date = datetime.strptime(start_date_str, "%m/%d/%y")
       end_date = datetime.strptime(end_date_str, "%m/%d/%y")

       biogas_filtered = gas_flow_rate_df[
           (gas_flow_rate_df['TimeStamp'] >= start_date) &
           (gas_flow_rate_df['TimeStamp'] <= end_date)
       ]
       substrate_filtered = substrate_feeding_df[
           (substrate_feeding_df['TimeStamp'] >= start_date) &
           (substrate_feeding_df['TimeStamp'] <= end_date)
       ]

       value_col = 'ValueNum'
       if is_smoothed_up:
           biogas_filtered['SmoothedValueNum'] = apply_smoothing(biogas_filtered[value_col], SMOOTHING_LEVEL)
           substrate_filtered['SmoothedValueNum'] = apply_smoothing(substrate_filtered[value_col], SMOOTHING_LEVEL)
           value_col = 'SmoothedValueNum'

       biogas_plot, = ax1_up_model.step(
           biogas_filtered['TimeStamp'],
           biogas_filtered[value_col],
           where="post",
           color='blue', linestyle='--', label='Biogas Production Rate (Step Upward)'
       )
       ax1_up_model.set_xlabel("Time")
       ax1_up_model.set_ylabel("Biogas Production Rate [m³/h]", color='blue')
       ax1_up_model.tick_params(axis="y", labelcolor='blue')

       ax1_up_model.set_ylim(75, 120)
       ax1_up_model.yaxis.set_major_locator(plt.MultipleLocator(5))
       ax1_up_model.yaxis.set_minor_locator(plt.MultipleLocator(1))
       ax1_up_model.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{int(x)}"))

       substrate_filtered['Derivative'] = substrate_filtered['ValueNum'].diff().clip(lower=0)
       substrate_filtered['FeedingRate'] = substrate_filtered['Derivative'].fillna(0)

       threshold = 0.1
       for i in range(1, len(substrate_filtered)):
           if substrate_filtered['FeedingRate'].iloc[i] < threshold and substrate_filtered['FeedingRate'].iloc[i - 1] > threshold:
               substrate_filtered.loc[substrate_filtered.index[i], 'FeedingRate'] = substrate_filtered['FeedingRate'].iloc[i - 1]

       processed_substrate = substrate_filtered[substrate_filtered['FeedingRate'] > 0]

       if not processed_substrate.empty:
           substrate_plot, = ax2_up_model.step(
               processed_substrate['TimeStamp'],
               processed_substrate['FeedingRate'],
               where="post",
               color='red', label='Substrate Feeding Rate (Step Upward)'
           )
       else:
           substrate_plot = None

       ax2_up_model.set_ylabel("Substrate Feeding Rate [t/h]", color='red')
       ax2_up_model.tick_params(axis="y", labelcolor='red', labelsize=8)
       
       ax2_up_model.set_ylim(0, 0.7)
       ax2_up_model.yaxis.set_major_locator(plt.MultipleLocator(0.1))
       ax2_up_model.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.1f}"))

       ax1_up_model.xaxis.set_major_formatter(mdates.DateFormatter('%d\n%H:%M'))
       ax1_up_model.tick_params(axis="x", rotation=0)

       def add_month_year_to_last_tick(ax):
           ticks = ax.get_xticks()
           if len(ticks) > 0:
               ticks_as_dates = [mdates.num2date(tick) for tick in ticks]
               labels = [tick.strftime('%d\n%H:%M') for tick in ticks_as_dates]
               labels[-1] += f"\n{ticks_as_dates[-1].strftime('%b %Y')}"
               ax.set_xticklabels(labels, ha='center')

       add_month_year_to_last_tick(ax1_up_model)

       fig_up_model.set_size_inches(10, 6)
       plt.margins(x=0.02)
       plt.tight_layout(pad=2.0, w_pad=0.5, h_pad=1.0)

       if substrate_plot:
           ax1_up_model.legend(handles=[biogas_plot, substrate_plot], loc='upper left', fontsize=9)
       else:
           ax1_up_model.legend(handles=[biogas_plot], loc='upper left', fontsize=9)

       ax1_up_model.grid(which='major', linestyle='-', linewidth=0.5)
       ax1_up_model.set_title("Time percentage method", fontsize=10, pad=15)
       
       canvas_up_model.draw()

   except Exception as e:
       print(f"Error during graph preprocessing and plotting: {e}")

# Create a frame with a grid layout inside preprocessing_tab
modelestimation_frame = tk.Frame(preprocess_tab, borderwidth=0, relief="flat", bg="#F0F0F0")
modelestimation_frame.grid(row=3, column=0, columnspan=2, padx=0, pady=10, sticky="nsew")

# Configure grid for modelestimation_frame to make it fully responsive
modelestimation_frame.grid_columnconfigure(0, weight=1)
modelestimation_frame.grid_columnconfigure(1, weight=0)  # Button in center
modelestimation_frame.grid_columnconfigure(2, weight=1)

# Button for Model estimation in preprocessing tab
modelestimation_button = tk.Button(modelestimation_frame, text="Model estimation", font=("Arial", 10), command=modelestimation_button_pushed)
modelestimation_button.grid(row=0, column=1, padx=10, pady=10, ipadx=5, ipady=5, sticky="")

# Configure the grid layout for the model_tab to make it responsive
model_tab.grid_columnconfigure(0, weight=1)
model_tab.grid_columnconfigure(1, weight=1)
model_tab.grid_rowconfigure(1, weight=1)

# Set up frames and widgets to expand properly
step_down_frame.grid_rowconfigure(0, weight=1)
step_down_frame.grid_columnconfigure(0, weight=1)
step_up_frame.grid_rowconfigure(0, weight=1)
step_up_frame.grid_columnconfigure(0, weight=1)

# Top Section: Two side-by-side plots
model_tab.grid_rowconfigure(0, weight=0)  # Row for labels
model_tab.grid_rowconfigure(1, weight=0)  # Row for titles
model_tab.grid_rowconfigure(2, weight=1)  # Row for plots
model_tab.grid_rowconfigure(3, weight=1)  # Row for tables
model_tab.grid_rowconfigure(4, weight=0)  # Row for button
model_tab.grid_columnconfigure(0, weight=1)
model_tab.grid_columnconfigure(1, weight=1)

# Row 0: Labels for Step Downwards and Step Upwards
step_down_label = tk.Label(model_tab, text="Preprocessed data for step downwards", 
                           font=("Arial", 12, "bold"), bg="#F0F0F0")
step_down_label.grid(row=0, column=0, padx=10, pady=5, sticky="n")

step_up_label = tk.Label(model_tab, text="Preprocessed data for step upwards", 
                         font=("Arial", 12, "bold"), bg="#F0F0F0")
step_up_label.grid(row=0, column=1, padx=10, pady=5, sticky="n")

# Row 1: Title for Time Percentage Method
time_percentage_title_left = tk.Label(model_tab, text="Time percentage method", 
                                      font=("Arial", 10, "italic"), bg="#F0F0F0")
time_percentage_title_left.grid(row=1, column=0, padx=10, pady=5, sticky="n")

time_percentage_title_right = tk.Label(model_tab, text="Time percentage method", 
                                       font=("Arial", 10, "italic"), bg="#F0F0F0")
time_percentage_title_right.grid(row=1, column=1, padx=10, pady=5, sticky="n")

# Row 2: Left Plot - Preprocessed Data for Step Downwards
step_down_model_frame = tk.Frame(model_tab, bg="#F0F0F0")
step_down_model_frame.grid(row=2, column=0, padx=10, pady=10, sticky="nsew")

fig_down_model, ax_down_model = plt.subplots()
canvas_down_model = FigureCanvasTkAgg(fig_down_model, master=step_down_model_frame)
canvas_down_model.get_tk_widget().grid(row=0, column=0, padx=10, pady=10, sticky="nsew")

# Row 2: Right Plot - Preprocessed Data for Step Upwards
step_up_model_frame = tk.Frame(model_tab, bg="#F0F0F0")
step_up_model_frame.grid(row=2, column=1, padx=10, pady=10, sticky="nsew")

fig_up_model, ax_up_model = plt.subplots()
canvas_up_model = FigureCanvasTkAgg(fig_up_model, master=step_up_model_frame)
canvas_up_model.get_tk_widget().grid(row=0, column=0, padx=10, pady=10, sticky="nsew")

# Function to dynamically resize the canvas and redraw the figures
def resize_canvas(event):
    # Dynamically adjust the figure size based on the actual frame size
    canvas_width_down = step_down_model_frame.winfo_width()
    canvas_height_down = step_down_model_frame.winfo_height()

    # Adjust Step Downwards figure
    fig_down_model.set_size_inches(canvas_width_down / 100, canvas_height_down / 100)
    canvas_down_model.draw()

    canvas_width_up = step_up_model_frame.winfo_width()
    canvas_height_up = step_up_model_frame.winfo_height()

    # Adjust Step Upwards figure
    fig_up_model.set_size_inches(canvas_width_up / 100, canvas_height_up / 100)
    canvas_up_model.draw()

# Bind the resize event to the frames
step_down_model_frame.bind("<Configure>", resize_canvas)
step_up_model_frame.bind("<Configure>", resize_canvas)

# Ensure the canvas widgets expand with their parent frames
canvas_down_model.get_tk_widget().grid(row=0, column=0, sticky="nsew")  # Ensure full stretching
canvas_up_model.get_tk_widget().grid(row=0, column=0, sticky="nsew")    # Ensure full stretching

# Ensure the frames stretch with the window
model_tab.grid_rowconfigure(2, weight=1)  # Plots row
model_tab.grid_columnconfigure(0, weight=1)  # Step Downwards plot
model_tab.grid_columnconfigure(1, weight=1)  # Step Upwards plot
step_down_model_frame.grid_rowconfigure(0, weight=1)  # Stretch plot area
step_down_model_frame.grid_columnconfigure(0, weight=1)  # Stretch plot area
step_up_model_frame.grid_rowconfigure(0, weight=1)  # Stretch plot area
step_up_model_frame.grid_columnconfigure(0, weight=1)  # Stretch plot area

# Row 3: Bottom Section - Two side-by-side tables with a centered title
similarity_frame = tk.Frame(model_tab, bg="#F0F0F0")
similarity_frame.grid(row=3, column=0, columnspan=2, padx=10, pady=10, sticky="nsew")

# Ensure the similarity_frame expands proportionally
similarity_frame.grid_columnconfigure(0, weight=1)  # Left table
similarity_frame.grid_columnconfigure(1, weight=1)  # Right table

# Single Centered Title for Correlation Metrics
correlation_title = tk.Label(
    similarity_frame,
    text="Correlation between selected model and preprocessed gas production flow rate",
    font=("Arial", 12, "bold"),
    bg="#F0F0F0",
)
correlation_title.grid(row=0, column=0, columnspan=2, padx=10, pady=5, sticky="n")

# Left Table: Similarity Metrics for Step Downwards
down_table_frame = tk.Frame(similarity_frame, bg="#FFFFFF", relief="groove", borderwidth=1)
down_table_frame.grid(row=1, column=0, padx=10, pady=10, sticky="nsew")

# Treeview for the Down Table
down_table = ttk.Treeview(down_table_frame, columns=("Metric", "Value"), show="headings", height=6)
down_table.heading("Metric", text="Similarity metrics", anchor="w")
down_table.heading("Value", text="Value", anchor="center")
down_table.column("Metric", width=250, anchor="w")
down_table.column("Value", width=100, anchor="center")

# Add vertical scrollbar for Down Table
down_vsb = ttk.Scrollbar(down_table_frame, orient="vertical", command=down_table.yview)
down_table.configure(yscrollcommand=down_vsb.set)
down_vsb.pack(side="right", fill="y")

down_table.pack(side="left", fill="both", expand=True)

# Right Table: Similarity Metrics for Step Upwards
up_table_frame = tk.Frame(similarity_frame, bg="#FFFFFF", relief="groove", borderwidth=1)
up_table_frame.grid(row=1, column=1, padx=10, pady=10, sticky="nsew")

# Treeview for the Up Table
up_table = ttk.Treeview(up_table_frame, columns=("Metric", "Value"), show="headings", height=6)
up_table.heading("Metric", text="Similarity metrics", anchor="w")
up_table.heading("Value", text="Value", anchor="center")
up_table.column("Metric", width=250, anchor="w")
up_table.column("Value", width=100, anchor="center")

# Add vertical scrollbar for Up Table
up_vsb = ttk.Scrollbar(up_table_frame, orient="vertical", command=up_table.yview)
up_table.configure(yscrollcommand=up_vsb.set)
up_vsb.pack(side="right", fill="y")

up_table.pack(side="left", fill="both", expand=True)

# Add "Select Model" Button below the tables
select_model_button = tk.Button(
    model_tab, text="Select Model", font=("Arial", 10, "bold"), relief="raised", command=lambda: print("Select Model clicked")
)
select_model_button.grid(row=4, column=0, columnspan=2, pady=10, ipadx=20, ipady=10)
      
# Run Tkinter Main Loop
root.mainloop()

