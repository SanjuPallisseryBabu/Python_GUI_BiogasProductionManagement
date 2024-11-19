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

# Main window
root = tk.Tk()
root.title("Python GUI App")
root.geometry('1000x900')

# Set the window icon
icon_path = 'logopython.ico'
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

        # Walk through all folders in the parent directory
        for root, dirs, files in os.walk(folder_path):
            for file in files:
                if file == "Gas flow rate Fermenter - current.csv":
                    gas_flow_rate_files.append(os.path.join(root, file))
                elif file == "Substrate feeding Fermenter - today.csv":
                    substrate_feeding_files.append(os.path.join(root, file))

        # Load all found files into DataFrames (or handle accordingly)
        try:
            for file in gas_flow_rate_files:
                df = pd.read_csv(file, parse_dates=['TimeStamp'], dayfirst=True)
                gas_flow_rate_df = pd.concat([gas_flow_rate_df, df], ignore_index=True)
                print(f"Loaded Gas Flow Rate File: {file}")
                print(df.head())  # Placeholder: display first few rows

            for file in substrate_feeding_files:
                df = pd.read_csv(file, parse_dates=['TimeStamp'], dayfirst=True)
                substrate_feeding_df = pd.concat([substrate_feeding_df, df], ignore_index=True)
                print(f"Loaded Substrate Feeding File: {file}")
                print(df.head())  # Placeholder: display first few rows

            # Update radio button to "Yes" when data load is successful
            radio_var.set("Yes")

            # If data loaded successfully, display a success message
            success_label.config(text="Data loaded successfully!")
        except Exception as e:
            success_label.config(text=f"Error: {e}")
    else:
        success_label.config(text=f"Error: The folder '{folder_path}' does not exist.")

def plot_step_graph(upwards=False):
    global gas_flow_rate_df, substrate_feeding_df,  fig1, fig2, canvas1, canvas2

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

        # Determine plotting direction
        where = 'pre' if upwards else 'post'
        plot_title = 'Before Data Preprocessing (Upwards)' if upwards else 'Before Data Preprocessing (Downwards)'

        # Plot Gas Flow Rate on the first y-axis
        ax1.step(filtered_gas_flow_rate_df['TimeStamp'], filtered_gas_flow_rate_df['ValueNum'], where=where, label='Biogas Production Rate', linestyle='-', color='b')
        ax1.set_xlabel('Time', fontsize=12)
        ax1.set_ylabel('Biogas Production Rate [m3/h]', color='b', fontsize=12)
        ax1.tick_params(axis='y', labelcolor='b')

        # Customize x-axis to show date above time
        locator = mdates.HourLocator(byhour=[0, 12])  # Set ticks for 00:00 and 12:00
        formatter = mdates.DateFormatter('%d\n%H:%M')  # Multi-line label: '07\n00:00'

        # Apply locator and formatter
        ax1.xaxis.set_major_locator(locator)
        ax1.xaxis.set_major_formatter(formatter)

        # Add month and year to the last tick
        def add_month_year_to_last_tick(ax):
            ticks = ax.get_xticks()
            if len(ticks) > 0:
                ticks_as_dates = [mdates.num2date(tick) for tick in ticks]
                labels = [tick.strftime('%d\n%H:%M') for tick in ticks_as_dates]  # Date above time
                # Add month and year to the last label
                labels[-1] += f"\n{ticks_as_dates[-1].strftime('%b %Y')}"  # Append 'Oct 2023' to the last label
                ax.set_xticklabels(labels, ha='center')

        # Apply the custom last tick adjustment
        add_month_year_to_last_tick(ax1)

        # Ensure straight x-axis labels
        plt.setp(ax1.xaxis.get_majorticklabels(), rotation=0, ha='center')  # No rotation, labels are centered

        # Create a second y-axis for Substrate Feed Rate
        ax2 = ax1.twinx()
        ax2.step(filtered_substrate_feeding_df['TimeStamp'], filtered_substrate_feeding_df['ValueNum'], where=where, label='Substrate Feeding', linestyle='-', color='r')
        ax2.set_ylabel('Substrate Feeding [t]', color='r', fontsize=12)
        ax2.tick_params(axis='y', labelcolor='r')

        # Adding legend
        lines, labels = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines + lines2, labels + labels2, loc='upper left')

        # Adding title and formatting
        plt.title(plot_title, fontsize=14)
        ax1.grid(True)
        fig.tight_layout()

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
status_label = tk.Label(root, text="No file selected")
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

# Function to change to the next tab (in this case, from Load to Preprocess)
def next_tab():
    notebook.select(preprocess_tab)

# Function to handle the Preprocessing Button click from the Load tab
def preprocessing_button_pushed():
    next_tab()  # Switch to the Preprocess tab
   # display_plots_in_columns()  # Display plots in the Preprocess tab

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


# Function to update Step Downward Plot
def update_step_downward_plot():
    global ax1_down, ax2_down

    # Clear the existing plot
    fig_down.clear()

    # Create primary axis
    ax1_down = fig_down.add_subplot(111)

    # Create secondary axis
    ax2_down = ax1_down.twinx()

    # Length of the data
    data_length = len(biogas_data)

    # Calculate start and end indices based on slider values
    start_ind = round(slider_down_start.get() / 100 * data_length)
    end_ind = round(slider_down_end.get() / 100 * data_length)

    # Ensure indices are within valid range
    start_ind = max(0, min(start_ind, data_length - 1))
    end_ind = max(start_ind, min(end_ind, data_length - 1))  # Ensure end_ind is at least start_ind
    #end_ind = max(0, min(end_ind, data_length))

    # Extract the data for the selected range
    biogas_segment = biogas_data.iloc[start_ind:end_ind]
    substrate_segment = substrate_data.iloc[start_ind:end_ind]

    # Plot Biogas Production Rate on primary y-axis
    ax1_down.step(biogas_segment['TimeStamp'], biogas_segment['ValueNum'], where="post",
                  color='blue', linestyle='--', label='Biogas Production Rate (Step Downward)')
    ax1_down.set_xlabel("Time")
    ax1_down.set_ylabel("Biogas Production Rate [m³/h]", color='blue')
    ax1_down.tick_params(axis="y", labelcolor='blue')

    # Plot Substrate Feeding on secondary y-axis
    ax2_down.step(substrate_segment['TimeStamp'], substrate_segment['ValueNum'], where="post",
                  color='red', label='Substrate Feeding (Step Downward)')
    ax2_down.set_ylabel("Substrate Feeding [t]", color='red')
    ax2_down.tick_params(axis="y", labelcolor='red')

    # Set x-axis major locator and formatter
    ax1_down.xaxis.set_major_locator(mdates.DayLocator(interval=1))
    ax1_down.xaxis.set_major_formatter(mdates.DateFormatter('%b %d, %H:%M'))
    plt.xticks(rotation=45, fontsize=8)

    # Add legends
    ax1_down.legend(loc='upper left')
    ax2_down.legend(loc='upper right')

    # Adjust layout
    fig_down.tight_layout()
    canvas_down.draw()

# Function to update Step Upward Plot
def update_step_upward_plot():
    global ax1_up, ax2_up

    # Clear the existing plot
    fig_up.clear()

    # Create primary axis
    ax1_up = fig_up.add_subplot(111)

    # Create secondary axis
    ax2_up = ax1_up.twinx()

    # Length of the data
    data_length = len(biogas_data)

    # Calculate start and end indices based on slider values
    start_ind = round(slider_up_start.get() / 100 * data_length)
    end_ind = round(slider_up_end.get() / 100 * data_length)

    # Ensure indices are within valid range
    start_ind = max(0, min(start_ind, data_length - 1))
    end_ind = max(0, min(end_ind, data_length))

    # Extract the data for the selected range
    biogas_segment = biogas_data.iloc[start_ind:end_ind]
    substrate_segment = substrate_data.iloc[start_ind:end_ind]

    # Plot Biogas Production Rate on primary y-axis
    ax1_up.step(biogas_segment['TimeStamp'], biogas_segment['ValueNum'], where="pre",
                color='blue', linestyle='--', label='Biogas Production Rate (Step Upward)')
    ax1_up.set_xlabel("Time")
    ax1_up.set_ylabel("Biogas Production Rate [m³/h]", color='blue')
    ax1_up.tick_params(axis="y", labelcolor='blue')

    # Plot Substrate Feeding on secondary y-axis
    ax2_up.step(substrate_segment['TimeStamp'], substrate_segment['ValueNum'], where="pre",
                color='red', label='Substrate Feeding (Step Upward)')
    ax2_up.set_ylabel("Substrate Feeding [t]", color='red')
    ax2_up.tick_params(axis="y", labelcolor='red')

    # Set x-axis major locator and formatter
    ax1_up.xaxis.set_major_locator(mdates.DayLocator(interval=1))
    ax1_up.xaxis.set_major_formatter(mdates.DateFormatter('%b %d, %H:%M'))
    plt.xticks(rotation=45, fontsize=8)

    # Add legends
    ax1_up.legend(loc='upper left')
    ax2_up.legend(loc='upper right')

    # Adjust layout
    fig_up.tight_layout()
    canvas_up.draw()

# Set up frames and widgets to expand properly
step_down_frame.grid_rowconfigure(0, weight=1)
step_down_frame.grid_columnconfigure(0, weight=1)
step_up_frame.grid_rowconfigure(0, weight=1)
step_up_frame.grid_columnconfigure(0, weight=1)

#########################################################################################

###########################################################
# Model estimation tab starts

# Function to perform a PT1 model estimation
def pt1_model_estimation(flowrate, feed, ind_sprung, feed_max, feed_min, flowrate_max, flowrate_min):
    """
    PT1 model estimation using step response parameters.
    """
    K = (flowrate_max - flowrate_min) / (feed_max - feed_min)  # Static gain
    T = 50  # Placeholder for time constant (adjust accordingly based on estimation)
    time = np.linspace(0, len(flowrate), len(flowrate))
    model = flowrate_min + K * (1 - np.exp(-time / T))
    return model

# Function to calculate similarity metrics
def calculate_similarity_metrics(actual, estimated):
    """
    Calculate similarity metrics between actual and estimated data.
    """
    r2 = r2_score(actual, estimated)
    return {
        'R^2': r2
    }

# Function to create the model estimation plot within the Tkinter GUI
def plot_model_estimation_in_frame(ax, time, actual, estimated, title):
    """
    Plot the actual and estimated data in the provided axis.
    """
    ax.clear()
    ax.plot(time, actual, label='Actual Data', color='b')
    ax.plot(time, estimated, label='Estimated Model', color='r', linestyle='--')
    ax.set_xlabel('Time')
    ax.set_ylabel('Gas Flow Rate')
    ax.set_title(title)
    ax.legend()
    ax.grid(True)

# Function to change to the Model Estimation tab from the Preprocess tab
def modelestimation_button_pushed():
    notebook.select(model_tab)  # Change to model estimation tab
    
    # Use preprocessed data segments for model estimation
    if hasattr(root, 'biogas_segment_down') and hasattr(root, 'substrate_segment_down'):
        time = root.biogas_segment_down['TimeStamp']
        flowrate = root.biogas_segment_down['ValueNum']
        feed = root.substrate_segment_down['ValueNum']

        # Estimation parameters (use appropriate analysis to determine these)
        ind_sprung = 10  # Example index where step change occurs
        feed_max = feed.max()
        feed_min = feed.min()
        flowrate_max = flowrate.max()
        flowrate_min = flowrate.min()

        # Perform PT1 model estimation
        estimated_model = pt1_model_estimation(flowrate, feed, ind_sprung, feed_max, feed_min, flowrate_max, flowrate_min)

        # Plotting in Tkinter frames
        plot_model_estimation_in_frame(ax6, time, flowrate, estimated_model, "PT1 Model Estimation - Step Downwards")
        canvas6.draw()

    if hasattr(root, 'biogas_segment_up') and hasattr(root, 'substrate_segment_up'):
        time = root.biogas_segment_up['TimeStamp']
        flowrate = root.biogas_segment_up['ValueNum']
        feed = root.substrate_segment_up['ValueNum']

        # Estimation parameters (use appropriate analysis to determine these)
        ind_sprung = 10  # Example index where step change occurs
        feed_max = feed.max()
        feed_min = feed.min()
        flowrate_max = flowrate.max()
        flowrate_min = flowrate.min()

        # Perform PT1 model estimation
        estimated_model = pt1_model_estimation(flowrate, feed, ind_sprung, feed_max, feed_min, flowrate_max, flowrate_min)

        # Plotting in Tkinter frames
        plot_model_estimation_in_frame(ax7, time, flowrate, estimated_model, "PT1 Model Estimation - Step Upwards")
        canvas7.draw()

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

# Add Titles to the First Row in Each Column (Row 1)
title_label_left = tk.Label(model_tab, text="Preprocessed Data for Step Downwards", font=("Arial", 12, "bold"))
title_label_left.grid(row=1, column=0, padx=10, pady=10, sticky="n")

title_label_right = tk.Label(model_tab, text="Preprocessed Data for Step Upwards", font=("Arial", 12, "bold"))
title_label_right.grid(row=1, column=1, padx=10, pady=10, sticky="n")


# Create a frame for the chart under 'Preprocessed Data for Step Downwards'
chart_frame_left = tk.Frame(model_tab, borderwidth=0, relief="flat", bg="#F0F0F0", width=588, height=325)
chart_frame_left.grid(row=2, column=0, padx=10, pady=10, sticky="nsew")
chart_frame_left.grid_propagate(False)  # Prevent the frame from resizing to fit its contents

# Create and set up Figure and Axes for the first plot
fig6, ax6 = plt.subplots()
ax6.set_title('Step Downwards Data')
ax6.set_xlabel('X')
ax6.set_ylabel('Y')

# Embed the first plot into chart_frame_left
canvas6 = FigureCanvasTkAgg(fig6, master=chart_frame_left)
canvas6.get_tk_widget().pack(fill="both", expand=True)  # Use grid to make it responsive within the frame

# Create a frame for the chart under 'Preprocessed Data for Step Upwards'
chart_frame_right = tk.Frame(model_tab, borderwidth=0, relief="flat", bg="#F0F0F0", width=588, height=325)
chart_frame_right.grid(row=2, column=1, padx=10, pady=10, sticky="nsew")
chart_frame_right.grid_propagate(False)

# Create and set up Figure and Axes for the second plot
fig7, ax7 = plt.subplots()
ax7.set_title('Step Upwards Data')
ax7.set_xlabel('X')
ax7.set_ylabel('Y')

# Embed the second plot into chart_frame_right
canvas7 = FigureCanvasTkAgg(fig7, master=chart_frame_right)
canvas7.get_tk_widget().pack(fill="both", expand=True)  # Use pack to make it responsive within the frame

# Add a centered label in row 3, spanning across both columns
correlation_label = tk.Label(
    model_tab, 
    text="Correlation between selected model and preprocessing gas production flow rate", 
    font=("Arial", 12, "bold")
)
correlation_label.grid(row=3, column=0, columnspan=2, padx=10, pady=20, sticky="n")

##
# Function to update Step Downward Plot
def update_step_downward_plot():
    global ax1_down, ax2_down

    # Clear the existing plot
    fig_down.clear()

    # Create primary axis
    ax1_down = fig_down.add_subplot(111)

    # Create secondary axis
    ax2_down = ax1_down.twinx()

    # Length of the data
    data_length = len(biogas_data)

    # Calculate start and end indices based on slider values
    start_ind = round(slider_down_start.get() / 100 * data_length)
    end_ind = round(slider_down_end.get() / 100 * data_length)

    # Ensure indices are within valid range
    start_ind = max(0, min(start_ind, data_length - 1))
    end_ind = max(start_ind, min(end_ind, data_length - 1))  # Ensure end_ind is at least start_ind

    # Extract the data for the selected range
    root.biogas_segment_down = biogas_data.iloc[start_ind:end_ind]
    root.substrate_segment_down = substrate_data.iloc[start_ind:end_ind]

    # Plot Biogas Production Rate on primary y-axis
    ax1_down.step(root.biogas_segment_down['TimeStamp'], root.biogas_segment_down['ValueNum'], where="post",
                  color='blue', linestyle='--', label='Biogas Production Rate (Step Downward)')
    ax1_down.set_xlabel("Time")
    ax1_down.set_ylabel("Biogas Production Rate [m³/h]", color='blue')
    ax1_down.tick_params(axis="y", labelcolor='blue')

    # Plot Substrate Feeding on secondary y-axis
    ax2_down.step(root.substrate_segment_down['TimeStamp'], root.substrate_segment_down['ValueNum'], where="post",
                  color='red', label='Substrate Feeding (Step Downward)')
    ax2_down.set_ylabel("Substrate Feeding [t]", color='red')
    ax2_down.tick_params(axis="y", labelcolor='red')

    # Set x-axis major locator and formatter
    ax1_down.xaxis.set_major_locator(mdates.DayLocator(interval=1))
    ax1_down.xaxis.set_major_formatter(mdates.DateFormatter('%b %d, %H:%M'))
    plt.xticks(rotation=45, fontsize=8)

    # Add legends
    ax1_down.legend(loc='upper left')
    ax2_down.legend(loc='upper right')

    # Adjust layout
    fig_down.tight_layout()
    canvas_down.draw()

# Function to update Step Upward Plot
def update_step_upward_plot():
    global ax1_up, ax2_up

    # Clear the existing plot
    fig_up.clear()

    # Create primary axis
    ax1_up = fig_up.add_subplot(111)

    # Create secondary axis
    ax2_up = ax1_up.twinx()

    # Length of the data
    data_length = len(biogas_data)

    # Calculate start and end indices based on slider values
    start_ind = round(slider_up_start.get() / 100 * data_length)
    end_ind = round(slider_up_end.get() / 100 * data_length)

    # Ensure indices are within valid range
    start_ind = max(0, min(start_ind, data_length - 1))
    end_ind = max(0, min(end_ind, data_length))

    # Extract the data for the selected range
    root.biogas_segment_up = biogas_data.iloc[start_ind:end_ind]
    root.substrate_segment_up = substrate_data.iloc[start_ind:end_ind]

    # Plot Biogas Production Rate on primary y-axis
    ax1_up.step(root.biogas_segment_up['TimeStamp'], root.biogas_segment_up['ValueNum'], where="pre",
                color='blue', linestyle='--', label='Biogas Production Rate (Step Upward)')
    ax1_up.set_xlabel("Time")
    ax1_up.set_ylabel("Biogas Production Rate [m³/h]", color='blue')
    ax1_up.tick_params(axis="y", labelcolor='blue')

    # Plot Substrate Feeding on secondary y-axis
    ax2_up.step(root.substrate_segment_up['TimeStamp'], root.substrate_segment_up['ValueNum'], where="pre",
                color='red', label='Substrate Feeding (Step Upward)')
    ax2_up.set_ylabel("Substrate Feeding [t]", color='red')
    ax2_up.tick_params(axis="y", labelcolor='red')

    # Set x-axis major locator and formatter
    ax1_up.xaxis.set_major_locator(mdates.DayLocator(interval=1))
    ax1_up.xaxis.set_major_formatter(mdates.DateFormatter('%b %d, %H:%M'))
    plt.xticks(rotation=45, fontsize=8)

    # Add legends
    ax1_up.legend(loc='upper left')
    ax2_up.legend(loc='upper right')

    # Adjust layout
    fig_up.tight_layout()
    canvas_up.draw()
##




# Create a frame with a grid layout inside model_tab for the "Select Model" button
selectmodel_frame = tk.Frame(model_tab, borderwidth=0, relief="flat", bg="#F0F0F0")
selectmodel_frame.grid(row=5, column=0, columnspan=2, padx=0, pady=10, sticky="nsew")


# Configure grid layout for the model_tab to ensure responsiveness
model_tab.grid_columnconfigure(0, weight=1)
model_tab.grid_columnconfigure(1, weight=1)




# Function to change to the Control System tab from the Model Estimation tab
def switch_to_control_system_tab():
    notebook.select(control_tab)  # Change to control system tab


# Configure grid for selectmodel_frame to make it fully responsive and center the button
selectmodel_frame.grid_columnconfigure(0, weight=1)
selectmodel_frame.grid_columnconfigure(1, weight=0)  # Center button in column 1
selectmodel_frame.grid_columnconfigure(2, weight=1)

# Button for Select Model in model estimation tab
selectmodel_button = tk.Button(selectmodel_frame, text="Select Model", font=("Arial", 10), command=switch_to_control_system_tab)
selectmodel_button.grid(row=0, column=1, padx=10, pady=10, ipadx=5, ipady=5, sticky="")

# Create a frame to hold Table 4 with specified styling and position
table4_frame = tk.Frame(model_tab, borderwidth=2, relief="solid", bg="#FFFFFF")
table4_frame.grid(row=4, column=0, padx=10, pady=10, sticky="nsew")

# Configure grid to make the table expandable
model_tab.grid_rowconfigure(4, weight=1)
model_tab.grid_columnconfigure(0, weight=1)


# Create a frame to hold Table 4 with specified styling and position
table4_frame = tk.Frame(model_tab, borderwidth=2, relief="solid", bg="#FFFFFF")
table4_frame.grid(row=4, column=0, padx=10, pady=10, sticky="nsew")

# Configure grid to make the table expandable
model_tab.grid_rowconfigure(4, weight=1)
model_tab.grid_columnconfigure(0, weight=1)
model_tab.grid_columnconfigure(1, weight=1)  # For Table 5's column

# Add Table 4 inside the frame
table4 = ttk.Treeview(table4_frame, columns=("Similarity metrics", "Value"), show="headings", height=5)
table4.heading("Similarity metrics", text="Similarity metrics")
table4.heading("Value", text="Value")
table4.column("Similarity metrics", anchor="center", width=180)
table4.column("Value", anchor="center", width=180)
table4.pack(expand=True, fill="both")

# Create a frame to hold Table 5 with specified styling and position
table5_frame = tk.Frame(model_tab, borderwidth=2, relief="solid", bg="#FFFFFF")
table5_frame.grid(row=4, column=1, padx=10, pady=10, sticky="nsew")

# Add Table 5 inside the frame
table5 = ttk.Treeview(table5_frame, columns=("Similarity metrics", "Value"), show="headings", height=5)
table5.heading("Similarity metrics", text="Similarity metrics")
table5.heading("Value", text="Value")
table5.column("Similarity metrics", anchor="center", width=180)
table5.column("Value", anchor="center", width=180)
table5.pack(expand=True, fill="both")

# Apply a style to the headers to set background color
style = ttk.Style()
style.configure("Treeview.Heading", background="#E0E0E0", font=("Arial", 10, "bold"))

#########################################################################################

#CONTROL SYSTEM

#Label feedback controller and saturations
# Add Titles to the First Row in Each Column (Row 1)
title_label_left = tk.Label(control_tab, text="Feedback Controller", font=("Arial", 12, "bold"))
title_label_left.grid(row=1, column=0, padx=10, pady=10, sticky="n")

title_label_right = tk.Label(control_tab, text="Saturations", font=("Arial", 12, "bold"))
title_label_right.grid(row=1, column=4, padx=10, pady=10, sticky="n")

# Configure the grid layout for the model_tab to make columns responsive
control_tab.grid_columnconfigure(0, weight=1)
control_tab.grid_columnconfigure(1, weight=1)


# Gas production flow rate, initial state arranged horizontally in the same row
initial_state_label = tk.Label(control_tab, text="Gas production flow rate, initial state")
initial_state_label.grid(row=2, column=0, padx=(10, 5), pady=5, sticky="e")  # Label aligned to the right

initial_state_entry = tk.Entry(control_tab, width=10)
initial_state_entry.grid(row=2, column=1, padx=5, pady=5, sticky="w")  # Entry next to label

initial_state_unit = tk.Label(control_tab, text="[m^3/h]")
initial_state_unit.grid(row=2, column=2, padx=5, pady=5, sticky="w")  # Unit label next to entry



# Gas production flow rate, setpoint
setpoint_label = tk.Label(control_tab, text="Gas production flow rate, setpoint")
setpoint_label.grid(row=2, column=3, padx=(20, 5), pady=5, sticky="ns")  # Adjusted padx for spacing

setpoint_entry = tk.Entry(control_tab, width=10)
setpoint_entry.grid(row=2, column=4, padx=5, pady=5, sticky="ns")

setpoint_unit = tk.Label(control_tab, text="[m^3/h]")
setpoint_unit.grid(row=2, column=5, padx=5, pady=5, sticky="ns")

# Substrate feeding rate (max)
feeding_rate_label = tk.Label(control_tab, text="Substrate feeding rate (max)")
feeding_rate_label.grid(row=2, column=6, padx=(20, 5), pady=5, sticky="ns")  # Adjusted padx for spacing

feeding_rate_entry = tk.Entry(control_tab, width=10)
feeding_rate_entry.grid(row=2, column=7, padx=5, pady=5, sticky="ns")

feeding_rate_unit = tk.Label(control_tab, text="[t/2h]")
feeding_rate_unit.grid(row=2, column=8, padx=5, pady=5, sticky="ns")

# Function to update flow rate initial state
def update_flowrate_init(event):
    try:
        flowrate_init = float(initial_state_entry.get())
        print(f"Flow rate initial state updated to: {flowrate_init}")
    except ValueError:
        print("Please enter a valid number.")

# Bind Enter key to update function for initial state entry
initial_state_entry.bind("<Return>", update_flowrate_init)





#########################################################################################
# Start the application
root.mainloop()
