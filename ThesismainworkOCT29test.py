import datetime
import tkinter as tk
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import matplotlib.dates as mdates
from tkinter import ttk
from tkinter import filedialog  # For file dialog
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from tkcalendar import DateEntry
from openpyxl.workbook import workbook
from openpyxl import load_workbook
from matplotlib.ticker import MaxNLocator

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

# CSV load
load_frame = tk.LabelFrame(load_tab,borderwidth=0, relief="flat",bg="#F0F0F0", padx=10, pady=0.5)
load_frame.grid(row=0, column=0, columnspan=2, padx=(10, 10), pady=0, sticky="nsew")

# Press button text
press_button = tk.Label(load_frame, text="Press the button to load CSV data", font=("Arial", 12, "bold"), background="#F0F0F0")
press_button.grid(row=0, column=0, padx=10, pady=(10,5), sticky="ew")

# Function to load the CSV files
def load_files():
    global biogas_data, substrate_data

# List to store selected files
selected_files = []

# Initialize empty variables to hold the data
biogas_data = None
substrate_data = None


# Function to load the CSV files
def load_files():
    global biogas_data, substrate_data

    # Open file dialog to select the first CSV file (Biogas production rate)
    biogas_file = filedialog.askopenfilename(title="Select Biogas Production CSV", filetypes=[("CSV files", "*.csv")])
    
    if biogas_file:
        biogas_data = pd.read_csv(biogas_file)
        selected_files.append(biogas_file)

    # Open file dialog to select the second CSV file (Substrate feeding)
    substrate_file = filedialog.askopenfilename(title="Select Substrate Feeding CSV", filetypes=[("CSV files", "*.csv")])
    
    if substrate_file:
        substrate_data = pd.read_csv(substrate_file)
        selected_files.append(substrate_file)

    # Convert the TimeStamp columns to datetime
    if biogas_data is not None:
        biogas_data['TimeStamp'] = pd.to_datetime(biogas_data['TimeStamp'], format='%d.%m.%Y %H:%M:%S')
    
    if substrate_data is not None:
        substrate_data['TimeStamp'] = pd.to_datetime(substrate_data['TimeStamp'], format='%d.%m.%Y %H:%M:%S')

    # Update status label with selected files
    if len(selected_files) == 2:
        status_label.config(text="Files selected: " + ", ".join(selected_files))
        radio_var.set("Yes")  # Enable "Yes" radio button if both files are loaded
    else:
        status_label.config(text="No file selected or only one file selected")
        radio_var.set("No")  # Reset to "No" if both files are not selected
      
    # Once the files are loaded, enable the date entry widgets and plot button
    start_date_down.config(state='normal')
    end_date_down.config(state='normal')
    plot_down_button.config(state='normal')

    start_date_up.config(state='normal')
    end_date_up.config(state='normal')
    plot_up_button.config(state='normal')

def on_enter(event):
    load_button['background'] = '#d0eaff'  # Light blue color on hover

def on_leave(event):
    load_button['background'] = 'SystemButtonFace'  # Default button color

# Function to create and display the upward step plot
def create_plot_step_upward():
    if biogas_data is not None and substrate_data is not None:
        # Get the start and end dates from the entry widgets
        start_date = start_date_up.get()
        end_date = end_date_up.get()
        try:
            # Convert the start and end dates to datetime
            start_date = pd.to_datetime(start_date)
            end_date = pd.to_datetime(end_date)

            # Filter the data based on the date range
            filtered_biogas = biogas_data[(biogas_data['TimeStamp'] >= start_date) & (biogas_data['TimeStamp'] <= end_date)]
            filtered_substrate = substrate_data[(substrate_data['TimeStamp'] >= start_date) & (substrate_data['TimeStamp'] <= end_date)]

            fig2, ax1 = plt.subplots(figsize=(9, 5))

            # Plot biogas production
            ax1.plot(filtered_biogas['TimeStamp'], filtered_biogas['ValueNum'], 'b--', label='Biogas production rate')
            ax1.set_xlabel('Time')
            ax1.set_ylabel('Biogas production rate [m³/h]', color='b')
            ax1.tick_params(axis='y', labelcolor='b')
            # Set the title for the upward plot
            ax1.set_title('Before data preprocessing', fontsize=10, fontweight='bold', fontname='Arial')

            # Second y-axis for substrate feeding
            ax2 = ax1.twinx()
            ax2.step(filtered_substrate['TimeStamp'], filtered_substrate['ValueNum'][::-1], 'r-', label='Substrate feeding (Step Upward)')
            ax2.set_ylabel('Substrate feeding [t]', color='r')
            ax2.tick_params(axis='y', labelcolor='r')

            # Add legend and grid
            fig2.tight_layout()
            fig2.legend(loc='upper right', bbox_to_anchor=(0.85, 0.85))
            plt.grid(True)

            # Clear the previous plot and add the new one to the canvas
            #canvas2.get_tk_widget().pack_forget()
            #canvas2.get_tk_widget().pack()
            canvas2.figure = fig2
            canvas2.draw()
        except Exception as e:
            error_label.config(text=f"Error: {e}")


# Function to create and display the plot
def create_plot_step_downward():
    if biogas_data is not None and substrate_data is not None:
        # Get the start and end dates from the entry widgets
        start_date = start_date_down.get()
        end_date = end_date_down.get()
        try:
            # Convert the start and end dates to datetime
            start_date = pd.to_datetime(start_date)
            end_date = pd.to_datetime(end_date)

            # Filter the data based on the date range
            filtered_biogas = biogas_data[(biogas_data['TimeStamp'] >= start_date) & (biogas_data['TimeStamp'] <= end_date)]
            filtered_substrate = substrate_data[(substrate_data['TimeStamp'] >= start_date) & (substrate_data['TimeStamp'] <= end_date)]

            fig1, ax1 = plt.subplots(figsize=(9, 5))

            # Plot biogas production
            ax1.plot(filtered_biogas['TimeStamp'], filtered_biogas['ValueNum'], 'b--', label='Biogas production rate')
            ax1.set_xlabel('Time')
            ax1.set_ylabel('Biogas production rate [m³/h]', color='b')
            ax1.tick_params(axis='y', labelcolor='b') 
            # Set the title for the first axis
            ax1.set_title('Before data preprocessing', fontsize=10, fontweight='bold', fontname='Arial')
            # Automatically arrange y-axis with up to 10 evenly spaced ticks
            ax1.yaxis.set_major_locator(MaxNLocator(integer=True, prune='both'))  # prune='both' trims the first and last tick if necessary


            # Second y-axis for substrate feeding
            ax2 = ax1.twinx()
            ax2.step(filtered_substrate['TimeStamp'], filtered_substrate['ValueNum'], 'r-', label='Substrate feeding')
            ax2.set_ylabel('Substrate feeding [t]', color='r')
            ax2.tick_params(axis='y', labelcolor='r')
            # Automatically arrange y-axis with up to 10 evenly spaced ticks
            ax2.yaxis.set_major_locator(MaxNLocator(integer=True, prune='both'))  # prune='both' trims the first and last tick if necessary

            # Add legend and grid
            fig1.tight_layout()
            fig1.legend(loc='upper right', bbox_to_anchor=(0.85, 0.85))
            # plt.title('Biogas Production Rate and Substrate Feeding Over Time')
            plt.grid(True)

            # Clear the previous plot and add the new one to the canvas
            #canvas1.get_tk_widget().pack_forget()
            #canvas1.get_tk_widget().grid()
            canvas1.figure = fig1
            canvas1.draw()
        except Exception as e:
            error_label.config(text=f"Error: {e}")



# Button
load_button = tk.Button(load_frame, text="Load Data", font=("Arial", 10), bd=1, relief="solid", command=load_files)
load_button.grid(row=1, column=0, padx=(20, 20), pady=3, ipadx=15, ipady=0) 

# Bind hover effects
load_button.bind("<Enter>", on_enter)  # When mouse enters the button      
load_button.bind("<Leave>", on_leave)  # When mouse leaves the button

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

plot_down_button = tk.Button(down_frame, text="Plot data step downwards", font=("Arial", 10), bd=1, relief="solid", command=create_plot_step_downward)
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

plot_up_button = tk.Button(up_frame, text="Plot data step upwards", font=("Arial", 10), bd=1, relief="solid", command=create_plot_step_upward)
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
chart_frame = tk.LabelFrame(load_tab, borderwidth=0, relief="flat", bg="#F0F0F0",width=600, height=300)
chart_frame.grid(row=3, column=0,columnspan=2,padx=10, pady=0, sticky="nsew")
chart_frame.grid_propagate(False)  # Prevent the frame from resizing to fit its contents

# Configure the chart frame for 2-column layout
chart_frame.grid_columnconfigure(0, weight=1)  # First column for the first chart
chart_frame.grid_columnconfigure(1, weight=1)  # Second column for the second chart
chart_frame.grid_rowconfigure(0, weight=1)     # Ensure the row expands to fit both charts


# Create the first chart (Before data processing)
fig1, ax1 = plt.subplots(figsize=(5,3), dpi=35)
fig1.patch.set_facecolor('#F0F0F0')  # Set the background color of the figure (outside the plot area)
ax1 = fig1.add_subplot(111)

# Apply tight layout for better spacing
#plt.tight_layout(pad=2.0, w_pad=0.5, h_pad=0.5, rect=[0, 0, 1, 1])  # Adjusted pad values

# Embed the first chart in the chart_frame
canvas1 = FigureCanvasTkAgg(fig1, master=chart_frame)
canvas1.get_tk_widget().grid(row=0, column=0, padx=10, pady=10, sticky="nsew")
canvas1.draw()

# Create the second chart (After data processing)
fig2, ax2 = plt.subplots(figsize=(5,3), dpi=35)
fig2.patch.set_facecolor('#F0F0F0')  # Set the background color of the figure (outside the plot area)
ax3 = fig2.add_subplot(111)

# Embed the second chart in the chart_frame
canvas2 = FigureCanvasTkAgg(fig2, master=chart_frame)
canvas2.get_tk_widget().grid(row=0, column=1, padx=10, pady=10, sticky="nsew")
canvas2.draw()

# Configure load_tab to expand the frame
load_tab.grid_rowconfigure(3, weight=1)
load_tab.grid_columnconfigure(0, weight=1)
load_tab.grid_columnconfigure(1, weight=1) 

# Create a frame with a grid layout inside load_tab
preprocess_frame = tk.LabelFrame(load_tab, borderwidth=0, relief="flat", bg="#F0F0F0")
preprocess_frame.grid(row=4, column=0,columnspan=4, padx=0, pady=0, sticky="nsew")

# Configure the grid layout inside preprocess_frame
preprocess_frame.grid_columnconfigure(0, weight=1)  # Empty column on the left
preprocess_frame.grid_columnconfigure(2, weight=1)  # Empty column on the right
preprocess_frame.grid_rowconfigure(0, weight=1)     # Row configuration to allow vertical centering

###########################################nexttab################################################

# Function to change tabs
def next_tab():
    current_index = notebook.index(notebook.select())
    next_index = (current_index + 1) % notebook.index("end")
    notebook.select(next_index)

# Button for preprocessing
preprocess_button = tk.Button(preprocess_frame, text="Preprocessing", font=("Arial", 10), command=next_tab)
preprocess_button.grid(row=0, column=1, padx=10, pady=10, ipadx=5, ipady=5, sticky="") # Leave sticky="" for it to stay centered

def on_enter(event):
    preprocess_button['background'] = '#d0eaff'  # Light blue color on hover

def on_leave(event):
    preprocess_button['background'] = 'SystemButtonFace'  # Default button color
    
# Bind hover effects
preprocess_button.bind("<Enter>", on_enter)  # When mouse enters the button
preprocess_button.bind("<Leave>", on_leave)  # When mouse leaves the button

# Making the columns responsive for horizontal centering
preprocess_tab.grid_columnconfigure(0, weight=1)
preprocess_tab.grid_columnconfigure(1, weight=1)


class App:
    def __init__(self, preprocess_tab, preprocess_frame):
        self.preprocess_tab = preprocess_tab
        self.preprocess_frame = preprocess_frame

        # Call method to set up graphs
        self.setup_graphs()

    def setup_graphs(self):

        # Configure the grid for responsive design
        self.preprocess_frame.columnconfigure(0, weight=1)  # Allow first column to expand
        self.preprocess_frame.columnconfigure(1, weight=1)  # Allow second column to expand
        self.preprocess_frame.rowconfigure(1, weight=1)     # Allow row with graphs to expand
        self.preprocess_frame.rowconfigure(3, weight=0)     # Row for sliders
        # Create two columns for the graphs
        self.create_graph_column("Step Downwards", 0)
        self.create_graph_column("Step Upwards", 1)

    def create_graph_column(self, title, column):
        # Create a label for the title
        label = ttk.Label(self.preprocess_frame, text=title, font=("Arial", 12))
        label.grid(row=0, column=column, padx=10, pady=5, sticky="w")

        # Create a Matplotlib figure and axes
        fig, ax = plt.subplots(figsize=(5, 3))
        ax.set_title(f"{title} Graph")
        ax.set_xlabel("Time")
        ax.set_ylabel("Flowrate")
        
        # Sample data for demonstration (replace this with your actual flowrate data)
        time_data = np.arange(100)  # Replace with actual time data
        flowrate_data = np.sin(time_data * 0.1) * 10  # Replace with actual flowrate data
        self.plot_data(ax, time_data, flowrate_data)

        # Embed the figure in the Tkinter frame
        canvas = FigureCanvasTkAgg(fig, self.preprocess_frame)
        canvas.get_tk_widget().grid(row=1, column=column, pady=10)

        # Create sliders for start and end
        self.create_sliders(column)

    def create_sliders(self, column):
        # Create start slider
        start_slider_label = ttk.Label(self.preprocess_frame, text="Start:")
        start_slider_label.grid(row=2, column=column, sticky="w")
        start_slider = ttk.Scale(self.preprocess_frame, from_=0, to=100, orient='horizontal',
                                  command=lambda value: self.update_graphs())
        start_slider.set(0)
        start_slider.grid(row=3, column=column, sticky="ew", padx=10)

        # Create end slider
        end_slider_label = ttk.Label(self.preprocess_frame, text="End:")
        end_slider_label.grid(row=4, column=column, sticky="w")
        end_slider = ttk.Scale(self.preprocess_frame, from_=0, to=100, orient='horizontal',
                                command=lambda value: self.update_graphs())
        end_slider.set(100)
        end_slider.grid(row=5, column=column, sticky="ew", padx=10)

    def plot_data(self, ax, time_data, flowrate_data):
        ax.clear()
        ax.plot(time_data, flowrate_data, label="Flowrate", color='blue')
        ax.legend()
        ax.set_xlabel("Time")
        ax.set_ylabel("Flowrate")
        ax.set_title("Flowrate Graph")

    def update_graphs(self):
        # Here you can add logic to update the graphs based on the slider values
        print("Sliders updated - graphs should refresh")

# Processing tab frame setup
preprocess_frame = tk.LabelFrame(preprocess_tab, borderwidth=0, relief="flat", bg="#F0F0F0", padx=10, pady=0.5)
preprocess_frame.grid(row=1, column=0, columnspan=2, padx=(10, 10), pady=0, sticky="ew")

# Label centered horizontally in the first column
preprocess_label_down = tk.Label(preprocess_tab, text="Preprocessed data for step downwards", font=("Arial", 16), anchor="center")
preprocess_label_down.grid(row=0, column=0, padx=10, pady=(10, 5), sticky="ew")

# Label centered horizontally in the second column
preprocess_label_up = tk.Label(preprocess_tab, text="Preprocessed data for step upwards", font=("Arial", 16), anchor="center")
preprocess_label_up.grid(row=0, column=1, padx=10, pady=(10, 5), sticky="ew")

# Initialize the App with the existing preprocess_tab and preprocess_frame
app = App(preprocess_tab, preprocess_frame)

# Start the application
root.mainloop()
