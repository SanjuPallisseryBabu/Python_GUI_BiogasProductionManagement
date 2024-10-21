import tkinter as tk
import matplotlib.pyplot as plt
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

def on_enter(event):
    load_button['background'] = '#d0eaff'  # Light blue color on hover

def on_leave(event):
    load_button['background'] = 'SystemButtonFace'  # Default button color


# Function to create and display the plot
def create_plot():
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

            fig, ax1 = plt.subplots(figsize=(10, 5))

            # Plot biogas production
            ax1.plot(filtered_biogas['TimeStamp'], filtered_biogas['ValueNum'], 'b--', label='Biogas production rate')
            ax1.set_xlabel('Time')
            ax1.set_ylabel('Biogas production rate [m³/h]', color='b')
            ax1.tick_params(axis='y', labelcolor='b')
            #ax1.set_yticks([20, 40, 60, 80, 100, 120, 140])  # Set custom y-ticks 

            # Set the title for the first axis
            ax1.set_title('Before data preprocessing', fontsize=20, fontweight='bold', fontname='Arial')


            # Automatically arrange y-axis with up to 10 evenly spaced ticks
            ax1.yaxis.set_major_locator(MaxNLocator(integer=True, prune='both'))  # prune='both' trims the first and last tick if necessary


            # Format the x-axis to show both date and time
            #ax1.xaxis.set_major_formatter(mdates.DateFormatter('%b %d, %H:%M'))
            #ax1.xaxis.set_major_locator(mdates.HourLocator(interval=12))  # Major ticks every 12 hours
            #plt.xticks(rotation=45)  # Rotate the x-axis labels for better readability

             # Add vertical lines for each biogas production point
            #for i, row in filtered_biogas.iterrows():
            #ax1.axvline(x=row['TimeStamp'], color='blue', linestyle=':', linewidth=0.8)

            # Add horizontal lines for specific biogas production rates
            #for y in [20, 40, 60, 80, 100, 120, 140]:  # Adjust these values as needed
            #ax1.axhline(y=y, color='gray', linestyle='--', linewidth=0.7)


            # Second y-axis for substrate feeding
            ax2 = ax1.twinx()
            ax2.step(filtered_substrate['TimeStamp'], filtered_substrate['ValueNum'], 'r-', label='Substrate feeding')
            ax2.set_ylabel('Substrate feeding [t]', color='r')
            ax2.tick_params(axis='y', labelcolor='r')
            #ax2.set_yticks([1, 2, 3, 4, 5])  # Set custom y-ticks

            # Automatically arrange y-axis with up to 10 evenly spaced ticks
            ax2.yaxis.set_major_locator(MaxNLocator(integer=True, prune='both'))  # prune='both' trims the first and last tick if necessary



            # Add legend and grid
            fig.tight_layout()
            fig.legend(loc='upper right', bbox_to_anchor=(0.85, 0.85))
            # plt.title('Biogas Production Rate and Substrate Feeding Over Time')
            plt.grid(True)

            # Clear the previous plot and add the new one to the canvas
            canvas1.get_tk_widget().pack_forget()
            canvas1.get_tk_widget().grid()
            canvas1.figure = fig
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

plot_down_button = tk.Button(down_frame, text="Plot data step downwards", font=("Arial", 10), bd=1, relief="solid", command=create_plot)
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

plot_up_button = tk.Button(up_frame, text="Plot data step upwards", font=("Arial", 10), bd=1, relief="solid", command=create_plot)
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
#fig1 = Figure(figsize=(3, 0), dpi=35)
fig1, ax1 = plt.subplots(figsize=(5,3), dpi=35)

fig1.patch.set_facecolor('#F0F0F0')  # Set the background color of the figure (outside the plot area)
ax1 = fig1.add_subplot(111)

#ax1.set_title('Before data preprocessing', fontsize=20, fontweight='bold', fontname='Arial')

# Apply tight layout for better spacing
plt.tight_layout(pad=2.0, w_pad=0.5, h_pad=0.5, rect=[0, 0, 1, 1])  # Adjusted pad values

canvas1 = FigureCanvasTkAgg(fig1, master=chart_frame)
canvas1.get_tk_widget().grid(row=0, column=0, padx=10, pady=10, sticky="nsew")

# Create the second chart (After data processing)
#fig2 = Figure(figsize=(3, 0), dpi=35)
#fig2.patch.set_facecolor('#F0F0F0')  # Set the background color of the figure (outside the plot area)
#ax2 = fig2.add_subplot(111)
#ax2.plot(after_processing, marker='o', color='#9AC1D9', label='After Processing')
#ax2.set_title('After data preprocessing', fontsize=20, fontweight='bold', fontname='Arial')
#ax2.set_xlabel('Time', fontsize=20)
#ax2.set_ylabel('Gas Production flow rate [m³/h]', fontsize=20)
#ax2.legend()

# Embed the first chart in the chart_frame
#canvas1 = FigureCanvasTkAgg(fig1, master=chart_frame)
canvas1.draw()
#canvas1.get_tk_widget().grid(row=0, column=0, padx=10, pady=10, sticky="nsew")

# Embed the second chart in the chart_frame
#canvas2 = FigureCanvasTkAgg(fig2, master=chart_frame)
#canvas2.draw()
#canvas2.get_tk_widget().grid(row=0, column=1, padx=10, pady=10, sticky="nsew")

# Configure load_tab to expand the frame
load_tab.grid_rowconfigure(3, weight=1)
load_tab.grid_columnconfigure(0, weight=1)
# Create a frame with a grid layout inside load_tab
preprocess_frame = tk.LabelFrame(load_tab, borderwidth=0, relief="flat", bg="#F0F0F0")
preprocess_frame.grid(row=4, column=0,columnspan=4, padx=0, pady=0, sticky="nsew")

# Configure the grid layout inside preprocess_frame
preprocess_frame.grid_columnconfigure(0, weight=1)  # Empty column on the left
preprocess_frame.grid_columnconfigure(2, weight=1)  # Empty column on the right
preprocess_frame.grid_rowconfigure(0, weight=1)     # Row configuration to allow vertical centering

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

# Preprocessing Tab UI
preprocess_label = tk.Label(preprocess_tab, text="This is the Preprocessing tab", font=("Arial", 16))
preprocess_label.pack(pady=20)

# Start the application
root.mainloop()
