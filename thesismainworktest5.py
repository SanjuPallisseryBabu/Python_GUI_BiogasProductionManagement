import tkinter as tk
from tkinter import ttk
from tkinter import filedialog  # For file dialog
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from tkcalendar import DateEntry
import csv



# Main window
root = tk.Tk()
root.title("Python GUI App")
root.geometry('1000x900')
toggle_var = tk.StringVar(value="No")
toggle_label = tk.Label(root, text="Data not loaded")
toggle_label.pack()

load_button = tk.Button(root, text="Load Data", command=loaddata_button_pushed)
load_button.pack()

# Set the window icon
icon_path = 'logopython.ico'
root.iconbitmap(icon_path)

# Set background color of the main window
root.config(bg='#FFFFFF')

# Tabs
notebook = ttk.Notebook(root)
notebook.grid(row=0, column=0, columnspan=2, padx=10, pady=10, sticky='nsew')

# Place the notebook widget in the grid
notebook.grid(row=0, column=0, sticky="nsew")

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
model_tab = tk.Frame(notebook)
feeding_tab = tk.Frame(notebook)

notebook.add(load_tab, text="Load Data")
notebook.add(preprocess_tab, text="Preprocessing")
notebook.add(model_tab, text="Model Estimation")
notebook.add(control_tab, text="Control System")
notebook.add(feeding_tab, text="Feeding Schedule")

# Function to load CSV data using the csv module
def load_csv_data():
    try:
        # Open file dialog to choose CSV file
        filepath = filedialog.askopenfilename(title="Select CSV file", filetypes=[("CSV files", "*.csv")])
        
        # Load the CSV using the built-in csv module
        if filepath:
            with open(filepath, newline='') as csvfile:
                data = list(csv.reader(csvfile))  # Read CSV into a list
            return data
        else:
            return None
    except Exception as e:
        print(f"Error loading CSV: {e}")
        return None

# Function to handle 'Load Data' button click
def loaddata_button_pushed():
    data = load_csv_data()
    if data is not None:        
        # Switch the toggle to "Yes"
        update_toggle(state=True)

       
    else:
        # Handle the case where data loading failed
        update_toggle(state=False)

# CSV load
load_frame = tk.LabelFrame(load_tab,borderwidth=0, relief="flat",bg="#F0F0F0", padx=10, pady=0.5)
load_frame.grid(row=0, column=0, columnspan=2, padx=(10, 10), pady=0, sticky="nsew")

# Press button text
press_button = tk.Label(load_frame, text="Press the button to load CSV data", font=("Arial", 12, "bold"), background="#F0F0F0")
press_button.grid(row=0, column=0, padx=10, pady=(10,5), sticky="ew")

# Button
load_button = tk.Button(load_frame, text="Load Data", font=("Arial", 10), bd=1, relief="solid", command=loaddata_button_pushed)
load_button.grid(row=1, column=0, padx=(20, 20), pady=3, ipadx=15, ipady=0) 

# Load frame2 for radiobutton and labels
load_frame2 = tk.Frame(load_frame)
load_frame2.grid(row=0, column=1, rowspan=2, padx=0, pady=(10,5), sticky="nsew")


# Load frame2 for radio buttons and labels
load_frame2 = tk.Frame(load_frame)
load_frame2.grid(row=0, column=1, rowspan=2, padx=0, pady=(10,5), sticky="nsew")

# Create variable for radio buttons
toggle_var = tk.StringVar(value="No")  # Default to "No"

# Radio button for "No"
no_radio = tk.Radiobutton(load_frame2, text="No", variable=toggle_var, value="No", bg="#F0F0F0", font=("Arial", 12))
no_radio.grid(row=0, column=0, padx=10, pady=(10,25), sticky="ns")

# Radio button for "Yes"
yes_radio = tk.Radiobutton(load_frame2, text="Yes", variable=toggle_var, value="Yes", bg="#F0F0F0", font=("Arial", 12))
yes_radio.grid(row=0, column=2, padx=0, pady=(10,25), sticky="ns")

def load_csv_data(filepath):
    try:
        # Assuming you're using pandas to load the CSV data
        import pandas as pd
        data = pd.read_csv(filepath)
        return data
    except Exception as e:
        messagebox.showerror("Error", f"Failed to load data: {str(e)}")
        return None

def loaddata_button_pushed():
    # Open a file dialog to select a CSV file
    filepath = filedialog.askopenfilename(
        title="Select CSV File", 
        filetypes=(("CSV files", "*.csv"), ("All files", "*.*"))
    )
    if filepath:  # If a file is selected
        data = load_csv_data(filepath)
        if data is not None:        
            # Switch the radio button to "Yes"
            toggle_var.set("Yes")
            toggle_label.config(text="Data load is complete")  # Update label text
        else:
            # Handle the case where data loading failed
            toggle_var.set("No")
            toggle_label.config(text="Data load failed")  # Update label text
    else:
        # Handle the case where no file was selected
        toggle_var.set("No")
        toggle_label.config(text="No file selected")  # Update label text


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

plot_down_button = tk.Button(down_frame, text="Plot data step downwards", font=("Arial", 10), bd=1, relief="solid")
plot_down_button.grid(row=2, column=0, columnspan=4, padx=10, pady=(20,0), sticky='n')

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
up_frame.grid_rowconfigure(0, weight=0)  # Ensure the title row does not expand

# Start and End date labels and DateEntry widgets in the same row for Step Up
start_label_up = tk.Label(up_frame, text="Start date", bg="#F0F0F0")
start_label_up.grid(row=1, column=0, padx=2, pady=5, sticky='e')

start_date_up = DateEntry(up_frame, width=12, background='darkblue', foreground='white', borderwidth=2)
start_date_up.grid(row=1, column=1, padx=(2,0), pady=5, sticky='w')

end_label_up = tk.Label(up_frame, text="End date", bg="#F0F0F0")
end_label_up.grid(row=1, column=2, padx=(0,2), pady=5, sticky='e')

end_date_up = DateEntry(up_frame, width=12, background='darkblue', foreground='white', borderwidth=2)
end_date_up.grid(row=1, column=3, padx=0, pady=5, sticky='w')

plot_up_button = tk.Button(up_frame, text="Plot data step upwards", font=("Arial", 10), bd=1, relief="solid")
plot_up_button.grid(row=2, column=0, columnspan=4, padx=10, pady=(20,0), sticky='n')

# Create a new frame for charts
chart_frame = tk.LabelFrame(load_tab, borderwidth=0, relief="flat", bg="#F0F0F0",width=600, height=300)
chart_frame.grid(row=3, column=0,columnspan=2,padx=10, pady=0, sticky="nsew")
chart_frame.grid_propagate(False)  # Prevent the frame from resizing to fit its contents

# Configure the chart frame for 2-column layout
chart_frame.grid_columnconfigure(0, weight=1)  # First column for the first chart
chart_frame.grid_columnconfigure(1, weight=1)  # Second column for the second chart
chart_frame.grid_rowconfigure(0, weight=1)     # Ensure the row expands to fit both charts

# Sample data for biogas production before and after processing
before_processing = [100, 120, 110, 130, 115]
after_processing = [80, 90, 100, 85, 95]

# Create the first chart (Before data processing)
fig1 = Figure(figsize=(3, 0), dpi=35)
fig1.patch.set_facecolor('#F0F0F0')  # Set the background color of the figure (outside the plot area)
ax1 = fig1.add_subplot(111)
ax1.plot(before_processing, marker='o', color='#9AC1D9', label='Before Processing')
ax1.set_title('Before data preprocessing', fontsize=20, fontweight='bold', fontname='Arial')
ax1.set_xlabel('Time', fontsize=20)
ax1.set_ylabel('Gas Production flow rate [m³/h]', fontsize=20)
ax1.legend()

# Create the second chart (After data processing)
fig2 = Figure(figsize=(3, 0), dpi=35)
fig2.patch.set_facecolor('#F0F0F0')  # Set the background color of the figure (outside the plot area)
ax2 = fig2.add_subplot(111)
ax2.plot(after_processing, marker='o', color='#9AC1D9', label='After Processing')
ax2.set_title('After data preprocessing', fontsize=20, fontweight='bold', fontname='Arial')
ax2.set_xlabel('Time', fontsize=20)
ax2.set_ylabel('Gas Production flow rate [m³/h]', fontsize=20)
ax2.legend()

# Embed the first chart in the chart_frame
canvas1 = FigureCanvasTkAgg(fig1, master=chart_frame)
canvas1.draw()
canvas1.get_tk_widget().grid(row=0, column=0, padx=10, pady=10, sticky="nsew")

# Embed the second chart in the chart_frame
canvas2 = FigureCanvasTkAgg(fig2, master=chart_frame)
canvas2.draw()
canvas2.get_tk_widget().grid(row=0, column=1, padx=10, pady=10, sticky="nsew")

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

# Add a button to the middle column (column 1)
button = tk.Button(preprocess_frame, text="Preprocessing", font=("Arial", 10, "bold"), bd=1, relief="solid", padx=0, pady=0)
button.grid(row=0, column=1, padx=10, pady=10, ipadx=5,ipady=5,sticky="")  # Leave sticky="" for it to stay centered

# Configure load_tab to expand the frame
load_tab.grid_rowconfigure(4, weight=1)
load_tab.grid_columnconfigure(0, weight=1)


# Run the main loop
root.mainloop()