import tkinter as tk
from tkinter import ttk, filedialog
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
import pandas as pd

# Initialize empty variables to hold the data
biogas_data = None
substrate_data = None

# Function to load the CSV files
def load_files():
    global biogas_data, substrate_data
    
    # Open file dialog to select the first CSV file (Biogas production rate)
    biogas_file = filedialog.askopenfilename(title="Select Biogas Production CSV", 
                                             filetypes=[("CSV files", "*.csv")])
    # Open file dialog to select the second CSV file (Substrate feeding)
    substrate_file = filedialog.askopenfilename(title="Select Substrate Feeding CSV", 
                                                filetypes=[("CSV files", "*.csv")])
    
    # Load the data from the selected CSV files
    biogas_data = pd.read_csv(biogas_file)
    substrate_data = pd.read_csv(substrate_file)
    
    # Convert the TimeStamp columns to datetime
    biogas_data['TimeStamp'] = pd.to_datetime(biogas_data['TimeStamp'], format='%d.%m.%Y %H:%M:%S')
    substrate_data['TimeStamp'] = pd.to_datetime(substrate_data['TimeStamp'], format='%d.%m.%Y %H:%M:%S')
    
    # Enable the date entry widgets and plot button once files are loaded
    start_date_entry.config(state='normal')
    end_date_entry.config(state='normal')
    plot_button.config(state='normal')

# Function to create and display the upward step plot
def create_plot_step_upward():
    if biogas_data is not None and substrate_data is not None:
        # Get the start and end dates from the entry widgets
        start_date = start_date_entry.get()
        end_date = end_date_entry.get()

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

            # Set the title for the upward plot
            ax1.set_title('Plot Data Step Upward', fontsize=20, fontweight='bold', fontname='Arial')

            # Second y-axis for substrate feeding
            ax2 = ax1.twinx()
            ax2.step(filtered_substrate['TimeStamp'], filtered_substrate['ValueNum'], 'r-', label='Substrate feeding (Step Upward)')
            ax2.set_ylabel('Substrate feeding [t]', color='r')
            ax2.tick_params(axis='y', labelcolor='r')

            # Add legend and grid
            fig.tight_layout()
            fig.legend(loc='upper right', bbox_to_anchor=(0.85, 0.85))
            plt.grid(True)

            # Clear the previous plot and add the new one to the canvas
            canvas.get_tk_widget().pack_forget()
            canvas.get_tk_widget().pack()
            canvas.figure = fig
            canvas.draw()
        except Exception as e:
            error_label.config(text=f"Error: {e}")

# Create the main window
window = tk.Tk()
window.title("Biogas Production and Substrate Feeding - Step Upward")

# Create a frame for the buttons and plot
frame = ttk.Frame(window)
frame.pack(padx=10, pady=10)

# Create the 'Load' button to select files
load_button = ttk.Button(frame, text="Load CSV Files", command=load_files)
load_button.pack(pady=10)

# Date entry fields for start and end date
ttk.Label(frame, text="Start Date (YYYY-MM-DD):").pack(pady=5)
start_date_entry = ttk.Entry(frame, state='disabled')
start_date_entry.pack()

ttk.Label(frame, text="End Date (YYYY-MM-DD):").pack(pady=5)
end_date_entry = ttk.Entry(frame, state='disabled')
end_date_entry.pack()

# Button to plot the upward step graph
plot_button = ttk.Button(frame, text="Plot Data Step Upward", state='disabled', command=create_plot_step_upward)
plot_button.pack(pady=10)

# Label to display errors if any
error_label = ttk.Label(frame, text="", foreground="red")
error_label.pack()

# Placeholder for the plot
fig, ax = plt.subplots(figsize=(10, 5))
canvas = FigureCanvasTkAgg(fig, master=frame)
canvas.get_tk_widget().pack()

# Start the Tkinter main loop
window.mainloop()
