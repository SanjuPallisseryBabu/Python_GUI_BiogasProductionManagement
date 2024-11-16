import tkinter as tk
from tkinter import ttk
from tkcalendar import DateEntry
import os
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter
import matplotlib.ticker as mticker

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

            # If data loaded successfully, display a success message
            success_label.config(text="Data loaded successfully!")
        except Exception as e:
            success_label.config(text=f"Error: {e}")
    else:
        success_label.config(text=f"Error: The folder '{folder_path}' does not exist.")

def plot_step_graph(upwards=False):
    global gas_flow_rate_df, substrate_feeding_df

    if gas_flow_rate_df.empty or substrate_feeding_df.empty:
        success_label.config(text="Error: Data not loaded. Please load the data first.")
        return

    try:
        # Get selected date range from calendar widgets
        start_date = start_date_entry.get_date()
        end_date = end_date_entry.get_date()

        # Filter the data based on the selected date range
        mask_gas = (gas_flow_rate_df['TimeStamp'] >= pd.Timestamp(start_date)) & (gas_flow_rate_df['TimeStamp'] <= pd.Timestamp(end_date))
        mask_substrate = (substrate_feeding_df['TimeStamp'] >= pd.Timestamp(start_date)) & (substrate_feeding_df['TimeStamp'] <= pd.Timestamp(end_date))

        filtered_gas_flow_rate_df = gas_flow_rate_df.loc[mask_gas]
        filtered_substrate_feeding_df = substrate_feeding_df.loc[mask_substrate]

        # Plotting the graph with two y-axes
        fig, ax1 = plt.subplots(figsize=(12, 6))

        # Determine plotting direction
        where = 'pre' if upwards else 'post'
        plot_title = 'Before Data Preprocessing (Upwards)' if upwards else 'Before Data Preprocessing (Downwards)'

        # Plot Gas Flow Rate on the first y-axis
        ax1.step(filtered_gas_flow_rate_df['TimeStamp'], filtered_gas_flow_rate_df['ValueNum'], where=where, label='Biogas Production Rate', linestyle='-', color='b')
        ax1.set_xlabel('Time', fontsize=12)
        ax1.set_ylabel('Biogas Production Rate [m3/h]', color='b', fontsize=12)
        ax1.tick_params(axis='y', labelcolor='b')
        ax1.xaxis.set_major_formatter(DateFormatter('%b %d, %H:%M'))  # Format as 'Oct 07, 00:00' and 'Oct 07, 12:00'  # Format as 'Oct 07, 00:00'
        ax1.xaxis.set_major_locator(mticker.MultipleLocator(0.5))  # Custom locator to ensure labels like 'Oct 07, 00:00' and 'Oct 07, 12:00' are evenly spaced  # Ensure dates are spaced well without repetition
        plt.xticks(rotation=0)  # Rotate x-axis labels for better readability

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
        plt.show()
    except Exception as e:
        success_label.config(text=f"Error while plotting: {e}")

# Create the main application window
root = tk.Tk()
root.title("Python GUI with Load and Plot Tab")
root.geometry("600x400")

# Create a Notebook widget for tabs
notebook = ttk.Notebook(root)
notebook.pack(expand=True, fill='both')

# Create the "Load" tab frame
load_tab = ttk.Frame(notebook)
notebook.add(load_tab, text='Load')

# Add a button to the "Load" tab to load data
load_data_button = ttk.Button(load_tab, text='Load Data', command=load_data)
load_data_button.pack(pady=20)

# Add a label to display success or error messages
success_label = ttk.Label(load_tab, text="")
success_label.pack(pady=10)

# Create a "Plot" tab frame
plot_tab = ttk.Frame(notebook)
notebook.add(plot_tab, text='Plot')

# Add date range selection widgets to the "Plot" tab
start_date_label = ttk.Label(plot_tab, text="Start Date:")
start_date_label.pack(pady=5)
start_date_entry = DateEntry(plot_tab, width=12, background='darkblue', foreground='white', borderwidth=2)
start_date_entry.pack(pady=5)

end_date_label = ttk.Label(plot_tab, text="End Date:")
end_date_label.pack(pady=5)
end_date_entry = DateEntry(plot_tab, width=12, background='darkblue', foreground='white', borderwidth=2)
end_date_entry.pack(pady=5)

# Add a button to the "Plot" tab to plot the downward graph
plot_down_button = ttk.Button(plot_tab, text='Plot Down', command=lambda: plot_step_graph(upwards=False))
plot_down_button.pack(pady=10)

# Add a button to the "Plot" tab to plot the upward graph
plot_up_button = ttk.Button(plot_tab, text='Plot Up', command=lambda: plot_step_graph(upwards=True))
plot_up_button.pack(pady=10)

# Run the application
root.mainloop()
