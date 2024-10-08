import tkinter as tk
from tkinter import filedialog

def load_file():
    # Open file dialog to select a CSV file
    filepath = filedialog.askopenfilename(
        title="Select CSV File", 
        filetypes=(("CSV files", "*.csv"), ("All files", "*.*"))
    )

    if filepath:  # If a file is selected
        radio_var.set("Yes")  # Set the radio button to "Yes"
        status_label.config(text="File selected: " + filepath)  # Show selected file path
    else:
        radio_var.set("No")  # Set the radio button to "No"
        status_label.config(text="No file selected")  # Update status

# Create the main window
root = tk.Tk()
root.title("CSV File Selector")

# Create a StringVar to hold the value of the radio buttons
radio_var = tk.StringVar(value="No")

# Create radio buttons for "Yes" and "No"
radio_yes = tk.Radiobutton(root, text="Yes", variable=radio_var, value="Yes")
radio_no = tk.Radiobutton(root, text="No", variable=radio_var, value="No")

# Place the radio buttons using grid layout
radio_yes.grid(row=0, column=0, sticky="w", padx=10, pady=5)
radio_no.grid(row=0, column=1, sticky="w", padx=10, pady=5)

# Create a button to load the file
load_button = tk.Button(root, text="Load CSV", command=load_file)
load_button.grid(row=1, column=0, columnspan=2, padx=10, pady=10)

# Label to display the file selection status
status_label = tk.Label(root, text="No file selected")
status_label.grid(row=2, column=0, columnspan=2, padx=10, pady=10)

# Run the main loop
root.mainloop()
