import tkinter as tk
from tkinter import filedialog, messagebox
import pandas as pd
from ydata_profiling import ProfileReport
import webbrowser
import os

def generate_report():
    file_path = filedialog.askopenfilename(
        title="Select a CSV file",
        filetypes=(("CSV files", "*.csv"), ("All files", "*.*"))
    )
    
    if not file_path:
        
        return
        
    try:
        
        status_label.config(text=f"Processing: {os.path.basename(file_path)}...")
        root.update_idletasks() 
        
        
        df = pd.read_csv(file_path)
        profile = ProfileReport(df, title=f"Analysis of {os.path.basename(file_path)}")
        
        
        report_path = "desktop_app_report.html"
        profile.to_file(report_path)
        webbrowser.open('file://' + os.path.realpath(report_path))
        
        status_label.config(text="Report generated and opened successfully!")
        
    except Exception as e:
        messagebox.showerror("Error", f"An error occurred:\n{e}")
        status_label.config(text="An error occurred.")


root = tk.Tk()
root.title("Data Report Generator")
root.geometry("400x150")


label = tk.Label(root, text="Select a CSV file to generate an analysis report.")

label.pack(pady=10)

analyze_button = tk.Button(root, text="Select and Analyze CSV", command=generate_report, width=30, height=2)
analyze_button.pack(pady=10)

status_label = tk.Label(root, text="", fg="blue")
status_label.pack()


root.mainloop()
