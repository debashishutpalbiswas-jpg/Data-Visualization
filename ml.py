import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# Main Application Class
class DataAnalysisApp:
    def __init__(self, root):
        """
        Initialize the main application window and its widgets.
        """
        self.root = root
        self.root.title("Data Analysis and Visualization Tool")
        self.root.geometry("1000x800")
        self.root.configure(bg="#f0f0f0")

        self.df = None
        self.file_path = ""

        # --- Main Layout Frames ---
        control_frame = tk.Frame(root, bg="#dcdcdc", padx=10, pady=10)
        control_frame.pack(side=tk.TOP, fill=tk.X)

        self.plot_frame = tk.Frame(root, bg="white", padx=10, pady=10)
        self.plot_frame.pack(side=tk.BOTTOM, fill=tk.BOTH, expand=True)

        # --- Control Widgets ---
        # File Loading
        self.load_button = ttk.Button(control_frame, text="Load CSV File", command=self.load_csv)
        self.load_button.grid(row=0, column=0, padx=5, pady=5, sticky="ew")

        self.file_label = ttk.Label(control_frame, text="No file loaded.", background="#dcdcdc", wraplength=400)
        self.file_label.grid(row=0, column=1, columnspan=3, padx=5, pady=5, sticky="w")
        
        # Column Input
        ttk.Label(control_frame, text="Enter Columns (comma-separated):", background="#dcdcdc").grid(row=1, column=0, padx=5, pady=5, sticky="w")
        self.column_entry = ttk.Entry(control_frame, width=60)
        self.column_entry.grid(row=1, column=1, columnspan=3, padx=5, pady=5, sticky="ew")

        # --- Analysis Buttons ---
        analysis_frame = ttk.LabelFrame(control_frame, text="Analysis Options", padding=(10, 5))
        analysis_frame.grid(row=2, column=0, columnspan=4, padx=5, pady=10, sticky="ew")

        self.knn_button = ttk.Button(analysis_frame, text="K-Means Clustering", command=self.run_kmeans_clustering)
        self.knn_button.pack(side=tk.LEFT, padx=5, expand=True, fill=tk.X)

        self.lr_button = ttk.Button(analysis_frame, text="Linear Regression", command=self.run_linear_regression)
        self.lr_button.pack(side=tk.LEFT, padx=5, expand=True, fill=tk.X)

        self.mlr_button = ttk.Button(analysis_frame, text="Multiple Linear Regression", command=self.run_multilinear_regression)
        self.mlr_button.pack(side=tk.LEFT, padx=5, expand=True, fill=tk.X)

        # --- Matplotlib Canvas ---
        self.canvas_widget = None

    def load_csv(self):
        """
        Opens a file dialog to select a CSV file and loads it into a pandas DataFrame.
        """
        self.file_path = filedialog.askopenfilename(
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
        )
        if not self.file_path:
            return
        
        try:
            self.df = pd.read_csv(self.file_path)
            self.file_label.config(text=f"Loaded: {self.file_path.split('/')[-1]}")
            messagebox.showinfo("Success", f"CSV file loaded successfully!\n\nColumns: {', '.join(self.df.columns)}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load file: {e}")
            self.df = None
            self.file_label.config(text="No file loaded.")

    def _get_and_validate_columns(self):
        """
        Helper function to get column names from the entry box and validate them.
        """
        if self.df is None:
            messagebox.showwarning("Warning", "Please load a CSV file first.")
            return None

        cols_str = self.column_entry.get()
        if not cols_str:
            messagebox.showwarning("Warning", "Please enter column names.")
            return None

        columns = [col.strip() for col in cols_str.split(',')]
        
        # Check if all specified columns exist in the DataFrame
        missing_cols = [col for col in columns if col not in self.df.columns]
        if missing_cols:
            messagebox.showerror("Error", f"The following columns were not found: {', '.join(missing_cols)}")
            return None
            
        return columns
    
    def _clear_plot(self):
        """Clears the previous plot from the plot frame."""
        if self.canvas_widget:
            self.canvas_widget.destroy()

    def run_kmeans_clustering(self):
        """
        Performs K-Means clustering on the selected columns and visualizes the result.
        """
        columns = self._get_and_validate_columns()
        if not columns or len(columns) != 2:
            messagebox.showerror("Error", "K-Means Clustering requires exactly 2 columns for visualization.")
            return

        try:
            data = self.df[columns].dropna()
            
            # Standardize the data
            scaler = StandardScaler()
            scaled_data = scaler.fit_transform(data)
            
            # Perform K-Means clustering (let's assume 3 clusters for simplicity)
            kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
            clusters = kmeans.fit_predict(scaled_data)

            self._clear_plot()
            fig, ax = plt.subplots(figsize=(8, 6))
            scatter = ax.scatter(scaled_data[:, 0], scaled_data[:, 1], c=clusters, cmap='viridis', marker='o')
            ax.set_title('K-Means Clustering Results (k=3)')
            ax.set_xlabel(f"Standardized {columns[0]}")
            ax.set_ylabel(f"Standardized {columns[1]}")
            ax.legend(handles=scatter.legend_elements()[0], labels=['Cluster 0', 'Cluster 1', 'Cluster 2'])
            ax.grid(True)
            
            self._embed_plot(fig)
        except Exception as e:
            messagebox.showerror("Clustering Error", f"An error occurred: {e}")

    def run_linear_regression(self):
        """
        Performs Linear Regression and visualizes the regression line.
        """
        columns = self._get_and_validate_columns()
        if not columns or len(columns) != 2:
            messagebox.showerror("Error", "Linear Regression requires exactly 2 columns: one independent and one dependent.")
            return

        try:
            independent_var, dependent_var = columns[0], columns[1]
            data = self.df[[independent_var, dependent_var]].dropna()
            X = data[[independent_var]].values
            y = data[dependent_var].values

            model = LinearRegression()
            model.fit(X, y)
            y_pred = model.predict(X)
            r2 = r2_score(y, y_pred)

            self._clear_plot()
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.scatter(X, y, color='blue', label='Actual Data')
            ax.plot(X, y_pred, color='red', linewidth=2, label='Regression Line')
            ax.set_title(f'Linear Regression\n(R-squared: {r2:.2f})')
            ax.set_xlabel(independent_var)
            ax.set_ylabel(dependent_var)
            ax.legend()
            ax.grid(True)
            
            self._embed_plot(fig)
        except Exception as e:
            messagebox.showerror("Regression Error", f"An error occurred: {e}")

    def run_multilinear_regression(self):
        """
        Performs Multiple Linear Regression and plots actual vs. predicted values.
        """
        columns = self._get_and_validate_columns()
        if not columns or len(columns) < 3:
            messagebox.showerror("Error", "Multiple Linear Regression requires at least 3 columns (2+ independent, 1 dependent).")
            return

        try:
            independent_vars = columns[:-1]
            dependent_var = columns[-1]
            data = self.df[columns].dropna()

            X = data[independent_vars]
            y = data[dependent_var]
            
            # Splitting data for a more realistic performance measure
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            model = LinearRegression()
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            r2 = r2_score(y_test, y_pred)

            self._clear_plot()
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.scatter(y_test, y_pred, alpha=0.7, edgecolors='k')
            # Add a line for perfect correlation
            ax.plot([y.min(), y.max()], [y.min(), y.max()], 'r--', lw=2)
            ax.set_xlabel("Actual Values")
            ax.set_ylabel("Predicted Values")
            ax.set_title(f"Multiple Linear Regression: Actual vs. Predicted\n(R-squared on test set: {r2:.2f})")
            ax.grid(True)

            self._embed_plot(fig)
        except Exception as e:
            messagebox.showerror("Regression Error", f"An error occurred: {e}")

    def _embed_plot(self, fig):
        """Embeds a matplotlib figure into the Tkinter window."""
        canvas = FigureCanvasTkAgg(fig, master=self.plot_frame)
        self.canvas_widget = canvas.get_tk_widget()
        self.canvas_widget.pack(fill=tk.BOTH, expand=True)
        canvas.draw()


if __name__ == "__main__":
    root = tk.Tk()
    app = DataAnalysisApp(root)
    root.mainloop()
