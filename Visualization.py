import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from statsmodels.tsa.seasonal import seasonal_decompose
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import numpy as np

class DataVisualizer:
    def __init__(self, df: pd.DataFrame):
        self.df = df

    def plot_histogram(self, column: str, bins: int = 10):
        plt.figure(figsize=(8, 5))
        sns.histplot(data=self.df, x=column, bins=bins, kde=True)
        plt.title(f'Histogram of {column}')
        plt.xlabel(column)
        plt.ylabel('Frequency')
        plt.show()

    def plot_bar(self, column: str):
        plt.figure(figsize=(8, 5))
        sns.countplot(data=self.df, x=column)
        plt.title(f'Bar Plot of {column}')
        plt.xlabel(column)
        plt.ylabel('Count')
        plt.show()

    def plot_box(self, column: str):
        plt.figure(figsize=(8, 5))
        sns.boxplot(data=self.df, y=column)
        plt.title(f'Box Plot of {column}')
        plt.ylabel(column)
        plt.show()

    def plot_scatter(self, col_x: str, col_y: str):
        plt.figure(figsize=(8, 5))
        sns.scatterplot(data=self.df, x=col_x, y=col_y)
        plt.title(f'Scatter Plot: {col_x} vs {col_y}')
        plt.xlabel(col_x)
        plt.ylabel(col_y)
        plt.show()

    def plot_line(self, col_x: str, col_y: str):
        plt.figure(figsize=(8, 5))
        sns.lineplot(data=self.df, x=col_x, y=col_y)
        plt.title(f'Line Plot: {col_x} vs {col_y}')
        plt.xlabel(col_x)
        plt.ylabel(col_y)
        plt.show()

    def plot_area(self, col_x: str, col_y: str):
        plt.figure(figsize=(8, 5))
        plt.fill_between(self.df[col_x], self.df[col_y], alpha=0.4)
        plt.plot(self.df[col_x], self.df[col_y], linewidth=2)
        plt.title(f'Area Plot: {col_x} vs {col_y}')
        plt.xlabel(col_x)
        plt.ylabel(col_y)
        plt.show()

    def plot_pie(self, column: str):
        plt.figure(figsize=(7, 7))
        self.df[column].value_counts().plot.pie(autopct='%1.1f%%', startangle=90)
        plt.title(f'Pie Chart of {column}')
        plt.ylabel('')
        plt.show()

    def plot_violin(self, column: str):
        plt.figure(figsize=(8, 5))
        sns.violinplot(data=self.df, y=column)
        plt.title(f'Violin Plot of {column}')
        plt.ylabel(column)
        plt.show()

    def plot_pairplot(self, columns: list = None):
        if columns:
            sns.pairplot(self.df[columns])
        else:
            sns.pairplot(self.df)
        plt.suptitle('Pair Plot', y=1.02)
        plt.show()

    def plot_correlation_heatmap(self):
        plt.figure(figsize=(10, 7))
        corr = self.df.corr(numeric_only=True)
        sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f")
        plt.title('Correlation Heatmap')
        plt.show()

    def plot_time_series(self, date_col: str, value_col: str):
        plt.figure(figsize=(10, 6))
        sns.lineplot(data=self.df, x=date_col, y=value_col)
        plt.title(f'Time Series Plot: {value_col} over time')
        plt.xlabel('Date')
        plt.ylabel(value_col)
        plt.show()

    def plot_rolling_average(self, date_col: str, value_col: str, window: int = 7):
        plt.figure(figsize=(10, 6))
        self.df_sorted = self.df.sort_values(by=date_col)
        rolling_avg = self.df_sorted[value_col].rolling(window=window).mean()
        plt.plot(self.df_sorted[date_col], self.df_sorted[value_col], label='Original')
        plt.plot(self.df_sorted[date_col], rolling_avg, label=f'{window}-Day Rolling Average', linewidth=2)
        plt.title(f'Rolling Average of {value_col}')
        plt.xlabel('Date')
        plt.ylabel(value_col)
        plt.legend()
        plt.show()

    def plot_seasonal_decompose(self, date_col: str, value_col: str, period: int = 12):
        self.df_sorted = self.df.sort_values(by=date_col)
        self.df_sorted = self.df_sorted.set_index(date_col)
        result = seasonal_decompose(self.df_sorted[value_col], model='additive', period=period)
        result.plot()
        plt.suptitle(f'Seasonal Decomposition of {value_col}', y=1.02)
        plt.show()

    def plot_3d_scatter(self, col_x: str, col_y: str, col_z: str):
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(self.df[col_x], self.df[col_y], self.df[col_z], c='skyblue', s=50)
        ax.set_xlabel(col_x)
        ax.set_ylabel(col_y)
        ax.set_zlabel(col_z)
        ax.set_title(f'3D Scatter Plot: {col_x}, {col_y}, {col_z}')
        plt.show()

    def plot_3d_line(self, col_x: str, col_y: str, col_z: str):
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection='3d')
        ax.plot(self.df[col_x], self.df[col_y], self.df[col_z], color='green', linewidth=2)
        ax.set_xlabel(col_x)
        ax.set_ylabel(col_y)
        ax.set_zlabel(col_z)
        ax.set_title(f'3D Line Plot: {col_x}, {col_y}, {col_z}')
        plt.show()

    def plot_3d_surface(self, col_x: str, col_y: str, col_z: str):
        X = self.df[col_x]
        Y = self.df[col_y]
        Z = self.df[col_z]

        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection='3d')

        X_unique = np.unique(X)
        Y_unique = np.unique(Y)
        X_grid, Y_grid = np.meshgrid(X_unique, Y_unique)
        Z_grid = np.zeros_like(X_grid, dtype=float)

        for i in range(len(X)):
            xi = np.where(X_unique == X.iloc[i])[0][0]
            yi = np.where(Y_unique == Y.iloc[i])[0][0]
            Z_grid[yi, xi] = Z.iloc[i]

        surf = ax.plot_surface(X_grid, Y_grid, Z_grid, cmap=cm.viridis, edgecolor='k', alpha=0.8)
        fig.colorbar(surf, shrink=0.5, aspect=5)
        ax.set_xlabel(col_x)
        ax.set_ylabel(col_y)
        ax.set_zlabel(col_z)
        ax.set_title(f'3D Surface Plot: {col_x}, {col_y}, {col_z}')
        plt.show()


class DataVizApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Data Visualization App")
        self.df = None
        self.visualizer = None

        self.upload_btn = tk.Button(root, text="Upload CSV", command=self.load_file)
        self.upload_btn.pack(pady=10)

        self.plot_type = tk.StringVar()
        self.plot_options = [
            "Histogram", "Bar Plot", "Box Plot", "Scatter Plot", "Line Plot", "Area Plot", 
            "Pie Chart", "Violin Plot", "Pair Plot", "Correlation Heatmap", 
            "Time Series", "Rolling Average", "Seasonal Decompose",
            "3D Scatter", "3D Line", "3D Surface"
        ]
        self.plot_menu = ttk.Combobox(root, textvariable=self.plot_type, values=self.plot_options)
        self.plot_menu.set("Select Plot Type")
        self.plot_menu.pack(pady=10)

        self.col_entry = tk.Entry(root, width=60)
        self.col_entry.pack(pady=5)
        self.col_entry.insert(0, "Enter column(s), comma separated")

        self.plot_btn = tk.Button(root, text="Generate Plot", command=self.generate_plot)
        self.plot_btn.pack(pady=20)

    def load_file(self):
        file_path = filedialog.askopenfilename(filetypes=[("CSV Files", "*.csv")])
        if file_path:
            try:
                self.df = pd.read_csv(file_path)
                self.visualizer = DataVisualizer(self.df)
                messagebox.showinfo("Success", "CSV file loaded successfully!")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load CSV: {e}")

    def generate_plot(self):
        if self.df is None:
            messagebox.showerror("Error", "Please upload a CSV first!")
            return

        plot_choice = self.plot_type.get()
        cols = [c.strip() for c in self.col_entry.get().split(",")]

        try:
            if plot_choice == "Histogram" and len(cols) == 1:
                self.visualizer.plot_histogram(cols[0])
            elif plot_choice == "Bar Plot" and len(cols) == 1:
                self.visualizer.plot_bar(cols[0])
            elif plot_choice == "Box Plot" and len(cols) == 1:
                self.visualizer.plot_box(cols[0])
            elif plot_choice == "Scatter Plot" and len(cols) == 2:
                self.visualizer.plot_scatter(cols[0], cols[1])
            elif plot_choice == "Line Plot" and len(cols) == 2:
                self.visualizer.plot_line(cols[0], cols[1])
            elif plot_choice == "Area Plot" and len(cols) == 2:
                self.visualizer.plot_area(cols[0], cols[1])
            elif plot_choice == "Pie Chart" and len(cols) == 1:
                self.visualizer.plot_pie(cols[0])
            elif plot_choice == "Violin Plot" and len(cols) == 1:
                self.visualizer.plot_violin(cols[0])
            elif plot_choice == "Pair Plot":
                self.visualizer.plot_pairplot(cols if len(cols) > 1 else None)
            elif plot_choice == "Correlation Heatmap":
                self.visualizer.plot_correlation_heatmap()
            elif plot_choice == "Time Series" and len(cols) == 2:
                self.visualizer.plot_time_series(cols[0], cols[1])
            elif plot_choice == "Rolling Average" and len(cols) == 2:
                self.visualizer.plot_rolling_average(cols[0], cols[1])
            elif plot_choice == "Seasonal Decompose" and len(cols) == 2:
                self.visualizer.plot_seasonal_decompose(cols[0], cols[1])
            elif plot_choice == "3D Scatter" and len(cols) == 3:
                self.visualizer.plot_3d_scatter(cols[0], cols[1], cols[2])
            elif plot_choice == "3D Line" and len(cols) == 3:
                self.visualizer.plot_3d_line(cols[0], cols[1], cols[2])
            elif plot_choice == "3D Surface" and len(cols) == 3:
                self.visualizer.plot_3d_surface(cols[0], cols[1], cols[2])
            else:
                messagebox.showerror("Error", "Invalid column input for selected plot.")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to generate plot: {e}")


if __name__ == "__main__":
    root = tk.Tk()
    app = DataVizApp(root)
    root.mainloop()