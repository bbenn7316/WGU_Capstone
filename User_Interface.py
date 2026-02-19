from datetime import time
from time import sleep
from tkinter import *
from tkinter import ttk
import numpy as np
from Data_Model_Creation import *
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import (FigureCanvasTkAgg)
import matplotlib.pyplot as plt

class model_ui:

    def __init__(self, root, df):
        self.df = df

        root.title("Model Representation UI")

        self.mainframe = ttk.Frame(root, padding=(3, 3, 3, 3))
        self.mainframe.grid(column=0, row=0, sticky=(N, W, E, S))

        self.mae = StringVar(self.mainframe)
        self.mse = StringVar(self.mainframe)
        self.r2 = StringVar(self.mainframe)
        self.genre = StringVar(self.mainframe)
        genre_entry = ttk.Entry(self.mainframe, width=10, textvariable=self.genre)
        genre_entry.grid(column=2, row=2, sticky=(W, E))
        self.x = []
        self.y = []

        self.b = ttk.Button(self.mainframe, text="Create", command=self.create_model)
        self.b.grid(column=3, row=2, sticky=W)
        self.rmselbl = ttk.Label(self.mainframe, textvariable=self.mse, borderwidth=2, relief="groove")
        self.rmselbl.grid(column=2, row=1, sticky="NWES")

        self.r2lbl = ttk.Label(self.mainframe, textvariable=self.r2, borderwidth=2, relief="groove")
        self.r2lbl.grid(column=3, row=1, sticky="NWES")

        self.maelbl = ttk.Label(self.mainframe, textvariable=self.mae, borderwidth=2, relief="groove")
        self.maelbl.grid(column=4, row=1, sticky="NWES")

        self.txt1 = ttk.Label(self.mainframe, text="MSE", borderwidth=2, relief="groove")
        self.txt1.grid(column=2, row=0, sticky="NWES")

        self.txt2 = ttk.Label(self.mainframe, text="R2 Score", borderwidth=2, relief="groove")
        self.txt2.grid(column=3, row=0, sticky="NWES")

        self.txt3 = ttk.Label(self.mainframe, text="MAE", borderwidth=2, relief="groove")
        self.txt3.grid(column=4, row=0, sticky="NWES")

        #self.mainframe2 = ttk.Frame(root, padding=(3, 3, 3, 3))
        #self.mainframe2.grid(column=0, row=1, sticky=(N, W, E, S))

        graph = plt.figure(figsize=(5, 4), dpi=100)
        self.ax = graph.add_subplot(111)
        self.ax.scatter(x=self.x, y=self.y, alpha=0.3)
        self.ax.set(xlabel="Positive Reviews", ylabel="Prediction")
        self.ax.set_title("LR Model For Genre - " + self.genre.get())
        self.canvas = FigureCanvasTkAgg(graph, master=self.mainframe)
        self.canvas.draw()
        self.canvas_grid = self.canvas.get_tk_widget()
        self.canvas_grid.grid(column=0, row=3, sticky=(N, W, E, S), padx=5, pady=5)

        bfl = Figure(figsize=(5, 4), dpi=100)
        self.ax2 = bfl.add_subplot(111)
        self.ax2.plot(self.x, self.y)
        self.ax2.set(xlabel="Positive Review", ylabel="Prediction")
        self.ax2.set_title("Best Fit Line For Genre - " + self.genre.get())
        self.canvas2 = FigureCanvasTkAgg(bfl, master=self.mainframe)
        self.canvas2.draw()
        self.canvas_grid2 = self.canvas2.get_tk_widget()
        self.canvas_grid2.grid(column=1, row=3, sticky=(N, W, E, S), padx=5, pady=5)

        root.columnconfigure(0, weight=1)
        root.rowconfigure(0, weight=1)

        self.mainframe.columnconfigure(0, weight=1)
        self.mainframe.rowconfigure(1, weight=1)
        for child in self.mainframe.winfo_children():
            child.grid_configure(padx=5, pady=5)

        genre_entry.focus()
        root.bind("<Return>", self.create_model)
        self.mainframe.pack()
        self.root = root
    def reset(self):
        self.x = []
        self.y = []
        self.mse.set("")
        self.r2.set("")
        self.mae.set("")
        self.ax.clear()
        self.ax2.clear()

    def create_model(self, *args):
        try:
            self.reset()
            userinput = self.genre.get()
            genredf = select_genre(userinput, self.df)
            print(1)
            genrex_train, genrex_test, genrey_train, genrey_test = separate(genredf)
            print(2)
            LR_model, train_pred, test_pred = create_MLM(genrex_train, genrex_test, genrey_train, genrey_test)
            print(3)
            rmse_train, r2_train, rmse_test, r2_test, mae_test = RMSE_R2(genrey_train, genrey_test, train_pred, test_pred)
            print(4)
            self.x = genrey_test
            self.y = test_pred

            self.mse.set(round(rmse_test, 3))
            self.r2.set(round(r2_test, 3))
            self.mae.set(round(mae_test, 3))

            self.ax.clear()
            self.ax.scatter(x=self.x, y=self.y, alpha=0.3)
            self.ax.set(xlabel="Positive Reviews", ylabel="Prediction")
            scattertitle = "LR Model For Genre - " + self.genre.get()
            self.ax.set_title(scattertitle)
            self.canvas.draw()

            self.ax2.clear()
            m, b = np.polyfit(np.array(genrey_train, dtype=float), train_pred, 1)
            line_x = np.linspace(min(genrey_train), max(genrey_train), 1000000)
            liney = (m * line_x) + b
            self.ax2.plot(line_x, liney, linewidth=2)
            self.ax2.set(xlabel="Positive Reviews", ylabel="Prediction")
            title = "Best Fit Line For Genre - " + self.genre.get()
            self.ax2.set_title(title)
            self.canvas2.draw()
            #self.root.update()

        except ValueError:
            pass
    def run(self):
        self.root.mainloop()

df = read_convert_json_dataset()
cleanup_df(df)
root = Tk()
model_ui(root, df).run()