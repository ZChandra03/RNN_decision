import sys
import numpy as np
from PyQt5.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget
from PyQt5.QtCore import QTimer
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import matplotlib.pyplot as plt

class MplCanvas(FigureCanvas):
    def __init__(self, parent=None, width=5, height=4, dpi=100):
        # Create the figure and axes
        fig, self.ax = plt.subplots(figsize=(width, height), dpi=dpi)
        super().__init__(fig)

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Animated Scatter Plot (PyQt5 + Matplotlib)")

        # Create the Matplotlib canvas
        self.canvas = MplCanvas(self, width=6, height=6, dpi=100)

        # Generate some “random walk” data: 50 points over 150 frames
        self.num_points = 50
        self.num_frames = 150
        self.data = np.cumsum(np.random.randn(self.num_frames, self.num_points, 2), axis=0)

        # Initial scatter
        self.scatter = self.canvas.ax.scatter(
            self.data[0, :, 0],
            self.data[0, :, 1],
            s=30,
            alpha=0.8
        )
        self.canvas.ax.set_xlim(self.data[:, :, 0].min() - 1, self.data[:, :, 0].max() + 1)
        self.canvas.ax.set_ylim(self.data[:, :, 1].min() - 1, self.data[:, :, 1].max() + 1)

        # Layout
        container = QWidget()
        layout = QVBoxLayout()
        layout.addWidget(self.canvas)
        container.setLayout(layout)
        self.setCentralWidget(container)

        # Timer for animation
        self.frame = 0
        self.timer = QTimer(self)
        self.timer.setInterval(50)  # milliseconds between frames
        self.timer.timeout.connect(self.update_plot)
        self.timer.start()

    def update_plot(self):
        self.frame = (self.frame + 1) % self.num_frames
        # Update scatter data
        self.scatter.set_offsets(self.data[self.frame])
        self.canvas.draw_idle()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    w = MainWindow()
    w.show()
    sys.exit(app.exec_())
