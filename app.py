import os
import pydicom
import numpy as np
from PyQt6.QtWidgets import QApplication, QMainWindow, QPushButton, QLabel, QFileDialog, QVBoxLayout, QWidget
from PyQt6.QtGui import QPixmap, QImage
from PyQt6.QtCore import Qt

class DICOMViewer(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Biomedical Viewer - Task 2")
        self.setGeometry(100, 100, 800, 600)

        self.label = QLabel("No image loaded yet", self)
        self.label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.button = QPushButton("Load DICOM Folder", self)
        self.button.clicked.connect(self.load_dicom_folder)

        layout = QVBoxLayout()
        layout.addWidget(self.label)
        layout.addWidget(self.button)

        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)

    def load_dicom_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Select DICOM Folder")
        if folder:
            # نجيب أول ملف .dcm من الفولدر
            files = [f for f in os.listdir(folder) if f.lower().endswith(".dcm")]
            if not files:
                self.label.setText("⚠️ No DICOM files found in this folder.")
                return

            path = os.path.join(folder, files[0])
            ds = pydicom.dcmread(path)
            pixel_array = ds.pixel_array

            # نحول الصورة من numpy إلى QImage
            image = self.numpy_to_qimage(pixel_array)
            pixmap = QPixmap.fromImage(image)
            self.label.setPixmap(pixmap.scaled(self.label.width(), self.label.height(), Qt.AspectRatioMode.KeepAspectRatio))

    def numpy_to_qimage(self, arr):
        arr = arr.astype(np.float32)
        arr = 255 * (arr - np.min(arr)) / (np.max(arr) - np.min(arr))
        arr = arr.astype(np.uint8)
        h, w = arr.shape
        return QImage(arr.data, w, h, w, QImage.Format.Format_Grayscale8)

if __name__ == "__main__":
    app = QApplication([])
    viewer = DICOMViewer()
    viewer.show()
    app.exec()

