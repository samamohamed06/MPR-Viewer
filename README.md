# 🧠 Medical Imaging Viewer (MPR + AI Detection)

## 📖 Overview

The **Medical Imaging Viewer** is an advanced visualization and analysis tool for 3D medical images (CT/MRI).
It provides **Multi-Planar Reconstruction (MPR)** capabilities, **AI-powered organ and orientation detection**, and intuitive tools for **ROI selection**, **zooming**, **scrolling**, and **surface visualization**.

This tool is designed for both **research** and **clinical prototyping** — combining powerful image processing with a user-friendly graphical interface.

---

## ✨ Key Features

### 🩻 Multi-Planar Reconstruction (MPR)

* Display images in **Axial**, **Coronal**, and **Sagittal** planes.
* **Oblique view** support for free-angle slicing.
* **Linked cursor synchronization** between all views.

📷 *Placeholder for MPR interface image*
`![MPR Interface](images/mpr_placeholder.png)`

---

### 🧬 AI-Powered Organ Detection

* Utilizes deep learning models for **automatic organ segmentation and detection**.
* Highlights detected regions with color overlays.
* Outputs confidence scores and bounding boxes for each organ.

📷 *Placeholder for AI detection visualization*
`![AI Detection Example](images/ai_detection_placeholder.png)`

---

### 🧭 AI Orientation Detection

* Automatically identifies and labels the image **orientation (Axial / Sagittal / Coronal)**.
* Supports volume re-alignment to correct orientation inconsistencies.
* Reduces preprocessing effort for 3D datasets.

📷 *Placeholder for orientation detection demo*
`![Orientation Detection](images/orientation_placeholder.png)`

---

### 📈 ROI (Region of Interest) Tools

* Draw, edit, and measure **Regions of Interest** interactively.
* Compute ROI statistics (mean intensity, area, volume).
* Export ROI masks for further analysis.

📷 *Placeholder for ROI example*
`![ROI Example](images/roi_placeholder.png)`

---

### 🖱️ Navigation & Interaction

* **Scrolling:** Navigate through slices easily using mouse wheel or keyboard shortcuts.
* **Zoom & Pan:** Zoom in/out smoothly and move across slices.
* **Linked Views:** Simultaneous navigation across all planes for synchronized exploration.

📷 *Placeholder for navigation demo*
`![Zoom and Scroll](images/navigation_placeholder.png)`

---

### 🌐 3D Surface & Volume Rendering

* Generate **3D surface meshes** or **volume renderings** from segmentation data.
* Adjustable rendering parameters for lighting and opacity.
* Export 3D models for use in visualization or printing.

📷 *Placeholder for 3D rendering*
`![3D Surface Rendering](images/3d_surface_placeholder.png)`

---

## ⚙️ Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/mpr-ai-viewer.git
cd mpr-ai-viewer

# Install dependencies
pip install -r requirements.txt
```

---

## 🚀 Usage

```bash
# Launch the application
python dicom_viewer_full_gui.py
```

Then load your **DICOM folder** or **NIfTI volume**, and explore the dataset using the interactive GUI.

---

## 🧩 Dependencies

* **Python 3.8+**
* **NumPy**, **SciPy**
* **nibabel**, **pydicom**
* **PyTorch / TorchXRayVision**
* **Tkinter** or **PyQt** (for GUI)
* **matplotlib** or **VTK** (for 3D rendering)

---

## 🧠 Future Improvements

* Integration with **cloud-based AI services** for faster inference.
* Support for **DICOM RTSTRUCT** and **segmentation mask export**.
* Enhanced **measurement tools** and **histogram analysis**.

---

## 👨‍💻 Authors

Developed by **[Sama Mohamed]**
📧 Contact: [[sama.mohamed.18@gmail.com](mailto:your.email@example.com)]

---

## 🪪 License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.
