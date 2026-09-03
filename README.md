# meV-RIXS Toolkit

**meV-RIXS Toolkit** is a graphical Python application for loading, processing, visualizing, calibrating, and exporting resonant inelastic X-ray scattering (RIXS) data for the meV-RIXS Spectrometer at the UE-112 PGM-1 Beamline at BESSY II.

The toolkit provides an interactive interface for working with detector images and RIXS spectra, including detector calibration, filtering and correction tools, interactive visualization, and the extraction of one-dimensional spectra.

The graphical user interface is based on **CustomTkinter** and **Matplotlib** and is designed to run on Windows and Linux.

---

## Features

The toolkit currently includes functionality for:

- Loading and organizing RIXS scan data
- Interactive visualization of 2D detector histograms
- Extraction and visualization of 1D spectra
- Detector tilt correction
- Reference-line and ridge determination
- Symmetrization of detector data
- Adjustable detector binning
- Median and local filtering
- Region-of-interest selection and zooming
- Intensity scaling and colormap selection
- Detector energy calibration
- Single- and double-Gaussian calibration models
- Polynomial calibration functions
- Automatic search for scan and associated metadata files
- Export of processed spectra and detector histograms
- Optional inclusion of metadata in exported HDF5 files
- Multiple datasets within the same application session
- Cross-platform graphical interface for Windows and Linux

---

## Repository structure

```text
meV-RIXS_Toolkit/
│
├── meV-RIXS_Toolkit.py   # Main application and graphical user interface
├── mev_viewer.py         # RIXS data processing and viewer functionality
├── icon9.ico             # Windows application icon
├── icon9.png             # Application icon used for Linux / GUI resources
├── requirements.txt      # Python dependencies
└── README.md
```

---

## Requirements

Python **3.10 or newer** is recommended.

The main Python dependencies are listed in `requirements.txt`.

Install them with:

```bash
python -m pip install -r requirements.txt
```

The application also uses Pillow for image handling:

```bash
python -m pip install pillow
```

For Python versions earlier than Python 3.11, `typing_extensions` may additionally be required:

```bash
python -m pip install typing_extensions
```


## Running from source

Clone the repository:

```bash
git clone https://github.com/tubomk/meV-RIXS_Toolkit.git
cd meV-RIXS_Toolkit
```

Install the required Python packages:

```bash
python -m pip install -r requirements.txt

```

Then start the application with:

```bash
python meV-RIXS_Toolkit.py
```
Maybe additional packages have to be installed...

---

# Creating a standalone executable

The application can be packaged into a **single standalone executable** using [PyInstaller](https://pyinstaller.org/).

Install PyInstaller first:

```bash
python -m pip install pyinstaller
```

The resulting executable will be created in the:

```text
dist/
```

directory.

> **Note:** PyInstaller executables are platform-specific.  
> A Windows executable must be built on Windows, while a Linux executable must be built on Linux.

---

## Windows

Make sure the following files are located in the same directory:

```text
meV-RIXS_Toolkit.py
mev_viewer.py
icon8.ico
icon8.png
```

Then run:

```powershell
pyinstaller --noconfirm --clean --onefile --windowed --name meV-RIXS_Toolkit --icon=icon8.ico --add-data "icon8.ico;." --add-data "icon8.png;." meV-RIXS_Toolkit.py
```

The resulting executable will be:

```text
dist/meV-RIXS_Toolkit.exe
```

The `--windowed` option prevents an additional console window from being opened together with the graphical application.

---

## Linux

Make sure `icon8.png` is located in the same directory as the Python scripts.

Then run:

```bash
python -m PyInstaller --noconfirm --clean --onefile --name meV-RIXS_Toolkit --add-data "icon8.png:." --hidden-import PIL._tkinter_finder meV-RIXS_Toolkit.py
```

The resulting executable will be:

```text
dist/meV-RIXS_Toolkit
```

If necessary, make the generated file executable:

```bash
chmod +x dist/meV-RIXS_Toolkit
```

It can then be started with:

```bash
./dist/meV-RIXS_Toolkit
```

---

## macOS

The same PyInstaller resource syntax as on Linux can be used:

```bash
python -m PyInstaller --noconfirm --clean --onefile --name meV-RIXS_Toolkit --add-data "icon8.png:." --hidden-import PIL._tkinter_finder meV-RIXS_Toolkit.py
```

PyInstaller builds should generally be created directly on the operating system on which they are intended to run.

---

## Standalone version

When built with PyInstaller using `--onefile`, the required Python interpreter and Python packages are bundled with the application.

Therefore, users of the packaged application do **not** need to install Python or the Python dependencies separately.

The standalone builds can be distributed through the **Releases** section of this repository.

---

## Development status

The meV-RIXS Toolkit is under active development. Features, file handling, calibration routines, and the graphical interface may change between versions.

---

## Author

**Marco Kliem**

Helmholtz-Zentrum Berlin für Materialien und Energie (HZB)

