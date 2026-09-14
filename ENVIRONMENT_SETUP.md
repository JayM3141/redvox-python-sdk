# RedVox Django Web Viewer - Environment Setup Documentation

## Current Configuration (Streamlined)

### **Python Environment**
- **Active Virtual Environment**: `.venv` (Python 3.13.12)
- **Removed**: `.venv312` (Python 3.12) - was broken, deleted to save disk space
- **System Python**: Python 3.13.12 available

### **Installed Dependencies (in .venv)**
All required packages for RedVox Django web viewer:

**Core Dependencies:**
- lz4 (4.4.5) - File compression/decompression
- numpy (2.5.2) - Numerical computing
- pandas (3.0.5) - Data analysis
- scipy (1.18.1) - Scientific computing
- pyarrow (25.0.1) - Data serialization
- protobuf (7.36.0) - Data structures
- matplotlib (3.11.1) - Signal Analysis visualization

**Web Framework:**
- Django (6.1) - Web framework
- asgiref (3.12.1) - ASGI support
- sqlparse (0.6.0) - SQL parsing
- tzdata (2026.3) - Timezone data

**Additional Dependencies:**
- requests (2.34.2) - HTTP library
- dataclasses-json (0.6.7) - JSON serialization
- pillow (12.3.0) - Image processing
- contourpy (1.3.3) - Contour plotting
- fonttools (4.63.0) - Font manipulation
- kiwisolver (1.5.0) - Expression solver
- pyparsing (3.3.2) - Python parsing
- python-dateutil (2.9.0) - Date utilities
- six (1.17.0) - Python 2/3 compatibility
- typing_extensions (4.16.0) - Type hints
- typing_inspect (0.9.0) - Type inspection
- marshmallow (3.26.2) - Object serialization
- packaging (26.3) - Package utilities
- certifi (2026.7.22) - SSL certificates
- charset_normalizer (3.5.1) - Character encoding
- idna (3.19) - IDNA support
- urllib3 (2.7.0) - URL handling

## Features Available

### **Fully Functional:**
1. **File Inspector** - Upload and inspect .rdvxz, .rdvxm, .json files
2. **DataWindow Builder** - Create time-bounded data windows
3. **Signal Analysis** - Audio spectrograms, waveforms, visualization
4. **File Converter** - Convert between API formats
5. **Validator** - Validate .rdvxm files
6. **CLI Runner** - Execute RedVox CLI commands
7. **Cloud Connect** - Access RedVox cloud services
8. **ML Inference Extraction** - Extract ML data from RDVXM files

## Startup Scripts

### **Main Launcher:**
```batch
launch_redvox_gui.bat
```
- Uses .venv (Python 3.13)
- Checks for required dependencies
- Creates necessary directories
- Starts Django server on http://127.0.0.1:8000
- Opens browser automatically

### **Diagnostic Tool:**
```batch
diagnose_redvox_gui.bat
```
- Checks Python installation
- Verifies all dependencies
- Tests Django configuration
- Reports system status

### **Dependency Installation:**
```batch
install_dependencies.bat
```
- Installs all RedVox dependencies
- Uses .venv if available
- Falls back to system Python

## Disk Space Optimization

### **Removed:**
- `.venv312` virtual environment (~100-200 MB saved)
- This environment was broken with Python path issues

### **Kept:**
- `.venv` (Python 3.13) - fully functional
- Only one virtual environment needed

## Quick Start

1. **Start the application:**
   ```batch
   cd C:\1-My-Data-Centre\2-Ingest\3-Data-Types\Audio\Redvox-Python-SDK\redvox-python-sdk
   launch_redvox_gui.bat
   ```

2. **Access the web interface:**
   - Browser opens automatically to http://127.0.0.1:8000
   - All features are fully functional

3. **Troubleshoot if needed:**
   ```batch
   diagnose_redvox_gui.bat
   ```

## Configuration Files

### **Django Settings:**
- `redvox_gui/settings.py` - Django configuration
- `redvox_gui/urls.py` - URL routing
- `redvox_gui/viewer/urls.py` - Viewer routes
- `redvox_gui/viewer/views.py` - View functions

### **Key Settings:**
- Debug mode: True
- Allowed hosts: '*' (for local development)
- Database: None (file-based sessions)
- Media root: `redvox_gui/media/`
- Session engine: File-based
- CSRF trusted origins: localhost, 127.0.0.1

## Storage Requirements

### **Current Setup:**
- `.venv` environment: ~200-300 MB
- RedVox SDK: ~50 MB
- Total Python environment: ~250-350 MB

### **Optimized:**
- Single virtual environment instead of multiple
- Only required dependencies installed
- No duplicate packages

## Compatibility

### **Python Version:**
- Python 3.13.12 (system and .venv)
- Django 6.1 (compatible with Python 3.13)
- All packages use latest compatible versions

### **Operating System:**
- Windows 10/11
- PowerShell compatible
- Command prompt compatible

## Maintenance

### **Regular Tasks:**
1. **Update dependencies:**
   ```batch
   cd C:\1-My-Data-Centre\2-Ingest\3-Data-Types\Audio\Redvox-Python-SDK\redvox-python-sdk
   .venv\Scripts\python.exe -m pip install --upgrade pip
   .venv\Scripts\python.exe -m pip install --upgrade redvox
   ```

2. **Clean up if needed:**
   ```batch
   cleanup_environment.bat
   ```

3. **Reinstall if broken:**
   ```batch
   Remove-Item -Recurse -Force .venv
   python -m venv .venv
   .venv\Scripts\activate
   pip install -e .
   ```

## Historical Context

Based on the codebase analysis, this setup addresses:

1. **LZ4/RDVXZ Format Support**: The original discussion about LZ4 and RDVXZ formats is resolved with proper lz4 installation
2. **ML Inference Extraction**: ML extraction capabilities are available through the RedVox SDK
3. **DataWindow Processing**: Full DataWindow functionality for time-bounded data analysis
4. **Django Web Interface**: Complete web viewer for all RedVox operations

The environment is now streamlined to use a single, functional Python 3.13 virtual environment with all required dependencies for the RedVox Django web viewer.