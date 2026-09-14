# RedVox Django Web Viewer - Troubleshooting Guide

## Common Issues and Solutions

### 1. ModuleNotFoundError: No module named 'lz4'

**Symptoms**: Error when trying to import RDVXM files: "ModuleNotFoundError: No module named 'lz4'"

**Causes**: The lz4 compression library is not installed in the Python environment.

**Solutions**:
1. **Quick Fix**: Run `quick_fix_lz4.bat` to install only the lz4 module
2. **Full Installation**: Run `install_dependencies.bat` to install all RedVox dependencies
3. **Manual Install**: Run `pip install lz4==4.3.3` in your virtual environment
4. **Verify Installation**: Run `diagnose_redvox_gui.bat` to check if lz4 is installed

**Note**: lz4 is a critical dependency for RedVox file processing. Both API 900 (.rdvxz) and API 1000/M (.rdvxm) formats require it.

### 2. ERR_FILE_NO_SPACE Error

**Symptoms**: When trying to upload files or use the inspect function, you get "ERR_FILE_NO_SPACE" error.

**Causes**:
- Insufficient disk space on the drive where temporary files are stored
- Browser cache issues
- Temporary directory permissions

**Solutions**:
1. **Check Disk Space**: Ensure you have sufficient free space on your C: drive
2. **Clear Browser Cache**: Clear your browser's cache and cookies
3. **Try Different Browser**: Use Chrome, Firefox, or Edge instead of your current browser
4. **Check Temporary Directory**: Ensure your TEMP directory is accessible and has space
5. **Run Diagnostic**: Run `diagnose_redvox_gui.bat` to check system status

### 3. Server Won't Start

**Symptoms**: Django server fails to start or immediately crashes.

**Solutions**:
1. **Check Virtual Environment**: Ensure the virtual environment is properly activated
2. **Install Dependencies**: Run `pip install -e .` in the project directory
3. **Check Port 8000**: Ensure port 8000 is not already in use
4. **Run Diagnostic**: Use `diagnose_redvox_gui.bat` to identify issues

### 4. File Upload Issues

**Symptoms**: File uploads fail or don't process correctly.

**Solutions**:
1. **Check File Size**: Ensure files are under 50MB (configured in settings)
2. **Check File Format**: Only .rdvxz, .rdvxm, and .json files are supported
3. **Check Permissions**: Ensure the media directory has write permissions
4. **Check Temporary Space**: Ensure sufficient space for temporary file processing

### 5. Page Not Found / 404 Errors

**Symptoms**: Navigation links don't work or show 404 errors.

**Solutions**:
1. **Check URL Configuration**: Ensure urls.py is properly configured
2. **Restart Server**: Stop and restart the Django server
3. **Clear Browser Cache**: Clear browser cache and reload
4. **Check Allowed Hosts**: Ensure your host is in ALLOWED_HOSTS settings

### 6. Session Issues

**Symptoms**: Login state or form data not persisting.

**Solutions**:
1. **Check Sessions Directory**: Ensure the sessions directory exists and is writable
2. **Clear Sessions**: Delete session files in the sessions directory
3. **Check Session Engine**: Ensure file-based sessions are working

## Running the Diagnostic Tool

Run the diagnostic tool to check your setup:

```batch
diagnose_redvox_gui.bat
```

This will check:
- Directory structure
- Virtual environment status
- Python and Django installation
- Required directories
- Django settings
- Disk space

## Manual Directory Setup

If automatic directory creation fails, manually create these directories:

```
redvox_gui/
├── sessions/
├── media/
│   └── data_windows/
└── staticfiles/
```

## Browser Recommendations

For best compatibility:
- **Chrome/Edge**: Most compatible with modern Django features
- **Firefox**: Good alternative
- **Safari**: May have some file upload limitations

## Network Settings

If running on a network (not localhost):

1. Update `ALLOWED_HOSTS` in `redvox_gui/settings.py`:
```python
ALLOWED_HOSTS = ['*']  # or specify your IP/domain
```

2. Update `CSRF_TRUSTED_ORIGINS`:
```python
CSRF_TRUSTED_ORIGINS = [
    'http://your-ip:8000',
    'http://localhost:8000',
]
```

### 7. Signal Analysis Error - Missing matplotlib

**Symptoms**: Error when accessing the Signal Analysis page: "No module named 'matplotlib'. Install matplotlib to enable Signal Analysis."

**Causes**: matplotlib is not installed in the Python environment.

**Solutions**:
1. **Quick Install**: Run `pip install matplotlib` in your virtual environment
2. **Full Installation**: Run `install_dependencies.bat` to install all dependencies including matplotlib
3. **Manual Install**: `pip install matplotlib scipy pandas pyarrow`

**Note**: matplotlib is specifically required for the Signal Analysis feature which includes:
- Audio spectrogram generation
- Waveform plotting
- Signal visualization
- Data analysis charts

### 8. Missing Dependencies

**Symptoms**: Various import errors for missing modules (lz4, numpy, django, etc.)

**Solutions**:
1. **Full Dependency Installation**: Run `install_dependencies.bat` to install all required packages
2. **Individual Installation**: Install specific missing packages:
   - `pip install lz4==4.3.3` (compression)
   - `pip install django` (web framework)
   - `pip install numpy==1.26.4` (numerical computing)
   - `pip install pandas==2.2.2` (data processing)
   - `pip install pyarrow==16.1.0` (data serialization)
   - `pip install protobuf==4.25.3` (data serialization)
3. **Project Installation**: Run `pip install -e .` in the project directory
4. **Check Virtual Environment**: Ensure you're using the correct Python environment

**Core RedVox Dependencies**:
- lz4 (required for file compression/decompression)
- numpy (required for data processing)
- pandas (required for data manipulation)
- scipy (required for scientific computing)
- pyarrow (required for data serialization)
- protobuf (required for data structures)
- django (required for web interface)
- matplotlib (required for Signal Analysis feature)
- requests (required for cloud services)

## Getting Help

If issues persist:
1. Check the Django server console for error messages
2. Run the diagnostic tool
3. Review Django logs in the server window
4. Ensure all dependencies are installed: `pip install -e .`

## Development Server vs Production

This setup uses Django's development server. For production:
- Use a proper WSGI server (Gunicorn, uWSGI)
- Configure a web server (Nginx, Apache)
- Set DEBUG=False in settings
- Use a proper database instead of file-based sessions
- Configure proper static file serving