# RedVox Python SDK — Django GUI

## Project Overview
The RedVox Python SDK is a library for reading, creating, editing, and writing RedVox data formats (API 900 and API 1000/M). It handles sensor data (audio, barometer, accelerometer, etc.) collected by RedVox devices.

A Django web GUI has been built on top of the SDK, providing a browser-based interface for inspecting, converting, and validating RedVox files.

## Architecture
- **Language**: Python 3.11
- **Web Framework**: Django 5.2 + Whitenoise for static files
- **SDK Library**: redvox 3.8.6 (installed in editable mode from `pyproject.toml`)
- **Frontend Styling**: Bootstrap 5.3 (CDN) + Bootstrap Icons
- **Port**: 5000

## Directory Structure
```
redvox/               — RedVox SDK source library
  api900/             — Legacy API 900 (.rdvxz) implementation
  api1000/            — Modern API 1000/M (.rdvxm) implementation
  common/             — Shared utilities, DataWindow, Station
  cloud/              — RedVox cloud service clients
  cli/                — redvox-cli command-line tool
redvox_gui/           — Django web application
  manage.py
  redvox_gui/         — Django project settings and URLs
    settings.py       — ALLOWED_HOSTS=*, port 5000, no DB
    urls.py
  viewer/             — Main Django app
    views.py          — All view logic (dashboard, inspect, converter, validator, CLI)
    urls.py
    templates/viewer/ — HTML templates (base, dashboard, inspect, converter, validator, cli_runner)
```

## Features
- **Dashboard** — SDK overview with stats and quick links
- **File Inspector** — Upload .rdvxz, .rdvxm, or .json; view packet metadata and sensors
- **File Converter** — Convert between API 900 ↔ API 1000/M and JSON formats; downloads result
- **Validator** — Validate .rdvxm files using redvox-cli validate-m
- **CLI Runner** — Run any redvox-cli command from the browser

## Running
```bash
cd redvox_gui && python manage.py runserver 0.0.0.0:5000
```

## SDK Dependencies
- numpy, pandas, pyarrow — data processing
- protobuf — data serialization
- scipy — scientific computing
- lz4 — compression
- requests, websocket-client — cloud connectivity
- django, whitenoise — web framework and static files
