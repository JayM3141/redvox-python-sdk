import os
import io
import json
import base64
import sys
import time
import tempfile
import traceback
import subprocess
import uuid
from pathlib import Path
from datetime import datetime, timezone, timedelta
from pprint import pformat

import numpy as np
from django.shortcuts import render, redirect
from django.http import JsonResponse, HttpResponse
from django.conf import settings

REDVOX_REPO_ROOT = Path(getattr(settings, 'REPO_ROOT', Path(__file__).resolve().parents[2]))

# ─── helpers ────────────────────────────────────────────────────────────────

def _b64_figure(fig):
    """Render a matplotlib figure to a base64 PNG string."""
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight', dpi=120, facecolor='#1e1e2e')
    buf.seek(0)
    return base64.b64encode(buf.read()).decode('utf-8')


def _read_rdvxz(path: str):
    from redvox.api900 import reader
    return reader.read_rdvxz_file(path)


def _read_rdvxm(path: str):
    from redvox.api1000.wrapped_redvox_packet.wrapped_packet import WrappedRedvoxPacketM
    with open(path, 'rb') as f:
        data = f.read()
    return WrappedRedvoxPacketM.from_compressed_bytes(data)


def _redvox_cli_env():
    env = os.environ.copy()
    repo_root = str(REDVOX_REPO_ROOT)
    pythonpath = env.get('PYTHONPATH')
    env['PYTHONPATH'] = repo_root if not pythonpath else os.pathsep.join([repo_root, pythonpath])
    return env


def _run_redvox_cli(args, timeout: int):
    return subprocess.run(
        [sys.executable, '-m', 'redvox.cli.cli', *args],
        capture_output=True,
        text=True,
        timeout=timeout,
        cwd=str(REDVOX_REPO_ROOT),
        env=_redvox_cli_env(),
    )


def _data_window_cache_root(ensure_exists: bool = False) -> Path:
    root = Path(settings.MEDIA_ROOT) / 'data_windows'
    if ensure_exists:
        root.mkdir(parents=True, exist_ok=True)
    return root


def _default_data_window_form_values() -> dict:
    return {
        'input_dir': '',
        'event_name': 'dw',
        'output_type': 'LZ4',
        'structured_layout': True,
        'start_datetime': '',
        'end_datetime': '',
        'start_buffer_seconds': '120',
        'end_buffer_seconds': '120',
        'drop_time_seconds': '0.2',
        'station_ids': '',
        'apply_correction': True,
        'use_model_correction': True,
        'copy_edge_points': 'COPY',
        'make_runme': False,
        'debug': False,
    }


def _data_window_form_values(post) -> dict:
    defaults = _default_data_window_form_values()
    return {
        'input_dir': post.get('input_dir', defaults['input_dir']).strip(),
        'event_name': post.get('event_name', defaults['event_name']).strip() or defaults['event_name'],
        'output_type': post.get('output_type', defaults['output_type']).strip().upper() or defaults['output_type'],
        'structured_layout': post.get('structured_layout') == 'on',
        'start_datetime': post.get('start_datetime', defaults['start_datetime']).strip(),
        'end_datetime': post.get('end_datetime', defaults['end_datetime']).strip(),
        'start_buffer_seconds': post.get('start_buffer_seconds', defaults['start_buffer_seconds']).strip() or defaults['start_buffer_seconds'],
        'end_buffer_seconds': post.get('end_buffer_seconds', defaults['end_buffer_seconds']).strip() or defaults['end_buffer_seconds'],
        'drop_time_seconds': post.get('drop_time_seconds', defaults['drop_time_seconds']).strip() or defaults['drop_time_seconds'],
        'station_ids': post.get('station_ids', defaults['station_ids']).strip(),
        'apply_correction': post.get('apply_correction') == 'on',
        'use_model_correction': post.get('use_model_correction') == 'on',
        'copy_edge_points': post.get('copy_edge_points', defaults['copy_edge_points']).strip().upper() or defaults['copy_edge_points'],
        'make_runme': post.get('make_runme') == 'on',
        'debug': post.get('debug') == 'on',
    }


def _sanitize_data_window_name(name: str) -> str:
    cleaned = ''.join(ch if ch.isalnum() or ch in {'-', '_'} else '_' for ch in (name or 'dw'))
    cleaned = cleaned.strip('._-')
    return cleaned or 'dw'


def _parse_data_window_datetime(value: str):
    if not value:
        return None
    dt_value = datetime.fromisoformat(value.replace('Z', '+00:00'))
    if dt_value.tzinfo is None:
        return dt_value.replace(tzinfo=timezone.utc)
    return dt_value.astimezone(timezone.utc)


def _parse_data_window_station_ids(raw: str):
    station_ids = [token for token in raw.replace(',', ' ').split() if token]
    return station_ids or None


def _format_utc_datetime(value) -> str:
    if not value:
        return '—'
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    else:
        value = value.astimezone(timezone.utc)
    return value.strftime('%Y-%m-%d %H:%M:%S UTC')


def _format_epoch_micros(value) -> str:
    try:
        numeric_value = float(value)
    except (TypeError, ValueError):
        return '—'
    if np.isnan(numeric_value) or np.isinf(numeric_value):
        return '—'
    return datetime.fromtimestamp(numeric_value / 1_000_000, tz=timezone.utc).strftime('%Y-%m-%d %H:%M:%S.%f UTC')


def _format_filesystem_datetime(timestamp: float) -> str:
    return datetime.fromtimestamp(timestamp, tz=timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')


def _looks_like_data_window_metadata(payload) -> bool:
    return isinstance(payload, dict) and {'event_name', 'config', 'out_type'}.issubset(payload.keys())


def _find_data_window_metadata_file(cache_target: Path):
    if cache_target.is_file():
        if cache_target.suffix.lower() != '.json':
            return None
        try:
            with open(cache_target, 'r', encoding='utf-8') as file_obj:
                payload = json.load(file_obj)
        except Exception:
            return None
        return cache_target if _looks_like_data_window_metadata(payload) else None
    if not cache_target.exists() or not cache_target.is_dir():
        return None
    json_files = sorted(path for path in cache_target.glob('*.json') if path.is_file())
    for path in json_files:
        try:
            with open(path, 'r', encoding='utf-8') as file_obj:
                payload = json.load(file_obj)
        except Exception:
            continue
        if _looks_like_data_window_metadata(payload):
            return path
    return None


def _summarize_station(station) -> dict:
    station_dict = station.as_dict()
    errors_dict = station_dict.get('errors') if isinstance(station_dict.get('errors'), dict) else {}
    return {
        'id': station.id() or '—',
        'uuid': station.uuid() or '—',
        'start_date': _format_epoch_micros(station.start_date()),
        'first_data_timestamp': _format_epoch_micros(station.first_data_timestamp()),
        'last_data_timestamp': _format_epoch_micros(station.last_data_timestamp()),
        'sensors': station_dict.get('sensors', []),
        'errors': errors_dict.get('errors', []),
        'error_count': errors_dict.get('num_errors', 0),
    }


def _summarize_data_window(data_window, metadata_path=None, cache_key: str = '') -> dict:
    config = data_window.config()
    errors_dict = data_window.errors().as_dict()
    cache_dir = metadata_path.parent if metadata_path else (Path(data_window.save_dir()) if data_window.save_dir() else None)
    saved_files = []
    if cache_dir and cache_dir.exists():
        saved_files = sorted(path.name for path in cache_dir.iterdir())
    return {
        'event_name': data_window.event_name,
        'out_type': str(data_window.out_type()).upper(),
        'sdk_version': data_window.sdk_version(),
        'cache_key': cache_key,
        'cache_dir': str(cache_dir) if cache_dir else '',
        'metadata_path': str(metadata_path) if metadata_path else '',
        'saved_artifact': '',
        'saved_files': saved_files,
        'station_count': len(data_window.stations()),
        'station_ids': data_window.station_ids(),
        'start_time': _format_epoch_micros(data_window.start_date()),
        'end_time': _format_epoch_micros(data_window.end_date()),
        'errors': errors_dict.get('errors', []),
        'error_count': errors_dict.get('num_errors', 0),
        'config': {
            'input_dir': config.input_dir,
            'structured_layout': config.structured_layout,
            'start_datetime': _format_utc_datetime(config.start_datetime),
            'end_datetime': _format_utc_datetime(config.end_datetime),
            'start_buffer_seconds': f'{config.start_buffer_td.total_seconds():g}',
            'end_buffer_seconds': f'{config.end_buffer_td.total_seconds():g}',
            'drop_time_seconds': f'{config.drop_time_s:g}',
            'station_ids': sorted(config.station_ids) if config.station_ids else [],
            'apply_correction': config.apply_correction,
            'use_model_correction': config.use_model_correction,
            'copy_edge_points': config.copy_edge_points.name,
        } if config else None,
        'stations': [_summarize_station(station) for station in data_window.stations()],
    }


def _list_cached_data_windows():
    cache_root = _data_window_cache_root()
    if not cache_root.exists():
        return []
    entries = []
    for child in sorted(cache_root.iterdir(), key=lambda item: item.stat().st_mtime, reverse=True):
        metadata_path = _find_data_window_metadata_file(child)
        if metadata_path is None:
            continue
        try:
            with open(metadata_path, 'r', encoding='utf-8') as file_obj:
                metadata = json.load(file_obj)
            config = metadata.get('config') if isinstance(metadata.get('config'), dict) else {}
            errors_dict = metadata.get('errors') if isinstance(metadata.get('errors'), dict) else {}
            entries.append({
                'cache_key': child.name,
                'display_name': metadata.get('event_name') or child.stem,
                'out_type': str(metadata.get('out_type', 'UNKNOWN')).upper(),
                'station_count': len(metadata.get('stations') or []),
                'modified': _format_filesystem_datetime(metadata_path.stat().st_mtime),
                'input_dir': config.get('input_dir', ''),
                'metadata_name': metadata_path.name,
                'location': str(child),
                'errors': errors_dict.get('errors', []),
            })
        except Exception as exc:
            entries.append({
                'cache_key': child.name,
                'display_name': child.stem,
                'out_type': 'UNKNOWN',
                'station_count': 0,
                'modified': _format_filesystem_datetime(metadata_path.stat().st_mtime),
                'input_dir': '',
                'metadata_name': metadata_path.name,
                'location': str(child),
                'errors': [str(exc)],
            })
    return entries


def _resolve_cached_data_window_metadata(cache_key: str) -> Path:
    if not cache_key:
        raise ValueError('Please choose a cached DataWindow to load.')
    cache_root = _data_window_cache_root()
    if not cache_root.exists():
        raise FileNotFoundError('No cached DataWindows are available yet.')
    root_resolved = cache_root.resolve()
    candidate = (cache_root / cache_key).resolve()
    if candidate != root_resolved and root_resolved not in candidate.parents:
        raise ValueError('Invalid cached DataWindow selection.')
    if not candidate.exists():
        raise FileNotFoundError(f'Cached DataWindow not found: {cache_key}')
    metadata_path = _find_data_window_metadata_file(candidate)
    if metadata_path is None:
        raise FileNotFoundError(f'No DataWindow metadata JSON found for cache entry: {cache_key}')
    return metadata_path


def _create_cached_data_window(form_values: dict) -> dict:
    from redvox.common.data_window import DataWindow, DataWindowConfig
    from redvox.common import gap_and_pad_utils as gpu

    input_dir_raw = form_values.get('input_dir', '').strip()
    if not input_dir_raw:
        raise ValueError('Input directory is required.')
    input_dir = Path(input_dir_raw).expanduser()
    if not input_dir.exists() or not input_dir.is_dir():
        raise ValueError(f'Input directory does not exist or is not a directory: {input_dir_raw}')

    event_name = _sanitize_data_window_name(form_values.get('event_name') or input_dir.name or 'dw')
    output_type = str(form_values.get('output_type', 'LZ4')).upper()
    if output_type not in {'LZ4', 'JSON', 'PARQUET'}:
        raise ValueError(f'Unsupported output type: {output_type}')

    try:
        copy_edge_points = gpu.DataPointCreationMode[form_values.get('copy_edge_points', 'COPY').upper()]
    except KeyError as exc:
        raise ValueError(f'Unsupported edge point mode: {form_values.get("copy_edge_points", "")}') from exc

    cache_key = f'{event_name}_{datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")}_{uuid.uuid4().hex[:8]}'
    output_dir = _data_window_cache_root(ensure_exists=True) / cache_key
    output_dir.mkdir(parents=True, exist_ok=True)

    config = DataWindowConfig(
        input_dir=str(input_dir.resolve()),
        structured_layout=form_values.get('structured_layout', True),
        start_datetime=_parse_data_window_datetime(form_values.get('start_datetime', '')),
        end_datetime=_parse_data_window_datetime(form_values.get('end_datetime', '')),
        start_buffer_td=timedelta(seconds=float(form_values.get('start_buffer_seconds', '120'))),
        end_buffer_td=timedelta(seconds=float(form_values.get('end_buffer_seconds', '120'))),
        drop_time_s=float(form_values.get('drop_time_seconds', '0.2')),
        station_ids=_parse_data_window_station_ids(form_values.get('station_ids', '')),
        apply_correction=form_values.get('apply_correction', True),
        use_model_correction=form_values.get('use_model_correction', True),
        copy_edge_points=copy_edge_points,
    )

    original_cwd = os.getcwd()
    try:
        data_window = DataWindow(
            event_name=event_name,
            config=config,
            output_dir=str(output_dir),
            out_type=output_type,
            make_runme=form_values.get('make_runme', False),
            debug=form_values.get('debug', False),
        )
        saved_artifact = data_window.save()
        metadata_path = _find_data_window_metadata_file(output_dir)
        if metadata_path is None:
            raise RuntimeError('DataWindow metadata JSON was not created.')
        summary = _summarize_data_window(data_window, metadata_path=metadata_path, cache_key=cache_key)
        summary['saved_artifact'] = str(saved_artifact) if saved_artifact else ''
        return summary
    finally:
        os.chdir(original_cwd)


def _load_cached_data_window_summary(cache_key: str) -> dict:
    from redvox.common.data_window import DataWindow

    metadata_path = _resolve_cached_data_window_metadata(cache_key)
    original_cwd = os.getcwd()
    try:
        data_window = DataWindow.load(str(metadata_path))
    finally:
        os.chdir(original_cwd)
    return _summarize_data_window(data_window, metadata_path=metadata_path, cache_key=cache_key)


# ─── Dashboard ──────────────────────────────────────────────────────────────

def dashboard(request):
    import redvox
    return render(request, 'viewer/dashboard.html', {
        'sdk_version': redvox.VERSION,
        'api900_sensors': [
            'Microphone', 'Barometer', 'Location', 'Accelerometer',
            'Gyroscope', 'Magnetometer', 'Light', 'Infrared', 'Time Sync',
        ],
        'api1000_sensors': [
            'Audio', 'Compressed Audio', 'Pressure', 'Location (GNSS)',
            'Accelerometer', 'Gyroscope', 'Magnetometer', 'Ambient Temp',
            'Humidity', 'Light', 'Proximity', 'Image',
        ],
    })


# ─── File Inspector ──────────────────────────────────────────────────────────

def inspect(request):
    context = {}
    if request.method == 'POST' and request.FILES.get('file'):
        uploaded = request.FILES['file']
        filename = uploaded.name
        ext = Path(filename).suffix.lower()
        allowed = {'.rdvxz', '.rdvxm', '.json'}
        if ext not in allowed:
            context['error'] = f'Unsupported file type: {ext}. Please upload .rdvxz, .rdvxm, or .json.'
            return render(request, 'viewer/inspect.html', context)

        with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp:
            for chunk in uploaded.chunks():
                tmp.write(chunk)
            tmp_path = tmp.name

        try:
            if ext == '.rdvxz':
                result = _inspect_rdvxz(tmp_path)
            elif ext == '.rdvxm':
                result = _inspect_rdvxm_file(tmp_path)
            else:
                result = _inspect_json(tmp_path)
            context['filename'] = filename
            context['result'] = result
        except Exception as e:
            context['error'] = str(e)
            context['traceback'] = traceback.format_exc()
        finally:
            os.unlink(tmp_path)

    return render(request, 'viewer/inspect.html', context)


def _inspect_rdvxz(path: str) -> dict:
    p = _read_rdvxz(path)
    sensors = []
    if p.has_microphone_sensor():
        sensors.append(f'Microphone ({p.microphone_sensor().sample_rate_hz()} Hz)')
    if p.has_barometer_sensor():
        sensors.append('Barometer')
    if p.has_location_sensor():
        sensors.append('Location')
    if p.has_time_synchronization_sensor():
        sensors.append('Time Synchronization')
    if p.has_accelerometer_sensor():
        sensors.append('Accelerometer')
    if p.has_gyroscope_sensor():
        sensors.append('Gyroscope')
    if p.has_magnetometer_sensor():
        sensors.append('Magnetometer')
    if p.has_light_sensor():
        sensors.append('Light')
    if p.has_infrared_sensor():
        sensors.append('Infrared')

    ts = p.app_file_start_timestamp_epoch_microseconds_utc()
    ts_human = datetime.utcfromtimestamp(ts / 1_000_000).strftime('%Y-%m-%d %H:%M:%S UTC') if ts else 'N/A'

    return {
        'type': 'API 900 (.rdvxz)',
        'color': '#1d4ed8',
        'fields': [
            ('API Version', p.api()),
            ('Device ID', p.redvox_id()),
            ('UUID', p.uuid()),
            ('Device Make', p.device_make()),
            ('Device Model', p.device_model()),
            ('Device OS', p.device_os()),
            ('OS Version', p.device_os_version()),
            ('App Version', p.app_version()),
            ('Start Timestamp', ts_human),
            ('Duration (s)', p.duration_s()),
            ('Battery (%)', p.battery_level_percent()),
            ('Temperature (°C)', p.device_temperature_c()),
        ],
        'sensors': sensors,
    }


def _inspect_rdvxm_file(path: str) -> dict:
    p = _read_rdvxm(path)
    si = p.get_station_information()
    timing = p.get_timing_information()
    sensors_obj = p.get_sensors()

    sensors = []
    if sensors_obj.has_audio():
        sr = sensors_obj.get_audio().get_sample_rate()
        sensors.append(f'Audio ({sr:.1f} Hz)')
    if sensors_obj.has_compressed_audio():
        sensors.append('Compressed Audio')
    if sensors_obj.has_pressure():
        sensors.append('Pressure (Barometer)')
    if sensors_obj.has_location():
        sensors.append('Location (GNSS)')
    has_best_location = getattr(sensors_obj, 'has_best_location', None)
    if callable(has_best_location) and has_best_location():
        sensors.append('Best Location')
    if sensors_obj.has_accelerometer():
        sensors.append('Accelerometer')
    if sensors_obj.has_gyroscope():
        sensors.append('Gyroscope')
    if sensors_obj.has_magnetometer():
        sensors.append('Magnetometer')
    if sensors_obj.has_ambient_temperature():
        sensors.append('Ambient Temperature')
    if sensors_obj.has_relative_humidity():
        sensors.append('Relative Humidity')
    if sensors_obj.has_light():
        sensors.append('Light')
    if sensors_obj.has_proximity():
        sensors.append('Proximity')
    if sensors_obj.has_image():
        sensors.append('Image')

    start_ts = timing.get_packet_start_mach_timestamp()
    start_ts_human = (
        datetime.utcfromtimestamp(start_ts / 1_000_000).strftime('%Y-%m-%d %H:%M:%S UTC')
        if start_ts and start_ts > 0 else 'N/A'
    )

    server_acq_ts = None
    get_server_acq = getattr(timing, 'get_server_acquisition_arrival_timestamp', None)
    if callable(get_server_acq):
        try:
            server_acq_ts = get_server_acq()
        except Exception:
            server_acq_ts = None

    server_acq_human = 'N/A'
    if isinstance(server_acq_ts, (int, float)) and server_acq_ts and server_acq_ts > 0:
        # Expected unit is microseconds since Unix epoch. Ignore obviously invalid values.
        if server_acq_ts >= 1_000_000_000_000:  # >= ~2001-09-09 in microseconds
            try:
                server_acq_human = datetime.utcfromtimestamp(server_acq_ts / 1_000_000).strftime('%Y-%m-%d %H:%M:%S UTC')
            except Exception:
                server_acq_human = str(server_acq_ts)
        else:
            server_acq_human = f'N/A ({server_acq_ts})'

    return {
        'type': 'API 1000/M (.rdvxm)',
        'color': '#7c3aed',
        'fields': [
            ('Station ID', si.get_id()),
            ('UUID', si.get_uuid()),
            ('Make', si.get_make()),
            ('Model', si.get_model()),
            ('OS', str(si.get_os()).split('.')[-1]),
            ('OS Version', si.get_os_version()),
            ('App Version', si.get_app_version()),
            ('Auth ID', si.get_auth_id()),
            ('Start Timestamp', start_ts_human),
            ('Server Acquire Time', server_acq_human),
        ],
        'sensors': sensors,
    }


def _inspect_json(path: str) -> dict:
    with open(path, 'r') as f:
        data = json.load(f)
    raw = json.dumps(data, indent=2)
    return {
        'type': 'JSON',
        'color': '#064e3b',
        'json_preview': raw[:8000],
        'truncated': len(raw) > 8000,
        'fields': [],
        'sensors': [],
    }


# ─── File Converter ──────────────────────────────────────────────────────────

def converter(request):
    context = {}
    if request.method == 'POST' and request.FILES.get('file'):
        uploaded = request.FILES['file']
        filename = uploaded.name
        ext = Path(filename).suffix.lower()
        action = request.POST.get('action', '')

        valid_actions = {
            '.rdvxz': ['rdvxz_to_rdvxm', 'rdvxz_to_json'],
            '.rdvxm': ['rdvxm_to_rdvxz', 'rdvxm_to_json'],
            '.json': ['json_to_rdvxz', 'json_to_rdvxm'],
        }

        if ext not in valid_actions:
            context['error'] = f'Unsupported file type: {ext}'
            return render(request, 'viewer/converter.html', context)
        if action not in valid_actions.get(ext, []):
            context['error'] = f'Invalid action "{action}" for file type {ext}'
            return render(request, 'viewer/converter.html', context)

        with tempfile.TemporaryDirectory() as tmpdir:
            in_path = os.path.join(tmpdir, filename)
            with open(in_path, 'wb') as f:
                for chunk in uploaded.chunks():
                    f.write(chunk)

            out_dir = os.path.join(tmpdir, 'output')
            os.makedirs(out_dir, exist_ok=True)

            cmd_map = {
                'rdvxz_to_rdvxm': ['rdvxz-to-rdvxm', '--out-dir', out_dir, in_path],
                'rdvxm_to_rdvxz': ['rdvxm-to-rdvxz', '--out-dir', out_dir, in_path],
                'rdvxz_to_json': ['rdvxz-to-json', '--out-dir', out_dir, in_path],
                'rdvxm_to_json': ['rdvxm-to-json', '--out-dir', out_dir, in_path],
                'json_to_rdvxz': ['json-to-rdvxz', '--out-dir', out_dir, in_path],
                'json_to_rdvxm': ['json-to-rdvxm', '--out-dir', out_dir, in_path],
            }

            try:
                result = _run_redvox_cli(cmd_map[action], timeout=30)
                out_files = list(Path(out_dir).glob('*'))
                if out_files:
                    out_file = out_files[0]
                    with open(out_file, 'rb') as f:
                        content = f.read()
                    response = HttpResponse(content, content_type='application/octet-stream')
                    response['Content-Disposition'] = f'attachment; filename="{out_file.name}"'
                    return response
                else:
                    stdout = result.stdout or ''
                    stderr = result.stderr or ''
                    context['error'] = (
                        f'Conversion produced no output file.\n'
                        f'stdout: {stdout}\nstderr: {stderr}'
                    )
            except subprocess.TimeoutExpired:
                context['error'] = 'Conversion timed out after 30 seconds.'
            except Exception as e:
                context['error'] = str(e)
                context['traceback'] = traceback.format_exc()

    return render(request, 'viewer/converter.html', context)


# ─── Validator ───────────────────────────────────────────────────────────────

def validator(request):
    context = {}
    if request.method == 'POST' and request.FILES.get('file'):
        uploaded = request.FILES['file']
        filename = uploaded.name
        ext = Path(filename).suffix.lower()

        if ext not in ('.rdvxm', '.rdvxz'):
            context['error'] = 'Validator supports .rdvxm and .rdvxz files.'
            return render(request, 'viewer/validator.html', context)

        with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp:
            for chunk in uploaded.chunks():
                tmp.write(chunk)
            tmp_path = tmp.name

        try:
            if ext == '.rdvxm':
                result = _run_redvox_cli(['validate-m', tmp_path], timeout=30)
            else:
                # For API 900, just try to read it as validation
                p = _read_rdvxz(tmp_path)
                class FakeResult:
                    returncode = 0
                    stdout = f"API 900 file valid.\nDevice: {p.redvox_id()} | Make: {p.device_make()} | OS: {p.device_os()}"
                    stderr = ''
                result = FakeResult()

            context['filename'] = filename
            context['ext'] = ext
            context['returncode'] = result.returncode
            context['stdout'] = (result.stdout or '').strip()
            stderr = result.stderr or ''
            stderr_lines = [
                line for line in stderr.splitlines()
                if 'GUI dependencies are not installed' not in line
                and 'cloud_data_retrieval.py' not in line
                and not line.strip().startswith('warnings.warn')
                and 'UserWarning' not in line
            ]
            context['stderr'] = '\n'.join(stderr_lines).strip()
            context['valid'] = result.returncode == 0
        except Exception as e:
            context['error'] = str(e)
            context['traceback'] = traceback.format_exc()
        finally:
            os.unlink(tmp_path)

    return render(request, 'viewer/validator.html', context)


# ─── CLI Runner ──────────────────────────────────────────────────────────────

def cli_runner(request):
    commands = [
        {'cmd': '--help', 'desc': 'Show help / available commands'},
        {'cmd': 'rdvxz-to-rdvxm --help', 'desc': 'Convert API 900 → API 1000/M (help)'},
        {'cmd': 'rdvxm-to-rdvxz --help', 'desc': 'Convert API 1000/M → API 900 (help)'},
        {'cmd': 'rdvxz-to-json --help', 'desc': 'Export API 900 → JSON (help)'},
        {'cmd': 'rdvxm-to-json --help', 'desc': 'Export API 1000/M → JSON (help)'},
        {'cmd': 'validate-m --help', 'desc': 'Validate API 1000/M file (help)'},
        {'cmd': 'sort-unstructured --help', 'desc': 'Sort unstructured RedVox files (help)'},
        {'cmd': 'print-z --help', 'desc': 'Print .rdvxz contents (help)'},
        {'cmd': 'print-m --help', 'desc': 'Print .rdvxm contents (help)'},
        {'cmd': 'data-req --help', 'desc': 'Data request from cloud (help)'},
        {'cmd': 'cloud-download --help', 'desc': 'Cloud data download (help)'},
    ]
    context = {'commands': commands}

    if request.method == 'POST':
        cmd = request.POST.get('command', '').strip()
        args = request.POST.get('args', '').strip()
        if not cmd:
            context['error'] = 'No command selected.'
            return render(request, 'viewer/cli_runner.html', context)

        cli_args = cmd.split() + (args.split() if args else [])
        try:
            result = _run_redvox_cli(cli_args, timeout=15)
            context['ran_cmd'] = ' '.join([sys.executable, '-m', 'redvox.cli.cli', *cli_args])
            context['stdout'] = result.stdout
            context['stderr'] = result.stderr
            context['returncode'] = result.returncode
        except subprocess.TimeoutExpired:
            context['error'] = 'Command timed out after 15 seconds.'
        except Exception as e:
            context['error'] = str(e)

    return render(request, 'viewer/cli_runner.html', context)


# ─── Signal Analysis ─────────────────────────────────────────────────────────

def analysis(request):
    context = {}
    if request.method == 'POST' and request.FILES.get('file'):
        uploaded = request.FILES['file']
        filename = uploaded.name
        ext = Path(filename).suffix.lower()

        if ext not in ('.rdvxz', '.rdvxm'):
            context['error'] = 'Signal analysis requires .rdvxz or .rdvxm files.'
            return render(request, 'viewer/analysis.html', context)

        with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp:
            for chunk in uploaded.chunks():
                tmp.write(chunk)
            tmp_path = tmp.name

        try:
            if ext == '.rdvxz':
                context.update(_analyze_rdvxz(tmp_path, filename))
            else:
                context.update(_analyze_rdvxm(tmp_path, filename))
        except Exception as e:
            context['error'] = str(e)
            context['traceback'] = traceback.format_exc()
        finally:
            os.unlink(tmp_path)

    return render(request, 'viewer/analysis.html', context)


def _analyze_rdvxz(path: str, filename: str) -> dict:
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ModuleNotFoundError as e:
        return {
            'filename': filename,
            'api': 'API 900',
            'error': (
                f'{e}. Install matplotlib to enable Signal Analysis. '
                'Example: pip install matplotlib'
            ),
        }
    from scipy import signal as scipy_signal

    p = _read_rdvxz(path)
    result = {'filename': filename, 'api': 'API 900'}

    if not p.has_microphone_sensor():
        result['warning'] = 'No microphone sensor found in this file. Cannot perform audio analysis.'
        return result

    mic = p.microphone_sensor()
    samples = np.array(mic.payload_values(), dtype=np.float64)
    sr = float(mic.sample_rate_hz())
    duration = len(samples) / sr
    t = np.linspace(0, duration, len(samples))

    result['stats'] = {
        'Sample Rate': f'{sr:.2f} Hz',
        'Num Samples': len(samples),
        'Duration': f'{duration:.3f} s',
        'Min Value': f'{samples.min():.4f}',
        'Max Value': f'{samples.max():.4f}',
        'Mean': f'{samples.mean():.4f}',
        'Std Dev': f'{samples.std():.4f}',
        'RMS': f'{np.sqrt(np.mean(samples**2)):.4f}',
    }

    # Waveform
    fig, ax = plt.subplots(figsize=(10, 2.5))
    ax.plot(t, samples, color='#89b4fa', linewidth=0.5)
    ax.set_xlabel('Time (s)', color='#cdd6f4')
    ax.set_ylabel('Amplitude', color='#cdd6f4')
    ax.set_title(f'Audio Waveform — {filename}', color='#cdd6f4')
    ax.tick_params(colors='#cdd6f4')
    for spine in ax.spines.values():
        spine.set_color('#313244')
    fig.patch.set_facecolor('#1e1e2e')
    ax.set_facecolor('#181825')
    result['waveform_img'] = _b64_figure(fig)
    plt.close(fig)

    # FFT
    fft_vals = np.abs(np.fft.rfft(samples))
    fft_freqs = np.fft.rfftfreq(len(samples), d=1.0 / sr)
    fig2, ax2 = plt.subplots(figsize=(10, 2.5))
    ax2.semilogy(fft_freqs, fft_vals + 1e-12, color='#a6e3a1', linewidth=0.8)
    ax2.set_xlabel('Frequency (Hz)', color='#cdd6f4')
    ax2.set_ylabel('Magnitude', color='#cdd6f4')
    ax2.set_title('FFT Spectrum', color='#cdd6f4')
    ax2.tick_params(colors='#cdd6f4')
    for spine in ax2.spines.values():
        spine.set_color('#313244')
    fig2.patch.set_facecolor('#1e1e2e')
    ax2.set_facecolor('#181825')
    result['fft_img'] = _b64_figure(fig2)
    plt.close(fig2)

    # Spectrogram
    if len(samples) >= 256:
        fig3, ax3 = plt.subplots(figsize=(10, 3))
        f, tt, Sxx = scipy_signal.spectrogram(samples, fs=sr, nperseg=min(256, len(samples) // 4))
        ax3.pcolormesh(tt, f, 10 * np.log10(Sxx + 1e-12), shading='gouraud', cmap='magma')
        ax3.set_ylabel('Frequency (Hz)', color='#cdd6f4')
        ax3.set_xlabel('Time (s)', color='#cdd6f4')
        ax3.set_title('Spectrogram', color='#cdd6f4')
        ax3.tick_params(colors='#cdd6f4')
        for spine in ax3.spines.values():
            spine.set_color('#313244')
        fig3.patch.set_facecolor('#1e1e2e')
        ax3.set_facecolor('#181825')
        result['spectrogram_img'] = _b64_figure(fig3)
        plt.close(fig3)

    # Peak frequency
    peak_idx = np.argmax(fft_vals)
    result['peak_freq'] = f'{fft_freqs[peak_idx]:.2f} Hz'

    return result


def _analyze_rdvxm(path: str, filename: str) -> dict:
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ModuleNotFoundError as e:
        return {
            'filename': filename,
            'api': 'API 1000/M',
            'error': (
                f'{e}. Install matplotlib to enable Signal Analysis. '
                'Example: pip install matplotlib'
            ),
        }
    from scipy import signal as scipy_signal

    p = _read_rdvxm(path)
    sensors_obj = p.get_sensors()
    result = {'filename': filename, 'api': 'API 1000/M'}

    if not sensors_obj.has_audio():
        result['warning'] = 'No audio sensor found. Cannot perform audio analysis.'
        # Try pressure analysis
        if sensors_obj.has_pressure():
            press = sensors_obj.get_pressure()
            samples = np.array(press.get_samples().get_values(), dtype=np.float64)
            sr = press.get_sample_rate()
            result['warning'] = None
            result['sensor_used'] = 'Pressure (Barometer)'
        else:
            return result
    else:
        audio = sensors_obj.get_audio()
        samples = np.array(audio.get_samples().get_values(), dtype=np.float64)
        sr = float(audio.get_sample_rate())
        result['sensor_used'] = 'Audio Microphone'

        try:
            from scipy.io import wavfile
            out_dir = Path(settings.MEDIA_ROOT) / 'generated'
            out_dir.mkdir(parents=True, exist_ok=True)
            out_name = f"{Path(filename).stem}_{uuid.uuid4().hex[:8]}.wav"
            out_path = out_dir / out_name

            # Normalize to int16 for browser playback
            if samples.size > 0:
                peak = float(np.max(np.abs(samples)))
                if peak > 0:
                    pcm = (samples / peak * 32767.0).astype(np.int16)
                else:
                    pcm = np.zeros_like(samples, dtype=np.int16)
                wavfile.write(str(out_path), int(sr), pcm)
                result['audio_url'] = f"/media/generated/{out_name}"
        except Exception:
            # Audio playback is optional; ignore failures and continue analysis.
            pass

    duration = len(samples) / sr if sr > 0 else 0
    t = np.linspace(0, duration, len(samples))

    result['stats'] = {
        'Sample Rate': f'{sr:.2f} Hz',
        'Num Samples': len(samples),
        'Duration': f'{duration:.3f} s',
        'Min Value': f'{samples.min():.4f}',
        'Max Value': f'{samples.max():.4f}',
        'Mean': f'{samples.mean():.6f}',
        'Std Dev': f'{samples.std():.6f}',
        'RMS': f'{np.sqrt(np.mean(samples**2)):.6f}',
    }

    # Waveform
    fig, ax = plt.subplots(figsize=(10, 2.5))
    ax.plot(t, samples, color='#cba6f7', linewidth=0.5)
    ax.set_xlabel('Time (s)', color='#cdd6f4')
    ax.set_ylabel('Amplitude', color='#cdd6f4')
    ax.set_title(f'Waveform — {filename} ({result.get("sensor_used", "audio")})', color='#cdd6f4')
    ax.tick_params(colors='#cdd6f4')
    for spine in ax.spines.values():
        spine.set_color('#313244')
    fig.patch.set_facecolor('#1e1e2e')
    ax.set_facecolor('#181825')
    result['waveform_img'] = _b64_figure(fig)
    plt.close(fig)

    # FFT
    if len(samples) > 1:
        fft_vals = np.abs(np.fft.rfft(samples))
        fft_freqs = np.fft.rfftfreq(len(samples), d=1.0 / sr) if sr > 0 else np.arange(len(fft_vals))
        fig2, ax2 = plt.subplots(figsize=(10, 2.5))
        ax2.semilogy(fft_freqs, fft_vals + 1e-12, color='#f38ba8', linewidth=0.8)
        ax2.set_xlabel('Frequency (Hz)', color='#cdd6f4')
        ax2.set_ylabel('Magnitude', color='#cdd6f4')
        ax2.set_title('FFT Spectrum', color='#cdd6f4')
        ax2.tick_params(colors='#cdd6f4')
        for spine in ax2.spines.values():
            spine.set_color('#313244')
        fig2.patch.set_facecolor('#1e1e2e')
        ax2.set_facecolor('#181825')
        result['fft_img'] = _b64_figure(fig2)
        plt.close(fig2)

        # Spectrogram
        if len(samples) >= 256 and sr > 0:
            fig3, ax3 = plt.subplots(figsize=(10, 3))
            f, tt, Sxx = scipy_signal.spectrogram(samples, fs=sr, nperseg=min(256, len(samples) // 4))
            ax3.pcolormesh(tt, f, 10 * np.log10(Sxx + 1e-12), shading='gouraud', cmap='magma')
            ax3.set_ylabel('Frequency (Hz)', color='#cdd6f4')
            ax3.set_xlabel('Time (s)', color='#cdd6f4')
            ax3.set_title('Spectrogram', color='#cdd6f4')
            ax3.tick_params(colors='#cdd6f4')
            for spine in ax3.spines.values():
                spine.set_color('#313244')
            fig3.patch.set_facecolor('#1e1e2e')
            ax3.set_facecolor('#181825')
            result['spectrogram_img'] = _b64_figure(fig3)
            plt.close(fig3)

        peak_idx = np.argmax(fft_vals)
        result['peak_freq'] = f'{fft_freqs[peak_idx]:.2f} Hz'

    return result


def data_window(request):
    context = {
        'form_values': _default_data_window_form_values(),
        'cache_entries': _list_cached_data_windows(),
    }

    if request.method == 'POST':
        action = request.POST.get('action', 'create').strip().lower() or 'create'
        try:
            if action == 'load':
                cache_key = request.POST.get('cache_key', '').strip()
                summary = _load_cached_data_window_summary(cache_key)
                context['data_window_summary'] = summary
                context['selected_cache_key'] = summary.get('cache_key', cache_key)
                context['success'] = f'Loaded DataWindow "{summary["event_name"]}".'
            else:
                form_values = _data_window_form_values(request.POST)
                summary = _create_cached_data_window(form_values)
                context['form_values'] = form_values
                context['data_window_summary'] = summary
                context['selected_cache_key'] = summary.get('cache_key')
                context['success'] = f'Created DataWindow "{summary["event_name"]}".'
        except Exception as e:
            if action != 'load':
                context['form_values'] = _data_window_form_values(request.POST)
            context['error'] = str(e)
            context['traceback'] = traceback.format_exc()
        context['cache_entries'] = _list_cached_data_windows()

    return render(request, 'viewer/data_window.html', context)


# ─── RedVox Cloud ─────────────────────────────────────────────────────────────

def cloud(request):
    context = {
        'auth_token': request.session.get('rv_auth_token'),
        'username': request.session.get('rv_username'),
        'host': request.session.get('rv_host', 'redvox.io'),
        'station_stats_text': None,
    }

    if request.method == 'POST':
        action = request.POST.get('action')

        if action == 'login':
            return _cloud_login(request, context)
        elif action == 'logout':
            request.session.pop('rv_auth_token', None)
            request.session.pop('rv_username', None)
            return redirect('cloud')
        elif action == 'station_stats':
            return _cloud_station_stats(request, context)
        elif action == 'validate_token':
            return _cloud_validate(request, context)

    return render(request, 'viewer/cloud.html', context)


def _cloud_login(request, context):
    username = request.POST.get('username', '').strip()
    password = request.POST.get('password', '').strip()
    host = request.POST.get('host', 'redvox.io').strip()
    port = int(request.POST.get('port', 8080))
    protocol = request.POST.get('protocol', 'https')

    try:
        from redvox.cloud.config import RedVoxConfig
        from redvox.cloud.client import CloudClient

        config = RedVoxConfig(
            username=username,
            password=password,
            protocol=protocol,
            host=host,
            port=port,
        )
        client = CloudClient(config)
        health = client.health_check()
        if not health:
            context['error'] = 'Cloud server health check failed. Check host/port settings.'
            client.close()
            return render(request, 'viewer/cloud.html', context)

        auth = client.authenticate_user(username, password)
        if auth is None or not auth.is_success():
            context['error'] = f'Authentication failed. Status: {getattr(auth, "status", "unknown")}. Check your credentials.'
            client.close()
            return render(request, 'viewer/cloud.html', context)

        request.session['rv_auth_token'] = auth.auth_token
        request.session['rv_username'] = username
        request.session['rv_host'] = host
        request.session['rv_port'] = port
        request.session['rv_protocol'] = protocol
        client.close()
        context['success'] = f'Logged in as {username} on {host}'
        context['auth_token'] = auth.auth_token
        context['username'] = username
    except Exception as e:
        context['error'] = f'Login error: {e}'
        context['traceback'] = traceback.format_exc()

    return render(request, 'viewer/cloud.html', context)


def _cloud_validate(request, context):
    auth_token = request.session.get('rv_auth_token')
    if not auth_token:
        context['error'] = 'Not logged in.'
        return render(request, 'viewer/cloud.html', context)
    try:
        from redvox.cloud.config import RedVoxConfig
        from redvox.cloud.client import CloudClient

        host = request.session.get('rv_host', 'redvox.io')
        port = request.session.get('rv_port', 8080)
        protocol = request.session.get('rv_protocol', 'https')
        config = RedVoxConfig.from_auth_token(auth_token, protocol=protocol, host=host, port=port)
        client = CloudClient(config)
        result = client.validate_own_auth_token()
        client.close()
        if result:
            context['success'] = f'Token valid. Claims: sub={result.sub}, tier={result.tier}, exp={result.exp}'
        else:
            context['error'] = 'Token validation failed. You may need to log in again.'
            request.session.pop('rv_auth_token', None)
            context['auth_token'] = None
    except Exception as e:
        context['error'] = f'Validation error: {e}'

    return render(request, 'viewer/cloud.html', context)


def _cloud_station_stats(request, context):
    auth_token = request.session.get('rv_auth_token')
    if not auth_token:
        context['error'] = 'Not logged in.'
        return render(request, 'viewer/cloud.html', context)
    try:
        from redvox.cloud.config import RedVoxConfig
        from redvox.cloud.client import CloudClient

        host = request.session.get('rv_host', 'redvox.io')
        port = request.session.get('rv_port', 8080)
        protocol = request.session.get('rv_protocol', 'https')
        station_ids_raw = request.POST.get('station_ids', '').strip()
        station_ids = [s for s in station_ids_raw.replace(',', ' ').split() if s]
        if not station_ids:
            context['error'] = 'Please provide one or more Station IDs.'
            return render(request, 'viewer/cloud.html', context)

        end_ts_s = int(request.POST.get('end_ts_s') or time.time())
        start_ts_s = int(request.POST.get('start_ts_s') or (end_ts_s - 3600))

        config = RedVoxConfig.from_auth_token(auth_token, protocol=protocol, host=host, port=port)
        client = CloudClient(config)
        stats = client.request_station_stats(start_ts_s, end_ts_s, station_ids)
        client.close()
        context['station_stats'] = stats
        if stats is None:
            context['station_stats_text'] = 'No station stats returned (None).'
            context['error'] = 'Station stats request returned no data. Verify station IDs and time range.'
        else:
            context['station_stats_text'] = pformat(stats)
            context['success'] = 'Retrieved station stats.'
    except Exception as e:
        context['error'] = f'Error fetching station stats: {e}'

    return render(request, 'viewer/cloud.html', context)


# ─── Sample Files ─────────────────────────────────────────────────────────────

def samples(request):
    samples_dir = Path(settings.MEDIA_ROOT) / 'samples'
    files = []
    if samples_dir.exists():
        for f in sorted(samples_dir.iterdir()):
            if f.is_file() and f.suffix in ('.rdvxz', '.rdvxm', '.json'):
                files.append({
                    'name': f.name,
                    'ext': f.suffix,
                    'size': f'{f.stat().st_size / 1024:.1f} KB',
                    'url': f'/media/samples/{f.name}',
                })
    return render(request, 'viewer/samples.html', {'files': files})


def download_sample(request, filename):
    safe_name = Path(filename).name
    path = Path(settings.MEDIA_ROOT) / 'samples' / safe_name
    if not path.exists() or not path.is_file():
        from django.http import Http404
        raise Http404
    with open(path, 'rb') as f:
        content = f.read()
    response = HttpResponse(content, content_type='application/octet-stream')
    response['Content-Disposition'] = f'attachment; filename="{safe_name}"'
    return response


# ─── API Info ─────────────────────────────────────────────────────────────────

def api_info(request):
    import redvox
    return JsonResponse({
        'version': redvox.VERSION,
        'api900_formats': ['.rdvxz'],
        'api1000_formats': ['.rdvxm'],
        'features': ['inspect', 'convert', 'validate', 'analysis', 'data_window', 'cloud', 'samples'],
        'signal_analysis': ['waveform', 'fft', 'spectrogram'],
        'cloud_endpoints': [
            'authenticate', 'validate_token', 'station_stats',
        ],
        'cli_commands': [
            'rdvxz-to-rdvxm', 'rdvxm-to-rdvxz',
            'rdvxz-to-json', 'rdvxm-to-json',
            'json-to-rdvxz', 'json-to-rdvxm',
            'sort-unstructured', 'print-z', 'print-m', 'validate-m',
            'data-req', 'data-req-report', 'cloud-download',
        ],
    })
