import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
from unittest.mock import patch

from django.conf import settings
from django.core.files.uploadedfile import SimpleUploadedFile
from django.test import SimpleTestCase

from . import views


def _sample_data_window_summary(cache_key: str = 'cache-1', event_name: str = 'demo-window'):
    return {
        'event_name': event_name,
        'out_type': 'LZ4',
        'sdk_version': 'test-sdk',
        'cache_key': cache_key,
        'cache_dir': 'C:/cache/data_windows/cache-1',
        'metadata_path': 'C:/cache/data_windows/cache-1/demo-window.json',
        'saved_artifact': 'C:/cache/data_windows/cache-1/demo-window.pkl.lz4',
        'saved_files': ['demo-window.json', 'demo-window.pkl.lz4'],
        'station_count': 1,
        'station_ids': ['1637610021'],
        'start_time': '2024-01-01 00:00:00.000000 UTC',
        'end_time': '2024-01-01 00:01:00.000000 UTC',
        'errors': [],
        'error_count': 0,
        'config': {
            'input_dir': 'C:/input',
            'structured_layout': True,
            'start_datetime': '2024-01-01 00:00:00 UTC',
            'end_datetime': '2024-01-01 00:01:00 UTC',
            'start_buffer_seconds': '120',
            'end_buffer_seconds': '120',
            'drop_time_seconds': '0.2',
            'station_ids': ['1637610021'],
            'apply_correction': True,
            'use_model_correction': True,
            'copy_edge_points': 'COPY',
        },
        'stations': [
            {
                'id': '1637610021',
                'uuid': 'uuid-1',
                'start_date': '2024-01-01 00:00:00.000000 UTC',
                'first_data_timestamp': '2024-01-01 00:00:00.000000 UTC',
                'last_data_timestamp': '2024-01-01 00:01:00.000000 UTC',
                'sensors': ['AUDIO', 'BAROMETER'],
                'errors': [],
                'error_count': 0,
            }
        ],
    }


class ViewerSettingsTests(SimpleTestCase):
    def test_repo_root_is_available_on_sys_path(self):
        self.assertIn(str(settings.REPO_ROOT), sys.path)


class ViewerRouteSmokeTests(SimpleTestCase):
    def test_main_routes_render(self):
        for path in ['/', '/inspect/', '/data_window/', '/converter/', '/validator/', '/cli/', '/analysis/', '/cloud/', '/samples/', '/api/info/']:
            with self.subTest(path=path):
                self.assertEqual(self.client.get(path).status_code, 200)


class ViewerDataWindowTests(SimpleTestCase):
    @patch('viewer.views._list_cached_data_windows', return_value=[])
    @patch('viewer.views._create_cached_data_window')
    def test_data_window_create_flow_uses_helper(self, create_cached_data_window, list_cached_data_windows):
        create_cached_data_window.return_value = _sample_data_window_summary(event_name='demo-window')

        response = self.client.post(
            '/data_window/',
            {
                'action': 'create',
                'input_dir': 'C:/input',
                'event_name': 'demo-window',
                'output_type': 'LZ4',
            },
        )

        self.assertEqual(response.status_code, 200)
        self.assertContains(response, 'Created DataWindow &quot;demo-window&quot;.')
        self.assertContains(response, 'demo-window.pkl.lz4')
        create_cached_data_window.assert_called_once()
        list_cached_data_windows.assert_called()

    @patch('viewer.views._list_cached_data_windows')
    @patch('viewer.views._load_cached_data_window_summary')
    def test_data_window_load_flow_uses_helper(self, load_cached_data_window_summary, list_cached_data_windows):
        list_cached_data_windows.return_value = [
            {
                'cache_key': 'cache-1',
                'display_name': 'demo-window',
                'out_type': 'LZ4',
                'station_count': 1,
                'modified': '2024-01-01 00:00:00 UTC',
                'input_dir': 'C:/input',
                'metadata_name': 'demo-window.json',
                'location': 'C:/cache/data_windows/cache-1',
                'errors': [],
            }
        ]
        load_cached_data_window_summary.return_value = _sample_data_window_summary(event_name='demo-window')

        response = self.client.post('/data_window/', {'action': 'load', 'cache_key': 'cache-1'})

        self.assertEqual(response.status_code, 200)
        self.assertContains(response, 'Loaded DataWindow &quot;demo-window&quot;.')
        self.assertContains(response, '1637610021')
        load_cached_data_window_summary.assert_called_once_with('cache-1')

    def test_list_cached_data_windows_uses_data_window_metadata_json(self):
        with tempfile.TemporaryDirectory() as media_root:
            cache_dir = Path(media_root) / 'data_windows' / 'cache-1'
            cache_dir.mkdir(parents=True, exist_ok=True)
            (cache_dir / 'station_1637610021.json').write_text(
                json.dumps({'id': '1637610021'}),
                encoding='utf-8',
            )
            (cache_dir / 'demo-window.json').write_text(
                json.dumps(
                    {
                        'event_name': 'demo-window',
                        'out_type': 'LZ4',
                        'config': {'input_dir': 'C:/input'},
                        'stations': ['station_1637610021'],
                        'errors': {'obj_class': 'DataWindow'},
                    }
                ),
                encoding='utf-8',
            )

            with self.settings(MEDIA_ROOT=media_root):
                entries = views._list_cached_data_windows()

        self.assertEqual(len(entries), 1)
        self.assertEqual(entries[0]['cache_key'], 'cache-1')
        self.assertEqual(entries[0]['display_name'], 'demo-window')
        self.assertEqual(entries[0]['metadata_name'], 'demo-window.json')
        self.assertEqual(entries[0]['input_dir'], 'C:/input')

    def test_create_cached_data_window_restores_cwd_after_save(self):
        class FakeErrors:
            def as_dict(self):
                return {'obj_class': 'DataWindow'}

        class FakeDataWindow:
            def __init__(self, event_name='dw', event_origin=None, config=None, output_dir='.', out_type='NONE', make_runme=False, debug=False):
                self.event_name = event_name
                self._config = config
                self._save_dir = output_dir
                self._out_type = out_type
                self._errors = FakeErrors()
                os.chdir(output_dir)

            def save(self):
                metadata_path = Path(self._save_dir) / f'{self.event_name}.json'
                metadata_path.write_text(
                    json.dumps(
                        {
                            'event_name': self.event_name,
                            'out_type': self._out_type,
                            'config': self._config.to_dict(),
                            'stations': [],
                            'errors': {'obj_class': 'DataWindow'},
                        }
                    ),
                    encoding='utf-8',
                )
                artifact_path = Path(self._save_dir) / f'{self.event_name}.pkl.lz4'
                artifact_path.write_bytes(b'test-bytes')
                return artifact_path

            def config(self):
                return self._config

            def errors(self):
                return self._errors

            def save_dir(self):
                return self._save_dir

            def out_type(self):
                return self._out_type

            def sdk_version(self):
                return 'fake-sdk'

            def stations(self):
                return []

            def station_ids(self):
                return []

            def start_date(self):
                return 1704067200000000

            def end_date(self):
                return 1704067260000000

        form_values = views._default_data_window_form_values()
        with tempfile.TemporaryDirectory() as media_root, tempfile.TemporaryDirectory() as input_dir:
            form_values.update({'input_dir': input_dir, 'event_name': 'demo-window', 'output_type': 'LZ4'})
            original_cwd = os.getcwd()
            with self.settings(MEDIA_ROOT=media_root):
                with patch('redvox.common.data_window.DataWindow', FakeDataWindow):
                    summary = views._create_cached_data_window(form_values)

            self.assertEqual(os.getcwd(), original_cwd)
            self.assertEqual(summary['event_name'], 'demo-window')
            self.assertTrue(summary['metadata_path'].endswith('demo-window.json'))
            self.assertTrue(summary['saved_artifact'].endswith('demo-window.pkl.lz4'))
            self.assertTrue(Path(summary['metadata_path']).exists())


class ViewerCliIntegrationTests(SimpleTestCase):
    def test_redvox_cli_env_includes_repo_root(self):
        with patch.dict('os.environ', {}, clear=True):
            env = views._redvox_cli_env()

        self.assertEqual(env['PYTHONPATH'], str(views.REDVOX_REPO_ROOT))

    @patch('viewer.views._run_redvox_cli')
    def test_converter_returns_download_when_cli_creates_output(self, run_redvox_cli):
        def fake_run(args, timeout):
            out_dir = Path(args[2])
            out_dir.mkdir(parents=True, exist_ok=True)
            (out_dir / 'converted.json').write_text('{"ok": true}', encoding='utf-8')
            return SimpleNamespace(returncode=0, stdout='ok', stderr='')

        run_redvox_cli.side_effect = fake_run

        response = self.client.post(
            '/converter/',
            {
                'action': 'rdvxz_to_json',
                'file': SimpleUploadedFile('example.rdvxz', b'example-bytes'),
            },
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response['Content-Disposition'], 'attachment; filename="converted.json"')
        self.assertEqual(response.content, b'{"ok": true}')
        run_redvox_cli.assert_called_once()
        self.assertEqual(run_redvox_cli.call_args.args[0][0], 'rdvxz-to-json')
        self.assertEqual(run_redvox_cli.call_args.kwargs['timeout'], 30)

    @patch('viewer.views._run_redvox_cli')
    def test_validator_uses_cli_helper_for_rdvxm_files(self, run_redvox_cli):
        run_redvox_cli.return_value = SimpleNamespace(
            returncode=0,
            stdout='validation ok',
            stderr='',
        )

        response = self.client.post(
            '/validator/',
            {'file': SimpleUploadedFile('example.rdvxm', b'compressed-bytes')},
        )

        self.assertEqual(response.status_code, 200)
        self.assertContains(response, 'validation ok')
        run_redvox_cli.assert_called_once()
        self.assertEqual(run_redvox_cli.call_args.args[0][0], 'validate-m')
        self.assertEqual(run_redvox_cli.call_args.kwargs['timeout'], 30)

    @patch('viewer.views._run_redvox_cli')
    def test_cli_runner_uses_cli_helper(self, run_redvox_cli):
        run_redvox_cli.return_value = SimpleNamespace(
            returncode=0,
            stdout='usage: redvox-cli',
            stderr='',
        )

        response = self.client.post('/cli/', {'command': '--help', 'args': ''})

        self.assertEqual(response.status_code, 200)
        self.assertContains(response, 'usage: redvox-cli')
        run_redvox_cli.assert_called_once_with(['--help'], timeout=15)

from django.contrib.auth.models import User
from .models import DashboardShareToken

class Phase2And3Tests(SimpleTestCase):
    def setUp(self):
        super().setUp()
        self.client = self.client_class()
        
    def test_gis_filter_endpoint_post(self):
        payload = {
            "type": "Feature",
            "geometry": {
                "type": "Polygon",
                "coordinates": [[
                    [-155.3, 19.3], [-155.1, 19.3],
                    [-155.1, 19.5], [-155.3, 19.5], [-155.3, 19.3]
                ]]
            }
        }
        response = self.client.post(
            '/api/gis/filter/', 
            json.dumps(payload),
            content_type='application/json'
        )
        self.assertEqual(response.status_code, 200)
        
    def test_gis_filter_endpoint_requires_post(self):
        response = self.client.get('/api/gis/filter/')
        self.assertEqual(response.status_code, 405)
        
    def test_export_hdf5_endpoint(self):
        response = self.client.get('/api/export/hdf5/')
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response['Content-Type'], 'application/x-hdf5')
        
    def test_export_netcdf_endpoint(self):
        response = self.client.get('/api/export/netcdf/')
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response['Content-Type'], 'application/x-netcdf')
