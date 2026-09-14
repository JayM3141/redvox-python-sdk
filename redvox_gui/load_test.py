import threading
import requests
import time
import uuid
import json

BASE_URL = "http://127.0.0.1:8000"
NUM_THREADS = 50

def test_ml_endpoint():
    try:
        start_time = time.time()
        resp = requests.post(f"{BASE_URL}/api/ml/analyze_audio/")
        if resp.status_code == 200:
            job_id = resp.json().get('job_id')
            # Poll once to simulate checking
            if job_id:
                requests.get(f"{BASE_URL}/api/ml/analyze_audio/?job_id={job_id}")
        return time.time() - start_time
    except Exception as e:
        return -1

def test_gis_endpoint():
    try:
        start_time = time.time()
        payload = {
            "type": "Feature",
            "geometry": {
                "type": "Polygon",
                "coordinates": [[[-155.5, 19.1], [-155.0, 19.1], [-155.0, 19.6], [-155.5, 19.6], [-155.5, 19.1]]]
            }
        }
        resp = requests.post(
            f"{BASE_URL}/api/gis/filter/",
            json=payload,
            headers={'Content-Type': 'application/json'}
        )
        return time.time() - start_time
    except Exception as e:
        return -1

def test_cloud_export_endpoint():
    try:
        start_time = time.time()
        resp = requests.post(
            f"{BASE_URL}/api/export/cloud/",
            data={'provider': 'aws', 'bucket': 'load-test-bucket'}
        )
        return time.time() - start_time
    except Exception as e:
        return -1


def run_load_test():
    print(f"Starting Load Test with {NUM_THREADS} concurrent threads...")
    
    results = {'ml': [], 'gis': [], 'cloud': []}
    threads = []
    
    def worker():
        # Each worker does one of each
        results['ml'].append(test_ml_endpoint())
        results['gis'].append(test_gis_endpoint())
        results['cloud'].append(test_cloud_export_endpoint())
        
    start_total = time.time()
    for i in range(NUM_THREADS):
        t = threading.Thread(target=worker)
        threads.append(t)
        t.start()
        
    for t in threads:
        t.join()
        
    total_time = time.time() - start_total
    
    def print_stats(name, rlist):
        valid = [r for r in rlist if r >= 0]
        failed = len(rlist) - len(valid)
        if valid:
            avg = sum(valid) / len(valid)
            print(f"[{name.upper()}] Avg Latency: {avg:.3f}s | Min: {min(valid):.3f}s | Max: {max(valid):.3f}s | Failures: {failed}")
        else:
            print(f"[{name.upper()}] All requests failed.")

    print(f"\n--- LOAD TEST RESULTS ---")
    print(f"Total Elapsed Time: {total_time:.2f}s for {NUM_THREADS*3} requests.")
    print_stats('ml', results['ml'])
    print_stats('gis', results['gis'])
    print_stats('cloud', results['cloud'])
    
if __name__ == "__main__":
    run_load_test()
