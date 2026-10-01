import os
import argparse
from gwpy.timeseries import TimeSeries
from gwosc.datasets import event_gps
import numpy as np
import warnings
warnings.filterwarnings('ignore')

def download_event(event_name, detector='H1', window=4096, sample_rate=4096, out_dir='data/raw'):
    print(f"Fetching {event_name} from {detector}...")
    try:
        gps = event_gps(event_name)
        start = int(gps - window/2)
        end = int(gps + window/2)
        
        # Download strain data
        ts = TimeSeries.fetch_open_data(detector, start, end, sample_rate=sample_rate)
        
        out_path = os.path.join(out_dir, f"{event_name}_{detector}.hdf5")
        ts.write(out_path, format='hdf5', overwrite=True)
        print(f"Saved to {out_path}")
        return ts
    except Exception as e:
        print(f"Failed to fetch {event_name}: {e}")
        return None

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir', type=str, default='data/raw')
    args = parser.parse_args()
    
    os.makedirs(args.dir, exist_ok=True)
    
    # We will download a few confirmed events and background noise segments
    events = ['GW150914', 'GW151226', 'GW170814', 'GW190521']
    for ev in events:
        download_event(ev, out_dir=args.dir)

