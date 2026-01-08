from ezc3d import c3d
import numpy as np

file_path = "./138_HealthyPiG_10.05/SUBJ01/SUBJ1 (1).c3d"

try:
    c = c3d(file_path)
    
    print("--- C3D File Exploration ---")
    print(f"File: {file_path}")
    
    # Header Information
    print("\n[Header Information]")
    header = c['header']
    print(f"Points Frame Rate: {header['points']['frame_rate']}")
    print(f"First Frame: {header['points']['first_frame']}")
    print(f"Last Frame: {header['points']['last_frame']}")
    
    # Parameters
    print("\n[Parameters]")
    params = c['parameters']
    
    # Point Parameters
    if 'POINT' in params:
        point_params = params['POINT']
        print(f"Point Units: {point_params['UNITS']['value'] if 'UNITS' in point_params else 'Unknown'}")
        
        if 'LABELS' in point_params:
            labels = point_params['LABELS']['value']
            print(f"Number of Point Labels: {len(labels)}")
            print("Point Labels:", labels)
        
        if 'DESCRIPTIONS' in point_params:
             print("Point Descriptions:", point_params['DESCRIPTIONS']['value'])

    # Analog Parameters
    if 'ANALOG' in params:
        analog_params = params['ANALOG']
        if 'LABELS' in analog_params:
            analog_labels = analog_params['LABELS']['value']
            print(f"\nNumber of Analog Labels: {len(analog_labels)}")
            print("Analog Labels:", analog_labels)
            
    # Data Structure
    print("\n[Data Structure]")
    data = c['data']
    
    if 'points' in data:
        points = data['points']
        print(f"Points Data Shape (XYZ x N_Points x N_Frames): {points.shape}")
        
        # Check for NaNs
        nan_count = np.isnan(points).sum()
        total_values = points.size
        print(f"Total NaN values in points: {nan_count} ({nan_count/total_values*100:.2f}%)")
        
        # Ranges
        print(f"X range: {np.nanmin(points[0])} to {np.nanmax(points[0])}")
        print(f"Y range: {np.nanmin(points[1])} to {np.nanmax(points[1])}")
        print(f"Z range: {np.nanmin(points[2])} to {np.nanmax(points[2])}")
        
        # Sample Data (Frame 0 for first 3 points)
        print("\n[Sample Data - Frame 0]")
        if points.shape[1] > 0:
            for i in range(min(5, points.shape[1])):
                label = labels[i] if i < len(labels) else f"Point {i}"
                print(f"{label}: {points[:, i, 0]}")

    if 'analogs' in data:
        analogs = data['analogs']
        print(f"Analog Data Shape (1 x N_Channels x N_Frames): {analogs.shape}")

    if 'platform' in data:
         print(f"Platform Data keys: {data['platform'].keys()}")
         
    # Metadata keys
    print("\n[Parameter Groups]")
    print(list(params.keys()))

except Exception as e:
    print(f"Error reading file: {e}")
