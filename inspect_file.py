from ezc3d import c3d
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np

c = c3d("./138_HealthyPiG_10.05/SUBJ01/SUBJ1 (1).c3d")

# for i, label in enumerate(c['parameters']['POINT']["LABELS"]["value"]):
#     print(f"Index: {i}, Label: {label}")

all_points = c['data']['points']

fig = plt.figure()
ax = fig.add_subplot(projection='3d')

# Get all labels
labels = c['parameters']['POINT']["LABELS"]["value"]

# Helper to get point data by label
def get_point_data(label):
    if label in labels:
        idx = labels.index(label)
        return all_points[:, idx, :]
    return None

# Define skeleton connections
skeleton = [
    ("CentreOfMass", "TRXO"), ("TRXO", "HEDO"), # Spine
    ("TRXO", "LCLO"), ("TRXO", "RCLO"), # Shoulders
    ("LCLO", "LHUO"), ("LHUO", "LRAO"), ("LRAO", "LHNO"), # Left Arm
    ("RCLO", "RHUO"), ("RHUO", "RRAO"), ("RRAO", "RHNO"), # Right Arm
    ("CentreOfMass", "LFEP"),("LFEP", "LFEO"), ("LFEO", "LTIO"), ("LTIO", "LFOO"), # Left Leg
    ("CentreOfMass", "RFEP"),("RFEP", "RFEO"), ("RFEO", "RTIO"), ("RTIO", "RFOO"), # Right Leg

]

lines_data = []
for p1_name, p2_name in skeleton:
    p1_data = get_point_data(p1_name)
    p2_data = get_point_data(p2_name)
    if p1_data is not None and p2_data is not None:
        lines_data.append((p1_data, p2_data))

lines = [ax.plot([], [], [], marker='o', color='blue')[0] for _ in lines_data]

# Create text labels for markers
unique_labels = set()
for p1, p2 in skeleton:
    unique_labels.add(p1)
    unique_labels.add(p2)

text_objects = {}
for label in unique_labels:
    import_data = get_point_data(label)
    if import_data is not None:
        text_objects[label] = ax.text(0, 0, 0, label, color='black')

zoom_factor = 1.0

def on_scroll(event):
    global zoom_factor
    if event.button == 'up':
        zoom_factor /= 1.1
    elif event.button == 'down':
        zoom_factor *= 1.1

fig.canvas.mpl_connect('scroll_event', on_scroll)

all_x = []
all_y = []
all_z = []
for p1_data, p2_data in lines_data:
    all_x.extend(p1_data[0, :])
    all_x.extend(p2_data[0, :])
    all_y.extend(p1_data[1, :])
    all_y.extend(p2_data[1, :])
    all_z.extend(p1_data[2, :])
    all_z.extend(p2_data[2, :])

max_extent = 0
for frame in range(all_points.shape[2]):
    frame_points = all_points[:, :, frame]
    
    # Filter out NaNs
    valid_mask = ~np.isnan(frame_points[0])
    if not np.any(valid_mask):
        continue
        
    valid_points = frame_points[:, valid_mask]
    
    center = np.mean(valid_points, axis=1)
    
    # Calculate max distance from center in any dimension
    dists = np.abs(valid_points - center[:, np.newaxis])
    max_dist = np.max(dists)
    if max_dist > max_extent:
        max_extent = max_dist

box_size = max_extent * 0.1

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

def update(frame):
    global zoom_factor
    current_points = []
    for line, (p1_data, p2_data) in zip(lines, lines_data):
        xs = [p1_data[0, frame], p2_data[0, frame]]
        ys = [p1_data[1, frame], p2_data[1, frame]]
        zs = [p1_data[2, frame], p2_data[2, frame]]
        line.set_data(xs, ys)
        line.set_3d_properties(zs)
        
        # Collect points to calculate center for this frame
        current_points.append([p1_data[0, frame], p1_data[1, frame], p1_data[2, frame]])
        current_points.append([p2_data[0, frame], p2_data[1, frame], p2_data[2, frame]])
    
    # Update text positions
    for label, text_obj in text_objects.items():
        data = get_point_data(label)
        if data is not None:
            x, y, z = data[0, frame], data[1, frame], data[2, frame]
            if not np.isnan(x):
                text_obj.set_position((x, y))
                text_obj.set_3d_properties(z)
                text_obj.set_visible(True)
            else:
                text_obj.set_visible(False)
    
    # Update camera center
    if current_points:
        current_points = np.array(current_points)
        # Handle NaNs if any point is missing
        valid_mask = ~np.isnan(current_points[:, 0])
        if np.any(valid_mask):
            valid_points = current_points[valid_mask]
            center = np.mean(valid_points, axis=0)
            
            current_box_size = box_size * zoom_factor
            ax.set_xlim(center[0] - current_box_size, center[0] + current_box_size)
            ax.set_ylim(center[1] - current_box_size, center[1] + current_box_size)
            ax.set_zlim(center[2] - current_box_size, center[2] + current_box_size)

    return lines + list(text_objects.values())

# Create animation
num_frames = all_points.shape[2]
ani = animation.FuncAnimation(fig, update, frames=num_frames, interval=50, blit=False)

plt.show()
