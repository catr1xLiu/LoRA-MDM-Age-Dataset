#!/usr/bin/env python3
"""
Real-time PLY sequence viewer using Open3D with proper world-space translation

Controls:
    SPACE     - Play/Pause
    N         - Next frame
    B         - Previous frame  
    R         - Reset to frame 0
    L         - Toggle loop
    T         - Toggle trail (show walking path)
    C         - Center camera on current position
    +/-       - Adjust speed
    Q         - Quit

Usage:
    python view_ply_open3d.py ./demo/demo_results
"""

import sys
import time
import numpy as np
import open3d as o3d
import joblib
from pathlib import Path

class PLYSequenceViewer:
    def __init__(self, ply_dir, fps=20):
        self.ply_dir = Path(ply_dir)
        self.fps = fps
        self.frame_delay = 1.0 / fps
        
        # Load all PLY and PKL files
        self.ply_files = sorted(self.ply_dir.glob('[0-9]*.ply'))
        self.pkl_files = sorted(self.ply_dir.glob('[0-9]*.pkl'))
        
        if not self.ply_files:
            raise ValueError(f"No PLY files found in {ply_dir}")
        
        if len(self.pkl_files) != len(self.ply_files):
            print(f"Warning: {len(self.ply_files)} PLY files but {len(self.pkl_files)} PKL files")
        
        print(f"Found {len(self.ply_files)} frames")
        
        # Playback state
        self.current_frame = 0
        self.is_playing = True
        self.loop = True
        self.speed_multiplier = 1.0
        self.show_trail = True
        self.follow_camera = False
        
        # Pre-load all meshes and translations
        print("Loading meshes and positions...")
        self.meshes = []
        self.translations = []
        
        for i, (ply_file, pkl_file) in enumerate(zip(self.ply_files, self.pkl_files)):
            # Load mesh
            mesh = o3d.io.read_triangle_mesh(str(ply_file))
            mesh.compute_vertex_normals()
            mesh.paint_uniform_color([0.68, 0.78, 0.91])  # Light blue
            
            # Load PKL to get root position
            try:
                pkl_data = joblib.load(str(pkl_file))
                
                # Try different keys for translation
                if 'root' in pkl_data:
                    translation = pkl_data['root']
                elif 'cam' in pkl_data:
                    translation = pkl_data['cam'].squeeze()
                elif 'transl' in pkl_data:
                    translation = pkl_data['transl'].squeeze()
                else:
                    print(f"Warning: No translation found in {pkl_file.name}, using zero")
                    translation = np.array([0.0, 0.0, 0.0])
                
                # Ensure it's a 1D array of length 3
                translation = np.array(translation).flatten()[:3]
                
            except Exception as e:
                print(f"Error loading {pkl_file.name}: {e}")
                translation = np.array([0.0, 0.0, 0.0])
            
            self.translations.append(translation)
            
            # Apply translation to mesh
            mesh.translate(translation)
            
            self.meshes.append(mesh)
            
            if (i + 1) % 20 == 0:
                print(f"  Loaded {i + 1}/{len(self.ply_files)}")
        
        print("All meshes loaded!")
        
        # Analyze motion path
        self.translations = np.array(self.translations)
        self._analyze_motion()
        
        # Create visualization elements
        self.floor = self._create_floor()
        self.coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
            size=0.5, origin=[0, 0, 0]
        )
        self.trail_line = self._create_trail()
        
        # Print controls
        self._print_controls()
    
    def _analyze_motion(self):
        """Analyze the walking motion"""
        print("\n" + "="*60)
        print("MOTION ANALYSIS:")
        print(f"  Start position: [{self.translations[0, 0]:.3f}, {self.translations[0, 1]:.3f}, {self.translations[0, 2]:.3f}]")
        print(f"  End position:   [{self.translations[-1, 0]:.3f}, {self.translations[-1, 1]:.3f}, {self.translations[-1, 2]:.3f}]")
        
        # Calculate total distance traveled
        distances = np.linalg.norm(np.diff(self.translations, axis=0), axis=1)
        total_distance = np.sum(distances)
        print(f"  Total distance: {total_distance:.3f} m")
        
        # Calculate average speed
        duration = len(self.translations) / self.fps
        avg_speed = total_distance / duration
        print(f"  Duration: {duration:.2f} s")
        print(f"  Average speed: {avg_speed:.3f} m/s ({avg_speed * 3.6:.2f} km/h)")
        
        # Bounds
        min_pos = self.translations.min(axis=0)
        max_pos = self.translations.max(axis=0)
        print(f"  Bounding box:")
        print(f"    X: [{min_pos[0]:.3f}, {max_pos[0]:.3f}] (range: {max_pos[0]-min_pos[0]:.3f} m)")
        print(f"    Y: [{min_pos[1]:.3f}, {max_pos[1]:.3f}] (range: {max_pos[1]-min_pos[1]:.3f} m)")
        print(f"    Z: [{min_pos[2]:.3f}, {max_pos[2]:.3f}] (range: {max_pos[2]-min_pos[2]:.3f} m)")
        print("="*60 + "\n")
    
    def _create_floor(self):
        """Create a grid floor centered at origin"""
        # Determine floor size based on motion bounds
        motion_range = self.translations.max(axis=0) - self.translations.min(axis=0)
        floor_size = max(10.0, motion_range[0] * 2, motion_range[2] * 2)
        
        # Create floor plane
        floor = o3d.geometry.TriangleMesh()
        
        half_size = floor_size / 2
        vertices = [
            [-half_size, 0, -half_size],
            [half_size, 0, -half_size],
            [half_size, 0, half_size],
            [-half_size, 0, half_size]
        ]
        
        triangles = [[0, 1, 2], [0, 2, 3]]
        
        floor.vertices = o3d.utility.Vector3dVector(vertices)
        floor.triangles = o3d.utility.Vector3iVector(triangles)
        floor.paint_uniform_color([0.9, 0.9, 0.9])
        floor.compute_vertex_normals()
        
        return floor
    
    def _create_grid(self):
        """Create grid lines on floor"""
        # Determine grid size
        motion_range = self.translations.max(axis=0) - self.translations.min(axis=0)
        grid_size = max(10.0, motion_range[0] * 2, motion_range[2] * 2) / 2
        
        points = []
        lines = []
        
        grid_step = 0.5
        
        idx = 0
        for i in np.arange(-grid_size, grid_size + grid_step, grid_step):
            # Lines parallel to X
            points.append([i, 0, -grid_size])
            points.append([i, 0, grid_size])
            lines.append([idx, idx + 1])
            idx += 2
            
            # Lines parallel to Z
            points.append([-grid_size, 0, i])
            points.append([grid_size, 0, i])
            lines.append([idx, idx + 1])
            idx += 2
        
        line_set = o3d.geometry.LineSet()
        line_set.points = o3d.utility.Vector3dVector(points)
        line_set.lines = o3d.utility.Vector2iVector(lines)
        line_set.paint_uniform_color([0.7, 0.7, 0.7])
        
        return line_set
    
    def _create_trail(self):
        """Create line showing the walking path"""
        if len(self.translations) < 2:
            return None
        
        # Create line set from root positions
        points = self.translations[:, :3].copy()
        # Lift trail slightly above floor for visibility
        points[:, 1] += 0.05
        
        lines = [[i, i+1] for i in range(len(points)-1)]
        
        line_set = o3d.geometry.LineSet()
        line_set.points = o3d.utility.Vector3dVector(points)
        line_set.lines = o3d.utility.Vector2iVector(lines)
        
        # Color gradient from blue (start) to red (end)
        colors = []
        for i in range(len(lines)):
            t = i / max(len(lines)-1, 1)
            # Blue to red gradient
            color = [t, 0, 1-t]
            colors.append(color)
        
        line_set.colors = o3d.utility.Vector3dVector(colors)
        
        return line_set
    
    def _print_controls(self):
        """Print control instructions"""
        print("="*60)
        print("CONTROLS:")
        print("  SPACE     - Play/Pause")
        print("  N         - Next frame")
        print("  B         - Previous frame")
        print("  R         - Reset to frame 0")
        print("  L         - Toggle loop")
        print("  T         - Toggle trail (walking path)")
        print("  C         - Center camera on current position")
        print("  F         - Toggle camera follow mode")
        print("  +         - Speed up")
        print("  -         - Slow down")
        print("  Mouse     - Rotate/zoom camera")
        print("  Q         - Quit")
        print("="*60 + "\n")
    
    def run(self):
        """Run the interactive viewer"""
        
        # Create visualizer
        vis = o3d.visualization.VisualizerWithKeyCallback()
        vis.create_window(window_name="PLY Sequence Viewer - Walking Motion", 
                         width=1280, height=720)
        
        # Add geometries
        vis.add_geometry(self.meshes[0])
        vis.add_geometry(self.floor)
        vis.add_geometry(self._create_grid())
        vis.add_geometry(self.coord_frame)
        
        if self.trail_line:
            vis.add_geometry(self.trail_line)
        
        # Setup camera to view the entire path
        ctr = vis.get_view_control()
        self._setup_camera_for_path(ctr)
        
        # Register key callbacks
        def toggle_play(vis):
            self.is_playing = not self.is_playing
            status = "Playing" if self.is_playing else "Paused"
            print(f"[{status}] Frame {self.current_frame}/{len(self.meshes)-1} | Speed: {self.speed_multiplier:.2f}x")
            return False
        
        def next_frame(vis):
            self.next_frame(vis)
            if self.follow_camera:
                self._update_camera_follow(vis)
            print(f"Frame {self.current_frame}/{len(self.meshes)-1}")
            return False
        
        def prev_frame(vis):
            self.prev_frame(vis)
            if self.follow_camera:
                self._update_camera_follow(vis)
            print(f"Frame {self.current_frame}/{len(self.meshes)-1}")
            return False
        
        def reset_frame(vis):
            self.current_frame = 0
            self.update_mesh(vis)
            if self.follow_camera:
                self._update_camera_follow(vis)
            print("Reset to frame 0")
            return False
        
        def toggle_loop(vis):
            self.loop = not self.loop
            status = "ON" if self.loop else "OFF"
            print(f"Loop: {status}")
            return False
        
        def toggle_trail(vis):
            self.show_trail = not self.show_trail
            status = "ON" if self.show_trail else "OFF"
            print(f"Trail: {status}")
            if self.show_trail and self.trail_line:
                vis.add_geometry(self.trail_line, reset_bounding_box=False)
            elif self.trail_line:
                vis.remove_geometry(self.trail_line, reset_bounding_box=False)
            return False
        
        def center_camera(vis):
            current_pos = self.translations[self.current_frame]
            ctr = vis.get_view_control()
            ctr.set_lookat(current_pos)
            print(f"Camera centered on position: [{current_pos[0]:.2f}, {current_pos[1]:.2f}, {current_pos[2]:.2f}]")
            return False
        
        def toggle_follow(vis):
            self.follow_camera = not self.follow_camera
            status = "ON" if self.follow_camera else "OFF"
            print(f"Camera follow: {status}")
            return False
        
        def speed_up(vis):
            self.speed_multiplier *= 1.5
            self.speed_multiplier = min(8.0, self.speed_multiplier)
            print(f"Speed: {self.speed_multiplier:.2f}x")
            return False
        
        def slow_down(vis):
            self.speed_multiplier /= 1.5
            self.speed_multiplier = max(0.125, self.speed_multiplier)
            print(f"Speed: {self.speed_multiplier:.2f}x")
            return False
        
        # Register callbacks
        vis.register_key_callback(32, toggle_play)   # SPACE
        vis.register_key_callback(78, next_frame)    # N
        vis.register_key_callback(66, prev_frame)    # B
        vis.register_key_callback(82, reset_frame)   # R
        vis.register_key_callback(76, toggle_loop)   # L
        vis.register_key_callback(84, toggle_trail)  # T
        vis.register_key_callback(67, center_camera) # C
        vis.register_key_callback(70, toggle_follow) # F
        vis.register_key_callback(61, speed_up)      # +
        vis.register_key_callback(45, slow_down)     # -
        
        # Playback loop
        last_update = time.time()
        
        while True:
            current_time = time.time()
            
            if self.is_playing:
                adjusted_delay = self.frame_delay / self.speed_multiplier
                
                if current_time - last_update >= adjusted_delay:
                    self.next_frame(vis)
                    if self.follow_camera:
                        self._update_camera_follow(vis)
                    last_update = current_time
            
            # Update visualization
            if not vis.poll_events():
                break
            vis.update_renderer()
            
            # Small sleep to prevent CPU spinning
            time.sleep(0.001)
        
        vis.destroy_window()
    
    def _setup_camera_for_path(self, ctr):
        """Position camera to view the entire walking path"""
        # Get path center and bounds
        path_center = self.translations.mean(axis=0)
        path_min = self.translations.min(axis=0)
        path_max = self.translations.max(axis=0)
        path_range = path_max - path_min
        
        # Position camera to see entire path
        # Camera looks from the side and slightly above
        camera_distance = max(5.0, path_range.max() * 2)
        
        ctr.set_front([0.3, -0.3, -0.9])
        ctr.set_up([0, 1, 0])
        ctr.set_lookat(path_center)
        ctr.set_zoom(0.4)
    
    def _update_camera_follow(self, vis):
        """Update camera to follow current position"""
        current_pos = self.translations[self.current_frame]
        ctr = vis.get_view_control()
        ctr.set_lookat(current_pos)
    
    def update_mesh(self, vis):
        """Update the displayed mesh"""
        vis.clear_geometries()
        vis.add_geometry(self.meshes[self.current_frame], reset_bounding_box=False)
        vis.add_geometry(self.floor, reset_bounding_box=False)
        vis.add_geometry(self._create_grid(), reset_bounding_box=False)
        vis.add_geometry(self.coord_frame, reset_bounding_box=False)
        
        if self.show_trail and self.trail_line:
            vis.add_geometry(self.trail_line, reset_bounding_box=False)
    
    def next_frame(self, vis):
        """Advance to next frame"""
        self.current_frame += 1
        if self.current_frame >= len(self.meshes):
            if self.loop:
                self.current_frame = 0
            else:
                self.current_frame = len(self.meshes) - 1
                self.is_playing = False
        self.update_mesh(vis)
    
    def prev_frame(self, vis):
        """Go to previous frame"""
        self.current_frame -= 1
        if self.current_frame < 0:
            if self.loop:
                self.current_frame = len(self.meshes) - 1
            else:
                self.current_frame = 0
        self.update_mesh(vis)


def main():
    if len(sys.argv) < 2:
        print("Usage: python view_ply_open3d.py <directory_with_ply_files> [fps]")
        print("\nExample:")
        print("  python view_ply_open3d.py ./demo/demo_results 20")
        sys.exit(1)
    
    ply_dir = sys.argv[1]
    fps = int(sys.argv[2]) if len(sys.argv) > 2 else 20
    
    viewer = PLYSequenceViewer(ply_dir, fps=fps)
    viewer.run()


if __name__ == '__main__':
    main()