#!/usr/bin/env python3
"""
Interactive tool to explore SMPL mesh vertices and find mirror-symmetrical vertex IDs.
Uses pyrender for GPU-accelerated rendering and matplotlib for interactive UI.
"""

import os
import sys
import argparse
import subprocess
import re
from pathlib import Path
import numpy as np
import torch
import trimesh
import trimesh.transformations
import pyrender
import matplotlib.pyplot as plt
import json
from matplotlib.widgets import Slider, Button, TextBox, CheckButtons
from scipy.spatial import KDTree

# Add local path to marker_vids if needed
sys.path.append(os.path.join(os.getcwd(), "human_body_prior", "src"))
try:
    import marker_vids

    KNOWN_MARKERS = marker_vids.all_marker_vids.get("smpl", {})
except ImportError:
    print(
        "Warning: marker_vids.py not found in human_body_prior/src. Marker jump feature disabled."
    )
    KNOWN_MARKERS = {}

NTU_25_MARKERS = {
    1: 1807,
    2: 3511,
    3: 3069,
    4: 336,
    5: 1291,
    9: 4773,
    6: 1573,
    10: 5044,
    7: 1923,
    11: 5385,
    8: 2226,
    12: 5688,
    13: 1801,
    17: 5263,
    14: 1046,
    18: 4530,
    15: 3321,
    19: 6721,
    16: 3366,
    20: 6766,
    21: 3495,
    22: 2297,
    24: 5758,
    23: 2710,
    25: 6170,
}

# ============================================================================
# GPU DETECTION AND RENDERING BACKEND (COPIED FROM visualize_markers_mesh.py)
# ============================================================================


def detect_gpus():
    """Detect available GPUs and return their information."""
    gpus = []
    dri_path = Path("/dev/dri")
    if not dri_path.exists():
        return gpus
    cards = sorted(
        [
            c
            for c in dri_path.glob("card*")
            if c.name.startswith("card") and c.name[4:].isdigit()
        ]
    )
    for card_idx, card in enumerate(cards):
        gpu_info = {
            "id": card_idx,
            "device": str(card),
            "vendor": "Unknown",
            "model": "Unknown",
        }
        try:
            result = subprocess.run(
                ["udevadm", "info", "--query=all", f"--name={card}"],
                capture_output=True,
                text=True,
                timeout=2,
            )
            if result.returncode == 0:
                output = result.stdout
                pci_match = re.search(r"ID_PATH=pci-([\S]+)", output)
                if pci_match:
                    pci_path = pci_match.group(1)
                    lspci_result = subprocess.run(
                        ["lspci", "-v", "-s", pci_path],
                        capture_output=True,
                        text=True,
                        timeout=2,
                    )
                    if lspci_result.returncode == 0:
                        lspci_output = lspci_result.stdout
                        vga_match = re.search(
                            r"VGA compatible controller: (.+)", lspci_output
                        )
                        if vga_match:
                            full_name = vga_match.group(1).strip()
                            if "NVIDIA" in full_name:
                                gpu_info["vendor"] = "NVIDIA"
                                gpu_info["model"] = full_name.replace(
                                    "NVIDIA Corporation", ""
                                ).strip()
                            elif "AMD" in full_name or "ATI" in full_name:
                                gpu_info["vendor"] = "AMD"
                                model = re.sub(
                                    r"Advanced Micro Devices, Inc\.\s*\[AMD/ATI\]\s*",
                                    "",
                                    full_name,
                                )
                                model = re.sub(r"\s*\(prog-if.*?\).*", "", model)
                                model = re.sub(r"\s*\(rev.*?\).*", "", model)
                                gpu_info["model"] = model.strip()
                            elif "Intel" in full_name:
                                gpu_info["vendor"] = "Intel"
                                gpu_info["model"] = full_name.replace(
                                    "Intel Corporation", ""
                                ).strip()
                            else:
                                gpu_info["model"] = full_name
        except:
            pass
        gpus.append(gpu_info)
    return gpus


def select_gpu(gpus):
    """Prompt user to select a GPU."""
    print("\n" + "=" * 60)
    print("Multiple GPUs detected. Please select one:")
    print("=" * 60)
    for gpu in gpus:
        print(f"  [{gpu['id']}] {gpu['vendor']} - {gpu['model']}")
    print("=" * 60)
    while True:
        try:
            choice = input(f"Select GPU [0-{len(gpus) - 1}]: ").strip()
            gpu_id = int(choice)
            if 0 <= gpu_id < len(gpus):
                return gpu_id
        except ValueError:
            pass


def setup_rendering_backend(gpu_id=None):
    """
    Setup the EGL rendering backend with GPU selection.

    Args:
        gpu_id: GPU device ID to use (None for auto-detection)

    Returns:
        int or None: Selected GPU ID (None if using default EGL device)
    """
    # Always use EGL backend
    os.environ["PYOPENGL_PLATFORM"] = "egl"

    # Detect available GPUs
    gpus = detect_gpus()

    if len(gpus) == 0:
        print("Warning: No GPU devices detected. Trying default EGL device...")
        return None

    if len(gpus) == 1:
        # Only one GPU, use it automatically
        gpu_id = 0
        print(f"Using GPU: {gpus[0]['vendor']} - {gpus[0]['model']}")
        os.environ["EGL_DEVICE_ID"] = str(gpu_id)
        return gpu_id

    # Multiple GPUs detected
    if gpu_id is None:
        # Prompt user to select
        gpu_id = select_gpu(gpus)
    elif gpu_id >= len(gpus):
        print(f"Error: GPU ID {gpu_id} not found. Available GPUs: 0-{len(gpus) - 1}")
        sys.exit(1)
    else:
        # Use specified GPU
        print(f"Using GPU: {gpus[gpu_id]['vendor']} - {gpus[gpu_id]['model']}")

    os.environ["EGL_DEVICE_ID"] = str(gpu_id)
    return gpu_id


def setup_renderer_live(width=1024, height=1024):
    """Initialize offscreen renderer."""
    try:
        return pyrender.OffscreenRenderer(width, height)
    except Exception as e:
        print(f"Renderer initialization failed: {e}")
        sys.exit(1)


# ============================================================================
# SMPL LOADING AND SYMMETRY
# ============================================================================


def get_smpl_tpose(model_path="data/smpl/", gender="neutral"):
    """Generate T-pose vertices for SMPL model."""
    import smplx

    model = smplx.create(
        model_path, model_type="smpl", gender=gender, ext="pkl", batch_size=1
    )
    with torch.no_grad():
        output = model(return_verts=True)
    return output.vertices[0].detach().cpu().numpy(), model.faces


def precompute_mirror_map(vertices):
    """Reflect vertices and find symmetry mapping."""
    reflected_verts = vertices.copy()
    reflected_verts[:, 0] = -reflected_verts[:, 0]
    tree = KDTree(vertices)
    distances, indices = tree.query(reflected_verts, k=1)
    return indices, distances


# ============================================================================
# EXPLORER APP
# ============================================================================


class VertexExplorer:
    def __init__(self, vertices, faces, renderer):
        self.vertices = vertices
        self.faces = faces
        self.renderer = renderer
        self.num_verts = vertices.shape[0]
        self.current_id = 0
        self.session_mappings = {}
        self.show_all_markers = False
        self.show_ntu_markers = False
        self.output_file = "new_marker_vids.json"

        # Camera state
        self.camera_azimuth = 0
        self.camera_elevation = 15
        self.cam_distance = 3.0
        self.last_depth_map = None
        self.last_pose = None
        self.yfov = np.pi / 4.0
        self.width = 1024
        self.height = 1024

        # Precompute mirror mapping
        print("Precomputing mirror symmetry map...")
        self.mirror_map, self.mirror_dist = precompute_mirror_map(vertices)

        # Setup Figure
        self.fig = plt.figure(figsize=(12, 10))
        self.ax = self.fig.add_axes((0.05, 0.2, 0.9, 0.75))
        self.ax.axis("off")

        self.img_plot = self.ax.imshow(np.zeros((1024, 1024, 3)))

        # UI Widgets
        ax_slider = plt.axes((0.15, 0.1, 0.65, 0.03))
        self.slider = Slider(
            ax_slider, "Vertex ID", 0, self.num_verts - 1, valinit=0, valfmt="%d"
        )
        self.slider.on_changed(self.update_from_slider)

        ax_minus = plt.axes((0.15, 0.05, 0.05, 0.04))
        self.btn_minus = Button(ax_minus, "-1")
        self.btn_minus.on_clicked(lambda x: self.step_id(-1))

        ax_plus = plt.axes((0.21, 0.05, 0.05, 0.04))
        self.btn_plus = Button(ax_plus, "+1")
        self.btn_plus.on_clicked(lambda x: self.step_id(1))

        ax_search = plt.axes((0.4, 0.05, 0.15, 0.04))
        self.txt_search = TextBox(ax_search, "Marker: ", initial="")
        self.txt_search.on_submit(self.on_search_submit)

        ax_save = plt.axes((0.57, 0.05, 0.08, 0.04))
        self.btn_save = Button(ax_save, "Map It")
        self.btn_save.on_clicked(self.save_mapping)

        ax_export = plt.axes((0.66, 0.05, 0.08, 0.04))
        self.btn_export = Button(ax_export, "Export")
        self.btn_export.on_clicked(self.export_to_json)

        ax_check = plt.axes((0.75, 0.05, 0.15, 0.04))
        self.check_markers = CheckButtons(
            ax_check, ["Markers", "NTU 25"], [False, False]
        )
        self.check_markers.on_clicked(self.toggle_markers)

        ax_png = plt.axes((0.91, 0.05, 0.06, 0.04))
        self.btn_png = Button(ax_png, "PNG")
        self.btn_png.on_clicked(self.export_front_view_png)

        # Controls info
        self.fig.text(
            0.05,
            0.02,
            "L/R: Rotate Azimuth | U/D: Rotate Elevation | Scroll: Zoom",
            fontsize=9,
        )
        self.info_text = self.fig.text(
            0.7, 0.03, "", fontsize=10, bbox=dict(facecolor="white", alpha=0.5)
        )

        # Event connections
        self.fig.canvas.mpl_connect("key_press_event", self.on_key)
        self.fig.canvas.mpl_connect("scroll_event", self.on_scroll)
        self.fig.canvas.mpl_connect("button_press_event", self.on_click)

        self.update_render()

    def render_scene(self):
        scene = pyrender.Scene(ambient_light=[0.5, 0.5, 0.5], bg_color=[255, 255, 255])

        # Body Mesh
        body_mesh = trimesh.Trimesh(vertices=self.vertices, faces=self.faces)
        # Matte light gray color, zero metalness, maximum roughness to minimize reflection
        mesh_material = pyrender.MetallicRoughnessMaterial(
            baseColorFactor=[0.5, 0.5, 0.5, 1.0],
            metallicFactor=0.0,
            roughnessFactor=1.0,
            doubleSided=True,
        )
        scene.add(
            pyrender.Mesh.from_trimesh(body_mesh, material=mesh_material, smooth=True)
        )

        # Highlight Selected (Red)
        v = self.vertices[self.current_id]
        sphere_red = trimesh.creation.icosphere(subdivisions=2, radius=0.015)
        sphere_red.vertices += v
        mat_red = pyrender.MetallicRoughnessMaterial(
            baseColorFactor=[1.0, 0.0, 0.0, 1.0], roughnessFactor=0.5
        )
        scene.add(pyrender.Mesh.from_trimesh(sphere_red, material=mat_red))

        # Highlight Mirror (Green)
        mid = self.mirror_map[self.current_id]
        m = self.vertices[mid]
        sphere_green = trimesh.creation.icosphere(subdivisions=2, radius=0.012)
        sphere_green.vertices += m
        mat_green = pyrender.MetallicRoughnessMaterial(
            baseColorFactor=[0.0, 1.0, 0.0, 1.0], roughnessFactor=0.5
        )
        scene.add(pyrender.Mesh.from_trimesh(sphere_green, material=mat_green))

        # Camera
        mesh_center = self.vertices.mean(axis=0)
        azimuth_rad = np.radians(self.camera_azimuth)
        elevation_rad = np.radians(self.camera_elevation)

        cam_x = mesh_center[0] + self.cam_distance * np.cos(elevation_rad) * np.sin(
            azimuth_rad
        )
        cam_z = mesh_center[2] + self.cam_distance * np.cos(elevation_rad) * np.cos(
            azimuth_rad
        )
        cam_y = mesh_center[1] + self.cam_distance * np.sin(elevation_rad)

        cam_pos = np.array([cam_x, cam_y, cam_z])
        forward = mesh_center - cam_pos
        forward /= np.linalg.norm(forward)

        # Robust right/up calculation
        world_up = np.array([0, 1, 0])
        right = np.cross(forward, world_up)
        if np.linalg.norm(right) < 1e-6:
            right = np.array([1, 0, 0])
        right /= np.linalg.norm(right)
        up = np.cross(right, forward)

        cp = np.eye(4)
        cp[:3, 0], cp[:3, 1], cp[:3, 2], cp[:3, 3] = right, up, -forward, cam_pos
        self.last_pose = cp
        scene.add(pyrender.PerspectiveCamera(yfov=self.yfov), pose=cp)

        # Balanced directional lights
        light = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=2.0)
        scene.add(light, pose=cp)

        # Show all known markers if toggled
        if self.show_all_markers:
            for name, vid in KNOWN_MARKERS.items():
                if vid >= self.num_verts:
                    continue
                m_pos = self.vertices[vid]
                s = trimesh.creation.icosphere(subdivisions=1, radius=0.008)
                s.vertices += m_pos
                # Blue for existing markers
                m_mat = pyrender.MetallicRoughnessMaterial(
                    baseColorFactor=[0.2, 0.2, 1.0, 0.6], roughnessFactor=0.8
                )
                scene.add(pyrender.Mesh.from_trimesh(s, material=m_mat))

        # Show session mappings
        for name, vid in self.session_mappings.items():
            m_pos = self.vertices[vid]
            s = trimesh.creation.icosphere(subdivisions=1, radius=0.01)
            s.vertices += m_pos
            # Yellow for session markers
            m_mat = pyrender.MetallicRoughnessMaterial(
                baseColorFactor=[1.0, 1.0, 0.0, 1.0], roughnessFactor=0.5
            )
            scene.add(pyrender.Mesh.from_trimesh(s, material=m_mat))

        # Show NTU markers
        if self.show_ntu_markers:
            for label, vid in NTU_25_MARKERS.items():
                m_pos = self.vertices[vid]
                s = trimesh.creation.icosphere(subdivisions=1, radius=0.02)
                s.vertices += m_pos
                # Bright Yellow for NTU
                m_mat = pyrender.MetallicRoughnessMaterial(
                    baseColorFactor=[1.0, 0.9, 0.0, 1.0], roughnessFactor=0.3
                )
                scene.add(pyrender.Mesh.from_trimesh(s, material=m_mat))

        # Constant fill light from above
        fill_light = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=1.0)
        fill_pose = np.eye(4)
        fill_pose[:3, 3] = [0, 5, 0]
        scene.add(fill_light, pose=fill_pose)

        color, depth = self.renderer.render(scene)
        self.last_depth_map = depth
        return color

    def update_render(self):
        color = self.render_scene()
        self.img_plot.set_data(color)

        vid = self.current_id
        v = self.vertices[vid]
        mid = self.mirror_map[vid]
        info = (
            f"Selected ID: {vid}\n"
            f"Coords: ({v[0]:.3f}, {v[1]:.3f}, {v[2]:.3f})\n"
            f"Mirror ID: {mid}\n"
            f"Mirror Dist: {self.mirror_dist[vid]:.4f}"
        )
        self.info_text.set_text(info)
        self.fig.canvas.draw_idle()

    def step_id(self, step):
        self.slider.set_val(np.clip(int(self.slider.val) + step, 0, self.num_verts - 1))

    def update_from_slider(self, val):
        self.current_id = int(val)
        self.update_render()

    def on_search_submit(self, text):
        name = text.strip().upper()
        if name in KNOWN_MARKERS:
            self.slider.set_val(KNOWN_MARKERS[name])
        elif name in self.session_mappings:
            self.slider.set_val(self.session_mappings[name])

    def save_mapping(self, event):
        name = self.txt_search.text.strip().upper()
        if not name:
            print("Please enter a marker name in the search box first.")
            return
        self.session_mappings[name] = self.current_id
        print(f"Mapped {name} to vertex {self.current_id}")
        self.update_render()

    def export_to_json(self, event):
        if not self.session_mappings:
            print("No new mappings to export.")
            return

        # Merge with existing if file exists
        data = {}
        if os.path.exists(self.output_file):
            with open(self.output_file, "r") as f:
                data = json.load(f)

        data.update(self.session_mappings)
        with open(self.output_file, "w") as f:
            json.dump(data, f, indent=4)
        print(f"Exported {len(self.session_mappings)} mappings to {self.output_file}")

    def toggle_markers(self, label):
        if label == "Markers":
            self.show_all_markers = not self.show_all_markers
        elif label == "NTU 25":
            self.show_ntu_markers = not self.show_ntu_markers
        self.update_render()

    def export_front_view_png(self, event=None):
        print("Exporting NTU front-view PNG...")
        # Save current view
        old_az, old_el, old_dist = (
            self.camera_azimuth,
            self.camera_elevation,
            self.cam_distance,
        )
        old_ntu = self.show_ntu_markers

        # Force NTU on
        self.show_ntu_markers = True

        # Set to Front view
        self.camera_azimuth = 0
        self.camera_elevation = 0
        self.cam_distance = 2.5

        # Render
        color = self.render_scene()

        # Create a new figure just for the export
        fig_export = plt.figure(figsize=(10, 10))
        ax_export = fig_export.add_subplot(111)
        ax_export.imshow(color)
        ax_export.axis("off")

        # Use simple orthographic-like projection for fixed front view
        # Image is 1024x1024. Center is 512, 512.
        # SMPL Y is UP, image Y is DOWN.
        # We need to calibrate the scale.
        mesh_center = self.vertices.mean(axis=0)

        # The scale calculation should match the pyrender perspective camera logic
        # at cam_distance=2.5 and yfov=pi/4.
        scale = (1.0 / (2.5 * np.tan(self.yfov / 2.0))) * 512

        for label, vid in NTU_25_MARKERS.items():
            v = self.vertices[vid]
            # Offset relative to center
            dx = v[0] - mesh_center[0]
            dy = v[1] - mesh_center[1]

            # Screen coords (X right, Y down)
            sx = 512 + (dx * scale)
            sy = 512 - (dy * scale)

            # Use black labels for better visibility
            ax_export.text(
                sx,
                sy,
                str(label),
                color="black",
                fontweight="bold",
                fontsize=11,
                ha="center",
                va="center",
                alpha=0.9,
            )

        plt.savefig("ntu_25_front_view.png", bbox_inches="tight", pad_inches=0, dpi=100)
        plt.close(fig_export)
        print("Saved to ntu_25_front_view.png")

        # Restore view
        self.camera_azimuth, self.camera_elevation, self.cam_distance = (
            old_az,
            old_el,
            old_dist,
        )
        self.show_ntu_markers = old_ntu
        if event is not None:
            self.update_render()

    def on_click(self, event):
        if event.inaxes != self.ax:
            return

        # event.xdata/ydata are in image coordinates from imshow
        ix, iy = int(event.xdata), int(event.ydata)
        if self.last_depth_map is None or self.last_pose is None:
            return

        # Get depth at click
        depth = self.last_depth_map[iy, ix]
        if depth == 0:  # Hit background
            return

        # Convert screen to world coordinates
        # 1. Normalized Device Coordinates (NDC)
        # Flip Y because image space is top-down
        nx = (2.0 * ix / self.width) - 1.0
        ny = 1.0 - (2.0 * iy / self.height)

        # 2. Camera space
        aspect = self.width / self.height
        tan_half_fov = np.tan(self.yfov / 2.0)

        # Perspective projection math
        # Z is negative in camera space (OpenGL convention)
        z_cam = -depth
        x_cam = nx * aspect * tan_half_fov * depth
        y_cam = ny * tan_half_fov * depth

        p_cam = np.array([x_cam, y_cam, z_cam, 1.0])

        # 3. World space
        p_world = self.last_pose @ p_cam
        p_world = p_world[:3]

        # Find nearest vertex to the 3D click point
        tree = KDTree(self.vertices)
        dist, idx = tree.query(p_world)
        if dist < 0.1:  # Only select if click is reasonably close to mesh
            self.slider.set_val(int(idx))

    def on_key(self, event):
        if event.key == "left":
            self.camera_azimuth -= 15
        elif event.key == "right":
            self.camera_azimuth += 15
        elif event.key == "up":
            self.camera_elevation = min(85, self.camera_elevation + 10)
        elif event.key == "down":
            self.camera_elevation = max(-85, self.camera_elevation - 10)
        self.update_render()

    def on_scroll(self, event):
        if event.button == "up":
            self.cam_distance = max(0.5, self.cam_distance - 0.2)
        elif event.button == "down":
            self.cam_distance = min(10.0, self.cam_distance + 0.2)
        self.update_render()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu-id", type=int, default=None)
    parser.add_argument(
        "--ntu-25-png",
        action="store_true",
        help="Generate NTU-25 front view PNG and exit",
    )
    args = parser.parse_args()

    print("Setting up GPU rendering...")
    setup_rendering_backend(args.gpu_id)
    renderer = setup_renderer_live()

    print("Loading SMPL neutral model...")
    verts, faces = get_smpl_tpose()

    explorer = VertexExplorer(verts, faces, renderer)

    if args.ntu_25_png:
        explorer.export_front_view_png()
        sys.exit(0)

    plt.show()
