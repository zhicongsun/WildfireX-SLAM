import sys
sys.path.append('external-libraries')
import airsim

import os
import pprint

import numpy as np
import time
import math
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import cv2

from datetime import datetime

import keyboard

class FireScanner:
	def __init__(self,
		camera_name="0",               # Camera name (AirSim camera key/index), e.g. "0", "bottom_center"
		vehicle_name="",               # Vehicle/drone name in AirSim, e.g. "UAV1"
		scan_mode="",                  # Scan mode: "zigzag" / "spiral" / "random"; empty -> use default
		fps=3,                        # Capture rate (frames per second)
		record_time=0.2,               # Dwell/record duration at each waypoint (seconds)
		# zigzag-scan
		down_orientation=None,         # Camera orientation: Euler deg (pitch, roll, yaw) or airsim.Quaternionr; None -> default (look-down)
		scan_height=30,               # Scan height (meters): Z offset relative to the fire; in NED, down is positive so look-down uses negative z
		scan_width=20,                 # Scan area width (meters)
		scan_length=20,                # Scan area length (meters)
		point_spacing=5,               # Waypoint spacing (meters): smaller -> denser path / finer sampling
		# spiral-scan
		spiral_radius=30.0,              # Spiral radius (meters)
		spiral_turns=3,                  # Number of spiral turns
		spiral_points_per_turn=36,       # Samples per turn
		spiral_phi_start_deg=5.0,        # Polar angle start from the top (degrees)
		spiral_phi_end_deg=60.0,        # Polar angle end toward the bottom (degrees)
		# search-scan
		random_start_north=40.0,         # Start offset north of the fire (meters)
		random_start_east=-40.0,         # Start offset east of the fire (meters)
		random_start_height=30.0,        # Start height above the fire (meters)
		random_path_points=15,           # Number of sampled waypoints along the random path
		random_path_curvature=0.3):      # Path curvature (0-1)

		self.center = None

		self.client = airsim.VehicleClient()
		self.client.confirmConnection()

		# Camera params
		self.camera_name = camera_name
		self.vehicle_name = vehicle_name
		self.fov_deg = self.get_fov_deg()

		# Record params
		self.fps = fps
		self.frame_time = 1.0 / fps
		self.record_time = record_time
		self.responses = []

		# Z-scan params (NED; D is down)
		self.scan_height = -float(scan_height)
		self.scan_width = float(scan_width)
		self.scan_length = float(scan_length)
		self.point_spacing = float(point_spacing)
		# camera look-down orientation
		if down_orientation is None:
			self.down_orientation = airsim.to_quaternion(math.radians(-90), math.radians(0), math.radians(0))
		elif isinstance(down_orientation, airsim.Quaternionr):
			self.down_orientation = down_orientation
		else:
			pd, rd, yd = down_orientation
			self.down_orientation = airsim.to_quaternion(math.radians(pd), math.radians(rd), math.radians(yd))

		# spiral params
		self.scan_mode = scan_mode
		self.spiral_radius = float(spiral_radius)
		self.spiral_turns = int(spiral_turns)
		self.spiral_points_per_turn = int(spiral_points_per_turn)
		self.spiral_phi_start_deg = float(spiral_phi_start_deg)
		self.spiral_phi_end_deg = float(spiral_phi_end_deg)

		# search-scan params
		self.random_start_north = float(random_start_north)
		self.random_start_east = float(random_start_east)
		self.random_start_height = -float(random_start_height)
		self.random_path_points = int(random_path_points)
		self.random_path_curvature = float(random_path_curvature)




	def get_fov_deg(self):
		try:
			info = self.client.simGetCameraInfo(self.camera_name, self.vehicle_name)
			return float(info.fov)
		except Exception:
			return 90.0

	def find_fire_location(self):
		print("Searching fire source...")
		names = self.client.simListSceneObjects()
		fire_objs = [name for name in names if "niagara" in name.lower()]
		if not fire_objs:
			print("No fire object found")
			return None
		fire_name = fire_objs[0]
		print(f"Fire object: {fire_name}")
		fire_pose = self.client.simGetObjectPose(fire_name)
		p = fire_pose.position
		print(f"Fire position: X={p.x_val:.2f}, Y={p.y_val:.2f}, Z={p.z_val:.2f}")
		return p



	def generate_3d_wildfire_field(self, t=0, fire_center=(0, 0, 0), wind_direction=(1, 0, 0), 
								wind_speed=5.0, fuel_density=1.0, humidity=0.3, grid_range=100, 
								intensity_threshold=0.0001, return_filtered=True, height_scale=15.0,
								use_elliptical_grid=True):
		"""
		Improved 3D wildfire field generator with optional elliptical grid support.
		"""
		
		# Fire center
		cx, cy, cz = fire_center
		
		# Wind direction normalization
		wind_x, wind_y, wind_z = wind_direction
		wind_magnitude = np.sqrt(wind_x**2 + wind_y**2 + wind_z**2)
		
		if wind_magnitude == 0:
			wind_x, wind_y, wind_z = 0, 0, 0
			wind_magnitude = 1
		
		wind_x, wind_y, wind_z = wind_x/wind_magnitude, wind_y/wind_magnitude, wind_z/wind_magnitude
		
		if use_elliptical_grid:
			# Create an elliptical grid. The ellipse axes depend on wind direction.
			base_spread = 2.0 * t
			wind_factor = wind_speed * 0.3
			spread_forward = base_spread + wind_factor * t
			spread_lateral = base_spread * 0.8
			
			# Ellipse semi-axes (scaled)
			a = spread_forward * 2  # Major axis (downwind direction)
			b = spread_lateral * 2  # Minor axis (crosswind direction)
			
			# Start with a larger square grid, then filter to the ellipse region.
			larger_range = max(a, b) * 1.5
			x_range = np.linspace(cx - larger_range, cx + larger_range, 150)
			y_range = np.linspace(cy - larger_range, cy + larger_range, 150)
			X, Y = np.meshgrid(x_range, y_range)
			
			# Compute ellipse mask in a fire-centered coordinate frame.
			X_centered = X - cx
			Y_centered = Y - cy
			
			# Wind direction angle in the XY plane
			wind_angle = np.arctan2(wind_y, wind_x)
			
			# Rotate coordinates to align with the ellipse main axis.
			X_rotated = X_centered * np.cos(-wind_angle) - Y_centered * np.sin(-wind_angle)
			Y_rotated = X_centered * np.sin(-wind_angle) + Y_centered * np.cos(-wind_angle)
			
			# Ellipse equation: (x/a)^2 + (y/b)^2 <= 1
			ellipse_mask = (X_rotated / a)**2 + (Y_rotated / b)**2 <= 1
			
			# Keep only points inside the ellipse.
			X = X[ellipse_mask]
			Y = Y[ellipse_mask]
			
			# Recompute distance for the filtered points only.
			distance = np.sqrt((X - cx)**2 + (Y - cy)**2)
			
		else:
			# Original square grid
			x_range = np.linspace(cx - grid_range, cx + grid_range, 100)
			y_range = np.linspace(cy - grid_range, cy + grid_range, 100)
			X, Y = np.meshgrid(x_range, y_range)
			distance = np.sqrt((X - cx)**2 + (Y - cy)**2)
		
		# Wildfire spread/intensity model (rest of the computation)
		base_spread = 2.0 * t
		wind_factor = wind_speed * 0.3
		fuel_factor = fuel_density * 1.2
		humidity_factor = 1.0 - humidity * 0.5
		time_decay = np.exp(-t / 300.0)
		
		spread_forward = base_spread + wind_factor * t
		spread_lateral = base_spread * 0.8
		
		# Compute intensity field
		wind_projection = (X - cx) * wind_x + (Y - cy) * wind_y
		forward_intensity = np.exp(-(wind_projection - spread_forward)**2 / (2 * (spread_forward * 0.3)**2))
		lateral_intensity = np.exp(-distance**2 / (2 * (spread_lateral * 0.5)**2))
		
		intensity = (forward_intensity * lateral_intensity * 
					fuel_factor * humidity_factor * time_decay)
		
		distance_decay = np.exp(-distance / 15.0)
		intensity *= distance_decay
		
		noise = np.random.normal(0, 0.05, X.shape)
		intensity += noise * intensity
		intensity = np.maximum(intensity, 0)
		
		Z = cz + intensity * height_scale
		
		if return_filtered:
			intensity_mask = intensity > intensity_threshold
			X_filtered = X[intensity_mask]
			Y_filtered = Y[intensity_mask]
			Z_filtered = Z[intensity_mask]
			intensity_filtered = intensity[intensity_mask]
			return X_filtered, Y_filtered, Z_filtered, intensity_filtered
		else:
			return X, Y, Z, intensity



	def reconstruct_grid_from_filtered(self,X_filtered, Y_filtered, Z_filtered, intensity_filtered, 
									fire_center, grid_range=100, grid_size=100):
		"""
		Reconstruct a 2D grid from filtered (scattered) samples for surface plotting.

		Args:
			X_filtered, Y_filtered, Z_filtered, intensity_filtered: Filtered scattered samples.
			fire_center: Fire center (cx, cy, cz).
			grid_range: Grid range around the center.
			grid_size: Grid resolution.

		Returns:
			X_grid, Y_grid, Z_grid, intensity_grid: Reconstructed 2D grid arrays.
		"""
		
		cx, cy, cz = fire_center
		
		# Create a new regular grid
		x_range = np.linspace(cx - grid_range, cx + grid_range, grid_size)
		y_range = np.linspace(cy - grid_range, cy + grid_range, grid_size)
		X_grid, Y_grid = np.meshgrid(x_range, y_range)
		
		# Initialize grids
		Z_grid = np.full_like(X_grid, cz)  # Initialize to fire center height
		intensity_grid = np.zeros_like(X_grid)
		
		# Reconstruct grids via interpolation
		from scipy.interpolate import griddata
		
		if len(X_filtered) > 0:
			# Prepare interpolation points
			points = np.column_stack((X_filtered, Y_filtered))
			
			# Interpolate Z
			Z_grid = griddata(points, Z_filtered, (X_grid, Y_grid), method='linear', fill_value=cz)
			
			# Interpolate intensity
			intensity_grid = griddata(points, intensity_filtered, (X_grid, Y_grid), method='linear', fill_value=0)
			
			# Replace NaNs from interpolation
			Z_grid = np.nan_to_num(Z_grid, nan=cz)
			intensity_grid = np.nan_to_num(intensity_grid, nan=0)
		
		return X_grid, Y_grid, Z_grid, intensity_grid

	def plot_3d_surface_only_simple_reconstruction(self,ax,t=60, fire_center=(0, 0, 0), wind_direction=(0, 0, 0), 
												wind_speed=5.0, fuel_density=1.0, humidity=0.3,height_scale=30.0):
		"""3D surface plot using a lightweight grid reconstruction approach."""
		
		# Get filtered data
		X_filtered, Y_filtered, Z_filtered, intensity_filtered = self.generate_3d_wildfire_field(
			t, fire_center, wind_direction, wind_speed, fuel_density, humidity, 
			return_filtered=True, height_scale=height_scale
		)
		
		# Simple reconstruction: infer bounds from samples and build a smaller grid
		if len(X_filtered) > 0:
			x_min, x_max = X_filtered.min(), X_filtered.max()
			y_min, y_max = Y_filtered.min(), Y_filtered.max()

			# Create a new grid
			grid_size = 50  # Smaller grid for performance
			x_range = np.linspace(x_min, x_max, grid_size)
			y_range = np.linspace(y_min, y_max, grid_size)
			X_grid, Y_grid = np.meshgrid(x_range, y_range)
			
			# Interpolation-based reconstruction
			from scipy.interpolate import griddata
			
			points = np.column_stack((X_filtered, Y_filtered))
			Z_grid = griddata(points, Z_filtered, (X_grid, Y_grid), method='linear', fill_value=fire_center[2])
			intensity_grid = griddata(points, intensity_filtered, (X_grid, Y_grid), method='linear', fill_value=0)
			
			# Handle NaNs
			Z_grid = np.nan_to_num(Z_grid, nan=fire_center[2])
			intensity_grid = np.nan_to_num(intensity_grid, nan=0)
			

			
			surf = ax.plot_surface(X_grid, Y_grid, Z_grid, facecolors=cm.hot(intensity_grid/intensity_grid.max()), 
								alpha=0.9, linewidth=0, antialiased=True)
			
			# Mark the fire center
			ax.scatter(fire_center[0], fire_center[1], fire_center[2], color='red', s=100, marker='*', label='Fire Center')

			
			return X_grid, Y_grid, Z_grid, intensity_grid
		else:
			print("No samples found above the intensity threshold.")
			return None, None, None, None

	def plot_3d_surface_only(self,ax,t=60, fire_center=(0, 0, 0), wind_direction=(1, 0, 0), 
							wind_speed=5.0, fuel_density=1.0, humidity=0.3, use_filtered=True, height_scale=30.0):
		"""Plot only the 3D surface."""
		if use_filtered:
			# Reconstruct grid from filtered samples
			X_filtered, Y_filtered, Z_filtered, intensity_filtered = self.generate_3d_wildfire_field(
				t, fire_center, wind_direction, wind_speed, fuel_density, humidity, 
				return_filtered=True, height_scale=height_scale
			)
			
			# Reconstruct 2D grids
			X_grid, Y_grid, Z_grid, intensity_grid = self.reconstruct_grid_from_filtered(
				X_filtered, Y_filtered, Z_filtered, intensity_filtered, fire_center
			)
			
			# Surface plot using reconstructed grid data
			surf = ax.plot_surface(X_grid, Y_grid, Z_grid, facecolors=cm.hot(intensity_grid/intensity_grid.max()), 
								alpha=0.9, linewidth=0, antialiased=True)
			
			# Filled intensity contours on the XY plane
			contourf = ax.contourf(X_grid, Y_grid, intensity_grid, levels=15, cmap='hot', alpha=0.2, zdir='z', zs=fire_center[2])
			
			# Mark the fire center
			ax.scatter(fire_center[0], fire_center[1], fire_center[2], color='red', s=100, marker='*', label='Fire Center')


		else:
			# Use full grid data (surface plot)
			X, Y, Z, intensity = self.generate_3d_wildfire_field(
				t, fire_center, wind_direction, wind_speed, fuel_density, humidity, 
				return_filtered=False, height_scale=height_scale
			)


			# Surface plot using full grid data
			surf = ax.plot_surface(X, Y, Z, facecolors=cm.hot(intensity/intensity.max()), 
								alpha=0.9, linewidth=0, antialiased=True)
			
			# Filled intensity contours on the XY plane
			contourf = ax.contourf(X, Y, intensity, levels=15, cmap='hot', alpha=0.2, zdir='z', zs=0)

		
		if use_filtered:
			return X_grid, Y_grid, Z_grid, intensity_grid
		else:
			return X, Y, Z, intensity

	def generate_z_pattern_points(self, fire_position):
		print("\nGenerating Z-scan waypoints...")
		start_x = fire_position.x_val - self.scan_width / 2
		end_x   = fire_position.x_val + self.scan_width / 2
		start_y = fire_position.y_val - self.scan_length / 2
		end_y   = fire_position.y_val + self.scan_length / 2

		num_x = int(self.scan_width / self.point_spacing) + 1
		num_y = int(self.scan_length / self.point_spacing) + 1

		points = []
		for i in range(num_y):
			y = start_y + i * self.point_spacing
			if i % 2 == 0:
				x_range = np.linspace(start_x, end_x, num_x)
			else:
				x_range = np.linspace(end_x, start_x, num_x)
			for x in x_range:
				points.append(airsim.Vector3r(x, y, fire_position.z_val + self.scan_height))
		print(f"Waypoints: {len(points)}")
		return points

	def generate_spiral_waypoints_and_orientations(self, fire_position):
		# Parameters
		R = self.spiral_radius
		turns = self.spiral_turns
		pts_per_turn = self.spiral_points_per_turn
		phi0 = math.radians(self.spiral_phi_start_deg)
		phi1 = math.radians(self.spiral_phi_end_deg)

		n_total = max(2, turns * pts_per_turn)
		waypoints = []
		orientations = []

		# Build spherical points in ENU, then convert to NED (Z_down = -Z_up)
		cx, cy, cz = fire_position.x_val, fire_position.y_val, fire_position.z_val

		for i in range(n_total):
			t = i / (n_total - 1) if n_total > 1 else 0.0
			theta = 2.0 * math.pi * turns * t
			phi = phi0 + (phi1 - phi0) * t

			# ENU: x=N, y=E, z_up
			x = R * math.sin(phi) * math.cos(theta)
			y = R * math.sin(phi) * math.sin(theta)
			z_up = R * math.cos(phi)

			# Convert to NED: Z_down = -Z_up
			wx = cx + x
			wy = cy + y
			wz = cz - z_up
			waypoints.append(airsim.Vector3r(wx, wy, wz))

			# Orientation: point toward the fire center
			dn = cx - wx
			de = cy - wy
			dd = cz - wz  # Note: in NED, down is positive; if the target is below, dd > 0

			yaw = math.atan2(de, dn)
			# Empirical: pitch toward target; in NED this typically uses a minus sign
			hyp = math.hypot(dn, de)
			pitch = -math.atan2(dd, hyp)
			roll = 0.0

			q = airsim.to_quaternion(pitch, roll, yaw)
			orientations.append(q)

		return waypoints, orientations


	def generate_random_path_to_fire(self, fire_position):
		"""
		Generate a smooth random path from a start offset to the fire location.
		"""
		print("\nGenerating smooth random path to fire...")
		
		# Compute start point (relative to fire)
		start_x = fire_position.x_val + self.random_start_north  # North
		start_y = fire_position.y_val + self.random_start_east   # East
		start_z = fire_position.z_val + self.random_start_height # Height
		
		start_point = airsim.Vector3r(start_x, start_y, start_z)
		end_point = airsim.Vector3r(fire_position.x_val, fire_position.y_val, fire_position.z_val + self.scan_height)
		
		# Generate Bezier control points with additional randomness
		# Start and end points
		p0 = np.array([start_point.x_val, start_point.y_val, start_point.z_val])
		p3 = np.array([end_point.x_val, end_point.y_val, end_point.z_val])
		
		# Path length
		path_length = np.linalg.norm(p3 - p0)
		
		# Add randomness: choose multiple intermediate control points
		num_control_points = np.random.randint(3, 6)  # Randomly 3-5 control points
		control_points = [p0]
		
		# Generate random control points between start and end
		for i in range(1, num_control_points - 1):
			t = i / (num_control_points - 1)
			
			# Base linear interpolation
			base_point = p0 + t * (p3 - p0)
			
			# Add random offset
			# Horizontal random offset
			horizontal_offset = np.random.uniform(-1, 1, 2) * path_length * 0.4 * self.random_path_curvature
			
			# Vertical random offset (larger variation)
			height_offset = np.random.uniform(-1, 1) * path_length * 0.3 * self.random_path_curvature
			
			# Nonlinear shaping (max offset near the middle)
			nonlinear_factor = np.sin(t * np.pi) * 0.5
			
			# Combine offsets
			random_offset = np.array([
				horizontal_offset[0] * nonlinear_factor,
				horizontal_offset[1] * nonlinear_factor,
				height_offset * nonlinear_factor
			])
			
			control_point = base_point + random_offset
			control_points.append(control_point)
		
		control_points.append(p3)
		
		# Add additional perturbations
		for i in range(1, len(control_points) - 1):
			# Small random perturbation per intermediate control point
			perturbation = np.random.uniform(-1, 1, 3) * path_length * 0.1 * self.random_path_curvature
			control_points[i] += perturbation
		
		# Generate path points using the Bezier curve
		waypoints = []
		orientations = []
		
		# Use more points for smooth interpolation, then downsample
		num_interpolation_points = self.random_path_points * 3
		
		for i in range(num_interpolation_points):
			t = i / (num_interpolation_points - 1)
			
			# De Casteljau algorithm for an Nth-order Bezier curve
			n = len(control_points) - 1
			points = np.array(control_points, dtype=float)
			
			for r in range(1, n + 1):
				for j in range(n - r + 1):
					points[j] = (1 - t) * points[j] + t * points[j + 1]
			
			waypoint = airsim.Vector3r(points[0][0], points[0][1], points[0][2])
			waypoints.append(waypoint)
			
			# Orientation: point toward the fire location
			dx = fire_position.x_val - waypoint.x_val
			dy = fire_position.y_val - waypoint.y_val
			# dz = (fire_position.z_val + self.scan_height) - waypoint.z_val
			dz = fire_position.z_val - waypoint.z_val
			
			# Yaw (horizontal)
			yaw = math.atan2(dy, dx)
			
			# Pitch (vertical)
			horizontal_dist = math.sqrt(dx**2 + dy**2)
			pitch = -math.atan2(dz, horizontal_dist)  # Minus sign for NED convention
			
			# Keep roll at 0
			roll = 0.0
			
			# Convert to quaternion
			orientation = airsim.to_quaternion(pitch, roll, yaw)
			orientations.append(orientation)
		
		# Downsample to key waypoints
		selected_indices = np.linspace(0, len(waypoints)-1, self.random_path_points, dtype=int)
		selected_waypoints = [waypoints[i] for i in selected_indices]
		selected_orientations = [orientations[i] for i in selected_indices]
		
		print(f"Smooth random path generated with {len(selected_waypoints)} waypoints")
		return selected_waypoints, selected_orientations


	def manual_collection_mode(self):
		"""
		Manual collection mode:
		- Press 'Z' to start/stop a collection segment
		- Press 'C' to exit manual mode
		"""
		print("\n=== Manual Collection Mode ===")
		print("Press 'Z' to start/stop collection")
		print("Press 'C' to exit manual mode")
		print("Press 'X' to quit the entire program")
		print("Press 'V' to replay last manual collection")
		print("Press 'M' to start center-scan")
		print("Press 'I' to set scan_mode -> z-scan")
		print("Press 'O' to set scan_mode -> spiral-scan")
		print("Press 'P' to set scan_mode -> search-scan")
		print("You can now control the drone with keyboard in AirSim")
		
		collection_active = False
		collection_count = 0
		all_waypoints = []
		all_orientations = []
		
		# Track key state to avoid repeated triggers
		z_pressed = False
		c_pressed = False
		x_pressed = False
		v_pressed = False
		m_pressed = False
		try:
			while True:
				# Non-blocking key polling
				# Scan mode hotkeys: u -> zigzag, o -> spiral, p -> random
				if keyboard.is_pressed('u'):
					self.scan_mode = "zigzag"; print("scan_mode -> z-scan")
				if keyboard.is_pressed('o'):
					self.scan_mode = "spiral"; print("scan_mode -> spiral-scan")
				if keyboard.is_pressed('p'):
					self.scan_mode = "random"; print("scan_mode -> search-scan")
				if keyboard.is_pressed('z') and not z_pressed:
					z_pressed = True
					if not collection_active:
						# Start a new collection segment
						collection_active = True
						collection_count += 1
						print(f"\n--- Starting Collection #{collection_count} ---")
						# Read current pose
						current_pose = self.client.simGetVehiclePose(self.vehicle_name)
						print(f"Current position: ({current_pose.position.x_val:.2f}, {current_pose.position.y_val:.2f}, {current_pose.position.z_val:.2f})")
						self.responses = []
						# Run the collection loop
						waypoints, orientations = self._manual_collection_loop(collection_count)
						collection_active = False
						# Accumulate waypoints/orientations for plotting
						all_waypoints.extend(waypoints)
						all_orientations.extend(orientations)
						# Save last segment for replay ('V')
						self.waypoints = waypoints
						self.orientations = orientations
						print(f"\n--- Ending Collection #{collection_count} ---")
						if all_waypoints:
							print(f"\nDisplaying collection path with {len(all_waypoints)} waypoints...")
							self._plot_manual_collection_path(all_waypoints, all_orientations)
							all_waypoints = []
							all_orientations = []
						# Persist collected data
						if self.responses:
							print(f"Saving {len(self.responses)} collected samples...")
							self.save_data()
							print("Data saved successfully!")
							# Clear buffer
							self.responses = []
						else:
							print("No data collected in this session.")
				elif not keyboard.is_pressed('z'):
					z_pressed = False

				if keyboard.is_pressed('v') and not v_pressed:
					v_pressed = True
					if not collection_active and getattr(self, 'waypoints', None):
						print("\n--- Replaying last manual collection ---")
						self.replay_last_manual_scan()
					else:
						print("Cannot replay: either collecting now or no previous manual data.")
				elif not keyboard.is_pressed('v'):
					v_pressed = False

				if keyboard.is_pressed('m') and not m_pressed:
					m_pressed = True
					if not collection_active:
						try:
							current_pose = self.client.simGetVehiclePose(self.vehicle_name)
							center = current_pose.position
							self.center = center
							if not self.scan_mode or self.scan_mode not in ["zigzag", "spiral", "random"]:
								print("scan_mode not set or invalid, default to 'zigzag'")
								self.scan_mode = "zigzag"
							print(f"\n--- Center-scan ({self.scan_mode}) from current position ---")
							self.execute_scan_from_center(center_position=center, scan_mode=self.scan_mode)
							# Auto-save after scan
							if self.responses:
								self.save_data()
								self.responses = []
						except Exception as e:
							print(f"Center-scan error: {e}")
					else:
						print("Cannot start center-scan while collecting. Press Z to end collection first.")
				elif not keyboard.is_pressed('m'):
					m_pressed = False


				if keyboard.is_pressed('c') and not c_pressed:
					c_pressed = True
					# Exit manual collection mode
					print("\n=== Exiting Manual Collection Mode ===")
					if all_waypoints:
						print(f"\nDisplaying collection path with {len(all_waypoints)} waypoints...")
						self._plot_manual_collection_path(all_waypoints, all_orientations)
						all_waypoints = []
						all_orientations = []
					if collection_active and self.responses:
						print("Saving final collection...")
						self.save_data()
						print("Final data saved successfully!")
					break
				elif not keyboard.is_pressed('c'):
					c_pressed = False
				
				if keyboard.is_pressed('x') and not x_pressed:
					x_pressed = True
					# Exit the entire program
					print("\n=== Exiting Program ===")
					# Display collection path
					if all_waypoints:
						print(f"\nDisplaying collection path with {len(all_waypoints)} waypoints...")
						self._plot_manual_collection_path(all_waypoints, all_orientations)
						all_waypoints = []
						all_orientations = []
					if collection_active and self.responses:
						print("Saving final collection...")
						self.save_data()
						print("Final data saved successfully!")
					return False
				elif not keyboard.is_pressed('x'):
					x_pressed = False
				# Short sleep to avoid high CPU usage
				time.sleep(0.1)
						
		except KeyboardInterrupt:
			print("\n\nManual collection interrupted by user")
			# Display collection path
			if all_waypoints:
				print(f"\nDisplaying collection path with {len(all_waypoints)} waypoints...")
				self._plot_manual_collection_path(all_waypoints, all_orientations)
				all_waypoints = []
				all_orientations = []
				
			if collection_active and self.responses:
				print("Saving final collection...")
				self.save_data()
		return True

	def replay_last_manual_scan(self):
		"""Replay the last manual segment using its saved waypoints/orientations."""
		if not getattr(self, 'waypoints', None) or len(self.waypoints) == 0:
			print("No previous manual waypoints to replay.")
			return
		print(f"Replaying manual scan with {len(self.waypoints)} waypoints...")
		self.responses = []
		for i, pt in enumerate(self.waypoints):
			try:
				self.hover_and_scan_fps(pt, i, len(self.waypoints),
					orientation=(self.orientations[i] if getattr(self, 'orientations', None) else None))
			except Exception as e:
				print(f"Replay WP {i+1} error: {e}")
				continue
		print("Replay complete. Saving...")
		self.save_data()

	def _manual_collection_loop(self, collection_number):
		"""
		Manual collection loop that samples data at the configured FPS.

		Returns:
			(waypoints, orientations) recorded during this collection segment.
		"""
		print(f"Collection #{collection_number} started. Press 'Z' again to stop.")
		
		# Frame duration
		frame_time = 1.0 / self.fps
		
		waypoints = []
		orientations = []
		
		# Wait for 'Z' release to avoid immediately stopping
		print("Waiting for Z key release...")
		while keyboard.is_pressed('z'):
			time.sleep(0.1)
		
		try:
			while True:
				# Stop when 'Z' is pressed again
				if keyboard.is_pressed('z'):
					# Wait for key release
					while keyboard.is_pressed('z'):
						time.sleep(0.1)
					break
				
				# Start sampling
				start_time = time.time()
				
				# Resume simulator
				self.client.simPause(False)
				
				# Wait for one frame duration
				time.sleep(frame_time)
				
				# Pause simulator
				self.client.simPause(True)
				
				# Read current pose
				current_pose = self.client.simGetVehiclePose(self.vehicle_name)
				
				# Record waypoint and orientation
				waypoints.append(current_pose.position)
				orientations.append(current_pose.orientation)
				
				# Capture images and pose fields
				response = []
				response1 = self.client.simGetImage(self.camera_name, airsim.ImageType.Scene,vehicle_name=self.vehicle_name)
				response2 = self.client.simGetImage(self.camera_name, airsim.ImageType.DepthPerspective,vehicle_name=self.vehicle_name)
				pose_vehicle = self.client.simGetVehiclePose(vehicle_name=self.vehicle_name)
				response3 = pose_vehicle.position.x_val
				response4 = pose_vehicle.position.y_val
				response5 = pose_vehicle.position.z_val
				response6 = pose_vehicle.orientation.w_val
				response7 = pose_vehicle.orientation.x_val
				response8 = pose_vehicle.orientation.y_val
				response9 = pose_vehicle.orientation.z_val
				response.append(response1)
				response.append(response2)
				response.append(response3)
				response.append(response4)
				response.append(response5)
				response.append(response6)
				response.append(response7)
				response.append(response8)
				response.append(response9)
				self.responses.append(response)

				
				# Actual sampling time
				actual_time = time.time() - start_time
				print(f"Collected frame {len(waypoints)} at {actual_time:.3f}s")
				
				# If record_time is set, wait before the next sample
				if self.record_time > 0:
					time.sleep(self.record_time)
				
		except Exception as e:
			print(f"Error during manual collection: {e}")
			print("Collection stopped due to error")
		
		print(f"Collection #{collection_number} completed with {len(waypoints)} waypoints")
		self.client.simPause(False)
		return waypoints, orientations


	def _plot_manual_collection_path(self, waypoints, orientations):
		"""
		Display the path for a manual collection segment.
		"""
		if not waypoints:
			print("No waypoints to plot")
			return
		
		# Find fire position for plotting
		fire_position = self.find_fire_location()
		if not fire_position:
			print("Warning: No fire found, using origin as fire position")
			fire_position = airsim.Vector3r(0, 0, 0)
		self.waypoints = waypoints
		self.orientations = orientations
		# Use the existing plotting helper
		self.plot_planned_path_with_frustums(
			waypoints,
			fire_position,
			save_path="planned_z_scan_path.png",
			title="Path for Data Collection",
			orientations = orientations,
			save_flag=False
		)
		
		print("Collection path plot displayed")

	# Entry point for manual mode
	def start_manual_mode(self):
		"""
		Start manual collection mode.
		"""
		print("Starting manual collection mode...")
		print("Make sure the drone is positioned where you want to collect data.")
		print("You can control the drone manually in AirSim while collecting.")
		
		# Start manual mode loop
		should_continue = self.manual_collection_mode()
		
		if should_continue:
			print("Manual mode ended. You can start other scanning modes.")
		else:
			print("Program will exit.")
			return False
		
		return True

	def move_to_position_paused(self, target_position, orientation=None):
		# Pause simulator and set pose (does not consume sim-time for motion)
		# print(f"Moving to {target_position} ... (sim paused)")
		self.client.simPause(True)
		q = orientation if orientation is not None else self.down_orientation
		target_pose = airsim.Pose(target_position, q)
		self.client.simSetVehiclePose(target_pose, True)
		time.sleep(0.5)  # Give the API time to apply (wall-clock time)
		print("Arrived.")

	def hover_and_scan_fps(self, position, point_index, total_points, orientation=None):
		# Workflow: move (paused) -> resume for 1/fps -> pause -> record data (wall-clock time)
		# print(f"\nWP {point_index + 1}/{total_points}: {position}")
		print(f"\nWP {point_index + 1}/{total_points}")

		self.move_to_position_paused(position, orientation=orientation)

		# Run the simulator for 1/fps seconds (simulate a single-frame capture)
		self.client.simPause(False)
		time.sleep(self.frame_time)

		# Pause simulator for recording (does not consume sim-time)
		self.client.simPause(True)
		response = []
		response1 = self.client.simGetImage(self.camera_name, airsim.ImageType.Scene,vehicle_name=self.vehicle_name)
		response2 = self.client.simGetImage(self.camera_name, airsim.ImageType.DepthPerspective,vehicle_name=self.vehicle_name)
		pose_vehicle = self.client.simGetVehiclePose(vehicle_name=self.vehicle_name)
		response3 = pose_vehicle.position.x_val
		response4 = pose_vehicle.position.y_val
		response5 = pose_vehicle.position.z_val
		response6 = pose_vehicle.orientation.w_val
		response7 = pose_vehicle.orientation.x_val
		response8 = pose_vehicle.orientation.y_val
		response9 = pose_vehicle.orientation.z_val
		response.append(response1)
		response.append(response2)
		response.append(response3)
		response.append(response4)
		response.append(response5)
		response.append(response6)
		response.append(response7)
		response.append(response8)
		response.append(response9)
		self.responses.append(response)
		time.sleep(self.record_time)
		print("Recorded.")
		

	def plot_planned_path_with_frustums(self, waypoints, fire_position,
										save_path="planned_z_scan_path.png",
										title="Planned Path for Data Collection",
										frustum_len=1.0, 
										frustum_color=(0.0, 0.0, 0.0, 0.0),
										line_color='r',
										line_width=0.3,
										orientations=None,
										save_flag=None):
		import numpy as np
		import matplotlib.pyplot as plt
		from mpl_toolkits.mplot3d.art3d import Poly3DCollection
		
		if not waypoints:
			print("No waypoints to plot")
			return

		# Unified coordinate transform: NED -> plotting coordinates
		z0 = fire_position.z_val
		def ned_to_plot(p):
			# Position transform: Z' = -(Z - Z_fire)
			if hasattr(p, 'x_val'):  # airsim.Vector3r-like object
				return np.array([p.x_val, -p.y_val, -(p.z_val - z0)], dtype=float)
			else:  # numpy array / sequence
				return np.array([p[0], -p[1], -(p[2] - z0)], dtype=float)

		# Path: computed in NED, then transformed
		pts = np.array([ned_to_plot(p) for p in waypoints], dtype=float)
		fire_pt = ned_to_plot(fire_position)

		# White background and transparent panes
		fig = plt.figure(figsize=(8, 6), facecolor='white')
		ax = fig.add_subplot(111, projection='3d')
		ax.set_facecolor('white')
		ax.xaxis.pane.fill = False
		ax.yaxis.pane.fill = False
		ax.zaxis.pane.fill = False

		# Path polyline
		ax.plot(pts[:, 0], pts[:, 1], pts[:, 2], '-o', ms=2, lw=1.5, label="Path")

		# Fire point (Z=0 in plotting frame)
		# ax.scatter([fire_pt[0]], [fire_pt[1]], [fire_pt[2]], c='r', s=60, label='Fire')
		fire_center=(fire_pt[0], fire_pt[1], fire_pt[2])
		print(f"fire_center: {fire_center}")

		self.plot_3d_surface_only_simple_reconstruction(ax,t=20, fire_center=fire_center, wind_direction=(0, 0, 0), 
					wind_speed=0.0, fuel_density=1.0, humidity=0.3, height_scale=1000.0)


		# Start and end markers
		start_point = pts[0]
		end_point = pts[-1]
		ax.scatter([start_point[0]], [start_point[1]], [start_point[2]], c='g', s=60, marker='^', label='Start')
		ax.scatter([end_point[0]], [end_point[1]], [end_point[2]], c='b', s=60, marker='^', label='End')
		
		# Start/end text annotation
		ax.text(start_point[0], start_point[1], start_point[2], ' START', fontsize=6, fontweight='bold', color='green')
		
		# Frustum parameters
		half_fov = math.radians(self.fov_deg) / 2.0
		half_w = frustum_len * math.tan(half_fov)

		frustum_faces = []

		# Compute rotated direction vectors from quaternion (in NED frame)
		def quat_to_rot(q):
			x, y, z, w = q.x_val, q.y_val, q.z_val, q.w_val
			xx, yy, zz = x*x, y*y, z*z
			xy, xz, yz = x*y, x*z, y*z
			wx, wy, wz = w*x, w*y, w*z
			return np.array([
				[1 - 2*(yy + zz), 2*(xy - wz),     2*(xz + wy)],
				[2*(xy + wz),     1 - 2*(xx + zz), 2*(yz - wx)],
				[2*(xz - wy),     2*(yz + wx),     1 - 2*(xx + yy)]
			], dtype=float)

		# If not provided, use look-down orientation for all points
		if orientations is None:
			orientations = [self.down_orientation] * len(waypoints)
		if len(orientations) != len(waypoints):
			print("Warning: orientations length != waypoints length, fallback to down_orientation")
			orientations = [self.down_orientation] * len(waypoints)

		# Basis vectors in NED:
		forward_ned = np.array([1.0, 0.0, 0.0], dtype=float)  # +N (nose / forward)
		right_ned = np.array([0.0, 1.0, 0.0], dtype=float)    # +E (right wing)
		down_ned = np.array([0.0, 0.0, 1.0], dtype=float)     # +D (down)


		# Normalize helper
		def normalize(v):
			norm = np.linalg.norm(v)
			return v / norm if norm > 1e-8 else v


		for idx, wp in enumerate(waypoints):
			# Use the per-waypoint quaternion
			R = quat_to_rot(orientations[idx])

			forward_rotated = normalize(R @ forward_ned)
			right_rotated   = normalize(R @ right_ned)
			down_rotated    = normalize(R @ down_ned)

			# Compute frustum in NED
			apex_ned = np.array([wp.x_val, wp.y_val, wp.z_val], dtype=float)
			
			# Base center: move along optical axis by frustum_len
			base_c_ned = apex_ned + forward_rotated * frustum_len
			
			# Base corners (square)
			c1_ned = base_c_ned + (-half_w) * right_rotated + (-half_w) * down_rotated
			c2_ned = base_c_ned + ( half_w) * right_rotated + (-half_w) * down_rotated
			c3_ned = base_c_ned + ( half_w) * right_rotated + ( half_w) * down_rotated
			c4_ned = base_c_ned + (-half_w) * right_rotated + ( half_w) * down_rotated

			# Convert to plotting coordinates
			apex = ned_to_plot(apex_ned)
			c1 = ned_to_plot(c1_ned)
			c2 = ned_to_plot(c2_ned)
			c3 = ned_to_plot(c3_ned)
			c4 = ned_to_plot(c4_ned)

			# Wireframe
			ax.plot([apex[0], c1[0]], [apex[1], c1[1]], [apex[2], c1[2]], color=line_color, lw=line_width)
			ax.plot([apex[0], c2[0]], [apex[1], c2[1]], [apex[2], c2[2]], color=line_color, lw=line_width)
			ax.plot([apex[0], c3[0]], [apex[1], c3[1]], [apex[2], c3[2]], color=line_color, lw=line_width)
			ax.plot([apex[0], c4[0]], [apex[1], c4[1]], [apex[2], c4[2]], color=line_color, lw=line_width)
			ax.plot([c1[0], c2[0]], [c1[1], c2[1]], [c1[2], c2[2]], color=line_color, lw=line_width)
			ax.plot([c2[0], c3[0]], [c2[1], c3[1]], [c2[2], c3[2]], color=line_color, lw=line_width)
			ax.plot([c3[0], c4[0]], [c3[1], c4[1]], [c3[2], c4[2]], color=line_color, lw=line_width)
			ax.plot([c4[0], c1[0]], [c4[1], c1[1]], [c4[2], c1[2]], color=line_color, lw=line_width)

			# Faces
			frustum_faces += [
				[apex, c1, c2], [apex, c2, c3], [apex, c3, c4], [apex, c4, c1],
				[c1, c2, c3, c4]
			]

		if frustum_faces:
			poly = Poly3DCollection(frustum_faces, facecolors=[frustum_color], edgecolors='none')
			ax.add_collection3d(poly)

		# Axes labels
		ax.set_xlabel("N (X)")
		ax.set_ylabel("E (Y)")
		ax.set_zlabel("Altitude (Z)")
		# Customize Y tick labels (display negatives)
		y_ticks = ax.get_yticks()
		y_tick_labels = [f'{-y:.1f}' for y in y_ticks]
		ax.set_yticklabels(y_tick_labels)

		# Manually set Y ticks to ensure enough tick marks
		y_limits = ax.get_ylim()
		y_min, y_max = y_limits[0], y_limits[1]
		# Generate more Y tick values
		y_ticks_manual = np.linspace(y_min, y_max, 8)  # 8 ticks
		ax.set_yticks(y_ticks_manual)
		# Reset Y tick labels
		y_tick_labels_manual = [f'{-y:.1f}' for y in y_ticks_manual]
		ax.set_yticklabels(y_tick_labels_manual)

		def set_equal_3d(ax):
			x_limits = ax.get_xlim3d()
			y_limits = ax.get_ylim3d()
			z_limits = ax.get_zlim3d()
			x_range = abs(x_limits[1] - x_limits[0])
			y_range = abs(y_limits[1] - y_limits[0])
			z_range = abs(z_limits[1] - z_limits[0])

			# Use a custom scaling ratio
			x_mid = np.mean(x_limits)
			y_mid = np.mean(y_limits)
			z_mid = np.mean(z_limits)

			# Expand Y relative to X for better aspect ratio
			y_expanded_range = x_range * 0.9
			max_range = max([x_range, y_expanded_range, z_range])
			ax.set_xlim3d([x_mid - max_range/2, x_mid + max_range/2])
			ax.set_ylim3d([y_mid - y_expanded_range/2, y_mid + y_expanded_range/2])
			# ax.set_zlim3d([z_mid - max_range/2, z_mid + max_range/2])
		
		# Legend styling
		ax.legend(loc='upper right', bbox_to_anchor=(1.15, 1.0), 
				fontsize=6,           # Font size
				markerscale=1,       # Marker scale
				labelspacing=0.9,      # Label spacing
				borderpad=0.5,         # Border padding
				handlelength=2.0,      # Handle length
				handleheight=1.0)      # Handle height
		if self.scan_mode == "manual":
			title = "Manual Flight"
		elif self.scan_mode == "zigzag":
			title = "Z-scan Flight"
		elif self.scan_mode == "spiral":
			title = "Spiral-scan Flight"
		elif self.scan_mode == "random":
			title = "Search-scan Flight"
			
		ax.set_title(title)
		set_equal_3d(ax)
		if save_flag:
			plt.savefig(save_path, dpi=600)
			print(f"Saved path figure: {save_path}")
			plt.close()
		else:
			plt.show()

	def execute_scan_from_center(self, center_position, orientations=None, scan_mode=None):
		print(f"Start center scan at {self.fps} fps ...")
		mode = scan_mode if scan_mode is not None else self.scan_mode

		if mode == "zigzag":
			waypoints = self.generate_z_pattern_points(center_position)
			if orientations is None:
				orientations = [self.down_orientation] * len(waypoints)
			self.waypoints = waypoints
			self.orientations = orientations
		elif mode == "spiral":
			waypoints, orientations_auto = self.generate_spiral_waypoints_and_orientations(center_position)
			orientations = orientations if orientations is not None else orientations_auto
			self.waypoints = waypoints
			self.orientations = orientations
		elif mode == "random":
			waypoints, orientations_auto = self.generate_random_path_to_fire(center_position)
			orientations = orientations if orientations is not None else orientations_auto
			self.waypoints = waypoints
			self.orientations = orientations
		else:
			print(f"Unknown scan_mode: {mode}, fallback to zigzag.")
			waypoints = self.generate_z_pattern_points(center_position)
			if orientations is None:
				orientations = [self.down_orientation] * len(waypoints)
			self.waypoints = waypoints
			self.orientations = orientations

		# Plot planned path
		self.plot_planned_path_with_frustums(
			waypoints, center_position,
			save_path="planned_center_scan_path.png",
			title="Planned Path (Center Scan)",
			orientations=orientations
		)

		print(f"Executing {mode} center-scan with {len(waypoints)} waypoints...")
		for i, pt in enumerate(waypoints):
			try:
				self.hover_and_scan_fps(pt, i, len(waypoints), orientation=orientations[i])
				progress = (i + 1) / len(waypoints) * 100
				print(f"Progress: {progress:.1f}%")
			except Exception as e:
				print(f"WP {i + 1} error: {e}")
				continue

		self.client.simPause(False)
		print("\nCenter-scan mission complete.")

	def execute_fire_scan_fps(self, orientations=None, scan_mode=None):
		print(f"Start fire scan mission at {self.fps} fps ...")
		fire_position = self.find_fire_location()
		if fire_position is None:
			print("Abort: no fire")
			return

		mode = scan_mode if scan_mode is not None else self.scan_mode

		if mode == "zigzag":
			waypoints = self.generate_z_pattern_points(fire_position)
			# If orientations are not provided, use look-down for all waypoints
			if orientations is None:
				orientations = [self.down_orientation] * len(waypoints)
			self.waypoints = waypoints
			self.orientations = orientations
		elif mode == "spiral":
			waypoints, orientations_auto = self.generate_spiral_waypoints_and_orientations(fire_position)
			# Allow external override; otherwise use auto orientations (pointing to fire center)
			orientations = orientations if orientations is not None else orientations_auto
			self.waypoints = waypoints
			self.orientations = orientations
		elif mode == "random":  # Random path mode
				waypoints, orientations_auto = self.generate_random_path_to_fire(fire_position)
				# Allow external override; otherwise use auto orientations (pointing to fire center)
				orientations = orientations if orientations is not None else orientations_auto
				self.waypoints = waypoints
				self.orientations = orientations
		else:
			print(f"Unknown scan_mode: {mode}, fallback to zigzag.")
			waypoints = self.generate_z_pattern_points(fire_position)
			if orientations is None:
				orientations = [self.down_orientation] * len(waypoints)
			self.waypoints = waypoints
			self.orientations = orientations
		# Plot planned path first (supports per-waypoint orientations)
		self.plot_planned_path_with_frustums(
			waypoints, fire_position,
			save_path="planned_z_scan_path.png",
			title="Planned Path for Data Collection",
			orientations=orientations
		)

		print(f"Executing {mode} scan with {len(waypoints)} waypoints...")
		for i, pt in enumerate(waypoints):
			try:
				self.hover_and_scan_fps(pt, i, len(waypoints), orientation=orientations[i])
				progress = (i + 1) / len(waypoints) * 100
				print(f"Progress: {progress:.1f}%")
			except Exception as e:
				print(f"WP {i + 1} error: {e}")
				continue

		self.client.simPause(False)
		print("\nMission complete.")
	
	def save_data(self):
		# Create output folder structure
		current_dir = os.getcwd()
		airsim_data_dir = os.path.join(current_dir, "airsim_data")

		# Create airsim_data folder (if missing)
		if not os.path.exists(airsim_data_dir):
			os.makedirs(airsim_data_dir)
			print(f"Create airsim_data file: {airsim_data_dir}")

		# Create a timestamped subfolder
		current_time = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
		timestamp_dir = os.path.join(airsim_data_dir, current_time)
		if not os.path.exists(timestamp_dir):
			os.makedirs(timestamp_dir)
			print(f"\nCreate file at time: {timestamp_dir}")

		# Create images subfolder
		images_dir = os.path.join(timestamp_dir, "images")
		if not os.path.exists(images_dir):
			os.makedirs(images_dir)
			print(f"Create images file: {images_dir}")

		# Create depths subfolder
		depths_dir = os.path.join(timestamp_dir, "depths")
		if not os.path.exists(depths_dir):
			os.makedirs(depths_dir)
			print(f"Create depths file: {depths_dir}")
		
		data_rows = []
		for i, response in enumerate(self.responses):
			scene_response = response[0]
			depth_perspective_response = response[1]
			pos_x = response[2]
			pos_y = response[3]
			pos_z = response[4]
			q_w = response[5]
			q_x = response[6]
			q_y = response[7]
			q_z = response[8]

			timestamp = int(datetime.utcnow().timestamp() * 1000)

			if scene_response:
				# Decode as an OpenCV image
				img_bgr = cv2.imdecode(np.array(bytearray(scene_response), dtype='uint8'), cv2.IMREAD_UNCHANGED)
    
				# Convert BGR to RGB
				img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
				
				# Generate filename (includes timestamp)
				scene_filename = f"img_{self.vehicle_name}_{self.camera_name}_{airsim.ImageType.Scene}_{timestamp}.png"
				filepath = os.path.join(images_dir, scene_filename)
				
				# Save image
				cv2.imwrite(filepath, cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR))
			else:
				print("Failed to retrieve scene image data.")

			if depth_perspective_response:
				# Decode depth image
				depth = cv2.imdecode(np.array(bytearray(depth_perspective_response), dtype='uint8'), cv2.IMREAD_UNCHANGED)
				depth_vis = cv2.cvtColor(depth, cv2.COLOR_BGR2RGB)


				depth_filename = f"img_{self.vehicle_name}_{self.camera_name}_{airsim.ImageType.DepthPerspective}_{timestamp}.png"
				depth_filepath = os.path.join(depths_dir, depth_filename)
				cv2.imwrite(depth_filepath, depth_vis)
			else:
				print("Failed to retrieve DepthPerspective response.")
			row = [
				self.vehicle_name,
				timestamp,
				pos_x,
				pos_y,
				pos_z,
				q_w,
				q_x,
				q_y,
				q_z,
				f"{scene_filename};{depth_filename}"
			]
			data_rows.append(row)
		    # Write trajectory file
		txt_filepath = os.path.join(timestamp_dir, f"airsim_rec.txt")
		
		with open(txt_filepath, 'w', encoding='utf-8') as f:
			# Header
			header = "VehicleName\tTimeStamp\tPOS_X\tPOS_Y\tPOS_Z\tQ_W\tQ_X\tQ_Y\tQ_Z\tImageFile\n"
			f.write(header)
			
			# Rows
			for row in data_rows:
				line = "\t".join([str(item) for item in row]) + "\n"
				f.write(line)
		
		print(f"Trajectory data saved to: {txt_filepath}")
		# Find reference position for plotting
		if self.center is None:
			fire_position = self.find_fire_location()
			if not fire_position:
				print("Warning: No fire found, using origin as fire position")
				fire_position = airsim.Vector3r(0, 0, 0)
			# if self.scan_mode == "manual":
			# Use the existing plotting helper
			self.plot_planned_path_with_frustums(
				self.waypoints,
				fire_position,
				save_path=timestamp_dir + "/path.png",
				title="Path for Data Collection",
				orientations = self.orientations,
				save_flag=True
			)
			self.center = None
		else:
			self.plot_planned_path_with_frustums(
				self.waypoints,
				self.center,
				save_path=timestamp_dir + "/path.png",
				title="Path for Data Collection",
				orientations = self.orientations,
				save_flag=True
			)


"""
Main Running
"""


def main():
	scan_mode = "zigzag"
	# scan_mode = "spiral"
	# scan_mode = "random"
	# scan_mode = "manual"


	if scan_mode == "zigzag":
		"""
		Z-scan
		"""
		scanner = FireScanner(
			camera_name="bottom_center",      # Camera name (AirSim camera key)
			vehicle_name="UAV1",              # Vehicle name
			scan_mode="zigzag",               # Scan mode: spiral / zigzag / random
			fps=3,                            # Capture rate (fps)
			record_time=0.2,                  # Dwell/record duration per waypoint (seconds)
			down_orientation=(-90, 0, 0),     # Camera orientation (Euler deg: pitch, roll, yaw); -90 means look-down; or pass airsim.Quaternionr
			scan_height=40,                   # Scan height (meters)
			scan_width=20,                    # Scan area width (meters)
			scan_length=20,                   # Scan area length (meters)
			point_spacing=5                   # Waypoint spacing (meters)
		)
		try:
			scanner.execute_fire_scan_fps()
			scanner.save_data()
		except KeyboardInterrupt:
			print("\nInterrupted")
			# scanner.client.simPause(False)
		finally:
			print("Done")





	if scan_mode == "spiral":
		"""
		Spirial-scan
		"""
		scanner = FireScanner(
		    camera_name="bottom_center",      # Camera name (AirSim camera key)
		    vehicle_name="UAV1",              # Vehicle name
		    scan_mode="spiral",               # Scan mode: spiral / zigzag / random
		    fps=3,                            # Capture rate (fps)
		    record_time=0.2,                  # Dwell/record duration per waypoint (seconds)
		    spiral_radius=45.0,               # Spiral radius (m)
		    spiral_turns=1,                   # Spiral turns
		    spiral_points_per_turn=20,        # Waypoints per turn (larger -> denser)
		    spiral_phi_start_deg=30.0,        # Start polar angle (deg)
		    spiral_phi_end_deg=30.0           # End polar angle (deg)
		)
		try:
			scanner.execute_fire_scan_fps()
			scanner.save_data()
		except KeyboardInterrupt:
			print("\nInterrupted")
			# scanner.client.simPause(False)
		finally:
			print("Done")




	if scan_mode == "random":		
		"""
		Search-scan
		# random_path_curvature:
		# 0.1: slight curvature, path close to straight
		# 0.3: moderate curvature, natural curved path
		# 0.5: strong curvature, more variation
		# 0.8: very strong curvature, highly irregular path
		"""
		scanner = FireScanner(
			camera_name="bottom_center",
			vehicle_name="UAV1",
			fps=3,
			record_time=0.2,
			down_orientation=(-90, 0, 0),     # Or pass airsim.Quaternionr directly
			scan_height=40,                  # Relative to fire: Z down offset
			scan_width=20,
			scan_length=20,
			point_spacing=10,
			# Random-path parameters
			scan_mode="random",               # Use random path mode
			random_start_north=-20.0,          # Start offset north of fire (m)
			random_start_east=-20.0,          # Start offset east of fire (m)
			random_start_height=25.0,         # Start height (m)
			random_path_points=15,            # Number of sampled waypoints
			random_path_curvature=0.5         # Path curvature
		)
		try:
			scanner.execute_fire_scan_fps()
			scanner.save_data()
		except KeyboardInterrupt:
			print("\nInterrupted")
			# scanner.client.simPause(False)
		finally:
			print("Done")




	if scan_mode == "manual":
		"""
		Manual-scan
		"""
		scanner = FireScanner(
			camera_name="bottom_center",
			vehicle_name="UAV1",
			scan_mode="manual",  # Manual mode
			fps=3,
			record_time=0.2
		)
		scanner.start_manual_mode()


if __name__ == "__main__":
	main()
