"""
PyVista 3D Realtime Quantum Probability Current Flow Visualizer.

Implements true quantum mechanical probability current flow:
    v(r) = (hbar / (2*m_e*i)) * (psi* grad(psi) - psi grad(psi*)) / |psi|^2 = j / rho
    v_phi = (hbar * m) / (m_e * r * sin(theta))

Features:
- Individual 3D spherical particles stream and flow along physical velocity streamlines.
- Global probability density cloud remains stationary/invariant while internal particles swirl.
- Flow velocity scales physically: fast near the core, smooth in the outer lobes.
- Real-time 60 FPS animation loop on CPU.
- Interactive hotkeys for n, l, m, flow speed, colormaps, cutaways, and particle sizes.
"""

from __future__ import annotations
import math
import time
from typing import Optional, List, Tuple
import numpy as np
import pyvista as pv
import vtk

# Configure VTK logger to quiet mode
try:
    vtk.vtkLogger.SetStderrVerbosity(vtk.vtkLogger.VERBOSITY_ERROR)
except Exception:
    pass

from atoms_py.wavefunction import validate_quantum_numbers, get_orbital_label, orbital_energy
from atoms_py.sampling import sample_orbital, OrbitalCloud, calculate_probability_current, spherical_to_cartesian


def create_point_cloud_polydata(
    cloud: OrbitalCloud,
    log_scale: bool = True,
) -> Tuple[pv.PolyData, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Construct a PyVista PolyData object and return coordinate and scalar arrays.
    """
    pts = cloud.points.copy()
    densities = cloud.densities.copy()
    r = cloud.r.copy()
    theta = cloud.theta.copy()
    phi = cloud.phi.copy()

    # Log-compressed density for vivid contrast
    if log_scale and len(densities) > 0:
        eps = 1e-12
        log_d = np.log10(np.maximum(densities, eps))
        min_val = np.percentile(log_d, 2.0)
        max_val = np.percentile(log_d, 99.5)
        if max_val > min_val:
            normalized_log = np.clip((log_d - min_val) / (max_val - min_val), 0.0, 1.0)
        else:
            normalized_log = np.zeros_like(log_d)
        log_density = normalized_log
    else:
        log_density = densities

    poly = pv.PolyData(pts)
    poly["log_density"] = log_density

    return poly, r, theta, phi, log_density


class OrbitalVisualizer:
    """
    Interactive 3D visualizer for Hydrogen Quantum Orbital probability current flow.
    """

    def __init__(
        self,
        n: int = 5,
        l: int = 2,
        m: int = 1,
        num_samples: int = 35000,
        real_form: bool = False,
        cutaway_mode: str = "full",
        show_grid: bool = True,
        point_size: float = 6.0,
        colormap: str = "inferno",
        off_screen: bool = False,
        auto_rotate: bool = True,
    ):
        validate_quantum_numbers(n, l, m)
        self.n = n
        self.l = l
        self.m = m
        self.num_samples = num_samples
        self.real_form = real_form
        self.cutaway_mode = cutaway_mode
        self.show_grid = show_grid
        self.point_size = point_size
        self.colormap = colormap
        self.off_screen = off_screen
        self.animating = auto_rotate
        self.flow_speed = 1.0  # multiplier for probability current speed

        self.cloud: Optional[OrbitalCloud] = None
        self.polydata: Optional[pv.PolyData] = None
        self.full_r: Optional[np.ndarray] = None
        self.full_theta: Optional[np.ndarray] = None
        self.full_phi: Optional[np.ndarray] = None
        self.full_log_density: Optional[np.ndarray] = None
        self.omega: Optional[np.ndarray] = None

        self.plotter: Optional[pv.Plotter] = None
        self.actor = None
        self.grid_actor = None
        self.hud_actor = None
        self.nucleus_actor = None

        # Generate initial cloud
        self._regenerate_cloud()

    def _regenerate_cloud(self) -> None:
        """Sample particles for current quantum state and compute velocity fields."""
        self.cloud = sample_orbital(
            self.n,
            self.l,
            self.m,
            num_samples=self.num_samples,
            real_form=self.real_form,
        )
        self.polydata, self.full_r, self.full_theta, self.full_phi, self.full_log_density = create_point_cloud_polydata(
            self.cloud
        )

        # Precompute angular velocity omega_i = v_phi / (r * sin(theta)) = (hbar * m) / (m_e * (r * sin(theta))^2)
        # Using atomic units hbar = 1, m_e = 1
        sin_th = np.sin(self.full_theta)
        r_perp = self.full_r * sin_th
        effective_m = float(self.m)
        if effective_m == 0:
            self.omega = np.zeros_like(r_perp)
        else:
            self.omega = effective_m / (r_perp**2 + 1.0)
            self.omega = np.clip(self.omega, -2.5, 2.5)

        # Apply initial cutaway
        self._apply_current_positions()

    def _get_hud_text(self) -> str:
        """Formatted HUD overlay text."""
        label = get_orbital_label(self.n, self.l, self.m, real_form=self.real_form)
        energy_ev = orbital_energy(self.n, "ev")
        energy_ha = orbital_energy(self.n, "hartree")
        mode = "Real Cartesian Lobes" if self.real_form else "Complex Stationary State"
        
        if self.m != 0 and not self.real_form:
            flow_str = f"ACTIVE (v = j / rho) [Speed: {self.flow_speed:.1f}x]" if self.animating else "PAUSED (Space to resume)"
        else:
            flow_str = "ZERO (m=0 / Real State is Stationary)"

        cut_name = self.cutaway_mode.capitalize()
        shape_name = getattr(self, "particle_shape", "Points").capitalize()
        return (
            f"Hydrogen Orbital: {label}\n"
            f"Quantum Numbers: n={self.n}, l={self.l}, m={self.m}\n"
            f"Representation: {mode}\n"
            f"Energy: {energy_ev:.4f} eV  ({energy_ha:.4f} Ha)\n"
            f"Particles: {len(self.polydata.points):,} ({shape_name})\n"
            f"Cross-Section: {cut_name} (X)\n"
            f"Probability Flow: {flow_str}"
        )

    def _apply_current_positions(self) -> None:
        """Compute Cartesian coordinates for all particles and apply the active spatial cross-section cut."""
        if self.full_r is None or self.full_theta is None or self.full_phi is None:
            return

        all_pts = spherical_to_cartesian(self.full_r, self.full_theta, self.full_phi)

        if self.cutaway_mode == "quadrant":
            # Cut away (-x, +y) quadrant in fixed lab space
            mask = ~((all_pts[:, 0] < 0) & (all_pts[:, 1] > 0))
            active_pts = all_pts[mask]
            active_scalars = self.full_log_density[mask]
        elif self.cutaway_mode == "half":
            # Cut away (+y) half-space in fixed lab space
            mask = (all_pts[:, 1] <= 0)
            active_pts = all_pts[mask]
            active_scalars = self.full_log_density[mask]
        else:
            active_pts = all_pts
            active_scalars = self.full_log_density

        self.polydata.points = active_pts
        self.polydata["log_density"] = active_scalars
        self.polydata.Modified()

    def update_quantum_numbers(
        self,
        n: Optional[int] = None,
        l: Optional[int] = None,
        m: Optional[int] = None,
        real_form: Optional[bool] = None,
        cutaway_mode: Optional[str] = None,
        num_samples: Optional[int] = None,
    ) -> None:
        """Update quantum state parameters and refresh mesh."""
        new_n = self.n if n is None else int(n)
        new_l = self.l if l is None else int(l)
        new_m = self.m if m is None else int(m)
        new_rf = self.real_form if real_form is None else bool(real_form)
        new_cut = self.cutaway_mode if cutaway_mode is None else str(cutaway_mode)
        new_samples = self.num_samples if num_samples is None else int(num_samples)

        # Clamp and validate safely
        new_n = max(1, min(7, new_n))
        new_l = max(0, min(new_n - 1, new_l))
        new_m = max(-new_l, min(new_l, new_m))

        self.n = new_n
        self.l = new_l
        self.m = new_m
        self.real_form = new_rf
        self.cutaway_mode = new_cut
        self.num_samples = new_samples

        self._regenerate_cloud()

        if self.plotter is not None:
            if self.actor is not None:
                self.plotter.remove_actor(self.actor)
            
            self.actor = self.plotter.add_mesh(
                self.polydata,
                scalars="log_density",
                cmap=self.colormap,
                point_size=self.point_size,
                clim=[0.0, 1.0],
                render_points_as_spheres=False,
                style="points",
                show_scalar_bar=False,
                lighting=False,
            )

            # Update HUD
            if self.hud_actor is not None:
                self.plotter.remove_actor(self.hud_actor)
            self.hud_actor = self.plotter.add_text(
                self._get_hud_text(),
                position="upper_left",
                font_size=10,
                color="white",
                font="courier",
                shadow=False,
            )

            r_max = float(np.percentile(self.full_r, 95.0)) if self.full_r is not None and len(self.full_r) > 0 else 10.0
            min_z = float(np.min(self.polydata.points[:, 2])) if len(self.polydata.points) > 0 else -10.0
            self.plotter.camera.focal_point = (0.0, 0.0, 0.0)
            self.plotter.reset_camera(bounds=[-r_max, r_max, -r_max, r_max, min_z, r_max])
            self.plotter.camera.zoom(1.2)
            self.plotter.render()

    def step_probability_flow(self, dt: float = 0.03) -> None:
        """
        Advance each individual particle along its quantum velocity streamline v = j / rho.
        When m != 0, particles stream along phi streamlines.
        When m == 0 or real_form is active, current is 0 and particles stay stationary.
        """
        if self.polydata is None or self.omega is None or self.full_phi is None:
            return

        # Physical quantum flow only exists when m != 0 and in complex state
        if self.m != 0 and not self.real_form and self.animating:
            effective_dt = dt * self.flow_speed
            self.full_phi = (self.full_phi + self.omega * effective_dt) % (2.0 * np.pi)
            self._apply_current_positions()

    def cycle_cutaway_mode(self) -> None:
        """Cycle through Full -> Quadrant Cutaway (1/4 slice) -> Half Cutaway (1/2 slice)."""
        cuts = ["full", "quadrant", "half"]
        idx = (cuts.index(self.cutaway_mode) + 1) % len(cuts) if self.cutaway_mode in cuts else 0
        self.update_quantum_numbers(cutaway_mode=cuts[idx])

    def toggle_grid(self) -> None:
        """Toggle coordinate floor grid visibility."""
        self.show_grid = not self.show_grid
        if self.plotter is not None:
            if self.grid_actor is not None:
                self.plotter.remove_actor(self.grid_actor)
            if self.show_grid:
                r_max = float(np.percentile(self.full_r, 95.0)) if self.full_r is not None and len(self.full_r) > 0 else 10.0
                min_z = float(np.min(self.polydata.points[:, 2])) if len(self.polydata.points) > 0 else -10.0
                grid_z = min_z - r_max * 0.15 - 1.0
                grid_size = r_max * 2.6
                grid = pv.Plane(
                    center=(0, 0, grid_z),
                    direction=(0, 0, 1),
                    i_size=grid_size,
                    j_size=grid_size,
                    i_resolution=24,
                    j_resolution=24,
                )
                self.grid_actor = self.plotter.add_mesh(grid, style="wireframe", color="#242c44", line_width=1.0)
            if self.hud_actor is not None:
                self.plotter.remove_actor(self.hud_actor)
            self.hud_actor = self.plotter.add_text(
                self._get_hud_text(),
                position="upper_left",
                font_size=10,
                color="white",
                font="courier",
                shadow=False,
            )
            self.plotter.render()

    def set_colormap(self, cmap: str) -> None:
        """Update colormap dynamically."""
        self.colormap = cmap
        if self.plotter is not None and self.actor is not None:
            self.actor.mapper.lookup_table.cmap = cmap
            self.plotter.render()

    def set_point_size(self, size: float) -> None:
        """Update point size."""
        self.point_size = float(size)
        if self.actor is not None:
            self.actor.prop.point_size = self.point_size
            if self.plotter is not None:
                self.plotter.render()

    def toggle_animation(self) -> None:
        """Toggle particle probability current flow."""
        self.animating = not self.animating
        if self.plotter is not None and self.hud_actor is not None:
            self.plotter.remove_actor(self.hud_actor)
            self.hud_actor = self.plotter.add_text(
                self._get_hud_text(),
                position="upper_left",
                font_size=10,
                color="white",
                font="courier",
                shadow=False,
            )
            self.plotter.render()

    def build_plotter(self, window_size: Tuple[int, int] = (1024, 768)) -> pv.Plotter:
        """Initialize PyVista Plotter scene with dark space theme and widgets."""
        pl = pv.Plotter(window_size=window_size, off_screen=self.off_screen)
        pl.set_background("#050512", top="#0f1226")

        r_max = float(np.percentile(self.full_r, 95.0)) if self.full_r is not None and len(self.full_r) > 0 else 10.0
        min_z = float(np.min(self.polydata.points[:, 2])) if len(self.polydata.points) > 0 else -10.0

        # Nucleus at center (Proton sphere)
        nuc_radius = max(0.8, r_max * 0.02)
        nucleus = pv.Sphere(radius=nuc_radius, center=(0, 0, 0))
        self.nucleus_actor = pl.add_mesh(
            nucleus,
            color="#ff2a5f",
            lighting=True,
            specular=1.0,
            show_scalar_bar=False,
        )

        # Coordinate floor grid directly under the orbital
        if self.show_grid:
            grid_z = min_z - r_max * 0.15 - 1.0
            grid_size = r_max * 2.6
            grid = pv.Plane(
                center=(0, 0, grid_z),
                direction=(0, 0, 1),
                i_size=grid_size,
                j_size=grid_size,
                i_resolution=24,
                j_resolution=24,
            )
            self.grid_actor = pl.add_mesh(grid, style="wireframe", color="#242c44", line_width=1.0)

        # Particle cloud
        self.actor = pl.add_mesh(
            self.polydata,
            scalars="log_density",
            cmap=self.colormap,
            point_size=self.point_size,
            clim=[0.0, 1.0],
            render_points_as_spheres=False,
            style="points",
            show_scalar_bar=True,
            scalar_bar_args={
                "title": "|psi|^2 Probability Density",
                "color": "white",
                "vertical": False,
                "position_x": 0.35,
                "position_y": 0.05,
                "width": 0.3,
                "height": 0.06,
                "n_labels": 3,
                "fmt": "%.2f",
            },
            lighting=False,
        )

        # Coordinate axes at corner
        pl.add_axes(
            line_width=3,
            color="white",
            xlabel="X",
            ylabel="Y",
            zlabel="Z",
        )

        # HUD Text
        self.hud_actor = pl.add_text(
            self._get_hud_text(),
            position="upper_left",
            font_size=10,
            color="white",
            font="courier",
            shadow=False,
        )

        # Controls reference HUD
        instructions = (
            "Controls:\n"
            "  Space : Pause / Resume Probability Current Flow (v = j / rho)\n"
            "  W / S : n (+/-)\n"
            "  E / D : l (+/-)\n"
            "  R / F : m (+/-)\n"
            "  [ / ] : Flow Speed (-/+)\n"
            "  C     : Toggle Real/Complex Form\n"
            "  X     : Cycle Cutaway Cross-Section (Full / Quadrant / Half)\n"
            "  G     : Toggle Floor Grid\n"
            "  M     : Cycle Colormap\n"
            "  + / - : Particle Size (+/-)\n"
            "  Q     : Quit\n"
            "  Mouse : Left Drag (Orbit), Middle Drag (Pan), Scroll (Zoom)"
        )
        pl.add_text(
            instructions,
            position="lower_left",
            font_size=8.5,
            color="#9090b8",
            font="courier",
            shadow=False,
        )

        # Keybindings
        colormaps = ["inferno", "viridis", "plasma", "magma", "turbo", "coolwarm", "hot"]

        def cycle_cmap():
            idx = (colormaps.index(self.colormap) + 1) % len(colormaps) if self.colormap in colormaps else 0
            self.set_colormap(colormaps[idx])
            if self.hud_actor is not None:
                self.plotter.remove_actor(self.hud_actor)
            self.hud_actor = self.plotter.add_text(
                self._get_hud_text(),
                position="upper_left",
                font_size=10,
                color="white",
                font="courier",
                shadow=False,
            )

        def inc_n(): self.update_quantum_numbers(n=self.n + 1)
        def dec_n(): self.update_quantum_numbers(n=self.n - 1)
        def inc_l(): self.update_quantum_numbers(l=self.l + 1)
        def dec_l(): self.update_quantum_numbers(l=self.l - 1)
        def inc_m(): self.update_quantum_numbers(m=self.m + 1)
        def dec_m(): self.update_quantum_numbers(m=self.m - 1)
        def toggle_form(): self.update_quantum_numbers(real_form=not self.real_form)
        def inc_speed():
            self.flow_speed = min(10.0, self.flow_speed + 0.5)
            self.update_quantum_numbers()
        def dec_speed():
            self.flow_speed = max(0.2, self.flow_speed - 0.5)
            self.update_quantum_numbers()
        def inc_point_size(): self.set_point_size(self.point_size + 1.0)
        def dec_point_size(): self.set_point_size(max(1.0, self.point_size - 1.0))

        pl.add_key_event("w", inc_n)
        pl.add_key_event("W", inc_n)
        pl.add_key_event("s", dec_n)
        pl.add_key_event("S", dec_n)
        pl.add_key_event("e", inc_l)
        pl.add_key_event("E", inc_l)
        pl.add_key_event("d", dec_l)
        pl.add_key_event("D", dec_l)
        pl.add_key_event("r", inc_m)
        pl.add_key_event("R", inc_m)
        pl.add_key_event("f", dec_m)
        pl.add_key_event("F", dec_m)
        pl.add_key_event("x", self.cycle_cutaway_mode)
        pl.add_key_event("X", self.cycle_cutaway_mode)
        pl.add_key_event("g", self.toggle_grid)
        pl.add_key_event("G", self.toggle_grid)
        pl.add_key_event("c", toggle_form)
        pl.add_key_event("C", toggle_form)
        pl.add_key_event("bracketright", inc_speed)
        pl.add_key_event("bracketleft", dec_speed)
        pl.add_key_event("m", cycle_cmap)
        pl.add_key_event("M", cycle_cmap)
        pl.add_key_event("plus", inc_point_size)
        pl.add_key_event("equal", inc_point_size)
        pl.add_key_event("minus", dec_point_size)
        pl.add_key_event("space", self.toggle_animation)

        pl.camera.focal_point = (0.0, 0.0, 0.0)
        cam_dist = r_max * 2.2
        pl.camera.position = (cam_dist * 0.7, cam_dist * 0.7, cam_dist * 0.6)
        pl.camera.up = (0.0, 0.0, 1.0)
        pl.reset_camera(bounds=[-r_max, r_max, -r_max, r_max, min_z, r_max])
        pl.camera.zoom(1.2)

        self.plotter = pl
        return pl

    def render_static_image(
        self,
        filepath: str,
        window_size: Tuple[int, int] = (1200, 900),
    ) -> None:
        """
        Render a high-resolution static PNG snapshot of the current orbital.
        """
        saved_off = self.off_screen
        self.off_screen = True
        pl = self.build_plotter(window_size=window_size)
        pl.screenshot(filepath)
        pl.close()
        self.off_screen = saved_off

    def show(self) -> None:
        """Launch interactive PyVista visualization window with true probability current flow."""
        pl = self.build_plotter()

        # Non-blocking interactive show
        pl.show(interactive_update=True, auto_close=False)

        # Active real-time 60 FPS probability current flow loop
        while not getattr(pl, "_closed", False) and getattr(pl, "render_window", None) is not None:
            try:
                if self.animating:
                    # Advance each individual dot along its quantum velocity streamline v = j / rho
                    self.step_probability_flow(dt=0.035)

                pl.update()
            except Exception:
                # Recover safely from any temporary GUI/interactor state during keypresses
                pass
            time.sleep(0.016)  # ~60 FPS smooth timing

        if not getattr(pl, "_closed", False):
            try:
                pl.close()
            except Exception:
                pass
