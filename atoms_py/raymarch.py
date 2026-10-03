"""
ModernGL + GLSL GPU Realtime Volume Raymarcher for Quantum Orbitals.

Performs volumetric raymarching through continuous 3D hydrogen orbital density fields
with real-time camera orbiting, GPU colormapping, and interactive controls.
"""

from __future__ import annotations
import math
import sys
from typing import Tuple
import numpy as np
import moderngl
from atoms_py.wavefunction import (
    validate_quantum_numbers,
    probability_density,
    get_orbital_label,
)

# GLSL Vertex Shader for fullscreen quad
VERTEX_SHADER = """
#version 330 core
in vec2 in_vert;
out vec2 v_uv;

void main() {
    v_uv = in_vert * 0.5 + 0.5;
    gl_Position = vec4(in_vert, 0.0, 1.0);
}
"""

# GLSL Fragment Shader for 3D Volume Raymarching
FRAGMENT_SHADER = """
#version 330 core
in vec2 v_uv;
out vec4 fragColor;

uniform sampler3D u_volume;
uniform vec3 u_camera_pos;
uniform vec3 u_camera_target;
uniform vec3 u_camera_up;
uniform vec2 u_resolution;
uniform float u_box_size;
uniform float u_density_scale;
uniform float u_step_size;
uniform int u_colormap_type; // 0: inferno, 1: plasma, 2: coolwarm

// Ray-box intersection (AABB)
vec2 intersectAABB(vec3 rayOrigin, vec3 rayDir, vec3 boxMin, vec3 boxMax) {
    vec3 tMin = (boxMin - rayOrigin) / rayDir;
    vec3 tMax = (boxMax - rayOrigin) / rayDir;
    vec3 t1 = min(tMin, tMax);
    vec3 t2 = max(tMin, tMax);
    float tNear = max(max(t1.x, t1.y), t1.z);
    float tFar = min(min(t2.x, t2.y), t2.z);
    return vec2(tNear, tFar);
}

// Thermal Inferno color palette
vec3 colormapInferno(float t) {
    t = clamp(t, 0.0, 1.0);
    vec3 c0 = vec3(0.001, 0.000, 0.015);
    vec3 c1 = vec3(0.258, 0.039, 0.408);
    vec3 c2 = vec3(0.576, 0.149, 0.404);
    vec3 c3 = vec3(0.867, 0.325, 0.227);
    vec3 c4 = vec3(0.988, 0.651, 0.212);
    vec3 c5 = vec3(0.988, 0.996, 0.749);

    if (t < 0.2) return mix(c0, c1, t / 0.2);
    if (t < 0.4) return mix(c1, c2, (t - 0.2) / 0.2);
    if (t < 0.6) return mix(c2, c3, (t - 0.4) / 0.2);
    if (t < 0.8) return mix(c3, c4, (t - 0.6) / 0.2);
    return mix(c4, c5, (t - 0.8) / 0.2);
}

// Plasma color palette
vec3 colormapPlasma(float t) {
    t = clamp(t, 0.0, 1.0);
    vec3 c0 = vec3(0.051, 0.031, 0.529);
    vec3 c1 = vec3(0.416, 0.000, 0.659);
    vec3 c2 = vec3(0.694, 0.165, 0.565);
    vec3 c3 = vec3(0.882, 0.392, 0.384);
    vec3 c4 = vec3(0.988, 0.651, 0.212);
    vec3 c5 = vec3(0.941, 0.976, 0.129);

    if (t < 0.2) return mix(c0, c1, t / 0.2);
    if (t < 0.4) return mix(c1, c2, (t - 0.2) / 0.2);
    if (t < 0.6) return mix(c2, c3, (t - 0.4) / 0.2);
    if (t < 0.8) return mix(c3, c4, (t - 0.6) / 0.2);
    return mix(c4, c5, (t - 0.8) / 0.2);
}

void main() {
    vec2 ndc = (gl_FragCoord.xy / u_resolution) * 2.0 - 1.0;
    ndc.x *= u_resolution.x / u_resolution.y;

    vec3 forward = normalize(u_camera_target - u_camera_pos);
    vec3 right = normalize(cross(forward, u_camera_up));
    vec3 up = cross(right, forward);

    vec3 rayDir = normalize(ndc.x * right + ndc.y * up + 2.0 * forward);

    vec3 boxHalf = vec3(u_box_size * 0.5);
    vec2 tHit = intersectAABB(u_camera_pos, rayDir, -boxHalf, boxHalf);

    // Dark cosmic background
    vec3 bgColor = mix(vec3(0.015, 0.015, 0.04), vec3(0.04, 0.04, 0.09), v_uv.y);

    if (tHit.x > tHit.y || tHit.y < 0.0) {
        fragColor = vec4(bgColor, 1.0);
        return;
    }

    float tStart = max(tHit.x, 0.0);
    float tEnd = tHit.y;

    vec3 accumColor = vec3(0.0);
    float transmittance = 1.0;
    float stepSize = u_step_size;

    for (float t = tStart; t < tEnd; t += stepSize) {
        vec3 p = u_camera_pos + t * rayDir;
        vec3 texCoord = (p + boxHalf) / u_box_size;

        if (texCoord.x < 0.0 || texCoord.x > 1.0 ||
            texCoord.y < 0.0 || texCoord.y > 1.0 ||
            texCoord.z < 0.0 || texCoord.z > 1.0) continue;

        float sampleVal = texture(u_volume, texCoord).r;
        if (sampleVal > 0.01) {
            float density = sampleVal * u_density_scale;
            float stepAlpha = 1.0 - exp(-density * stepSize * 15.0);

            vec3 emission = (u_colormap_type == 1) ? colormapPlasma(sampleVal) : colormapInferno(sampleVal);
            accumColor += transmittance * emission * stepAlpha;
            transmittance *= (1.0 - stepAlpha);

            if (transmittance < 0.02) break;
        }
    }

    vec3 finalColor = accumColor + transmittance * bgColor;
    fragColor = vec4(finalColor, 1.0);
}
"""


def compute_3d_orbital_volume(
    n: int,
    l: int,
    m: int,
    grid_size: int = 96,
    box_size: float = 30.0,
    real_form: bool = True,
) -> np.ndarray:
    """
    Generate 3D density grid normalized to [0, 1] for GPU texture uploading.
    """
    validate_quantum_numbers(n, l, m)
    lin = np.linspace(-box_size / 2.0, box_size / 2.0, grid_size, dtype=np.float32)
    # Standard 3D Cartesian coordinates: X, Y, Z
    X, Y, Z = np.meshgrid(lin, lin, lin, indexing="ij")

    R = np.sqrt(X**2 + Y**2 + Z**2)
    R_safe = np.maximum(R, 1e-8)
    Theta = np.arccos(np.clip(Z / R_safe, -1.0, 1.0))
    Phi = (np.arctan2(Y, X) + 2.0 * np.pi) % (2.0 * np.pi)

    density = probability_density(n, l, m, R, Theta, Phi, real_form=real_form)

    # Log scale normalization
    eps = 1e-10
    log_d = np.log10(np.maximum(density, eps))
    min_val = np.percentile(log_d, 5.0)
    max_val = np.percentile(log_d, 99.8)

    if max_val > min_val:
        norm_density = np.clip((log_d - min_val) / (max_val - min_val), 0.0, 1.0)
    else:
        norm_density = np.zeros_like(log_d)

    return norm_density.astype(np.float32)


class ModernGLRaymarcher:
    """ModernGL-based GPU volume raymarcher window application."""

    def __init__(
        self,
        n: int = 2,
        l: int = 1,
        m: int = 0,
        real_form: bool = True,
        grid_size: int = 96,
        window_size: Tuple[int, int] = (1024, 768),
    ):
        self.n = n
        self.l = l
        self.m = m
        self.real_form = real_form
        self.grid_size = grid_size
        self.window_size = window_size
        self.colormap_type = 0  # 0: Inferno, 1: Plasma

        self.box_size = max(28.0, (2.8 * (n ** 2) + 12.0 * n))
        self.density_scale = 1.0
        self.step_size = self.box_size / 180.0

        # Camera
        self.azimuth = math.radians(45)
        self.elevation = math.radians(25)
        self.distance = self.box_size * 1.6
        self.target = np.array([0.0, 0.0, 0.0], dtype=np.float32)

    def render_offscreen_frame(self) -> np.ndarray:
        """Render a single frame headless via ModernGL standalone context."""
        ctx = moderngl.create_context(standalone=True)
        fbo = ctx.framebuffer(
            color_attachments=[ctx.texture(self.window_size, 4)]
        )
        fbo.use()

        # Compute volume
        vol_data = compute_3d_orbital_volume(
            self.n, self.l, self.m, grid_size=self.grid_size, box_size=self.box_size, real_form=self.real_form
        )
        texture3d = ctx.texture3d(
            (self.grid_size, self.grid_size, self.grid_size),
            1,
            vol_data.tobytes(),
            dtype="f4",
        )
        texture3d.filter = (moderngl.LINEAR, moderngl.LINEAR)
        texture3d.use(location=0)

        program = ctx.program(vertex_shader=VERTEX_SHADER, fragment_shader=FRAGMENT_SHADER)

        quad_verts = np.array([-1.0, -1.0, 1.0, -1.0, -1.0, 1.0, 1.0, 1.0], dtype=np.float32)
        vbo = ctx.buffer(quad_verts.tobytes())
        vao = ctx.vertex_array(program, [(vbo, "2f", "in_vert")])

        cam_x = self.distance * math.cos(self.elevation) * math.sin(self.azimuth)
        cam_y = self.distance * math.cos(self.elevation) * math.cos(self.azimuth)
        cam_z = self.distance * math.sin(self.elevation)

        program["u_volume"].value = 0
        program["u_camera_pos"].value = (cam_x, cam_y, cam_z)
        program["u_camera_target"].value = tuple(self.target)
        program["u_camera_up"].value = (0.0, 0.0, 1.0)
        program["u_resolution"].value = tuple(self.window_size)
        program["u_box_size"].value = float(self.box_size)
        program["u_density_scale"].value = float(self.density_scale)
        program["u_step_size"].value = float(self.step_size)
        program["u_colormap_type"].value = self.colormap_type

        vao.render(moderngl.TRIANGLE_STRIP)

        raw = fbo.read(components=4)
        img = np.frombuffer(raw, dtype=np.uint8).reshape((self.window_size[1], self.window_size[0], 4))
        img = np.flipud(img)

        fbo.release()
        texture3d.release()
        program.release()
        vao.release()
        vbo.release()
        ctx.release()

        return img

    def save_image(self, filepath: str) -> None:
        """Export image using PIL / Pillow."""
        import PIL.Image as Image
        img_arr = self.render_offscreen_frame()
        img = Image.fromarray(img_arr)
        img.save(filepath)
        print(f"ModernGL raymarched image saved to {filepath}")
