# Analysis and Architecture Notes: Atoms (C++ / OpenGL)

This document summarizes the internal mechanisms of the reference C++ / OpenGL repository (`https://github.com/kavan010/Atoms`) by Kavan Patel, covering the 2D Bohr simulation, the 3D realtime point-cloud renderer, and the 3D GPU raytracer.

---

## 1. 2D Bohr Model Simulation (`src/atom.cpp` & `src/wave_atom_2d.cpp`)

### `atom.cpp` (Planetary Bohr Model with Photon Absorption/Emission)
* **Atomic Hierarchy & Physics**:
  * Represents hydrogen-like atoms (`Atom`) with a central positive nucleus (proton, charge $+1$) and orbiting electrons (charge $-1$).
  * Electron orbits are quantized at discrete radial distances: $r = n \cdot r_{\text{orbit}}$, where $n \in \{1, 2, 3, \dots\}$ is the principal quantum number.
  * Quantum energy levels follow the Rydberg formula:
    $$E_n = -\frac{13.6\,\text{eV}}{n^2}$$
* **Wave Packets & Transitions**:
  * Light waves are represented as propagating transverse wave packets:
    $$y_{\text{disp}} = A \sin(k \cdot |\mathbf{r}| - \omega t)$$
  * When incoming photon packets collide with an electron matching the resonance energy $\Delta E = E_{n+1} - E_n$, the electron absorbs the photon, raises its quantum level ($n \leftarrow n + 1$), and enters an excited state.
  * After an excited timer expires, the electron spontaneously decays ($n \leftarrow n - 1$) and emits a photon packet with energy $\Delta E$ in a random direction.
* **Many-body Dynamics**:
  * Atoms exert inter-atomic repulsive forces ($\mathbf{F} \propto \frac{\hat{\mathbf{r}}}{r}$) to prevent overlap, coupled with velocity damping and soft boundary collision repulsion.

### `wave_atom_2d.cpp` (De Broglie Matter Waves)
* Visualizes the wave-particle duality by modulating the circular orbit with a standing de Broglie matter wave:
  $$r(\theta) = r_{\text{base}} + A \sin(N_{\text{oscillations}} \cdot \theta)$$
  where $N_{\text{oscillations}} = -\frac{13.6}{E}$.
* Demonstrates constructive interference when the orbit circumference is an integer multiple of the de Broglie wavelength.

---

## 2. 3D Realtime Point-Cloud Renderer (`src/atom_realtime.cpp`)

### Quantum Wavefunction Sampling
* Evaluates the non-relativistic hydrogen wavefunction in spherical coordinates $(r, \theta, \phi)$:
  $$\psi_{nlm}(r, \theta, \phi) = R_{nl}(r) Y_l^m(\theta, \phi)$$
* **Radial Distribution $R_{nl}(r)$**:
  * Expressed using associated Laguerre polynomials $L_{n-l-1}^{2l+1}(\rho)$ where $\rho = \frac{2r}{n a_0}$:
    $$R_{nl}(r) = \sqrt{\left(\frac{2}{n a_0}\right)^3 \frac{(n-l-1)!}{2n (n+l)!}} e^{-\rho / 2} \rho^l L_{n-l-1}^{2l+1}(\rho)$$
  * Evaluated via recurrence relations:
    $$j L_j^\alpha(x) = (2j - 1 + \alpha - x) L_{j-1}^\alpha(x) - (j - 1 + \alpha) L_{j-2}^\alpha(x)$$
  * Samples $r$ via **Inverse Transform Sampling (CDF Inversion)** on a discrete 4096-point grid using the radial probability density $P(r) = r^2 |R_{nl}(r)|^2$.
* **Angular Distribution $Y_l^m(\theta, \phi)$**:
  * Evaluates Associated Legendre polynomials $P_l^m(\cos\theta)$ via recurrence relations.
  * Samples $\theta$ using CDF inversion of $P(\theta) = \sin\theta |P_l^m(\cos\theta)|^2$.
  * Samples $\phi$ uniformly in $[0, 2\pi)$ for complex states.
* **Probability Flow / Current**:
  * Calculates quantum probability current $\mathbf{j} = \frac{\hbar}{m_e} \operatorname{Im}(\psi^* \nabla \psi) = \frac{\hbar m}{m_e r \sin\theta} \hat{\boldsymbol{\phi}}$ to drive particle orbital velocity around the $z$-axis in realtime.
* **Color Mapping**:
  * Assigns point color based on the local probability density $|\psi(r, \theta, \phi)|^2$ using a perceptual thermal ramp (*Heat/Fire / Inferno*), highlighting high-density regions (lobes and nodes).

---

## 3. 3D Raytracer (`src/atom_raytracer.cpp`)

* **GPU Raymarching / Raytracing Pipeline**:
  * Generates $N$ sampled sphere coordinates and stores them in a Shader Storage Buffer Object (`SSBO`, `layout(std430, binding = 0)`).
  * Executes a fullscreen fragment shader that casts primary rays $\mathbf{r}(t) = \mathbf{o} + t \mathbf{d}$ from the camera through each pixel.
  * Performs analytical ray-sphere intersection tests against the particle buffer to find the nearest hit point $t_{\min}$.
  * Computes surface normals $\mathbf{n} = \frac{\mathbf{p} - \mathbf{c}}{R}$, shadow rays with early exit (`any_hit`), and Phong-like diffuse and ambient illumination from a point light source.

---

## Key Takeaways for Python Implementation

1. **Vectorization**:
   * Replace iterative C++ per-particle loops with vectorized NumPy inverse-CDF sampling and SciPy special functions (`scipy.special.genlaguerre`, `sph_harm_y` / `sph_harm`).
2. **Real & Complex Orbitals**:
   * Support both standard complex spherical harmonics and chemical real spherical harmonics ($p_x, p_y, p_z, d_{xy}, d_{x^2-y^2}, d_{z^2}$, etc.).
3. **Interactive 3D Point-Cloud Visualizer**:
   * Use PyVista with dynamic actors, point size, perceptual colormaps (`inferno`, `plasma`, `magma`, `fire`), and live slider controls for $n, l, m$, sample count $N$, and orbital phase.
4. **2D Bohr & De Broglie Visualizer**:
   * Implement Matplotlib-based interactive 2D simulations for Bohr energy levels, photon emission/absorption, and de Broglie standing wave quantization.
5. **ModernGL / Raymarcher Stretch**:
   * Implement a high-performance ModernGL realtime volume raymarcher / point visualizer for smooth GPU-accelerated visualization.
