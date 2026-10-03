/**
 * Quantum Hydrogen Orbital 3D - WebGL Three.js Engine
 * Clean Scientific HUD & Physical Probability Current Flow Visualizer
 */

// =============================================================================
// 1. Colormap LUT Definitions (RGB Normalized)
// =============================================================================
const COLORMAPS = {
  inferno: [
    [0.0, 0.0, 0.04], [0.15, 0.03, 0.28], [0.35, 0.06, 0.44], 
    [0.55, 0.14, 0.48], [0.73, 0.25, 0.44], [0.89, 0.41, 0.31], 
    [0.98, 0.65, 0.15], [0.99, 0.89, 0.43], [0.99, 1.0, 0.65]
  ],
  viridis: [
    [0.27, 0.0, 0.33], [0.28, 0.16, 0.47], [0.23, 0.32, 0.55],
    [0.17, 0.47, 0.56], [0.12, 0.63, 0.53], [0.22, 0.79, 0.41],
    [0.55, 0.89, 0.24], [0.99, 0.91, 0.14]
  ],
  plasma: [
    [0.05, 0.03, 0.53], [0.29, 0.0, 0.64], [0.5, 0.01, 0.66],
    [0.69, 0.17, 0.56], [0.85, 0.35, 0.41], [0.96, 0.57, 0.24],
    [0.99, 0.81, 0.19], [0.94, 0.98, 0.13]
  ],
  magma: [
    [0.0, 0.0, 0.04], [0.11, 0.06, 0.28], [0.32, 0.09, 0.48],
    [0.57, 0.15, 0.56], [0.81, 0.27, 0.51], [0.98, 0.47, 0.45],
    [0.99, 0.73, 0.57], [0.99, 0.95, 0.82]
  ],
  cyberpunk: [
    [0.02, 0.02, 0.08], [0.08, 0.15, 0.45], [0.0, 0.65, 0.85],
    [0.0, 0.95, 0.95], [0.85, 0.1, 0.75], [1.0, 0.25, 0.85],
    [1.0, 0.85, 0.95]
  ],
  emerald: [
    [0.01, 0.05, 0.02], [0.04, 0.22, 0.12], [0.08, 0.48, 0.25],
    [0.15, 0.78, 0.42], [0.45, 0.95, 0.65], [0.85, 1.0, 0.88]
  ],
  cosmic: [
    [0.02, 0.01, 0.08], [0.12, 0.08, 0.35], [0.25, 0.25, 0.75],
    [0.2, 0.65, 0.95], [0.45, 0.9, 1.0], [0.9, 0.98, 1.0]
  ],
  solar: [
    [0.05, 0.01, 0.01], [0.35, 0.02, 0.05], [0.75, 0.12, 0.05],
    [0.95, 0.42, 0.05], [1.0, 0.75, 0.15], [1.0, 0.95, 0.65]
  ],
  turbo: [
    [0.19, 0.07, 0.23], [0.19, 0.39, 0.98], [0.1, 0.73, 0.84],
    [0.26, 0.94, 0.46], [0.65, 0.96, 0.16], [0.98, 0.77, 0.12],
    [0.96, 0.42, 0.11], [0.75, 0.15, 0.06]
  ],
  coolwarm: [
    [0.23, 0.30, 0.75], [0.44, 0.59, 0.96], [0.71, 0.82, 0.99],
    [0.87, 0.87, 0.87], [0.98, 0.68, 0.57], [0.89, 0.34, 0.28],
    [0.70, 0.01, 0.15]
  ],
  hot: [
    [0.04, 0.0, 0.0], [0.4, 0.0, 0.0], [0.8, 0.0, 0.0],
    [1.0, 0.3, 0.0], [1.0, 0.6, 0.0], [1.0, 0.9, 0.0],
    [1.0, 1.0, 0.5], [1.0, 1.0, 1.0]
  ]
};

const COLORMAP_KEYS = Object.keys(COLORMAPS);

function sampleColormap(cmapName, t) {
  const lut = COLORMAPS[cmapName] || COLORMAPS.inferno;
  const clampedT = Math.max(0.0, Math.min(1.0, t));
  const scaled = clampedT * (lut.length - 1);
  const idx = Math.floor(scaled);
  const frac = scaled - idx;

  if (idx >= lut.length - 1) {
    return lut[lut.length - 1];
  }

  const c0 = lut[idx];
  const c1 = lut[idx + 1];
  return [
    c0[0] + (c1[0] - c0[0]) * frac,
    c0[1] + (c1[1] - c0[1]) * frac,
    c0[2] + (c1[2] - c0[2]) * frac
  ];
}

// Generate circular particle texture with smooth anti-aliased edge
function createCrispPointTexture() {
  const canvas = document.createElement('canvas');
  canvas.width = 64;
  canvas.height = 64;
  const ctx = canvas.getContext('2d');

  ctx.beginPath();
  ctx.arc(32, 32, 28, 0, Math.PI * 2);
  ctx.fillStyle = '#ffffff';
  ctx.fill();

  const texture = new THREE.CanvasTexture(canvas);
  texture.needsUpdate = true;
  return texture;
}

// =============================================================================
// 2. Quantum App State & Three.js Engine
// =============================================================================
class QuantumOrbitalApp {
  constructor() {
    this.n = 5;
    this.l = 2;
    this.m = 1;
    this.numSamples = 25000;
    this.realForm = false;
    this.cutawayMode = 'full'; // 'full', 'quadrant', 'half'
    this.colormap = 'inferno';
    this.pointSize = 2.0;
    this.flowSpeed = 1.0;
    this.isAnimating = true;

    // Particle memory buffers
    this.fullR = null;
    this.fullTheta = null;
    this.fullPhi = null;
    this.fullDensity = null;
    this.fullOmega = null;
    this.rMax = 25.0;

    // Three.js scene elements
    this.scene = null;
    this.camera = null;
    this.renderer = null;
    this.controls = null;
    this.pointCloud = null;
    this.nucleusMesh = null;
    this.pointTexture = createCrispPointTexture();

    // Timing
    this.lastTime = performance.now();

    this.initScene();
    this.initUI();
    this.fetchOrbitalData();
    this.animate = this.animate.bind(this);
    requestAnimationFrame(this.animate);
  }

  initScene() {
    const canvas = document.getElementById('webgl-canvas');
    this.renderer = new THREE.WebGLRenderer({
      canvas,
      antialias: true,
      powerPreference: 'high-performance',
      preserveDrawingBuffer: true
    });
    this.renderer.setSize(window.innerWidth, window.innerHeight);
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2.0));
    this.renderer.setClearColor(0x04040d, 1.0);

    this.scene = new THREE.Scene();

    // Perspective Camera with Z-axis as UP (matching quantum mechanics coordinates)
    this.camera = new THREE.PerspectiveCamera(45, window.innerWidth / window.innerHeight, 0.1, 2000);
    this.camera.up.set(0, 0, 1);
    this.camera.position.set(45, -45, 35);

    // Orbit Controls configured for Z-Up
    this.controls = new THREE.OrbitControls(this.camera, this.renderer.domElement);
    this.controls.enableDamping = true;
    this.controls.dampingFactor = 0.08;
    this.controls.rotateSpeed = 0.9;
    this.controls.zoomSpeed = 1.2;
    this.controls.panSpeed = 0.9;
    this.controls.maxDistance = 800;
    this.controls.minDistance = 1;

    // Lighting for Nucleus
    const ambientLight = new THREE.AmbientLight(0x334466, 1.5);
    this.scene.add(ambientLight);

    const dirLight = new THREE.DirectionalLight(0xffffff, 1.2);
    dirLight.position.set(20, -30, 40);
    this.scene.add(dirLight);

    // Nucleus Sphere (Red proton at 0,0,0)
    const nucGeo = new THREE.SphereGeometry(0.8, 32, 32);
    const nucMat = new THREE.MeshStandardMaterial({
      color: 0xff2a5f,
      emissive: 0x440011,
      roughness: 0.3,
      metalness: 0.6
    });
    this.nucleusMesh = new THREE.Mesh(nucGeo, nucMat);
    this.scene.add(this.nucleusMesh);

    // Window resize
    window.addEventListener('resize', () => {
      this.camera.aspect = window.innerWidth / window.innerHeight;
      this.camera.updateProjectionMatrix();
      this.renderer.setSize(window.innerWidth, window.innerHeight);
    });
  }

  async fetchOrbitalData() {
    this.setLoading(true);
    try {
      const url = `/api/sample?n=${this.n}&l=${this.l}&m=${this.m}&samples=${this.numSamples}&real_form=${this.realForm}`;
      const response = await fetch(url);
      if (!response.ok) {
        throw new Error(`Server returned ${response.status}`);
      }
      const data = await response.json();

      this.fullR = new Float32Array(data.r);
      this.fullTheta = new Float32Array(data.theta);
      this.fullPhi = new Float32Array(data.phi);
      this.fullDensity = new Float32Array(data.log_density);
      this.fullOmega = new Float32Array(data.omega);
      this.rMax = data.r_max || 25.0;

      this.rebuildPointCloud();
      this.updateHUD(data);
      this.alignCamera('iso');
    } catch (err) {
      console.error('Failed to fetch orbital cloud:', err);
    } finally {
      this.setLoading(false);
    }
  }

  rebuildPointCloud() {
    if (this.pointCloud) {
      this.scene.remove(this.pointCloud);
      this.pointCloud.geometry.dispose();
      this.pointCloud.material.dispose();
      this.pointCloud = null;
    }

    if (!this.fullR) return;

    const count = this.fullR.length;
    const geometry = new THREE.BufferGeometry();
    const positions = new Float32Array(count * 3);
    const colors = new Float32Array(count * 3);

    // Calculate initial 3D positions and crisp colors
    for (let i = 0; i < count; i++) {
      const r = this.fullR[i];
      const th = this.fullTheta[i];
      const ph = this.fullPhi[i];
      const sinTh = Math.sin(th);

      positions[3 * i] = r * sinTh * Math.cos(ph);
      positions[3 * i + 1] = r * sinTh * Math.sin(ph);
      positions[3 * i + 2] = r * Math.cos(th);

      const rgb = sampleColormap(this.colormap, this.fullDensity[i]);
      colors[3 * i] = rgb[0];
      colors[3 * i + 1] = rgb[1];
      colors[3 * i + 2] = rgb[2];
    }

    geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
    geometry.setAttribute('color', new THREE.BufferAttribute(colors, 3));

    // Crisp standard normal blending with point sprites (No blown-out white blobs)
    const material = new THREE.PointsMaterial({
      size: this.pointSize,
      vertexColors: true,
      map: this.pointTexture,
      transparent: true,
      alphaTest: 0.1,
      opacity: 0.95,
      depthWrite: true,
      sizeAttenuation: true
    });

    this.pointCloud = new THREE.Points(geometry, material);
    this.scene.add(this.pointCloud);

    // Nucleus scaling
    const nucRadius = Math.max(0.6, this.rMax * 0.02);
    this.nucleusMesh.scale.set(nucRadius, nucRadius, nucRadius);
  }

  updateColors() {
    if (!this.pointCloud || !this.fullDensity) return;
    const colors = this.pointCloud.geometry.attributes.color.array;
    const count = this.fullDensity.length;

    for (let i = 0; i < count; i++) {
      const rgb = sampleColormap(this.colormap, this.fullDensity[i]);
      colors[3 * i] = rgb[0];
      colors[3 * i + 1] = rgb[1];
      colors[3 * i + 2] = rgb[2];
    }
    this.pointCloud.geometry.attributes.color.needsUpdate = true;
    this.updateScalarBar();
  }

  updateScalarBar() {
    const bar = document.getElementById('scalar-bar');
    if (!bar) return;
    const lut = COLORMAPS[this.colormap] || COLORMAPS.inferno;
    const stops = lut.map((c, idx) => {
      const pct = (idx / (lut.length - 1)) * 100;
      const r = Math.round(c[0] * 255);
      const g = Math.round(c[1] * 255);
      const b = Math.round(c[2] * 255);
      return `rgb(${r}, ${g}, ${b}) ${pct.toFixed(1)}%`;
    });
    bar.style.background = `linear-gradient(to right, ${stops.join(', ')})`;
    const badge = document.getElementById('val-cmap-badge');
    if (badge) badge.textContent = this.colormap;
    const nameSpan = document.getElementById('val-cmap-name');
    if (nameSpan) nameSpan.textContent = this.colormap;
    const drawerSelect = document.getElementById('drawer-select-cmap');
    if (drawerSelect) drawerSelect.value = this.colormap;
  }

  /**
   * Perfectly align camera view to exact physical axes
   */
  alignCamera(view = 'iso') {
    const dist = this.rMax * 2.2;
    this.camera.up.set(0, 0, 1);

    if (view === 'top') {
      // Top view (looking down along +Z onto XY equatorial plane)
      this.camera.position.set(0.0001, 0, dist * 1.3);
      this.camera.up.set(0, 1, 0);
    } else if (view === 'front') {
      // Front view (looking directly at XZ plane)
      this.camera.position.set(0, -dist * 1.3, 0);
      this.camera.up.set(0, 0, 1);
    } else if (view === 'side') {
      // Side view (looking directly at YZ plane)
      this.camera.position.set(dist * 1.3, 0, 0);
      this.camera.up.set(0, 0, 1);
    } else {
      // Isometric 3D
      this.camera.position.set(dist * 0.75, -dist * 0.75, dist * 0.65);
      this.camera.up.set(0, 0, 1);
    }

    this.controls.target.set(0, 0, 0);
    this.controls.update();
  }

  animate(now) {
    requestAnimationFrame(this.animate);

    const dt = Math.min((now - this.lastTime) * 0.001, 0.1);
    this.lastTime = now;

    // Advance physical probability current flow
    if (this.isAnimating && this.pointCloud && this.fullPhi && this.fullOmega) {
      const positions = this.pointCloud.geometry.attributes.position.array;
      const count = this.fullPhi.length;
      const effectiveSpeed = this.flowSpeed * dt * 2.0;

      for (let i = 0; i < count; i++) {
        let ph = (this.fullPhi[i] + this.fullOmega[i] * effectiveSpeed) % (Math.PI * 2.0);
        if (ph < 0) ph += Math.PI * 2.0;
        this.fullPhi[i] = ph;

        const r = this.fullR[i];
        const th = this.fullTheta[i];
        const sinTh = Math.sin(th);

        const x = r * sinTh * Math.cos(ph);
        const y = r * sinTh * Math.sin(ph);
        const z = r * Math.cos(th);

        // Dynamic stationary spatial cutaways
        let isHidden = false;
        if (this.cutawayMode === 'quadrant') {
          if (x < 0 && y > 0) isHidden = true;
        } else if (this.cutawayMode === 'half') {
          if (y > 0) isHidden = true;
        }

        if (isHidden) {
          positions[3 * i] = 999999;
          positions[3 * i + 1] = 999999;
          positions[3 * i + 2] = 999999;
        } else {
          positions[3 * i] = x;
          positions[3 * i + 1] = y;
          positions[3 * i + 2] = z;
        }
      }

      this.pointCloud.geometry.attributes.position.needsUpdate = true;
    }

    this.controls.update();
    this.renderer.render(this.scene, this.camera);
  }

  // ===========================================================================
  // 3. UI Sync & Interactive Hotkeys
  // ===========================================================================
  initUI() {
    const updateState = () => {
      this.fetchOrbitalData();
      this.syncDrawer();
    };

    // Hotkey Handlers
    const incN = () => { if (this.n < 7) { this.n++; if (this.l >= this.n) this.l = this.n - 1; updateState(); } };
    const decN = () => { if (this.n > 1) { this.n--; if (this.l >= this.n) this.l = this.n - 1; updateState(); } };
    const incL = () => { if (this.l < this.n - 1) { this.l++; if (this.m > this.l) this.m = this.l; updateState(); } };
    const decL = () => { if (this.l > 0) { this.l--; if (this.m > this.l) this.m = this.l; if (this.m < -this.l) this.m = -this.l; updateState(); } };
    const incM = () => { if (this.m < this.l) { this.m++; updateState(); } };
    const decM = () => { if (this.m > -this.l) { this.m--; updateState(); } };
    const toggleForm = () => { this.realForm = !this.realForm; updateState(); };
    const cycleCutaway = () => {
      const cuts = ['full', 'quadrant', 'half'];
      this.cutawayMode = cuts[(cuts.indexOf(this.cutawayMode) + 1) % cuts.length];
      document.getElementById('val-cutaway').textContent = `${this.cutawayMode.charAt(0).toUpperCase() + this.cutawayMode.slice(1)} (X)`;
    };
    const cycleCmap = () => {
      const idx = (COLORMAP_KEYS.indexOf(this.colormap) + 1) % COLORMAP_KEYS.length;
      this.colormap = COLORMAP_KEYS[idx];
      this.updateColors();
      const selectEl = document.getElementById('drawer-select-cmap');
      if (selectEl) selectEl.value = this.colormap;
    };
    const toggleAnimation = () => {
      this.isAnimating = !this.isAnimating;
      this.updateFlowHUD();
    };
    const incSpeed = () => {
      this.flowSpeed = Math.min(10.0, this.flowSpeed + 0.5);
      this.updateFlowHUD();
      this.syncDrawer();
    };
    const decSpeed = () => {
      this.flowSpeed = Math.max(0.2, this.flowSpeed - 0.5);
      this.updateFlowHUD();
      this.syncDrawer();
    };
    const incSize = () => {
      this.pointSize = Math.min(8.0, this.pointSize + 0.5);
      if (this.pointCloud) this.pointCloud.material.size = this.pointSize;
      this.syncDrawer();
    };
    const decSize = () => {
      this.pointSize = Math.max(0.8, this.pointSize - 0.5);
      if (this.pointCloud) this.pointCloud.material.size = this.pointSize;
      this.syncDrawer();
    };

    // Keyboard Hotkey Listener
    window.addEventListener('keydown', (e) => {
      if (e.code === 'Space') { e.preventDefault(); toggleAnimation(); }
      else if (e.key === 'w' || e.key === 'W') incN();
      else if (e.key === 's' || e.key === 'S') decN();
      else if (e.key === 'e' || e.key === 'E') incL();
      else if (e.key === 'd' || e.key === 'D') decL();
      else if (e.key === 'r' || e.key === 'R') incM();
      else if (e.key === 'f' || e.key === 'F') decM();
      else if (e.key === 'c' || e.key === 'C') toggleForm();
      else if (e.key === 'x' || e.key === 'X') cycleCutaway();
      else if (e.key === 'm' || e.key === 'M') cycleCmap();
      else if (e.key === '1') this.alignCamera('top');
      else if (e.key === '2') this.alignCamera('front');
      else if (e.key === '3') this.alignCamera('side');
      else if (e.key === '0') this.alignCamera('iso');
      else if (e.key === '+' || e.key === '=') incSize();
      else if (e.key === '-' || e.key === '_') decSize();
      else if (e.key === ']') incSpeed();
      else if (e.key === '[') decSpeed();
    });

    // View Alignment Toolbar Buttons
    document.getElementById('btn-view-top').addEventListener('click', () => this.alignCamera('top'));
    document.getElementById('btn-view-front').addEventListener('click', () => this.alignCamera('front'));
    document.getElementById('btn-view-side').addEventListener('click', () => this.alignCamera('side'));
    document.getElementById('btn-reset-view').addEventListener('click', () => this.alignCamera('iso'));

    // Screenshot Snapshot
    document.getElementById('btn-snapshot').addEventListener('click', () => {
      this.renderer.render(this.scene, this.camera);
      const link = document.createElement('a');
      link.download = `quantum_orbital_n${this.n}_l${this.l}_m${this.m}.png`;
      link.href = this.renderer.domElement.toDataURL('image/png');
      link.click();
    });

    // Drawer Toggle
    const drawer = document.getElementById('sliders-drawer');
    document.getElementById('btn-toggle-menu').addEventListener('click', () => {
      drawer.classList.toggle('active');
    });
    document.getElementById('btn-close-drawer').addEventListener('click', () => {
      drawer.classList.remove('active');
    });

    // Click on Scalar Bar to Cycle Palette
    const scalarContainer = document.getElementById('scalar-container');
    if (scalarContainer) {
      scalarContainer.addEventListener('click', () => cycleCmap());
    }

    // Drawer Sliders
    const dN = document.getElementById('drawer-slider-n');
    const dL = document.getElementById('drawer-slider-l');
    const dM = document.getElementById('drawer-slider-m');
    const dSpeed = document.getElementById('drawer-slider-speed');
    const dSize = document.getElementById('drawer-slider-size');
    const dSamples = document.getElementById('drawer-slider-samples');
    const dCmap = document.getElementById('drawer-select-cmap');

    dN.addEventListener('input', () => { this.n = parseInt(dN.value, 10); if (this.l >= this.n) this.l = this.n - 1; updateState(); });
    dL.addEventListener('input', () => { this.l = parseInt(dL.value, 10); if (this.m > this.l) this.m = this.l; if (this.m < -this.l) this.m = -this.l; updateState(); });
    dM.addEventListener('input', () => { this.m = parseInt(dM.value, 10); updateState(); });
    dSpeed.addEventListener('input', () => { this.flowSpeed = parseFloat(dSpeed.value); this.updateFlowHUD(); this.syncDrawer(); });
    dSize.addEventListener('input', () => { this.pointSize = parseFloat(dSize.value); if (this.pointCloud) this.pointCloud.material.size = this.pointSize; this.syncDrawer(); });
    dSamples.addEventListener('change', () => { this.numSamples = parseInt(dSamples.value, 10); updateState(); });
    if (dCmap) dCmap.addEventListener('change', () => { this.colormap = dCmap.value; this.updateColors(); });

    this.updateScalarBar();
  }

  syncDrawer() {
    const dN = document.getElementById('drawer-slider-n');
    const dL = document.getElementById('drawer-slider-l');
    const dM = document.getElementById('drawer-slider-m');
    const dSpeed = document.getElementById('drawer-slider-speed');
    const dSize = document.getElementById('drawer-slider-size');
    const dSamples = document.getElementById('drawer-slider-samples');
    const dCmap = document.getElementById('drawer-select-cmap');

    if (dN) { dN.value = this.n; document.getElementById('drawer-val-n').textContent = this.n; }
    if (dL) { dL.max = this.n - 1; dL.value = this.l; document.getElementById('drawer-val-l').textContent = this.l; }
    if (dM) { dM.min = -this.l; dM.max = this.l; dM.value = this.m; document.getElementById('drawer-val-m').textContent = this.m; }
    if (dSpeed) { dSpeed.value = this.flowSpeed; document.getElementById('drawer-val-speed').textContent = `${this.flowSpeed.toFixed(1)}x`; }
    if (dSize) { dSize.value = this.pointSize; document.getElementById('drawer-val-size').textContent = this.pointSize.toFixed(1); }
    if (dSamples) { dSamples.value = this.numSamples; document.getElementById('drawer-val-samples').textContent = `${Math.round(this.numSamples/1000)}k`; }
    if (dCmap) { dCmap.value = this.colormap; }
  }

  updateFlowHUD() {
    const flowEl = document.getElementById('val-flow');
    if (!flowEl) return;
    if (this.m !== 0 && !this.realForm) {
      if (this.isAnimating) {
        flowEl.textContent = `ACTIVE (v = j / rho) [Speed: ${this.flowSpeed.toFixed(1)}x]`;
        flowEl.className = 'hud-val highlight';
      } else {
        flowEl.textContent = 'PAUSED (Space to resume)';
        flowEl.className = 'hud-val';
      }
    } else {
      flowEl.textContent = 'ZERO (m=0 / Real State is Stationary)';
      flowEl.className = 'hud-val';
    }
  }

  updateHUD(data) {
    document.getElementById('val-label').textContent = data.label;
    document.getElementById('val-qnums').textContent = `n=${data.n}, l=${data.l}, m=${data.m}`;
    document.getElementById('val-rep').textContent = data.real_form ? 'Real Cartesian Lobes' : 'Complex Stationary State';
    document.getElementById('val-energy').textContent = `${data.energy_ev.toFixed(4)} eV  (${data.energy_ha.toFixed(4)} Ha)`;
    document.getElementById('val-particles').textContent = `${data.num_particles.toLocaleString()} (Points)`;
    document.getElementById('val-cutaway').textContent = `${this.cutawayMode.charAt(0).toUpperCase() + this.cutawayMode.slice(1)} (X)`;
    this.updateFlowHUD();
    this.syncDrawer();
  }

  setLoading(active) {
    const spinner = document.getElementById('loading-spinner');
    if (spinner) spinner.classList.toggle('active', active);
  }
}

// Initialize on page load
window.addEventListener('DOMContentLoaded', () => {
  new QuantumOrbitalApp();
});
