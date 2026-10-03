"""
Script to generate static renders of 1s, 2p, 3d, 4f orbitals.
"""

import os
from atoms_py.render_points import OrbitalVisualizer

def generate_static_orbitals():
    os.makedirs("renders", exist_ok=True)
    
    orbitals = [
        (1, 0, 0, True, "1s"),
        (2, 0, 0, True, "2s"),
        (2, 1, 0, True, "2pz"),
        (2, 1, 1, True, "2px"),
        (3, 2, 0, True, "3dz2"),
        (3, 2, 2, True, "3dx2y2"),
        (3, 2, 1, True, "3dxz"),
        (4, 3, 0, True, "4fz3"),
    ]

    for n, l, m, rf, name in orbitals:
        print(f"Rendering {name} (n={n}, l={l}, m={m})...")
        v = OrbitalVisualizer(n=n, l=l, m=m, real_form=rf, num_samples=60000, off_screen=True)
        out_path = os.path.abspath(f"renders/orbital_{name}.png")
        v.render_static_image(out_path)
        print(f"  -> Saved {out_path}")

if __name__ == "__main__":
    generate_static_orbitals()
