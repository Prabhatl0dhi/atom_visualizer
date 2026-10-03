"""
Command-line interface and entry point for the Hydrogen Orbital Visualizer.

Usage:
    python -m atoms_py.app -n 5 -l 2 -m 1          # Exactly matches MinutePhysics video
    python -m atoms_py.app -n 2 -l 1 -m 1          # 2p complex swirling state
    python -m atoms_py.app -n 3 -l 2 -m 2          # 3d orbital
"""

from __future__ import annotations
import argparse
import sys
from atoms_py.wavefunction import validate_quantum_numbers, get_orbital_label
from atoms_py.render_points import OrbitalVisualizer


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Hydrogen Quantum Orbital 3D Probability Current Flow Visualizer",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "-n", "--principal",
        type=int,
        default=5,
        help="Principal quantum number n (n >= 1)",
    )
    parser.add_argument(
        "-l", "--azimuthal",
        type=int,
        default=2,
        help="Orbital angular momentum quantum number l (0 <= l < n)",
    )
    parser.add_argument(
        "-m", "--magnetic",
        type=int,
        default=1,
        help="Magnetic quantum number m (-l <= m <= l)",
    )
    parser.add_argument(
        "-N", "--samples",
        type=int,
        default=35000,
        help="Number of 3D spherical particles in cloud",
    )
    parser.add_argument(
        "--real-form",
        action="store_true",
        help="Use real Cartesian orbitals instead of complex stationary flow states",
    )
    parser.add_argument(
        "--cutaway",
        type=str,
        default="full",
        choices=["full", "quadrant", "half"],
        help="Cross-section cutaway view: 'full', 'quadrant' (1/4 slice cut), or 'half' (1/2 slice cut)",
    )
    parser.add_argument(
        "--no-grid",
        action="store_true",
        help="Hide coordinate floor grid",
    )
    parser.add_argument(
        "--colormap",
        type=str,
        default="viridis",
        choices=["viridis", "inferno", "plasma", "magma", "turbo", "coolwarm", "hot"],
        help="Color ramp for probability density |psi|^2",
    )
    parser.add_argument(
        "--point-size",
        type=float,
        default=6.0,
        help="Render size for 3D sphere particles",
    )
    parser.add_argument(
        "--speed",
        type=float,
        default=1.0,
        help="Probability current flow speed multiplier",
    )
    parser.add_argument(
        "--save-image",
        type=str,
        default=None,
        help="Path to export a static PNG screenshot instead of interactive window",
    )
    parser.add_argument(
        "--resolution",
        type=int,
        nargs=2,
        default=[1200, 900],
        metavar=("WIDTH", "HEIGHT"),
        help="Window or export image resolution",
    )

    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    n = args.principal
    l = args.azimuthal
    m = args.magnetic
    real_form = args.real_form

    try:
        validate_quantum_numbers(n, l, m)
    except (ValueError, TypeError) as err:
        print(f"\n[Error] Invalid quantum numbers: {err}", file=sys.stderr)
        sys.exit(1)

    label = get_orbital_label(n, l, m, real_form=real_form)
    print(f"\n=======================================================")
    print(f" Quantum Orbital Flow Visualizer: {label}")
    print(f" Quantum Numbers: (n={n}, l={l}, m={m})")
    print(f" Formula: v = (hbar / 2mi) * (psi* grad(psi) - psi grad(psi*)) / |psi|^2")
    print(f" Particles: {args.samples:,} (3D Spheres) | Colormap: {args.colormap}")
    print(f" Cutaway: {args.cutaway.upper()} | Floor Grid: {'OFF' if args.no_grid else 'ON'}")
    print(f" Flow Speed: {args.speed:.1f}x (Space = Pause/Resume, [ / ] = Speed)")
    print(f"=======================================================\n")

    vis = OrbitalVisualizer(
        n=n,
        l=l,
        m=m,
        num_samples=args.samples,
        real_form=real_form,
        cutaway_mode=args.cutaway,
        show_grid=not args.no_grid,
        point_size=args.point_size,
        colormap=args.colormap,
        off_screen=bool(args.save_image),
        auto_rotate=True,
    )
    vis.flow_speed = args.speed

    if args.save_image:
        print(f"Rendering snapshot to {args.save_image} ...")
        vis.render_static_image(args.save_image, window_size=tuple(args.resolution))
        print("Done!")
    else:
        print("Launching interactive PyVista 3D window...")
        print("Hotkeys:")
        print("  Space : Pause / Resume probability current flow")
        print("  [ / ] : Flow speed (-/+)")
        print("  W / S : n (+/-)")
        print("  E / D : l (+/-)")
        print("  R / F : m (+/-)")
        print("  X     : Cycle cutaway cross-section (Full -> Quadrant -> Half)")
        print("  G     : Toggle floor grid")
        print("  C     : Toggle Real/Complex Form")
        print("  M     : Cycle colormap")
        print("  + / - : Particle size (+/-)")
        vis.show()


if __name__ == "__main__":
    main()
