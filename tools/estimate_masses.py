#!/usr/bin/env python3
"""Rebuild Gray's mass model from its STL meshes plus datasheet component masses.

WHY
---
The SolidWorks-exported URDF totals 1.254 kg, which cannot be the whole robot - the
twelve DS3218MG servos alone are ~720 g. The robot is currently disassembled and
cannot be weighed, so the model is rebuilt from geometry we already have.

WHAT THE ORIGINAL URDF ACTUALLY IS
----------------------------------
Not garbage - incomplete. A least-squares fit over all 13 links recovers SolidWorks'
assumptions to an RMS error of 0.02 g:

    printed parts : SOLID plastic at 1.0227 g/cm3   (generic ABS/nylon, not resin)
    servos        : 64.3 g each, with 8 of 12 modelled

That fit is exact, so the URDF represents the printed structure plus 8 servos, and
nothing else. Missing: the 4 hip-pitch servos, the battery, the Pi, all electronics,
and every fastener. This tool adds them and re-densities the printed parts to SLA
resin.

GEOMETRY FACTS VERIFIED BEFORE RELYING ON ANY OF IT
---------------------------------------------------
  - URDF link meshes are in METRES; robot/STLs/*.STL are in MILLIMETRES.
  - The four legs are geometrically identical to 0.1 cm3 - the model is symmetric.
  - Links flagged "not watertight" are MULTI-BODY STLs, not broken geometry. The
    exporter merged each link's sub-assembly into one file; hole-filling changes
    their volume by 0.00% and the divergence-theorem volume sums bodies correctly.
  - SOME LINK MESHES CONTAIN SERVO GEOMETRY. Splitting fl_top yields a largest body
    of 73.92 cm3 - exactly Left_Thigh.STL - plus bodies measuring 40.0 x 20.3 and
    40.2 x 20.2 mm, i.e. the DS3218MG's 40 x 20 footprint. Baked-in servo count:
    base_link 4, each *_top 1, hips and shanks 0. Their displaced volume is removed
    before applying resin density, so a servo is never counted twice.
  - Servo-free links cross-check exactly: fl_bottom (46.4 cm3) == Leg_Universal
    (34.1) + Feet (12.0) = 46.0, and fl_hip (46.3) ~= HipJoint (42.5).
  - COM: mesh centroids match SolidWorks' inertial origins to 0.1 mm on every link
    WITHOUT baked-in servos, and differ by a consistent 12.5 mm on the four *_top
    links - the signature of SolidWorks correctly treating the servo as denser than
    the plastic. We therefore keep SolidWorks' COM and inertia axes and rescale them,
    rather than recomputing from a uniform-density mesh.

Run:
    python tools/estimate_masses.py                 # nominal (hollowed parts)
    python tools/estimate_masses.py --fill 1.0      # if parts turn out solid
    python tools/estimate_masses.py --write-yaml    # emit gray/config/robot.yaml
"""

from __future__ import annotations

import argparse
import os
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field

import numpy as np
import trimesh

SRC_URDF = "robot/++Main Body URDF/gray.urdf"
MESH_DIR = "robot/++Main Body URDF/meshes"
OUT_YAML = "gray/config/robot.yaml"

# --- Material ---------------------------------------------------------------------
# Cured SLA photopolymer, typically 1.10-1.20 g/cm3, printed at ~50-60% fill for
# structural integrity. NOTE: "50-60% infill" is FDM phrasing; in SLA it most likely
# means hollowed with an internal lattice. If the parts are solid, pass --fill 1.0,
# which roughly doubles printed mass. Settle it with a scale during reassembly.
RESIN_DENSITY_G_CM3 = 1.15
FILL_FRACTION = 0.55

# Displaced volume of one DS3218MG as modelled in CAD (40 x 20 x 40.5 mm body),
# measured from the sub-bodies inside fl_top.
SERVO_VOLUME_CM3 = 27.0
SERVO_MASS_G = 60.0  # datasheet; SolidWorks had used 64.3 g

# Servo geometry PRESENT IN each link mesh. Distinct from EXTRA_COMPONENTS below,
# which lists mass that exists on the robot but is absent from the CAD.
SERVOS_IN_MESH: dict[str, int] = {
    "base_link": 4,
    **{f"{leg}_top": 1 for leg in ("fl", "fr", "bl", "br")},
}


@dataclass(frozen=True)
class Component:
    name: str
    grams: float
    note: str = ""


# Mass on the real robot that the CAD never modelled.
#
# The 4 hip-pitch servos are absent from every link mesh. On a SpotMicro-style leg
# they mount on the hip bracket and drive the thigh, so that is where they are placed.
# ASSUMPTION - verify during reassembly; a servo one link outboard of this materially
# changes that leg's swing dynamics.
EXTRA_COMPONENTS: dict[str, list[Component]] = {
    "base_link": [
        Component("Raspberry Pi 4B", 46.0),
        Component("PCA9685 servo driver", 10.0),
        Component("IMU breakout", 3.0, "MPU-9255 / BNO055"),
        Component("TFmini-S lidar", 5.0),
        Component("Pi Camera", 3.0),
        Component("2S LiPo 5000mAh", 280.0, "power system not yet built - Phase 4"),
        Component("fasteners/rods/bearings", 100.0, "estimate, distributed"),
    ],
    **{
        f"{leg}_hip": [Component("DS3218MG (hip pitch)", SERVO_MASS_G, "not in CAD")]
        for leg in ("fl", "fr", "bl", "br")
    },
}


@dataclass
class Link:
    name: str
    mesh_volume_cm3: float
    resin_volume_cm3: float
    servos_in_mesh: int
    resin_g: float
    baked_servo_g: float
    extra_g: float
    sw_mass_g: float
    com_m: np.ndarray            # from SolidWorks - accounts for servo density
    inertia: np.ndarray          # kg.m^2 about COM, rescaled to total mass
    extras: list[Component] = field(default_factory=list)

    @property
    def total_g(self) -> float:
        return self.resin_g + self.baked_servo_g + self.extra_g


def load_source_urdf() -> dict[str, dict]:
    """SolidWorks mass, COM and inertia tensor per link."""
    root = ET.parse(SRC_URDF).getroot()
    out = {}
    for link in root.findall("link"):
        inertial = link.find("inertial")
        if inertial is None:
            continue
        i = inertial.find("inertia")
        out[link.get("name")] = {
            "mass_g": float(inertial.find("mass").get("value")) * 1000.0,
            "com": np.array([float(x) for x in inertial.find("origin").get("xyz").split()]),
            "inertia": np.array([
                [float(i.get("ixx")), float(i.get("ixy")), float(i.get("ixz"))],
                [float(i.get("ixy")), float(i.get("iyy")), float(i.get("iyz"))],
                [float(i.get("ixz")), float(i.get("iyz")), float(i.get("izz"))],
            ]),
        }
    return out


def fit_solidworks_assumptions(src: dict, volumes: dict) -> tuple[float, float, float]:
    """Recover the density and servo mass SolidWorks used. Pure validation."""
    rows, masses = [], []
    for name, d in src.items():
        ns = SERVOS_IN_MESH.get(name, 0)
        rows.append([volumes[name] - ns * SERVO_VOLUME_CM3, ns])
        masses.append(d["mass_g"])
    A, b = np.array(rows), np.array(masses)
    (rho, servo), *_ = np.linalg.lstsq(A, b, rcond=None)
    rms = float(np.sqrt(((b - A @ [rho, servo]) ** 2).mean()))
    return float(rho), float(servo), rms


def build(effective_g_cm3: float) -> list[Link]:
    src = load_source_urdf()
    volumes = {
        n: abs(float(trimesh.load(f"{MESH_DIR}/{n}.STL", force="mesh").volume)) * 1e6
        for n in src
    }

    links = []
    for name, d in src.items():
        ns = SERVOS_IN_MESH.get(name, 0)
        resin_v = max(volumes[name] - ns * SERVO_VOLUME_CM3, 0.0)
        resin_g = resin_v * effective_g_cm3
        baked_g = ns * SERVO_MASS_G
        extras = EXTRA_COMPONENTS.get(name, [])
        extra_g = sum(c.grams for c in extras)
        total_g = resin_g + baked_g + extra_g

        # Keep SolidWorks' inertia shape; rescale to the corrected mass. Same-shape
        # assumption - the added components are taken to follow the existing mass
        # distribution. Weakest for base_link, where battery and Pi dominate; the
        # +/-2 cm COM randomisation during training is sized to absorb it.
        inertia = d["inertia"] * (total_g / d["mass_g"])

        links.append(Link(
            name=name, mesh_volume_cm3=volumes[name], resin_volume_cm3=resin_v,
            servos_in_mesh=ns, resin_g=resin_g, baked_servo_g=baked_g,
            extra_g=extra_g, sw_mass_g=d["mass_g"], com_m=d["com"],
            inertia=inertia, extras=extras,
        ))
    return sorted(links, key=lambda l: (l.name != "base_link", l.name))


def main() -> None:
    ap = argparse.ArgumentParser(description="Rebuild Gray's mass model.")
    ap.add_argument("--density", type=float, default=RESIN_DENSITY_G_CM3,
                    help="cured resin density, g/cm3 (default: %(default)s)")
    ap.add_argument("--fill", type=float, default=FILL_FRACTION,
                    help="fill fraction 0-1; 1.0 = solid (default: %(default)s)")
    ap.add_argument("--write-yaml", action="store_true", help=f"write {OUT_YAML}")
    args = ap.parse_args()

    effective = args.density * args.fill
    links = build(effective)

    src = load_source_urdf()
    vols = {l.name: l.mesh_volume_cm3 for l in links}
    rho, servo_g, rms = fit_solidworks_assumptions(src, vols)
    print("Recovered SolidWorks assumptions (validation):")
    print(f"  printed density {rho:.4f} g/cm3   servo {servo_g:.1f} g   "
          f"RMS residual {rms:.3f} g")
    print(f"  -> the 1.254 kg URDF is printed structure + 8 of 12 servos, "
          f"nothing else\n")

    print(f"SLA resin {args.density:.2f} g/cm3 x fill {args.fill:.2f} "
          f"= {effective:.3f} g/cm3 effective\n")
    print(f"{'link':11s} {'mesh':>7s} {'resin':>7s} {'srv':>4s} | {'resin':>7s} "
          f"{'baked':>7s} {'extra':>7s} {'TOTAL':>8s} | {'was':>7s}")
    print(f"{'':11s} {'cm3':>7s} {'cm3':>7s} {'':>4s} | {'g':>7s} {'g':>7s} "
          f"{'g':>7s} {'g':>8s} | {'g':>7s}")
    print("-" * 88)
    for l in links:
        print(f"{l.name:11s} {l.mesh_volume_cm3:7.1f} {l.resin_volume_cm3:7.1f} "
              f"{l.servos_in_mesh:4d} | {l.resin_g:7.1f} {l.baked_servo_g:7.1f} "
              f"{l.extra_g:7.1f} {l.total_g:8.1f} | {l.sw_mass_g:7.1f}")
    print("-" * 88)
    t = lambda f: sum(f(l) for l in links)
    total = t(lambda l: l.total_g)
    print(f"{'TOTAL':11s} {t(lambda l:l.mesh_volume_cm3):7.1f} "
          f"{t(lambda l:l.resin_volume_cm3):7.1f} {t(lambda l:l.servos_in_mesh):4d} | "
          f"{t(lambda l:l.resin_g):7.1f} {t(lambda l:l.baked_servo_g):7.1f} "
          f"{t(lambda l:l.extra_g):7.1f} {total:8.1f} | {t(lambda l:l.sw_mass_g):7.1f}")

    n_servos = t(lambda l: l.servos_in_mesh) + sum(
        1 for l in links for c in l.extras if "DS3218MG" in c.name)
    print(f"\n  total {total/1000:.3f} kg   (servos accounted for: {n_servos}/12)")
    print(f"  Stanford Pupper, same 12-DOF hobby-servo class: 2.1 kg")

    leg = sum(l.total_g for l in links if l.name != "base_link") / 4
    body = next(l.total_g for l in links if l.name == "base_link")
    print(f"  body {body/1000:.3f} kg + 4 legs x {leg/1000:.3f} kg "
          f"= {100*4*leg/total:.0f}% of mass in the legs")

    if args.write_yaml:
        write_yaml(links, args, effective, total)


def write_yaml(links, args, effective, total_g) -> None:
    os.makedirs(os.path.dirname(OUT_YAML), exist_ok=True)
    out = [
        "# Gray - physical parameters. GENERATED by tools/estimate_masses.py",
        "#",
        "# Masses are ESTIMATED: mesh volume x SLA resin density, plus datasheet",
        "# masses for bought parts. The robot is disassembled and could not be",
        "# weighed. Re-run with measured values during Phase 4 and retrain.",
        "#",
        "# COM and inertia come from the SolidWorks export, which correctly treated",
        "# servos as denser than plastic; inertia is rescaled to the corrected mass.",
        "",
        "material:",
        f"  resin_density_g_cm3: {args.density}",
        f"  fill_fraction: {args.fill}",
        f"  effective_density_g_cm3: {effective:.4f}",
        "",
        "servo:",
        "  model: DS3218MG",
        "  count: 12",
        "  range_deg: 270",
        "  limit_rad: 2.3562         # +/-135 deg",
        "  effort_nm: 1.96           # 20 kg.cm @ 6.8 V",
        "  velocity_rad_s: 6.54      # 0.16 s / 60 deg",
        "  mass_g: 60.0",
        "  control: position         # PWM target only - NO position feedback",
        "  control_hz: 50            # PCA9685 servo PWM period is 20 ms",
        "",
        f"total_mass_kg: {total_g/1000:.4f}",
        "",
        "links:",
    ]
    for l in links:
        I = l.inertia
        out += [
            f"  {l.name}:",
            f"    mass_kg: {l.total_g/1000:.6f}",
            f"    resin_g: {l.resin_g:.2f}",
            f"    baked_servo_g: {l.baked_servo_g:.2f}",
            f"    extra_g: {l.extra_g:.2f}",
            f"    mesh_volume_cm3: {l.mesh_volume_cm3:.2f}",
            f"    resin_volume_cm3: {l.resin_volume_cm3:.2f}",
            f"    servos_in_mesh: {l.servos_in_mesh}",
            "    com_m: [{:.8f}, {:.8f}, {:.8f}]".format(*l.com_m),
            "    inertia:",
            f"      ixx: {I[0,0]:.9e}",
            f"      ixy: {I[0,1]:.9e}",
            f"      ixz: {I[0,2]:.9e}",
            f"      iyy: {I[1,1]:.9e}",
            f"      iyz: {I[1,2]:.9e}",
            f"      izz: {I[2,2]:.9e}",
        ]
        if l.extras:
            out.append("    extra_components:")
            out += [f"      - {{name: {c.name!r}, grams: {c.grams}}}" for c in l.extras]
    with open(OUT_YAML, "w") as fh:
        fh.write("\n".join(out) + "\n")
    print(f"\n  wrote {OUT_YAML}")


if __name__ == "__main__":
    main()
