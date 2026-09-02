#!/usr/bin/env python3
"""Inspect the source wheel-leg URDF without modifying it."""

from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path
import argparse
import json
import xml.etree.ElementTree as ET


DEFAULT_URDF = Path(__file__).resolve().parents[2] / "wheel_leg_urdf4" / "urdf" / "wheel_leg_urdf4.urdf"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--urdf", type=Path, default=DEFAULT_URDF)
    parser.add_argument("--json", action="store_true", help="print full JSON audit")
    args = parser.parse_args()

    audit = inspect_urdf(args.urdf)
    if args.json:
        print(json.dumps(audit, indent=2, sort_keys=True))
    else:
        print_summary(audit)


def inspect_urdf(path: Path) -> dict:
    root = ET.parse(path).getroot()
    links = []
    joints = []
    child_links = set()

    for link in root.findall("link"):
        mass = None
        inertial = link.find("inertial")
        if inertial is not None and inertial.find("mass") is not None:
            mass = float(inertial.find("mass").attrib["value"])
        links.append(
            {
                "name": link.attrib["name"],
                "mass": mass,
                "visual_count": len(link.findall("visual")),
                "collision_count": len(link.findall("collision")),
            }
        )

    for joint in root.findall("joint"):
        parent = joint.find("parent").attrib["link"]
        child = joint.find("child").attrib["link"]
        child_links.add(child)
        origin = joint.find("origin")
        axis = joint.find("axis")
        limit = joint.find("limit")
        joints.append(
            {
                "name": joint.attrib["name"],
                "type": joint.attrib.get("type"),
                "parent": parent,
                "child": child,
                "origin_xyz": origin.attrib.get("xyz") if origin is not None else None,
                "origin_rpy": origin.attrib.get("rpy") if origin is not None else None,
                "axis": axis.attrib.get("xyz") if axis is not None else None,
                "limit": limit.attrib if limit is not None else None,
                "mimic": joint.find("mimic") is not None,
            }
        )

    base_candidates = [link["name"] for link in links if link["name"] not in child_links]
    return {
        "path": str(path),
        "robot_name": root.attrib.get("name"),
        "base_candidates": base_candidates,
        "links": links,
        "joints": joints,
    }


def print_summary(audit: dict) -> None:
    joints = audit["joints"]
    links = audit["links"]
    print(f"URDF: {audit['path']}")
    print(f"robot: {audit['robot_name']}")
    print(f"base candidates: {', '.join(audit['base_candidates'])}")
    print(f"links: {len(links)}")
    print(f"joints: {len(joints)}")
    print(f"joint types: {dict(Counter(joint['type'] for joint in joints))}")

    missing_limits = [joint["name"] for joint in joints if joint["limit"] is None]
    mimic = [joint["name"] for joint in joints if joint["mimic"]]
    heavy_links = [link for link in links if link["mass"] is not None and link["mass"] > 5.0]
    print(f"joints without limits: {len(missing_limits)}")
    print(f"mimic joints: {mimic or 'none'}")
    print("heavy links (>5 kg):")
    for link in heavy_links:
        print(f"  {link['name']}: {link['mass']:.6g} kg")

    by_parent = defaultdict(list)
    for joint in joints:
        by_parent[joint["parent"]].append(joint)
    print("topology:")
    for parent, parent_joints in sorted(by_parent.items()):
        children = ", ".join(f"{joint['name']}->{joint['child']}" for joint in parent_joints)
        print(f"  {parent}: {children}")


if __name__ == "__main__":
    main()

