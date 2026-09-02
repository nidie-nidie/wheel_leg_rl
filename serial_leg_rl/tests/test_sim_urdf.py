import unittest
import xml.etree.ElementTree as ET

from serial_leg_rl.assets.sim_urdf import prepare_sim_urdf
from serial_leg_rl.assets.local_terrain import prepare_local_flat_terrain_usd


class SimUrdfTests(unittest.TestCase):
    def test_prepare_sim_urdf_rewrites_package_mesh_paths(self):
        sim_urdf_path = prepare_sim_urdf()
        text = sim_urdf_path.read_text(encoding="utf-8")
        self.assertNotIn("package://wheel_leg_urdf4", text)
        self.assertIn("/tmp/serial_leg_rl_isaac_assets/meshes/base_link.stl", text)

    def test_prepare_sim_urdf_removes_dummy_meshes_and_reduces_dummy_mass(self):
        sim_urdf_path = prepare_sim_urdf()
        root = ET.parse(sim_urdf_path).getroot()
        dummy_links = [link for link in root.findall("link") if "dummy" in link.attrib.get("name", "")]
        self.assertGreater(len(dummy_links), 0)
        for link in dummy_links:
            self.assertIsNone(link.find("visual"))
            self.assertIsNone(link.find("collision"))
            mass = link.find("inertial/mass")
            self.assertIsNotNone(mass)
            self.assertEqual(mass.attrib["value"], "0.001")

    def test_prepare_local_flat_terrain_usd_creates_mesh_asset(self):
        terrain_path = prepare_local_flat_terrain_usd()
        text = terrain_path.read_text(encoding="utf-8")
        self.assertIn('def Xform "terrain"', text)
        self.assertIn('def Mesh "collision_mesh"', text)
        self.assertIn("PhysicsCollisionAPI", text)


if __name__ == "__main__":
    unittest.main()
