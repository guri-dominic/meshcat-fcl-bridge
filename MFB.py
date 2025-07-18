from dataclasses import dataclass
from typing import List, Optional, Tuple

import fcl
import meshcat
import meshcat.geometry as mg
import numpy as np
from numpy.typing import NDArray
from spatialmath import SE3, SO3


@dataclass
class GeometryRecord:
    meshcat_path: str
    fcl_obj: fcl.CollisionObject
    geom_type: str  # "Sphere", "Box", "Cylinder", "Capsule", etc.
    size: Optional[Tuple[float, float, float]] = None
    radius: Optional[float] = None
    length: Optional[float] = None


class GeometryManager:
    def __init__(self, zmq_url: Optional[str] = None):
        # Offsets to align MeshCat (Y-up) to FCL (Z-up)
        self.meshcat_offsets = {
            "Cylinder": SE3.Rx(-np.pi / 2),
            "Capsule": SE3.Rx(-np.pi / 2),
        }
        # Initialize MeshCat
        if zmq_url is None:
            self._vis = None
        else:
            self._vis = meshcat.Visualizer(zmq_url=zmq_url)
            print(f"MeshCat URL: {self._vis.url()}")
        # Record storage
        self._records: dict[str, GeometryRecord] = {}

    def add_sphere(self, key: str, radius: float, pose: SE3 = SE3()):
        path = f"/{key}"
        # MeshCat
        if self._vis is not None:
            self._vis[path].set_object(mg.Sphere(radius))
            self._vis[path].set_transform(pose.A)
        # FCL
        obj = fcl.CollisionObject(
            fcl.Sphere(radius), fcl.Transform(pose.A[:3, :3], pose.A[:3, 3])
        )
        self._records[key] = GeometryRecord(path, obj, "Sphere", radius=radius)

    def add_box(self, key: str, size: Tuple[float, float, float], pose: SE3 = SE3()):
        path = f"/{key}"
        if self._vis is not None:
            self._vis[path].set_object(mg.Box(size))
            self._vis[path].set_transform(pose.A)
        obj = fcl.CollisionObject(
            fcl.Box(*size), fcl.Transform(pose.A[:3, :3], pose.A[:3, 3])
        )
        self._records[key] = GeometryRecord(path, obj, "Box", size=size)

    def add_cylinder(self, key: str, radius: float, length: float, pose: SE3 = SE3()):
        path = f"/{key}"
        # MeshCat: apply Y->Z offset
        off = self.meshcat_offsets["Cylinder"]
        if self._vis is not None:
            self._vis[path].set_object(mg.Cylinder(length, radius))
            self._vis[path].set_transform((pose * off).A)
        # FCL
        obj = fcl.CollisionObject(
            fcl.Cylinder(radius, length), fcl.Transform(pose.A[:3, :3], pose.A[:3, 3])
        )
        self._records[key] = GeometryRecord(
            path, obj, "Cylinder", radius=radius, length=length
        )

    def add_capsule(self, key: str, radius: float, length: float, pose: SE3 = SE3()):
        path = f"/{key}"
        # MeshCat body
        if self._vis is not None:
            off = self.meshcat_offsets["Capsule"]
            self._vis[f"{path}/body"].set_object(mg.Cylinder(length, radius))
            self._vis[f"{path}/body"].set_transform((pose * off).A)
            # MeshCat caps
            p1 = pose * SE3(0, 0, length / 2)
            self._vis[f"{path}/cap1"].set_object(mg.Sphere(radius))
            self._vis[f"{path}/cap1"].set_transform(p1.A)
            p2 = pose * SE3(0, 0, -length / 2)
            # print(f"p1={p1.t}  |  p2={p2.t}  |  length={length}  |  radius={radius}")
            self._vis[f"{path}/cap2"].set_object(mg.Sphere(radius))
            self._vis[f"{path}/cap2"].set_transform(p2.A)
        # FCL
        obj = fcl.CollisionObject(
            fcl.Capsule(radius, length), fcl.Transform(pose.A[:3, :3], pose.A[:3, 3])
        )
        self._records[key] = GeometryRecord(
            path, obj, "Capsule", radius=radius, length=length
        )

    def update(self, key: str, pose: SE3):
        rec = self._records[key]
        if self._vis is not None:
            if rec.geom_type in ("Cylinder", "Capsule"):
                off = self.meshcat_offsets[rec.geom_type]
                if rec.geom_type == "Cylinder":
                    self._vis[rec.meshcat_path].set_transform((pose * off).A)
                else:
                    # Capsule: update body and caps
                    self._vis[f"{rec.meshcat_path}/body"].set_transform((pose * off).A)
                    half = rec.length / 2
                    self._vis[f"{rec.meshcat_path}/cap1"].set_transform(
                        (pose * SE3.Tz(half)).A
                    )
                    self._vis[f"{rec.meshcat_path}/cap2"].set_transform(
                        (pose * SE3.Tz(-half)).A
                    )
            else:
                self._vis[rec.meshcat_path].set_transform(pose.A)
        # FCL
        rec.fcl_obj.setTransform(fcl.Transform(pose.A[:3, :3], pose.A[:3, 3]))

    def in_collision(self, key1: str, key2: str) -> bool:
        a = self._records[key1].fcl_obj
        b = self._records[key2].fcl_obj
        req = fcl.CollisionRequest()
        res = fcl.CollisionResult()
        fcl.collide(a, b, req, res)
        return res.is_collision

    def collision_manager(self, filter=lambda ss: True):
        manager = fcl.DynamicAABBTreeCollisionManager()
        objects = [rec.fcl_obj for key, rec in self._records.items() if filter(key)]
        manager.registerObjects(objects)
        manager.setup()
        return manager

    def clear(self, key: Optional[str] = None):
        if self._vis is not None:
            if key:
                rec = self._records.pop(key, None)
                if rec:
                    try:
                        self._vis.delete(rec.meshcat_path)
                    except Exception as _e:
                        self._vis[rec.meshcat_path].delete()
                return
            # clear entire scene
            self._vis.delete()
        self._records.clear()

    def add_mesh(self, key: str, mesh_file: str, pose: SE3 = SE3()):
        import trimesh

        mesh = trimesh.load(mesh_file, force="mesh")
        verts = np.array(mesh.vertices, dtype=np.float64)
        faces = np.array(mesh.faces, dtype=np.int32)
        path = f"/{key}"
        if self._vis is not None:
            self._vis[path].set_object(mg.TriangularMeshGeometry(verts, faces))
            self._vis[path].set_transform(pose.A)
        model = fcl.BVHModel()
        model.beginModel(len(verts), len(faces))
        model.addSubModel(verts.tolist(), faces.tolist())
        model.endModel()
        obj = fcl.CollisionObject(model, fcl.Transform(pose.A[:3, :3], pose.A[:3, 3]))
        self._records[key] = GeometryRecord(path, obj, "Mesh")

    def add_chain(self, key_prefix: str, chain: List[SE3 | NDArray], radius: float):
        """
        Add a kinematic chain of segments defined by 3D chain or SE3 poses.
        Each consecutive pair of chain gets a cylinder.
        """
        for i, (p0, p1) in enumerate(zip(chain, chain[1:])):
            v = p1 - p0
            length = np.linalg.norm(v)
            if length < 1e-6:
                continue
            midpoint = (p0 + p1) / 2
            # Compute orientation: align z-axis to segment direction
            z = v / length
            if abs(z[2]) < 0.9:
                x = np.cross([0, 0, 1], z)
            else:
                x = np.cross([0, 1, 0], z)
            x = x / np.linalg.norm(x)
            y = np.cross(z, x)
            R = np.column_stack((x, y, z))
            # pose = SE3(R, midpoint)
            pose = SE3.Trans(midpoint) * SE3(SO3(R))
            # self.add_capsule(f"{key_prefix}_seg{i}", radius, length, pose)
            self.add_cylinder(f"{key_prefix}_seg{i}", radius, length, pose)
            if i > 1:
                # print(f"{key_prefix}_seg{i}_joint")
                self.add_sphere(f"{key_prefix}_seg{i}_joint", radius, SE3.Trans(p0))
