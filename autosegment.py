import argparse
from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(
    description="Tutorial on using the differential IK controller."
)
parser.add_argument(
    "--robot", type=str, default="pi_robot", help="Name of the robot to load"
)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.headless = True
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import torch
import numpy as np
import open3d as o3d
from pxr import UsdGeom
from scipy.spatial.transform import Rotation as R
from pathlib import Path

import isaaclab.sim as sim_utils
import omni.usd
import isaaclab.utils.math as math

from isaacsim.core.prims import RigidPrim, GeometryPrim
from polaris.splat_renderer.scene.gaussian_model import GaussianModel

sim_cfg = sim_utils.SimulationCfg(
    dt=0.01,
    device="cuda",
)
sim = sim_utils.SimulationContext(sim_cfg)

def save_ply(model, path):
    from plyfile import PlyData, PlyElement
    
    xyz = model._xyz.detach().cpu().numpy()
    normals = np.zeros_like(xyz)
    f_dc = model._features_dc.detach().cpu().numpy().reshape(len(xyz), -1)
    f_rest = model._features_rest.detach().cpu().numpy().reshape(len(xyz), -1)
    opacity = model._opacity.detach().cpu().numpy().reshape(-1, 1)
    scaling = model._scaling.detach().cpu().numpy()
    rotation = model._rotation.detach().cpu().numpy()

    assert xyz.shape[1] == 3, f"xyz: {xyz.shape}"
    assert scaling.shape[1] == 3, f"Expected 3DGS scaling [N,3], got {scaling.shape}"
    assert rotation.shape[1] == 4, f"rotation: {rotation.shape}"
    assert opacity.shape[1] == 1, f"opacity: {opacity.shape}"

    dtype_full = (
        [('x', 'f4'), ('y', 'f4'), ('z', 'f4')]
        + [('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4')]
        + [(f'f_dc_{i}', 'f4') for i in range(f_dc.shape[1])]
        + [(f'f_rest_{i}', 'f4') for i in range(f_rest.shape[1])]
        + [('opacity', 'f4')]
        + [(f'scale_{i}', 'f4') for i in range(scaling.shape[1])]
        + [(f'rot_{i}', 'f4') for i in range(rotation.shape[1])]
    )

    elements = np.empty(len(xyz), dtype=dtype_full)
    attrs = np.concatenate([xyz, normals, f_dc, f_rest, opacity, scaling, rotation], axis=1)
    elements[:] = list(map(tuple, attrs))

    el = PlyElement.describe(elements, 'vertex')
    PlyData([el]).write(str(path))

# cfg = sim_utils.UsdFileCfg(usd_path=f"./data/assets/{args_cli.robot}/mesh.usdz")
cfg = sim_utils.UsdFileCfg(usd_path=args_cli.robot)
cfg.func("/World/test", cfg)


def get_robot_mesh_paths(stage):
    visited = set()
    meshes = set()

    def traverse(prim):
        if prim.IsValid():
            if UsdGeom.Mesh(prim) and "collision" not in prim.GetName():
                meshes.add(str(prim.GetPath()))

            for child in prim.GetChildren():
                traverse(child)

    traverse(stage.GetPseudoRoot())
    return meshes


# Example usage
stage = omni.usd.get_context().get_stage()
path_list = get_robot_mesh_paths(stage)
print(path_list)

sim.reset()
sim.step()

base_path = "/World/test/"
transforms = {}
for path in path_list:
    # view = RigidPrim(path)
    view = GeometryPrim(path)

    pos, rot = view.get_world_poses()
    scales = view.get_world_scales()
    assert torch.allclose(scales, scales[0,0]), "All scales must be the same"

    scale = scales[0,0].item()
    transforms[path] = (pos, rot, scale)


def get_meshes(stage, transforms):
    meshes = {}
    for path in transforms:
        prim = stage.GetPrimAtPath(path)
        if prim:
            mesh = UsdGeom.Mesh(prim)
            if mesh:
                points = mesh.GetPointsAttr().Get()
                face_vertex_indices = mesh.GetFaceVertexIndicesAttr().Get()
                faces = np.array(face_vertex_indices).reshape(-1, 3)
                vertices = np.array(points).reshape(-1, 3)

                pos, rot, scale= transforms[path]
                vertices = vertices * scale
                pos, rot = pos.detach().cpu().numpy(), rot.detach().cpu().numpy()
                pos, rot = pos.squeeze(), rot.squeeze()
                rot = R.from_quat(np.roll(rot, -1)).as_matrix()

                vertices = (rot @ vertices.T).T + pos

                mesh = o3d.geometry.TriangleMesh()
                mesh.vertices = o3d.utility.Vector3dVector(vertices)
                mesh.triangles = o3d.utility.Vector3iVector(faces)
                meshes[path] = (mesh, pos, rot)

    return meshes


# Convert HSV to RGB (0-1 range) using numpy and a standard HSV to RGB conversion
def hsv_to_rgb(hsv):
    h, s, v = hsv
    i = int(np.floor(h * 6))  # Determine the color sector (0-5)
    f = h * 6 - i  # Fractional part of hue
    p = v * (1 - s)
    q = v * (1 - f * s)
    t = v * (1 - (1 - f) * s)
    i = i % 6
    if i == 0:
        return [v, t, p]
    elif i == 1:
        return [q, v, p]
    elif i == 2:
        return [p, v, t]
    elif i == 3:
        return [p, q, v]
    elif i == 4:
        return [t, p, v]
    elif i == 5:
        return [v, p, q]


def generate_distinct_colors(n):
    # Generate 'n' evenly spaced hues
    hues = np.linspace(0, 1, n, endpoint=False)  # evenly space hues from 0 to 1
    colors = np.array([[(hue, 1, 1)] for hue in hues])  # Set saturation and value to 1
    colors_rgb = np.array([hsv_to_rgb(hsv[0]) for hsv in colors])
    return colors_rgb


def segment_pcd(o3d_meshes, pcd):
    scenes = []
    for mesh in o3d_meshes:
        scene = o3d.t.geometry.RaycastingScene()
        mesh_ = o3d.t.geometry.TriangleMesh.from_legacy(mesh)
        scene.add_triangles(mesh_)
        scenes.append(scene)

    assignment = -1 * np.ones(len(pcd))
    distances = np.zeros((len(pcd), len(scenes)))
    for i, scene in enumerate(scenes):
        is_inside = scene.compute_occupancy(
            o3d.core.Tensor(pcd[:, :3], dtype=o3d.core.Dtype.Float32)
        ).numpy()
        dist = scene.compute_distance(
            o3d.core.Tensor(pcd[:, :3], dtype=o3d.core.Dtype.Float32)
        ).numpy()
        distances[:, i] = dist
        assignment[is_inside == 1] = i

    assignment[assignment == -1] = np.argmin(distances[assignment == -1], axis=1)
    assignment = assignment.astype(int)

    n = np.max(assignment) + 1
    print(np.unique(assignment, return_counts=True))
    colors_options = generate_distinct_colors(n)

    # construct colors
    colors = []
    for i, a in enumerate(assignment):
        colors.append(colors_options[a])
    colors = np.array(colors)

    segmented_pcd = o3d.geometry.PointCloud()
    segmented_pcd.points = o3d.utility.Vector3dVector(pcd[:, :3])
    segmented_pcd.colors = o3d.utility.Vector3dVector(colors)
    o3d.visualization.draw_geometries([segmented_pcd] + o3d_meshes)

    return assignment


def split_model(model, assignments, meshes):
    path2model = {}
    for i, (mesh_path, (mesh, pos, rot)) in enumerate(meshes.items()):
        indices = np.where(assignments == i)[0]

        # mesh_parent = stage.GetPrimAtPath(mesh_path[:-1])
        # translate = mesh_parent.GetAttribute("xformOp:translate").Get()
        translate = -pos
        rotate = np.linalg.inv(rot)
        # rotate = R.from_matrix(rot).inv().as_matrix()
        # rotate = np.roll(rotate, 1)

        translate = torch.tensor(translate).cuda().float()
        rotate = torch.tensor(rotate).cuda().float()

        new_model = GaussianModel(3)
        # new_model._xyz = (rotate @ model._xyz[indices].T).T + translate
        new_model._xyz = (rotate @ (model._xyz[indices] + translate).T).T
        # new_model._xyz = model._xyz[indices] - translate
        new_model._features_rest = model._features_rest[indices]
        new_model._scaling = model._scaling[indices]

        # temp_rot = model._rotation[indices].detach().cpu().numpy()
        rotation_quat = math.quat_from_matrix(rotate)

        new_model._rotation = math.quat_mul(
            rotation_quat.repeat(len(model._rotation[indices]), 1),
            model._rotation[indices],
        )

        new_model._features_dc = model._features_dc[indices]
        new_model._opacity = model._opacity[indices]
        path2model[mesh_path] = new_model
    return path2model


meshes = get_meshes(stage, transforms)


model = GaussianModel(3)
# model.load_ply("./data/pi_robot/splat_real.ply")
# model.load_ply(f"./data/assets/{args_cli.robot}/splat.ply")
model.load_ply(Path(args_cli.robot).parent / "splat.ply")
pcd = model.get_xyz.detach().cpu().numpy()

assignments = segment_pcd([m[0] for m in meshes.values()], pcd)

path2model = split_model(model, assignments, meshes)
# save_dir = Path(f"./data/assets/{args_cli.robot}/SEGMENTED")
save_dir = Path(args_cli.robot).parent / "SEGMENTED_test"
save_dir.mkdir(exist_ok=True)
for mesh_path, model in path2model.items():
    mesh_path = mesh_path.split("/World/test/")[-1]
    mesh_path = mesh_path.replace("/", "-")
    save_ply(model, save_dir / f"{mesh_path}.ply")

# geom = o3d.geometry.PointCloud()
# geom.points = o3d.utility.Vector3dVector(pcd)
# o3d.visualization.draw_geometries([geom] + list(meshes.values()))


