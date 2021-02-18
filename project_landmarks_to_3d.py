import concurrent.futures
import itertools
import json
import logging
from pathlib import Path

import cv2
import hydra
import numpy as np
import open3d as o3d
import tifffile
from omegaconf import OmegaConf, DictConfig
from tqdm import tqdm
from scipy.spatial.transform import Rotation as R

FACE_MESH_DOWNSAMPLE_RADIUS = 0.003
LANDMARK_SPHERE_RADIUS = 0.0015  # Size of the sph


def _process_file(f_json: Path, f_img: Path, f_depth, dir_output: Path, visualize_mesh: bool = False):
    """Project facial landmarks on rgb image and save output visualization
    Args:
        f_json (Path): Json file containing camera intrinsics
        f_img (Path): Image to distort
        f_depth (Path): Depth image
        dir_output (Path): Which dir to store outputs in

    Note:
        The camera space co-ordinates for points is in the Blender camera notation: Y: up, negative-Z: fwd
        To cast to screen space, we convert them to computer vision camera notation: Y: down, Z: fwd
    """
    # Load images and metadata
    with f_json.open() as json_file:
        metadata = json.load(json_file)
        metadata = OmegaConf.create(metadata)

    # Extract all landmarks in camera space as a list of lists
    landmarks_cam = [OmegaConf.to_container(landmark.camera_space_pos) for landmark in metadata.landmarks]
    landmarks_cam = np.array(landmarks_cam)  # Shape: (68, 3)

    # Read RGB, Depth images
    rgb = cv2.imread(str(f_img))
    rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
    depth = tifffile.imread(str(f_depth))

    # Project RGB image to 3D
    h, w, c = rgb.shape
    xx, yy = np.meshgrid(np.arange(h), np.arange(w))
    img_pxs = np.stack((xx, yy), axis=2)  # shape (H, W, 2)
    img_pxs = img_pxs.reshape((-1, 2))  # Shape: [N, 2]
    img_pxs = np.concatenate([img_pxs, np.ones((h * w, 1))], axis=1)  # Add Homogenous coord. Shape: [N, 3]
    rgb_pxs = rgb.reshape((-1, 3)).astype(np.float32) / 255.0  # Convert to [0,1] range. Shape: [N, 3]

    depth_pxs = depth.reshape((-1))  # Shape: [N]
    depth_pxs[np.isinf(depth_pxs)] = 0.0
    depth_pxs[np.isnan(depth_pxs)] = 0.0

    # Filter valid pxs
    valid_pxs = ~(depth_pxs < 1e-6)  # Depth with value of 0 is not needed to construct ptcloud
    depth_pxs = depth_pxs[valid_pxs]
    img_pxs = img_pxs[valid_pxs, :]
    rgb_pxs = rgb_pxs[valid_pxs, :]

    # Cast to 3D, scale by depth.
    intrinsics = np.array(metadata.camera.intrinsics, dtype=np.float32)
    intrinsics_inv = np.linalg.inv(intrinsics)
    img_pts = (intrinsics_inv @ img_pxs.T).T  # Cast to 3D, depth=1. Shape: [N, 3]
    img_pts[:, 2] *= depth_pxs  # Scale the depth

    # Convert to houdini coordinate system: x: right, y: up, z: behind.
    # Projecting to intrinsics converts to camera coord system of x: right, y: down, z: forward
    # This corresponds to 180 deg rot around x-axis
    r = R.from_euler("x", 180, degrees=True)
    rot_mat = r.as_matrix()
    img_pts = (rot_mat @ img_pts.T).T

    # Construct pointcloud of face
    pcd_face = o3d.geometry.PointCloud()
    pcd_face.points = o3d.utility.Vector3dVector(img_pts)
    pcd_face.colors = o3d.utility.Vector3dVector(rgb_pxs)
    # Downsample and estimate normals
    pcd_face = pcd_face.voxel_down_sample(voxel_size=FACE_MESH_DOWNSAMPLE_RADIUS)
    pcd_face.estimate_normals()
    pcd_face.orient_normals_towards_camera_location()

    # Construct mesh of face from pointcloud
    radii = [0.005, 0.01]
    face_mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
        pcd_face, o3d.utility.DoubleVector(radii)
    )

    # Add a red spherical mesh for each landmark
    landmark_mesh = o3d.geometry.TriangleMesh()
    for landmark_ in landmarks_cam:
        sphere = o3d.geometry.TriangleMesh.create_sphere(radius=LANDMARK_SPHERE_RADIUS)
        sphere.paint_uniform_color(np.array([[1], [0], [0]], dtype=np.float64))
        sphere.translate(landmark_)
        landmark_mesh += sphere
    face_mesh += landmark_mesh

    # VISUALIZE THE DATA FOR DEBUGGING
    if visualize_mesh:
        o3d.visualization.draw_geometries([landmark_mesh, pcd_face], mesh_show_back_face=True)

    # Save mesh of face with landmarks.
    out_filename = dir_output / f"{f_img.stem}.face_mesh.ply"
    o3d.io.write_triangle_mesh(str(out_filename), face_mesh)


def get_render_id_from_path(path: Path):
    return int(str(path.name).split(".")[0])


@hydra.main(config_path=".", config_name="config")
def main(cfg: DictConfig):
    """Project facial landmarks on rgb image and create output visualization of the landmarks"""
    log = logging.getLogger(__name__)

    if int(cfg.workers) > 0:
        max_workers = int(cfg.workers)
    else:
        max_workers = None

    # Read input/output parameters
    dir_input = Path(cfg.dir.input)
    if not dir_input.is_dir():
        raise ValueError(f"Not a directory: {dir_input}")
    log.info(f"Input Dir: {dir_input}")

    if cfg.dir.output is None:
        dir_output = dir_input
    else:
        dir_output = Path(cfg.dir.output)
        if not dir_output.exists():
            dir_output.mkdir(parents=True)
    log.info(f"Output Dir: {dir_output}")

    ext_rgb = cfg.file_ext.rgb
    log.info(f"RGB img File Ext: {ext_rgb}")

    ext_depth = cfg.file_ext.depth
    log.info(f"Depth img File Ext: {ext_depth}")

    ext_info = cfg.file_ext.info
    ext_info_type = ext_info.split(".")[-1]
    if ext_info_type != "json":
        raise ValueError(f"Unsupported filetype: {ext_info_type}. Info files must be of type json")

    info_filenames = sorted(dir_input.glob("*" + ext_info), key=get_render_id_from_path)
    num_json = len(info_filenames)
    log.info(f"Num Info Files: {num_json}")
    if num_json < 1:
        raise ValueError(
            f"No info json files found. Searched:\n" f'  dir: "{dir_input}"\n' f'  file extention: "{ext_info}"'
        )

    rgb_filenames = sorted(dir_input.glob("*" + ext_rgb), key=get_render_id_from_path)
    num_images = len(rgb_filenames)
    log.info(f"Num Input Files: {num_images}")
    if num_images != num_json:
        raise ValueError(
            f"Unequal number of json files ({num_json}) and " f'rgb images ({num_images}) in dir: "{dir_input}"'
        )

    depth_filenames = sorted(dir_input.glob("*" + ext_depth), key=get_render_id_from_path)
    num_images = len(depth_filenames)
    log.info(f"Num Depth Files: {num_images}")
    if num_images != num_json:
        raise ValueError(
            f"Unequal number of json files ({num_json}) and " f'depth images ({num_images}) in dir: "{dir_input}"'
        )

    # Process files
    render_id = int(cfg.landmarks_3d.render_id)
    visualize_mesh = cfg.landmarks_3d.visualize
    if render_id > -1:
        # If a specific render id given, process only that render id
        info_file = dir_input / (f"{render_id}" + ext_info)
        rgb_file = dir_input / (f"{render_id}" + ext_rgb)
        depth_file = dir_input / (f"{render_id}" + ext_depth)
        _process_file(info_file, rgb_file, depth_file, dir_output, visualize_mesh)
    else:
        # Process all the files using multiple processes
        if visualize_mesh:
            raise ValueError(
                f"Visualization cannot be true when processing all the files."
                f"Please pass landmarks_3d.visualize=false"
            )
        with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
            with tqdm(total=len(info_filenames)) as pbar:
                for _ in executor.map(
                    _process_file,
                    info_filenames,
                    rgb_filenames,
                    depth_filenames,
                    itertools.repeat(dir_output),
                    itertools.repeat(visualize_mesh),
                ):
                    # Catch any error raised in processes
                    pbar.update()


if __name__ == "__main__":
    main()
