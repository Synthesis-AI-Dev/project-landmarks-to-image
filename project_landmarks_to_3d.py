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


def _process_file(f_json: Path, f_img: Path, f_depth, dir_output: Path):
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
    # img = cv2.imread(str(f_img), cv2.IMREAD_COLOR)
    # depth_img = cv2.imread(str(f_img), cv2.IMREAD_UNCHANGED | cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
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
    depth_pxs = depth.reshape((-1))  # Shape: [N]
    depth_pxs[np.isinf(depth_pxs)] = 0.0
    depth_pxs[np.isnan(depth_pxs)] = 0.0
    rgb_pxs = rgb.reshape((-1, 3)).astype(np.float32) / 255.0  # Convert to [0,1] range. Shape: [N, 3]
    valid_pxs = ~(depth_pxs < 1e-6)  # Depth with value of 0 is not needed to construct ptcloud

    # Cast to 3D, scale by depth.
    intrinsics = np.array(metadata.camera.intrinsics, dtype=np.float32)
    intrinsics_inv = np.linalg.inv(intrinsics)
    img_pts = (intrinsics_inv @ img_pxs.T).T  # Cast to 3D, depth=1. Shape: [N, 3]
    img_pts[:, 2] *= depth_pxs  # Scale the depth
    # Filter valid pxs
    img_pts = img_pts[valid_pxs, :]
    rgb_pxs = rgb_pxs[valid_pxs, :]

    # Convert to houdini coordinate system: x: right, y: up, z: behind.
    # Projecting to intrinsics converts to camera coord system of x: right, y: down, z: forward
    # This corresponds to 180 deg rot around x-axis
    r = R.from_euler('x', 180, degrees=True)
    rot_mat = r.as_matrix()
    img_pts = (rot_mat @ img_pts.T).T

    # Add the landmarks to points. Make them red in color
    img_pts_with_landmarks = np.concatenate([img_pts, landmarks_cam], axis=0)
    col_red = np.zeros(landmarks_cam.shape, dtype=np.float32)
    col_red[:, 0] = 1.0
    rgb_with_landmarks = np.concatenate([rgb_pxs, col_red], axis=0)

    # construct pointcloud of face
    pcd_face = o3d.geometry.PointCloud()
    pcd_face.points = o3d.utility.Vector3dVector(img_pts_with_landmarks)
    pcd_face.colors = o3d.utility.Vector3dVector(rgb_with_landmarks)
    # downsample and estimate normals
    pcd_face = pcd_face.voxel_down_sample(voxel_size=0.005)
    pcd_face.estimate_normals()
    pcd_face.orient_normals_towards_camera_location()

    # construct mesh of face
    radii = [0.005, 0.01]
    rec_mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
        pcd_face, o3d.utility.DoubleVector(radii))
    o3d.visualization.draw_geometries([pcd_face, rec_mesh])

    out_filename = dir_output / f"{f_img.stem}.face.ply"
    o3d.io.write_point_cloud(str(out_filename), pcd_face)

    # Construct pointcloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(landmarks_cam)
    out_filename = dir_output / f"{f_img.stem}.landmarks.ply"
    o3d.io.write_point_cloud(str(out_filename), pcd)


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

    # with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
    #     with tqdm(total=len(info_filenames)) as pbar:
    #         for _ in executor.map(
    #             _process_file, info_filenames, rgb_filenames, depth_filenames, itertools.repeat(dir_output)
    #         ):
    #             # Catch any error raised in processes
    #             pbar.update()
    img_idx = 14
    _process_file(info_filenames[img_idx], rgb_filenames[img_idx], depth_filenames[img_idx], dir_output)


if __name__ == "__main__":
    main()
