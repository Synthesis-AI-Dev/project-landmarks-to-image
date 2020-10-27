import concurrent.futures
import itertools
import json
import logging
from pathlib import Path

import cv2
import hydra
import numpy as np
from omegaconf import OmegaConf, DictConfig
from tqdm import tqdm
import open3d as o3d


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
    landmarks = [OmegaConf.to_container(landmark.camera_space_pos) for landmark in metadata.landmarks]
    landmarks = np.array(landmarks)  # Shape: (68, 3)

    # Construct pointcloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(landmarks)
    out_filename = dir_output / f"{f_img.stem}.landmarks.ply"
    o3d.io.write_point_cloud(str(out_filename), pcd)


@hydra.main(config_path='.', config_name='config')
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
        raise ValueError(f'Not a directory: {dir_input}')
    log.info(f'Input Dir: {dir_input}')

    if cfg.dir.output is None:
        dir_output = dir_input
    else:
        dir_output = Path(cfg.dir.output)
        if not dir_output.exists():
            dir_output.mkdir(parents=True)
    log.info(f'Output Dir: {dir_output}')

    ext_input = cfg.file_ext.input
    log.info(f'RGB img File Ext: {ext_input}')

    ext_depth = cfg.file_ext.depth
    log.info(f'Depth img File Ext: {ext_depth}')

    ext_info = cfg.file_ext.info
    ext_info_type = ext_info.split('.')[-1]
    if ext_info_type != 'json':
        raise ValueError(f'Unsupported filetype: {ext_info_type}. Info files must be of type json')

    info_filenames = sorted(dir_input.glob('*' + ext_info))
    num_json = len(info_filenames)
    log.info(f'Num Info Files: {num_json}')
    if num_json < 1:
        raise ValueError(f'No info json files found. Searched:\n'
                         f'  dir: "{dir_input}"\n'
                         f'  file extention: "{ext_info}"')

    input_filenames = sorted(dir_input.glob('*' + ext_input))
    num_images = len(input_filenames)
    log.info(f'Num Input Files: {num_images}')
    if num_images != num_json:
        raise ValueError(f'Unequal number of json files ({num_json}) and '
                         f'rgb images ({num_images}) in dir: "{dir_input}"')

    depth_filenames = sorted(dir_input.glob('*' + ext_depth))
    num_images = len(depth_filenames)
    log.info(f'Num Depth Files: {num_images}')
    if num_images != num_json:
        raise ValueError(f'Unequal number of json files ({num_json}) and '
                         f'depth images ({num_images}) in dir: "{dir_input}"')

    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        with tqdm(total=len(info_filenames)) as pbar:
            for _ in executor.map(_process_file, info_filenames, input_filenames, depth_filenames,
                                  itertools.repeat(dir_output)):
                # Catch any error raised in processes
                pbar.update()


if __name__ == "__main__":
    main()
