import concurrent.futures
import itertools
import json
import logging
from pathlib import Path
from typing import List, Sequence

import cv2
import hydra
import numpy as np
from omegaconf import OmegaConf, DictConfig
from tqdm import tqdm


def apply_transformation_to_point(transform_a2b: np.ndarray, point_a: list) -> list:
    """This function transforms a point from reference frame A to reference frame B
    Args:
        transform_a2b (numpy.ndarray): The 4x4 transformation matrix that defines the transform from ref
                                       frame A to ref frame B.
        point_a (list or tuple): A point (x,y,z) in ref frame A.
    Returns:
        list: point_a in ref frame B. Length: 3
    """
    if not (isinstance(point_a, list) or isinstance(point_a, tuple)):
        raise ValueError('Input point must be a list/tuple. Given: ', type(point_a))
    if len(point_a) != 3:
        raise ValueError('Input point must be a list/tuple of length 3. Given: ', point_a)
    if not isinstance(transform_a2b, np.ndarray):
        raise ValueError('Input transform must be numpy ndarray. Given: ', type(transform_a2b))
    if transform_a2b.shape != (4, 4):
        raise ValueError('Input transform must be of shape (4, 4). Given: ', transform_a2b.shape)

    point_a = np.array(point_a, dtype=np.float64)
    # Add homogenous coordinate to enable multiplication with transform
    point_a = np.append(point_a, 1.0)
    # Apply transformation
    point_b = np.dot(transform_a2b.astype(np.float64), point_a)
    # Discard homogenous coord
    point_b = point_b / point_b[-1]
    point_b = point_b[:3]

    return point_b.astype(np.float32).tolist()


def project_point_to_pixel(cam_intr: np.ndarray, point: List[float]) -> np.ndarray:
    """Projects a point in camera space to pixels

    Args:
        cam_intr (numpy.ndarray): Camera intrinsics matrix
        point (list[float]): A point in camera space in format [X, Y, Z]

    Returns:
        numpy.ndarray: The pixel coordinates the point projects to
    """
    # Convert from Houdini's camera space to computer vision camera space
    # Blender camera -> Y: up,   negative-Z: fwd
    # CV camera      -> Y: down,          Z: fwd
    rx_blender2cv = np.array([[1, 0, 0, 0],
                             [0, -1, 0, 0],
                             [0, 0, -1, 0],
                             [0, 0, 0, 1]])
    point_cam = apply_transformation_to_point(rx_blender2cv, point)
    point_cam = np.array(point_cam)

    point_px = cam_intr @ point_cam  # Project to pixels
    point_px = point_px / point_px[-1]  # Normalize pixel coordinate
    point_px = point_px[:2]  # Discard homogenous coordinate

    return point_px


def draw_landmarks(img_in: np.ndarray, landmarks: Sequence[np.ndarray]) -> np.ndarray:
    """Draw a circle for each landmark on an image for visualization"""
    img = img_in.copy()
    for landmark in landmarks:
        x, y = landmark.round().astype(np.int)
        img = cv2.circle(img, (x, y), radius=2, color=(0, 0, 255), thickness=-1)

    return img


def _process_file(f_json: Path, f_img: Path, dir_output: Path):
    """Project facial landmarks on rgb image and save output visualization
    Args:
        f_json (Path): Json file containing camera intrinsics
        f_img (Path): Image to distort
        dir_output (Path): Which dir to store outputs in

    Note:
        The camera space co-ordinates for points is in the Blender camera notation: Y: up, negative-Z: fwd
        To cast to screen space, we convert them to computer vision camera notation: Y: down, Z: fwd
    """
    # Load RGB image and image metadata
    img = cv2.imread(str(f_img), cv2.IMREAD_UNCHANGED | cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
    with f_json.open() as json_file:
        metadata = json.load(json_file)
        metadata = OmegaConf.create(metadata)

    # Extract all landmarks in camera space as a list of lists
    landmarks = [OmegaConf.to_container(landmark.camera_space_pos) for landmark in metadata.landmarks]

    # Project each landmark to pixel
    cam_intr = np.array(metadata.camera.intrinsics, dtype=np.float32)
    landmarks = [project_point_to_pixel(cam_intr, landmark) for landmark in landmarks]

    # Visualize each landmark
    rgb_viz = draw_landmarks(img, landmarks)

    # Save Result
    out_filename = dir_output / f"{f_img.stem}.landmarks{f_img.suffix}"
    retval = cv2.imwrite(str(out_filename), rgb_viz)
    if not retval:
        raise RuntimeError(f'Error in saving file {out_filename}')


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
    log.info(f'Input File Ext: {ext_input}')

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
    if num_images < 1:
        raise ValueError(f'No images found. Searched:\n'
                         f'  dir: "{dir_input}"\n'
                         f'  file extention: "{ext_input}"')

    if num_images != num_json:
        raise ValueError(f'Unequal number of json files ({num_json}) and images ({num_images}) in dir: "{dir_input}"')

    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        with tqdm(total=len(info_filenames)) as pbar:
            for _ in executor.map(_process_file, info_filenames, input_filenames, itertools.repeat(dir_output)):
                # Catch any error raised in processes
                pbar.update()


if __name__ == "__main__":
    main()
