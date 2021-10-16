#!/usr/bin/env python3

# pylint: disable=line-too-long,logging-format-interpolation,no-member

"""
Motion and object detection with OpenCV.

With much help from https://www.pyimagesearch.com/ tutorials and cvlib.

Caches images for a few frames before and after it detects movement.
"""

r"""
# TODO: fix --ignore-drive, it's broken

# TODO: treat /mnt/d/some/path/ the same as D:\some\path for --ignore-drive

# TODO: have args as provided by argparse take priority over those in the config (currently it is vv)

# TODO: add other output streams - not just to files, to cloud, sFTP server or email

# TODO: for hue color change detection, ignore pixels that clip to black/white or are pure gray (no hue)

# TODO: multiprocess progress bars - one for each process

# TODO: stack extra output frames

# TODO: object detection only in areas of frame near bounding boxes for motion

# TODO: allow drawing areas on frame to specify masks

# TODO: add play/pause/stop and other playback controls with keyboard shortcuts
"""

import sys
import os

import warnings
warnings.filterwarnings(action="ignore", message=r"Passing \(type, 1\) or '1type' as a synonym of type is deprecated", category=FutureWarning, module=".*")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import signal
import time
from datetime import datetime
import math

from argparse import ArgumentParser, Namespace
from configparser import ConfigParser
from pathlib import Path
from packaging import version
from ast import literal_eval
import json
from jsonschema import validate
from time import strptime

# to get YOLOv4 data
try:
    import importlib.resources as importlib_resources
except ImportError:
    # Try backported to PY<37 `importlib_resources`.
    import importlib_resources  # type: ignore
from . import data

from typing import List, Dict, Any, Union, Optional, Tuple, Deque, Set, Callable, Iterable, IO

from collections import deque
import copy

from functools import partial
from multiprocessing import Pool, Event, Pipe
from multiprocessing.connection import Connection
import progressbar
from progressbar import ProgressBar
from .DummyProgressBar import DummyProgressBar

from mem_top import mem_top
from orderedset import OrderedSet

from numpy import array as np_array
from numpy import int32 as np_int32
from numpy import ndarray as np_ndarray
from numpy import float32 as np_float32
from numpy import ones as np_ones

import cv2
import imutils
import cvlib as cv

import logging
import traceback


# pylint: disable=invalid-name
log = logging.getLogger(__name__)
log.setLevel(logging.INFO)
# pylint: enable=invalid-name

LINE_BUFFERED: int = 1

DUMMY_PROGRESS_BAR: DummyProgressBar = DummyProgressBar()

# Color constants
BLACK = (0x00, 0x00, 0x00)
RED = (0x00, 0x00, 0xFF)
GREEN = (0x00, 0xFF, 0x00)
BLUE = (0xFF, 0x00, 0x00)
CYAN = (0x7F, 0x7F, 0x00)

MASK_SCHEMA = {
    "type": "array",
    "items": {
        "type": "array",
        "minItems": 2,
        "items": {
            "type": "array",
            "minItems": 2,
            "maxItems": 2,
            "items": {
                "type": "integer"
            }
        }
    }
}

FIND_MOTION_PATH = os.path.dirname(os.path.realpath(__file__))

CASCADE_LOOKUP = {
    "frontalcatface": "Cat 1",
    "frontalcatface_extended": "Cat 2",
    "frontalface_alt": "Face 1",
    "frontalface_alt2": "Face 2",
    "frontalface_alt_tree": "Face 3",
    "frontalface_default": "Face 4",
    "fullbody": "Person",
    #   "lefteye_2splits": "Eye 1",
    #   "licence_plate_rus_16stages": "Car number plate 1",
    "lowerbody": "Legs",
    #   "eye": "Eye 2",
    #   "eye_tree_eyeglasses": "Glasses",
    "profileface": "Face 5",
    #   "righteye_2splits": "Eye 3",
    #   "russian_plate_number": "Car number plate 2",
    #   "smile": "Smile",
    #   "upperbody": "Torso"
}

unpaused = Event()


def init_worker(event) -> None:
    """
    Suppress signal handling in the worker processes so that they don't capture SIGINT (ctrl-c).

    Set an event so we can pause workers.
    """
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    global unpaused
    unpaused = event


class VideoError(Exception):
    """
    An error when processing a video.
    """
    def __init__(self, _msg) -> None:
        super()


class VideoInfo(object):
    """
    Class to read in a video, and get metadata out
    """
    def __init__(self, filename: str=None, log_level=logging.INFO) -> None:
        self.filename: Optional[str] = filename
        self.cap: cv2.VideoCapture = None
        self.amount_of_frames: int = 0
        self.frame_width: int = 0
        self.frame_height: int = 0

        self.log = logging.getLogger('find_motion.VideoInfo')
        self.log.setLevel(log_level)

        self.loaded = self._load_video()

    def __str__(self) -> str:
        return "File: {}; {} frames; {}x{}px".format(self.filename, self.amount_of_frames, self.frame_width, self.frame_height)

    def _load_video(self) -> bool:
        """
        Open the input video file, get the video info.
        """
        self.cap = cv2.VideoCapture(self.filename)
        try:
            self._get_video_info()
        except VideoError as e:
            self.log.error(str(e))
            return False
        return True

    def _get_video_info(self) -> None:
        """
        Set some metrics from the video.
        """
        self.log.debug(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.amount_of_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        if self.amount_of_frames == 0 or self.frame_width == 0 or self.frame_height == 0:
            broken = 'frames' if self.amount_of_frames == 0 else 'height/width'
            raise VideoError("Video info malformed - {} is 0: {}".format(broken, self.filename))
        return


class VideoFrame(object):
    """
    Encapsulates a single video frame, and the processing done to it.
    """
    def __init__(self, frame,
                 show: bool, show_extras: bool, gaussian: Tuple[int, int],
                 mask_areas: List[Any], scale: float = 1.0,
                 threshold: int = 0, box_size: int = 50,
                 no_shade: bool = False, no_hue: bool = False, no_edges: bool = False) -> None:
        self.raw: np_ndarray = frame
        self.show: bool = show
        self.show_extras: bool = show_extras
        self.gaussian: Tuple[int, int] = gaussian
        self.mask_areas = mask_areas
        self.scale: float = scale
        self.threshold_value = threshold
        self.box_size = box_size

        self.no_shade: bool = no_shade
        self.no_hue: bool = no_hue
        self.no_edges: bool = no_edges

        self.frame: Optional[np_ndarray] = self.raw.copy() if self.show else None
        self.in_cache: bool = False

        self.contours: List = []
        self.color_contours: List = []
        self.edges_contours: List = []

        self.frame_delta: np_ndarray = None
        self.color_delta: np_ndarray = None
        self.edges_delta: np_ndarray = None

        self.mini: np_ndarray = None
        self.hue: np_ndarray = None
        self.mini_blur: np_ndarray = None
        self.gray: np_ndarray = None
        self.gray_blur: np_ndarray = None
        self.resized: np_ndarray = None
        self.edges: np_ndarray = None

        self.thresh: np_ndarray = None
        self.color_thresh: np_ndarray = None
        self.edges_thresh: np_ndarray = None

    def find_edges(self):
        """Find edges in a frame."""
        self.edges = cv2.convertScaleAbs(cv2.Laplacian(self.mini, cv2.CV_32F))

    def make_hue(self) -> None:
        """Make the hue frame."""
        hsv = cv2.cvtColor(self.mini_blur, cv2.COLOR_BGR2HSV)
        self.hue = hsv[:, :, 0]

    def diff(self, ref_frame: np_ndarray, ref_color: np_ndarray, ref_edges: np_ndarray) -> None:
        """Find the diff between this frame and the reference frame."""
        if not self.no_shade:
            self.frame_delta = cv2.absdiff(self.gray_blur, ref_frame)
        if not self.no_hue:
            self.color_delta = cv2.absdiff(self.hue, ref_color)
        if not self.no_edges:
            self.edges_delta = cv2.cvtColor(cv2.absdiff(self.edges, ref_edges), cv2.COLOR_BGR2GRAY)

    def threshold(self) -> None:
        """Find the threshold of the diff."""
        if not self.no_shade:
            self.thresh = cv2.threshold(self.frame_delta, thresh=self.threshold_value, maxval=255, type=cv2.THRESH_BINARY)[1]
        if not self.no_hue:
            self.color_thresh = cv2.threshold(self.color_delta, thresh=self.threshold_value * 2, maxval=255, type=cv2.THRESH_BINARY)[1]
        if not self.no_edges:
            self.edges_thresh = cv2.threshold(self.edges_delta, thresh=self.threshold_value * 2, maxval=255, type=cv2.THRESH_BINARY)[1]

    def find_contours(self) -> None:
        """
        Find edges of the shapes in the thresholded image.

        Dilate the thresholded grayscale and edges images to fill in holes, then find contours.
        """

        if not self.no_shade:
            self.thresh = cv2.dilate(self.thresh, kernel=None, iterations=2)

            try:
                cnts, _hierarchy = cv2.findContours(
                    self.thresh, mode=cv2.RETR_EXTERNAL,
                    method=cv2.CHAIN_APPROX_SIMPLE
                )[-2:]
            except Exception as e:
                log.error(str(e))
                return
            self.contours = cnts

        if not self.no_hue:
            try:
                color_cnts, _hierarchy = cv2.findContours(
                    self.color_thresh, mode=cv2.RETR_EXTERNAL,
                    method=cv2.CHAIN_APPROX_SIMPLE
                )[-2:]
            except Exception as e:
                log.error(str(e))
                return
            self.color_contours = color_cnts

        if not self.no_edges:
            self.thresh = cv2.dilate(self.edges_thresh, kernel=None, iterations=2)

            try:
                edges_cnts, _hierarchy = cv2.findContours(
                    self.edges_thresh, mode=cv2.RETR_EXTERNAL,
                    method=cv2.CHAIN_APPROX_SIMPLE
                )[-2:]
            except Exception as e:
                log.error(str(e))
                return
            self.edges_contours = edges_cnts

    def make_mini(self) -> None:
        """Make a small version of the frame."""
        if self.mini is None:
            try:
                self.mini = imutils.resize(self.raw, width=self.box_size)
            except cv2.error as cv_err:
                log.error("Error resizing %s with width %d: %s", self.raw.shape, self.box_size, cv_err)

    def blur_frame(self) -> None:
        """Blur the gray scale and hue frames."""
        if not self.no_hue:
            self.mini_blur = cv2.GaussianBlur(self.mini, self.gaussian, 0)
        if not self.no_shade:
            self.gray = cv2.cvtColor(self.mini, cv2.COLOR_BGR2GRAY)
            self.gray_blur = cv2.GaussianBlur(self.gray, self.gaussian, 0)

    def mask_off_areas(self) -> None:
        """Mask off the gray scale, hue and edge frames."""
        if not self.no_shade:
            self.mask_off_frame(self.gray_blur, self.scale)
        if not self.no_hue:
            self.mask_off_frame(self.mini_blur, self.scale)
        if not self.no_edges:
            self.mask_off_frame(self.edges, self.scale)

    def mask_off_frame(self, frame: np_ndarray, scale: float) -> None:
        for area in self.mask_areas:
            scaled_area = VideoMotion.scale_area(area, scale)
            dim = len(scaled_area)
            if dim == 2:
                cv2.rectangle(frame,
                              *scaled_area,
                              BLACK, cv2.FILLED)
            else:
                pts = np_array(scaled_area, np_int32)
                cv2.fillConvexPoly(frame,
                                   pts,
                                   BLACK)

    def cleanup(self) -> None:
        """
        Actively destroy the frame.
        """
        log.debug('Cleanup frame')
        for attr in ('frame', 'thresh', 'contours', 'frame_delta', 'blur'):
            if hasattr(self, attr):
                delattr(self, attr)
        if self.in_cache:
            log.debug('Frame in cache, not cleaning up up raw frame or deleting frame object')
            return
        if hasattr(self, 'raw'):
            del self.raw


class VideoMotion(object):
    """
    Read in a video, detect motion in it, and write out just the motion to a new file.

    Detect any objects present when motion is seen.
    """
    # pylint: disable=too-many-instance-attributes,too-many-arguments
    def __init__(self, filename: Union[str, int],
                 connection: Optional[Connection]=None,
                 outdir: str='', fps: int=30,
                 box_size: int=100, min_box_scale: int=50, cache_time: float=3.0, min_time: float=0.25,
                 threshold: int=7, avg: float=0.05, blur_scale: int=20,
                 mask_areas: list=None, show: bool=False, show_extras: bool=False,
                 codec: str='MJPG', log_level: int=logging.INFO,
                 mem: bool=False, cleanup: bool=False,
                 multiprocess: bool=False,
                 cascades: List[str]=None,
                 haarcascades_path: str="/usr/local/share/opencv4/haarcascades",
                 yolov3: bool=False, yolov4: bool=False, yolo_tiny: bool=False,
                 confidence: float=0.25,
                 yolo_path: Optional[str]=None,
                 no_object_detection: bool=False,
                 always_object_detection: bool=False,
                 object_detect_frame_interval: int=10,
                 no_shade: bool=False, no_hue: bool=False, no_edges: bool = False,
                 no_output: bool=False) -> None:
        self.filename = filename
        self.connection = connection

        if self.filename is None:
            raise Exception('Filename required')

        self.log = logging.getLogger('find_motion.VideoMotion')
        self.log.setLevel(log_level)

        self.log.debug("Reading from {}".format(self.filename))

        self.multiprocess = multiprocess

        if not self.multiprocess:
            log.debug('Single process')

        self.outfile: cv2.VideoWriter = None    # type: ignore
        self.outfiles: int = 0
        self.outfile_name: str = ''
        self.outdir: str = os.path.normpath(outdir)

        self.no_shade: bool = no_shade
        self.no_hue: bool = no_hue
        self.no_edges: bool = no_edges
        self.motion_types_count: int = sum([not self.no_shade, not self.no_hue, not self.no_edges])

        self.fps: int = fps
        self.box_size: int = box_size
        self.min_box_scale: int = min_box_scale
        self.min_area: int = -1
        self.max_area: int = -1
        self.gaussian_scale: int = blur_scale
        self.cache_frames = int(cache_time * fps)
        self.min_movement_frames: int = int(min_time * fps)
        self.delta_thresh: int = threshold
        self.avg: float = avg
        self.mask_areas: List[Any] = mask_areas if mask_areas is not None else []
        self.show: bool = show
        self.show_extras: bool = show_extras
        self.no_output: bool = no_output

        self.log.debug('Caching {} frames, min motion {} frames'.format(self.cache_frames, self.min_movement_frames))

        self.no_object_detection: bool = no_object_detection
        self.always_object_detection: bool = always_object_detection
        self.object_detect_frame_interval: int = object_detect_frame_interval
        self.cascade_names: Optional[List[str]] = cascades if not self.no_object_detection else []
        self.haarcascades_path: str = haarcascades_path
        self.yolov3: bool = yolov3
        self.yolov4: bool = yolov4
        self.tiny: bool = yolo_tiny
        self.confidence: float = confidence
        self.yolo_path: Optional[str] = yolo_path
        self.names: Optional[List] = None
        self.cfg: Optional[str] = None
        self.weights: Optional[str] = None
        self.net: Optional[Any] = None

        if version.parse(cv2.__version__) <= version.parse("4.3") and self.yolov4:
            self.yolov4 = False
            self.yolov3 = True
            self.log.warning("Using YOLOv3")
            self.log.warning("Cannot use YOLOV4, OpenCV2 needs to be at 4.4.0 or above")
            self.log.warning("Try building from source if you rely on the opencv-python package from pypi, and that is not yet at that level, or update the package with pip")

        if self.yolov4:
            self._prepare_yolov4()

        self.cascades: Optional[Dict[str, Any]] = None
        self._load_cascades()
        self.log.debug(str(self.cascades))

        self.codec: str = codec
        self.debug: bool = log_level == logging.DEBUG
        self.mem: bool = mem
        self.cleanup_flag: bool = cleanup

        self.log.debug(self.codec)

        # initialised in _load_video
        self.amount_of_frames: int = -1
        self.frame_width: int = -1
        self.frame_height: int = -1
        self.scale: float = -1.0

        self.current_frame: VideoFrame
        self.ref_frame: np_ndarray = None
        self.ref_scaled: np_ndarray = None
        self.ref_color: np_ndarray = None
        self.ref_color_scaled: np_ndarray = None
        self.ref_edges: np_ndarray = None
        self.ref_edges_scaled: np_ndarray = None
        self.frame_cache: Deque[VideoFrame]

        self.wrote_frames: Optional[bool] = False
        self.err_msg: str = ''

        self.movement: bool = False
        self.movement_decay: int = 0
        self.movement_counter: int = 0

        self.object_counter: int = 0
        self.last_objects: Dict[str, List] = {}
        self.seen_objects: Set[str] = set()

        # Initialisation functions
        self._calc_min_area()
        self._make_gaussian()
        self.loaded = self._load_video()

    def _load_cascades(self) -> None:
        """
        Load object recognition cascades named in options.
        """
        if self.cascade_names is not None:
            if 'ALL' in self.cascade_names:
                cascades = copy.copy(CASCADE_LOOKUP)
            else:
                cascades = {c: CASCADE_LOOKUP[c] for c in self.cascade_names if c in CASCADE_LOOKUP}

            self.log.debug(str(cascades))
            for cascade, title in cascades.items():
                self.log.debug('{}: {}'.format(title, os.path.join(self.haarcascades_path, 'haarcascade_{}.xml'.format(cascade))))
            self.cascades = {title: cv2.CascadeClassifier(os.path.join(self.haarcascades_path, 'haarcascade_{}.xml'.format(cascade))) for cascade, title in cascades.items()}
        else:
            self.log.debug('No cascades')
            self.cascades = dict()

    def _prepare_yolov4(self) -> None:
        """
        Read in config, weights and names, and set up model
        """
        # TODO: download and cache yolov4.cfg and yolov4.weights from a suitable place
        # TODO: also do that for YOLOv3, merge functions - and let user specify the prefix for the model (e.g. "yolo3")
        # TODO: allow a different name for the names file
        # use built-in cfg/weights unless a path is provided
        try:
            cfg_file = f"yolov4{'-tiny' if self.tiny else ''}.cfg"
            weights_file = f"yolov4{'-tiny' if self.tiny else ''}.weights"
            names_file = "coco.names"
            if self.yolo_path:
                cfg_path = Path(self.yolo_path)
                self.cfg = cfg_path.joinpath(cfg_file).as_posix()  # type: ignore
                self.weights = cfg_path.joinpath(weights_file).as_posix()  # type: ignore
                names_file_path = cfg_path.joinpath(names_file)
            else:
                with importlib_resources.path(data, cfg_file) as cfg_file_path:  # type: ignore
                    self.cfg = cfg_file_path.as_posix()
                with importlib_resources.path(data, weights_file) as weights_file_path:  # type: ignore
                    self.weights = weights_file_path.as_posix()
                with importlib_resources.path(data, names_file) as _names_file_path:  # type: ignore
                    names_file_path = _names_file_path.as_posix()
        except Exception as err:
            self.log.error("Failed to read in YOLOv4 configuration: %s", err)
            self.yolov4 = False
            return

        self.names = [n for n in open(names_file_path, "r", encoding="utf-8").read().split('\n') if n is not None and n != ""]  # type: ignore

        # set up the detection network
        try:
            self.net = cv2.dnn_DetectionModel(self.cfg, self.weights)
            self.net.setInputSize(608, 608)
            self.net.setInputScale(1.0 / 255)
            self.net.setInputSwapRB(True)
        except Exception as err:
            self.log.error("Failed to set up YOLOv4 model: %s", err)
            self.yolov4 = False
            return

    def _calc_min_area(self) -> None:
        """
        Set the minimum motion area based on the box size.
        """
        self.min_area = int(math.pow(self.box_size / self.min_box_scale, 2))

    def _load_video(self) -> bool:
        """
        Open the input video file, set up the ref frame and frame cache, get the video info and set the scale
        """
        self.cap = cv2.VideoCapture(self.filename)
        self.frame_cache = deque(maxlen=self.cache_frames)

        try:
            self._get_video_info()
        except VideoError as e:
            self.log.error(str(e))
            return False
        self.scale = self.box_size / self.frame_width
        self.max_area = int((self.frame_width * self.frame_height) / 2 * self.scale)
        return True

    def _get_video_info(self) -> None:
        """
        Set some metrics from the video
        """
        self.log.debug(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))

        self.amount_of_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        if self.frame_width == 0 or self.frame_height == 0:
            broken = 'width' if self.frame_width == 0 else 'height'
            raise VideoError("Video info malformed - {} is 0: {}".format(broken, self.filename))
        if self.amount_of_frames == 0:
            log.warning('Video info malformed - frames reported as 0')
        elif self.amount_of_frames == -1:
            log.debug('Streaming - frames reported as -1')
        return

    def _make_outfile(self) -> None:
        """
        Create an output file based on the input filename and the output directory
        """
        if self.no_output:
            return

        self.outfiles += 1

        if self.outfiles > 1 and self.outfile is not None:
            self.outfile.release()

        outname = str(self.filename) + '_' + str(self.outfiles)

        if self.outdir == '':
            self.outfile_name = outname + '_motion.avi'
        else:
            self.outfile_name = os.path.join(self.outdir,
                                             os.path.basename(outname)) + '_motion.avi'

        if self.debug:
            self.log.debug("Writing to {}".format(self.outfile_name))

        try:
            self.outfile = cv2.VideoWriter(self.outfile_name,
                                           cv2.VideoWriter_fourcc(*self.codec),
                                           self.fps, (self.frame_width, self.frame_height))
        except Exception as e:
            self.log.error('Failed to create output file: {}'.format(e))
            raise e

        self.log.debug('Made output file')

    def _make_gaussian(self) -> None:
        """
        Make a gaussian for the blur using the box size as a guide.
        """
        gaussian_size = int(self.box_size / self.gaussian_scale)
        gaussian_size = gaussian_size + 1 if gaussian_size % 2 == 0 else gaussian_size
        self.gaussian = (gaussian_size, gaussian_size)

    def make_mini(self, frame=None) -> None:
        """
        Make a small version of the frame.
        """
        frame = self.current_frame if frame is None else frame
        frame.make_mini()

    def find_edges(self, frame=None) -> None:
        """
        Find edges in the frame.
        """
        frame = self.current_frame if frame is None else frame
        frame.find_edges()

    def blur_frame(self, frame=None) -> None:
        """
        Shrink, grayscale and blur the frame
        """
        frame = self.current_frame if frame is None else frame
        frame.blur_frame()

    def read(self) -> bool:
        """
        Read a frame from the capture member
        """
        (ret, frame) = self.cap.read()
        if not ret:
            return False

        self.current_frame = VideoFrame(frame, self.show, self.show_extras, self.gaussian,
                                        self.mask_areas, self.scale,
                                        self.delta_thresh, self.box_size,
                                        self.no_shade, self.no_hue)
        return True

    def output_frame(self, frame: VideoFrame=None) -> None:
        """
        Put a frame out to screen (if required) and file

        Initialise the output file if necessary
        """
        frame = self.current_frame if frame is None else frame

        if self.show:
            self.log.debug('Showing frame on output')
            cv2.imshow('frame', frame.frame)

        if not self.wrote_frames:
            self._make_outfile()
            self.wrote_frames = True

        if not self.no_output:
            try:
                self.outfile.write(frame.raw)
            except Exception as e:
                self.log.warning('Having to create output file due to exception: {}'.format(e))
                self._make_outfile()
                self.outfile.write(frame.raw)

    def output_raw_frame(self, frame: np_ndarray=None) -> None:
        """
        Output a raw frame, not a VideoFrame
        """
        if not self.wrote_frames:
            self._make_outfile()
            self.wrote_frames = True

        if not self.no_output:
            try:
                self.outfile.write(frame)
            except Exception as e:
                self.log.warning('Having to create output file due to exception: {}'.format(e))
                self._make_outfile()
                self.outfile.write(frame)


    def decide_output(self) -> None:
        """
        Decide if we are going to put out this frame
        """
        self.log.debug('Deciding output')

        if (self.movement_counter >= self.min_movement_frames) or (self.movement_decay > 0):
            self.log.debug('There is movement')
            # reset counter
            self.movement_counter = 0
            # show cached frames
            if self.movement:
                self.movement_decay = self.cache_frames

                for frame in self.frame_cache:
                    if frame is not None:
                        self.log.debug('Outputting cached raw frame')
                        self.output_raw_frame(frame.raw)
                        if self.cleanup_flag:
                            frame.in_cache = False
                            frame.cleanup()
                            del frame

                self.frame_cache.clear()

            # identify objects
            if self.movement and not self.no_object_detection or self.always_object_detection:
                objects = self.find_objects()   # TODO: pass args to set params
                if objects is not None and objects:
                    self.log.debug("Saw {} in motion".format(objects))
                    self.seen_objects.update(objects)

            # draw the text
            if self.show:
                self.draw_text()

            self.log.debug('Outputting frame')

            self.output_frame()
        else:
            self.log.debug('No movement, putting in cache')
            if self.cleanup_flag:
                self.cleanup_cache()
            self.frame_cache.append(self.current_frame)
            self.current_frame.in_cache = True

    def cleanup_cache(self) -> None:
        cache_size = len(self.frame_cache)
        self.log.debug(str(cache_size))
        if cache_size == self.cache_frames:
            self.log.debug('Clearing first cache entry')
            delete_frame = self.frame_cache.popleft()
            if delete_frame is not None:
                delete_frame.in_cache = False
                delete_frame.cleanup()
                del delete_frame

    def is_open(self) -> bool:
        """
        Say if the capture member is open.
        """
        return self.cap.isOpened()

    @staticmethod
    def scale_area(area: Tuple[Tuple[int, int], Tuple[int, int]], scale: float) -> List:
        """
        Scale the area by the scale factor.
        """
        return [(int(a[0] * scale), int(a[1] * scale)) for a in area]

    def make_hue(self, frame: VideoFrame=None) -> None:
        """
        Make a record of the hue of the frame.
        """
        frame = self.current_frame if frame is None else frame
        frame.make_hue()

    def mask_off_areas(self, frame: VideoFrame=None) -> None:
        """
        Draw black polygons over the masked off areas.
        """
        frame = self.current_frame if frame is None else frame
        frame.mask_off_areas()

    def find_diff(self, frame: VideoFrame=None) -> None:
        """
        Find the difference between this frame and the moving average.

        Locate the contours around the thresholded difference

        Update the moving average
        """
        frame = self.current_frame if frame is None else frame

        if frame.gray_blur is None and frame.hue is None and frame.edges is None:
            raise Exception("Blur and hue and edge frames are all None")
        else:
            # Make reference frames if they don't already exist
            if self.ref_frame is None and not self.no_shade:
                self.ref_frame = frame.gray_blur.copy().astype(np_float32)
            if self.ref_color is None and not self.no_hue:
                self.ref_color = frame.hue.copy().astype(np_float32)
            if self.ref_edges is None and not self.no_edges:
                self.ref_edges = frame.edges.copy().astype(np_float32)

            # compute the absolute difference between the current frame and ref frame
            if not self.no_shade:
                self.ref_scaled = cv2.convertScaleAbs(self.ref_frame)
            if not self.no_hue:
                self.ref_color_scaled = cv2.convertScaleAbs(self.ref_color)
            if not self.no_edges:
                self.ref_edges_scaled = cv2.convertScaleAbs(self.ref_edges)
            frame.diff(self.ref_scaled, self.ref_color_scaled, self.ref_edges_scaled)

            # threshold the difference
            frame.threshold()

            # update reference frame using weighted average
            if not self.no_shade:
                cv2.accumulateWeighted(frame.gray_blur, self.ref_frame, self.avg)
            if not self.no_hue:
                cv2.accumulateWeighted(frame.hue, self.ref_color, self.avg)
            if not self.no_edges:
                cv2.accumulateWeighted(frame.edges, self.ref_edges, self.avg)

            # find contours from the diff data
            frame.find_contours()

    def find_movement(self, frame: VideoFrame=None) -> None:
        """
        Locate contours that are big enough to count as movement
        """
        frame = self.current_frame if frame is None else frame

        self.movement = False
        self.movement_decay -= 1 if self.movement_decay > 0 else 0

        if not self.no_shade and frame.contours is not None and len(frame.contours) > 0:
            self.process_contours(frame, frame.contours, GREEN)

        if not self.no_hue and frame.color_contours is not None and len(frame.color_contours) > 0:
            self.process_contours(frame, frame.color_contours, BLUE)

        if not self.no_edges and frame.edges_contours is not None and len(frame.edges_contours) > 0:
            self.process_contours(frame, frame.edges_contours, CYAN)

        if self.movement:
            self.movement_counter += 1

        return

    def process_contours(self, frame, contours, color):
        # loop over the contours
        for contour in contours:
            # if the contour is too small, or too large, ignore it
            try:
                area = cv2.contourArea(contour)
            except Exception as e:
                self.log.error(str(e))
                continue

            if area < (self.min_area * 4):
                continue

            if area > self.max_area:
                continue

            # compute the bounding box for the contour, draw it on the frame,
            # and update the text

            if self.show:
                box = self.make_box(contour, frame)
                self.draw_box(box, frame, color)

            self.movement = True

    # TODO: allow tweaking object detection settings - width, scale, neightbours and confidence
    def find_objects(self, frame: VideoFrame=None, width=300, scaleFactor=1.1, minNeighbours=5, nmsthreshold=0.5) -> Set[str]:
        frame = self.current_frame if frame is None else frame

        # TODO: can we reuse an already done resize?
        frame.resized = imutils.resize(frame.raw, width=width)
        scale: float = self.frame_width / float(width)
        frame.mask_off_frame(frame.resized, 1 / scale)

        self.object_counter += 1
        if self.object_counter != self.object_detect_frame_interval:
            if self.show:
                self.draw_objects(frame.frame, scale)
            return set()
        self.object_counter = 0

        self.log.debug('Looking for objects')

        self.last_objects = {}

        self.log.debug(str(self.cascades))

        # with opencv cascades
        if self.cascades is not None:
            for title, cascade in self.cascades.items():
                self.log.debug('Looking for {}'.format(title))
                found = cascade.detectMultiScale(frame.resized, scaleFactor=scaleFactor, minNeighbors=minNeighbours)
                if len(found) > 0:
                    if title not in self.last_objects:
                        self.last_objects[title] = []
                    for rect in found:
                        area = VideoMotion.make_area_from_rect(rect)
                        self.last_objects[title].append(area)

        # with cvlib, using yolov3
        if self.yolov3:
            model = f"yolov3{'-tiny' if self.tiny else ''}"
            self.log.debug(f'Common objects with {model}')
            bboxes, labels, _confs = cv.detect_common_objects(frame.resized, confidence=self.confidence, model=model)
            self.log.debug('Common objects: done')
            cvlib_objects = list(zip(bboxes, labels))
            for box, label in cvlib_objects:
                if label not in self.last_objects:
                    self.last_objects[label] = []
                self.last_objects[label].append(VideoMotion.make_area_from_box(tuple(box)))

        # with opencv2, v4.4.0, packaged yolov4.cfg and yolov4.weights
        # based on code snippet in https://github.com/opencv/opencv/pull/17185 that enabled support for YOLOv4 in OpenCV
        # TODO: only yolov4-tiny is working - get yolov4 working
        if self.yolov4 is not None and self.net is not None and self.names is not None:
            # do detection in this frame
            classes, confidences, boxes = self.net.detect(frame.resized, confThreshold=self.confidence, nmsThreshold=nmsthreshold)
            log.debug(f"Classes: {type(classes)} {classes}")
            log.debug(f"Confidences: {type(confidences)} {confidences}")
            log.debug(f"Boxes: {type(boxes)} {boxes}")
            for classid, _confidence, box in zip(
                    classes.flatten() if hasattr(classes, 'flatten') else [],
                    confidences.flatten() if hasattr(confidences, 'flatten') else [],
                    boxes):
                label = self.names[classid]
                if label not in self.last_objects:
                    self.last_objects[label] = []
                self.last_objects[label].append(VideoMotion.make_area_from_box_2(tuple(box)))

        if self.show:
            self.draw_objects(frame.frame, self.frame_width / float(width))

        return set(self.last_objects.keys())

    def draw_objects(self, image, scale) -> None:
        for title, areas in self.last_objects.items():
            for area in areas:
                if self.show:
                    scaled_area = VideoMotion.scale_area(area, scale)
                    cv2.rectangle(image, *scaled_area, RED, 3)
                    cv2.putText(image,
                                title,
                                VideoMotion.find_centre(scaled_area),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.5, RED, 2
                                )

                self.log.debug('{} found'.format(title))

    @staticmethod
    def find_centre(area: List) -> Tuple[int, int]:
        return (
            (area[0][0] + area[1][0]) // 2,
            (area[0][1] + area[1][1]) // 2
        )

    def draw_text(self, frame: VideoFrame=None) -> None:
        """
        Put the status text on the frame
        """
        if self.show:
            frame = self.current_frame if frame is None else frame
            # draw the text
            cv2.putText(frame.frame,
                        "Status: {}".format('motion' if self.movement else 'quiet'),
                        (10, 20),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, RED, 2)
        return

    def make_box(self, contour, frame: VideoFrame=None) -> Tuple[Tuple[int, int], Tuple[int, int]]:
        """
        Make a bounding box from a contour.
        """
        frame = self.current_frame if frame is None else frame
        return VideoMotion.make_area_from_rect(cv2.boundingRect(contour))

    @staticmethod
    def make_area_from_box(object_tuple: Tuple[Any, ...]) -> Tuple[Tuple[int, int], Tuple[int, int]]:
        # pylint: disable=invalid-name
        (left, top, right, bottom) = object_tuple
        area = ((left, top), (bottom, right))
        return area

    @staticmethod
    def make_area_from_box_2(object_tuple: Tuple[Any, ...]) -> Tuple[Tuple[int, int], Tuple[int, int]]:
        (left, top, width, height) = object_tuple
        area = ((left, top), (left + width, top + height))
        return area

    @staticmethod
    def make_area_from_rect(object_tuple: Tuple[int, int, int, int]) -> Tuple[Tuple[int, int], Tuple[int, int]]:
        # pylint: disable=invalid-name
        (x, y, w, h) = object_tuple
        area = ((x, y), (x + w, y + h))
        return area

    def draw_box(self, area, frame: VideoFrame=None, color: Tuple[int, int, int]=GREEN) -> None:
        """Draw a rectangle on the frame with the chosen color."""
        if self.show:
            frame = self.current_frame if frame is None else frame
            cv2.rectangle(frame.frame, *VideoMotion.scale_area(area, 1 / self.scale), color, 2)

    @staticmethod
    def key_pressed(key: str) -> bool:
        """Say if we pressed the key we asked for."""
        return cv2.waitKey(1) & 0xFF == ord(key)

    def cleanup(self) -> None:
        """Close the input file, output file, and get rid of OpenCV windows."""
        if self.cap is not None:
            self.cap.release()

        if self.outfile is not None:
            self.outfile.release()

        if self.cleanup_flag:
            self.current_frame.in_cache = False
            self.current_frame.cleanup()
            del self.current_frame

            del self.ref_frame

            for frame in self.frame_cache:
                if frame is not None:
                    frame.in_cache = False
                    frame.cleanup()
                    del frame
            self.frame_cache.clear()
            del self.frame_cache

        return

    def find_motion(self) -> Tuple[Optional[bool], str, Tuple[str, ...]]:
        """Main loop. Find motion in frames."""
        while self.is_open():

            # pause/unpause
            if self.multiprocess:
                self.log.debug('Waiting...')
                unpaused.wait()

            if VideoMotion.key_pressed(' '):
                if self.multiprocess:
                    if self.connection is not None:
                        # signal to main process that we are pausing
                        self.log.debug('Pausing')
                        self.connection.send('stop')
                else:
                    while True:
                        if VideoMotion.key_pressed(' '):
                            break
                        time.sleep(1)

            # reading starts
            if not self.read():
                self.log.debug('Reading did not succeed')
                break

            self.make_mini()
            self.blur_frame()
            self.mask_off_areas()
            if not self.no_hue:
                self.make_hue()
            if not self.no_edges:
                self.find_edges()
            self.find_diff()

            self.log.debug('Blurred frame, masked off, and diff made')

            # draw contours and set movement
            try:
                self.find_movement()
            except Exception as e:
                self.log.error('find_movement: {}'.format(e))

            self.log.debug('Searched for movement')

            if self.mem:
                self.log.info(mem_top())

            self.decide_output()

            self.log.debug('Decided output')

            if self.show_extras:
                self.show_frames()

            self.log.debug('Decided to show frames or not')

            self.current_frame.cleanup()

            if VideoMotion.key_pressed('q'):
                self.wrote_frames = None
                self.err_msg = 'Closing video at user request'
                break

        self.log.debug('Cleaning up video')

        self.cleanup()
#        self.frame_cache.clear()

        return self.wrote_frames, self.err_msg, tuple(self.seen_objects)

    def show_frames(self) -> None:
        if self.frame_height > 0 and self.frame_width > 0:
            try:
                cf: VideoFrame = self.current_frame
                if cf.thresh is not None:
                    self.log.debug('Showing threshold frame')
                    cv2.imshow('thresh', cf.thresh)
                if cf.gray is not None:
                    self.log.debug('Showing gray frame')
                    cv2.imshow('gray', cf.gray)
                if cf.gray_blur is not None:
                    self.log.debug('Showing blur frame')
                    cv2.imshow('blur', cf.gray_blur)
                if cf.raw is not None:
                    self.log.debug('Showing raw frame')
                    cv2.imshow('raw', cf.raw)
                if cf.hue is not None:
                    cv2.imshow("Hues", cf.hue)
                if cf.color_delta is not None:
                    cv2.imshow("color delta", cf.color_delta)
                if cf.edges_delta is not None:
                    cv2.imshow("edges delta", cf.edges_delta)
                if cf.color_thresh is not None:
                    cv2.imshow("color thresh", cf.color_thresh)
                if cf.edges_thresh is not None:
                    cv2.imshow("edges thresh", cf.edges_thresh)
                if cf.frame_delta is not None:
                    cv2.imshow("delta", cf.frame_delta)
                if cf.edges is not None:
                    cv2.imshow("edges", cf.edges)
                if cf.resized is not None:
                    cv2.imshow("resized", cf.resized)

                self.log.debug('Showing ref frames')
                if self.ref_scaled is not None:
                    cv2.imshow('avg gray', self.ref_scaled)
                if self.ref_color_scaled is not None:
                    cv2.imshow("avg color", self.ref_color_scaled)
                if self.ref_edges_scaled is not None:
                    cv2.imshow("avg edges", self.ref_edges_scaled)
            except Exception as e:
                self.log.error('Oops: {}'.format(e))
        else:
            self.log.warning('Not showing frames, height or width is 0')


def find_files(directory: str) -> List[str]:
    """
    Finds files in the directory, recursively, sorts them by last modified time
    """
    return [os.path.normpath(os.path.abspath(os.path.join(dirpath, f))) for dirpath, _dnames, fnames in os.walk(directory) for f in fnames if f != 'progress.log'] if directory is not None else []


def verify_files(file_list: List[str]) -> List[str]:
    """
    Locates files with given file names, returns in a list of tuples with their last modified time
    """
    return [os.path.normpath(os.path.abspath(f)) for f in file_list if os.path.isfile(f)]


def sort_files_by_time(file_list: List[str], priority_intervals: List[Tuple[time.struct_time, time.struct_time]]) -> "OrderedSet[str]":
    """
    Sort files by modified time.

    Take into account if a file falls in one of the priority periods, do it first.

    Method: find files in the intervals, starting with the highest priority one first.

    Go through the files, putting any matches at the start of the OrderedSet.

    Repeat until we get to the end of the intervals, then put any remaining files onto the set in order.
    """
    sorted_files: List[Tuple[str, float]] = [f for f in sorted([(f, os.path.getmtime(f)) for f in file_list], key=lambda f: f[1])]

    file_set: "OrderedSet[str]" = OrderedSet()

    for time_interval in priority_intervals:
        for vid_file in sorted_files:
            if in_interval(vid_file, time_interval) and vid_file not in file_set:
                file_set.add(vid_file)

    for vid_file in sorted_files:
        if vid_file not in file_set:
            file_set.add(vid_file)

    if log.getEffectiveLevel() == logging.DEBUG:
        for f in file_set:
            log.debug(datetime.fromtimestamp(f[1]).strftime("%H:%M:%S"))

    return file_set


def in_interval(vid_file: Tuple[str, float], time_interval: Tuple[time.struct_time, time.struct_time]) -> bool:
    """
    Check if the mtime (epoch) of the file is in the time interval given
    """
    file_time = ClockTime(time.gmtime(vid_file[1]))
    interval = (ClockTime(time_interval[0]), ClockTime(time_interval[1]))
    return interval[0] <= file_time and file_time < interval[1]


class ClockTime(object):
    def __init__(self, struct_time: time.struct_time) -> None:
        self.hour: int = struct_time.tm_hour
        self.min: int = struct_time.tm_min
        self.sec: int = struct_time.tm_sec

    def __str__(self) -> str:
        return "{:02d}:{:02d}:{:02d}".format(self.hour, self.min, self.sec)

    # typehints left out on 'other' since 'ClockTime' fails and using 'object' provides little benefit
    def __lt__(self, other) -> bool:
        return self.hour < other.hour \
            or self.hour == other.hour and self.min < other.min \
            or self.hour == other.hour and self.min == other.min and self.sec < other.sec

    def __le__(self, other) -> bool:
        return self.__eq__(other) or \
            self.__lt__(other)

    def __eq__(self, other) -> bool:
        return self.hour == other.hour and \
            self.min == other.min and \
            self.sec == other.sec

    def __ne__(self, other) -> bool:
        return not self.__eq__(other)

    def __gt__(self, other) -> bool:
        return self.hour > other.hour \
            or self.hour == other.hour and self.min > other.min \
            or self.hour == other.hour and self.min == other.min and self.sec > other.sec

    def __ge__(self, other) -> bool:
        return self.__eq__(other) or \
            self.__gt__(other)


def run_vid(filename: Union[str, int], connection: Optional[Connection]=None, **kwargs) -> Tuple[Optional[bool], Union[str, int], Optional[str], Optional[Tuple[str, ...]]]:
    """
    Video creation and runner function to pass to multiprocessing pool
    """
    wrote_frames = None
    seen_objects = None
    err_msg = None
    try:
        log.debug(kwargs)
        vid = VideoMotion(filename=filename, connection=connection, **kwargs)
        if vid.loaded:
            log.debug('Video loaded')
            wrote_frames, err_msg, seen_objects = vid.find_motion()
        else:

            err_msg = 'Video did not load successfully'
    except Exception as e:
        err_msg = 'Error processing video {}: {}'.format(filename, e)
        traceback.print_exc(file=sys.stdout)
        wrote_frames = None
    log.debug("Finished video %s", filename)
    return (wrote_frames, filename, err_msg, seen_objects)


def get_progress(log_file: str) -> Set[str]:
    """
    Load the progress log file, get the list of files.

    Strip off any comment at the end of the filename.
    """
    try:
        with open(log_file, 'r') as progress_log:
            done_files = {f.split(' // ')[0].strip() for f in progress_log.readlines()}
            return done_files
    except FileNotFoundError:
        return set()


def run_pool(job: Callable[..., Any], processes: int, files: Iterable[str]=None, pbar: Union[ProgressBar, DummyProgressBar]=DUMMY_PROGRESS_BAR, progress_log: IO[str]=None) -> None:
    """
    Create and run a pool of workers.
    """
    if not files:
        raise ValueError('More than 0 files needed')

    num_files: int = len(list(files))
    done: int = 0
    files_written: Set = set()
    results: list = []

    pool = None

    try:
        pool = Pool(processes=processes, initializer=partial(init_worker, unpaused))
        parent_conns: List[Connection] = []
        unpaused.set()

        for filename in files:
            parent_conn, child_conn = Pipe()
            parent_conns.append(parent_conn)
            results.append(pool.apply_async(job, (filename, child_conn)))

        num_err = 0
        num_wrote = 0

        while True:
            files_done = {res.get() for res in results if res.ready()}
            # log.debug(files_done)
            num_done = len(files_done)

            if num_done > done:
                done = num_done

            if done > 0:
                new = files_done.difference(files_written)
                files_written.update(new)

                for wrote_frames, filename, err_msg, seen_objects in new:
                    log.debug('Done {}{}'.format(filename, '' if wrote_frames else ' (no output)'))

                    if err_msg:
                        log.error('Error processing {}: {}'.format(filename, err_msg))
                        log.debug('Saw objects: {}'.format(seen_objects))
                        num_err += 1
                    else:
                        if progress_log is not None:
                            print("{} // {}".format(filename, seen_objects), file=progress_log)

                    if wrote_frames:
                        num_wrote += 1

            pbar.update(done)

            if num_done == num_files:
                log.debug("All processes completed. {} errors, wrote {} files".format(num_err, num_wrote))
                break

            for connection in parent_conns:
                try:
                    if connection.poll():
                        msg = connection.recv()
                        log.info(msg)
                        if msg == "stop":
                            log.info("Stopping")
                            unpaused.clear()
                        elif msg == "go":
                            log.info("Going")
                            unpaused.set()
                except (EOFError, ConnectionError):
                    pass

            # TODO: play/pause button with text saying "press space to stop/play"
            cv2.imshow("Unpause", np_ones([300, 300, 1]))

            if VideoMotion.key_pressed(' '):
                if unpaused.is_set():
                    log.info("Pausing - global")
                    unpaused.clear()
                else:
                    log.info("Unpausing - global")
                    unpaused.set()

            time.sleep(1)
    except KeyboardInterrupt:
        log.warning('Ending processing at user request')

    if pool is not None:
        pool.terminate()


def run_map(job: Callable, files: Iterable[str], pbar: Union[ProgressBar, DummyProgressBar]=DUMMY_PROGRESS_BAR, progress_log: IO[str]=None) -> None:
    if not files:
        raise ValueError('More than 0 files needed')

    log.debug('Processing each file one-by-one')

    files_processed: Iterable[Tuple[Optional[bool], Union[str, int], Optional[str], Optional[Tuple[str, ...]]]] = map(job, files)
    done: int = 0

    try:
        for wrote_frames, filename, err_msg, seen_objects in files_processed:
            if pbar is not None:
                done += 1
                pbar.update(done)

            log.debug('Done {}{}'.format(filename, '' if wrote_frames else ' (no output: {})'.format(err_msg)))

            if err_msg:
                log.error('Error processing {}: {}'.format(filename, err_msg))
                log.debug('Saw objects: {}'.format(seen_objects))
            else:
                if progress_log is not None:
                    print("{} // {}".format(filename, seen_objects), file=progress_log)
    except KeyboardInterrupt:
        log.warning('Ending processing at user request')


def run_stream(job: Callable, processes: int, cameras: List[int], progress_log: IO[str]=None) -> None:
    """
    Process one or more streams of input video with a Pool.
    """
    if not cameras:
        raise ValueError('More than 0 cameras needed')

    log.debug('Cameras: {}'.format(cameras))

    num_cameras: int = len(list(cameras))
    done: int = 0
    files_written: Set = set()
    results: list = []
    pool = None

    if num_cameras == 1 and processes == 1:
        log.debug('Single process, single camera stream')
        job(cameras[0])
    else:
        try:
            pool = Pool(processes=processes, initializer=partial(init_worker, unpaused))

            for camera in cameras:
                results.append(pool.apply_async(job, (camera,)))

            log.debug('Running jobs')

            while True:
                files_done = {res.get() for res in results if res.ready()}
                num_done = len(files_done)

                if num_done > done:
                    done = num_done

                if done > 0:
                    new = files_done.difference(files_written)
                    files_written.update(new)

                    for status, stream, err_msg, seen_objects in new:
                        log.debug('Done {}{}'.format(stream, '' if status else ' (no output)'))

                        if err_msg:
                            log.error('Ended processing camera {}: {}'.format(stream, err_msg))
                            log.debug('Saw objects: {}'.format(seen_objects))

                        print('Finished streaming from camera {}'.format(stream), file=progress_log)

                if num_done == num_cameras:
                    log.debug("All processes completed")
                    break

                time.sleep(1)
        except KeyboardInterrupt:
            log.warning('Ending processing at user request')
            # TODO: summarise output from streams effectively

        if pool is not None:
            pool.terminate()


def test_files(files) -> None:
    """Test loading from files."""
    for f in files:
        log.debug("{}: {}".format(f[0], datetime.fromtimestamp(f[1]).isoformat()))
        try:
            log.debug(str(VideoInfo(f[0], log_level=logging.DEBUG)))
        except Exception as e:
            log.error("File {}: {}".format(f[0], e))


def test_stream(cameras) -> None:
    """Test streaming from cameras."""
    for camera in cameras:
        try:
            log.debug("Camera stream: " + str(VideoInfo(camera, log_level=logging.DEBUG)))
        except Exception as e:
            log.error("Failed to open camera {}: {}".format(camera, e))


def main() -> None:
    """
    Run app.
    """
    parser: ArgumentParser = ArgumentParser(description="Find motion and objects in video", epilog="Available specific object detection: {}".format(list(CASCADE_LOOKUP.keys())))
    get_args(parser)
    args: Namespace = parser.parse_args()

    logging.basicConfig()
    if args.debug:
        log.setLevel(logging.DEBUG)

    run(args, parser.print_help)


def make_pbar_widgets(num_files: int) -> List:
    """
    Create progressbar widgets.
    """
    return [
        progressbar.Counter(), '/', str(num_files), ' ',
        progressbar.Percentage(), ' ',
        progressbar.Bar(), ' ',
        progressbar.Timer(), ' ',
        progressbar.ETA(),
    ]


def make_progressbar(progress: bool=False, num_files: int=0) -> ProgressBar:
    """
    Create progressbar.
    """
    return ProgressBar(max_value=num_files,
                       redirect_stdout=True,
                       redirect_stderr=True,
                       widgets=make_pbar_widgets(num_files)
                       ) if progress else DUMMY_PROGRESS_BAR


def read_masks(masks_file: str) -> List[Tuple]:
    """Read in masks from file."""
    try:
        with open(masks_file, 'r') as mf:
            masks = json.load(mf)
            validate(masks, MASK_SCHEMA)
            out_masks = []

            for mask in masks:
                log.debug('Mask area: {}'.format(mask))
                out_masks.append(tuple([tuple(coord) for coord in mask]))

            return out_masks

    except Exception as e:
        log.error('Masks file not read ({}): {}'.format(masks_file, e))
        return []


def set_log_file(input_dir: str=None, output_dir: str=None) -> str:
    """Log file in output directory, or input directory, or current directory, in that order of preference."""
    return os.path.normpath(os.path.join(output_dir if output_dir is not None else input_dir if input_dir is not None else '.', 'progress.log'))


def run(args: Namespace, print_help: Callable=lambda x: None) -> None:
    """
    Run app using an argparse Namespace.
    """
    if args.config:
        log.debug('Processing config: {}'.format(args.config))
        process_config(args.config, args)

    if args.no_shade and args.no_hue and args.no_edges:
        log.error("You need to leave one of shade, hue or edge detection switched on")
        sys.exit(2)

    if args.debug or args.test:
        log.setLevel(logging.DEBUG)

    if args.progress:
        progressbar.streams.wrap_stderr()

    if not args.files and not args.input_dir and not args.cameras:
        # no input: help message, exit
        print_help()
        sys.exit(2)

    if args.output_dir and not os.path.isdir(args.output_dir):
        os.mkdir(args.output_dir)
        log.warning('Created output directory {}'.format(args.output_dir))

    masks: list = args.masks if args.masks else []

    if args.masks_file:
        masks.extend(read_masks(args.masks_file))

    log.debug(str(masks))

    log_file: str = set_log_file(args.input_dir, args.output_dir)

    job = partial(run_vid,
                  outdir=args.output_dir, mask_areas=masks,
                  show=args.show, show_extras=args.show_extras,
                  codec=args.codec,
                  log_level=logging.DEBUG if args.debug else logging.INFO,
                  mem=args.mem, cleanup=args.cleanup,
                  blur_scale=args.blur_scale, box_size=args.box_size, min_box_scale=args.min_box_scale,
                  threshold=args.threshold, avg=args.avg,
                  fps=args.fps, min_time=args.mintime, cache_time=args.cachetime,
                  multiprocess=args.processes > 1,
                  cascades=args.cascade_object, haarcascades_path=args.haarcascades_path,
                  yolov3=args.yolov3, yolov4=args.yolov4, yolo_tiny=args.yolo_tiny,
                  confidence=args.confidence,
                  yolo_path=args.yolo_path,
                  no_object_detection=args.no_object_detection,
                  always_object_detection=args.always_object_detection,
                  object_detect_frame_interval=args.object_detect_frame_interval,
                  no_shade=args.no_shade,
                  no_hue=args.no_hue,
                  no_output=args.no_output)

    try:
        if args.cameras:
            if args.test:
                test_stream(args.cameras)
                sys.exit(0)

            # processing camera streams
            with open(log_file, 'a+', LINE_BUFFERED) as progress_log:
                run_stream(job, args.processes, args.cameras, progress_log)
        else:
            # processing input files

            # sort out time ordering priority
            time_order = process_times(args.time_order)
            log.debug(str(time_order))

            # find files on disk
            in_files: List[str] = verify_files(args.files)
            in_files.extend(find_files(args.input_dir))

            # sort them
            files: OrderedSet = sort_files_by_time(in_files, time_order)

            found_files_num = len(files)

            log.debug("{} files found".format(str(found_files_num)))

            if not args.ignore_progress:
                files = process_progress(files, log_file, args.ignore_drive)
            else:
                log.debug('Ignoring previous progress, processing all found files')

            num_files: int = len(files)

            log.debug('Processing {} files'.format(num_files))
            log.debug(str(files))

            if args.test:
                test_files(files)
                sys.exit(0)

            do_files: List[str] = [f[0] for f in files]

            with make_progressbar(args.progress, num_files) as pbar:
                pbar.update(0)
                with open(log_file, 'a+', LINE_BUFFERED) as progress_log:
                    if args.processes > 1:
                        run_pool(job, args.processes, do_files, pbar, progress_log)
                    else:
                        run_map(job, do_files, pbar, progress_log)
    except ValueError as e:
        log.error(str(e))
        sys.exit(1)


# TODO: fix ignore_drive
def process_progress(files: "OrderedSet[str]", log_file: str, ignore_drive: bool=False) -> "OrderedSet[str]":
    """Remove any previously processed files."""
    found_files_num = len(files)
    done_files: Set[str] = get_progress(log_file)
    log.debug("{} done files".format(str(len(done_files))))
    if not ignore_drive:
        files = OrderedSet([f for f in files if f[0] not in done_files])
    else:
        files = OrderedSet([f for f in files if not map(lambda x: os.path.splitdrive(x[0])[1] == os.path.splitdrive(x), done_files)])
    log.debug("{} files removed".format(str(found_files_num - len(files))))
    return files


def process_times(time_order: List[str]) -> List[Tuple[time.struct_time, time.struct_time]]:
    """
    Turn config-provided time orders into usable data structures.s
    """
    times: List[Tuple[time.struct_time, time.struct_time]] = []

    if time_order is None:
        return times

    for time_slot in time_order:
        try:
            res: List[time.struct_time] = list(map(lambda t: strptime(t, '%H:%M'), time_slot.split('-')))
            interval: Tuple[time.struct_time, time.struct_time] = (res[0], res[1])
        except ValueError as e:
            log.error('Time interval {} misparsed: {}'.format(time_slot, e))
            continue
        times.append(interval)
    return times


def process_config(config_file: str, args: Namespace) -> Namespace:
    """
    Read an INI style config, converting data to appropriate types.

    TODO: apply argparse validation to the config values
    """
    config: ConfigParser = ConfigParser()
    config.read(config_file)
    for setting, value in config['settings'].items():
        setting = setting.replace('-', '_')
        use_value: Any = value
        if setting in ('processes', 'blur_scale', 'min_box_scale', 'threshold', 'fps', 'box_size', 'object_detection_frame_interval'):
            use_value = int(value)
        if setting in ('mintime', 'cachetime', 'avg', 'confidence'):
            use_value = float(value)
        if setting in ('mem', 'progress', 'cleanup', 'debug', 'test', 'show', 'show_extras', 'ignore_progress', 'ignore_drive', 'no_object_detection', 'always_object_detection', 'no_output', 'yolov4', 'yolov3', 'yolo_tiny', 'no_shade', 'no_hue', 'no_object_detection', 'no_shade', 'no_hue', 'no_edges'):
            if value == 'True':
                use_value = True
            elif value == 'False':
                use_value = False
            else:
                raise ValueError('{} must be True or False'.format(setting))
        if setting in ('masks', 'cameras', 'time_order', 'cascade_object'):
            use_value = literal_eval(value)
            # TODO: validate that this is a list of tuples of int (masks) or a list of ints (cameras) or a list of strings (time_order, cascade_object)
        args.__setattr__(setting, use_value)
    log.debug(str(vars(args)))
    return args


def get_args(parser: ArgumentParser) -> None:
    """
    Set how to process command line arguments.
    """
    # TODO: add sections to make help easier
    parser.add_argument('files', nargs='*', help='Video files to find motion in')

    parser.add_argument('--config', '-c', help='Config in INI format')

    parser.add_argument('--cameras', nargs='*', type=int, help='0-indexed number of camera to stream from')
    parser.add_argument('--input-dir', '-i', help='Input directory to process')
    parser.add_argument('--output-dir', '-o', default='', help='Output directory for processed files')
    parser.add_argument('--ignore-progress', '-I', action='store_true', default=False, help='Ignore progress log')
    parser.add_argument('--ignore-drive', '-D', action='store_true', default=False, help='Ignore drive letter in progress log')
    parser.add_argument('--no-output', '-nop', action='store_true', default=False, help='Do not write any files (apart from progress log)')

    parser.add_argument('--codec', '-k', default='MP42', help='Codec to write files with')
    parser.add_argument('--fps', '-f', type=int, default=30, help='Frames per second of input files')

    parser.add_argument('--time-order', '-to', nargs='*', help='Time ranges in priority order for processing. Express as "HH:MM-HH:MM"')

    parser.add_argument('--masks', '-m', nargs='*', type=literal_eval, help='Areas to mask off in video')
    parser.add_argument('--masks-file', '-mf', help='File holding mask coordinates (JSON)')

    parser.add_argument('--cascade-object', '-O', nargs='*', type=str, help='Specific types of objects to detect using haar cascades (slow!)')
    parser.add_argument('--haarcascades-path', '-cp', type=str, default="/usr/local/share/opencv4/haarcascades", help='Specific types of objects to detect using haar cascades (slow!)')
    parser.add_argument('--yolov3', '-y3', action='store_true', help='Use YOLOv3 object detection')
    parser.add_argument('--yolov4', '-y4', action='store_true', help='Use YOLOv4 object detection')
    parser.add_argument('--yolo-tiny', '-yt', action='store_true', help='Use fast common object detection')
    parser.add_argument('--confidence', '-ct', type=float, default=0.25, help='Confidence threshold for object detection')
    parser.add_argument('--yolo-path', '-yp', help='A path to the cfg/weights files used by yolo3 and yolo4 (e.g. yolo4.cfg and yolo4.weights)')
    parser.add_argument('--no-object-detection', '-no', action='store_true', help="Don't do any object detection")
    parser.add_argument('--always-object-detection', '-ao', action='store_true', help="Always do any object detection")
    parser.add_argument('--object-detect-frame-interval', '-oi', type=int, default=1, help='Do object detection every N frames')

    parser.add_argument('--no-shade', '-ns', action="store_true", help="Don't use gray scale shade to detect motion")
    parser.add_argument('--no-hue', '-nh', action="store_true", help="Don't use color hue to detect motion")
    parser.add_argument('--no-edges', '-ne', action="store_true", help="Don't use edges to detect motion")

    parser.add_argument('--blur-scale', '-b', type=int, default=20, help='Scale of gaussian blur size compared to video width (used as 1/blur_scale)')
    parser.add_argument('--box-size', '-B', type=int, default=100, help='Pixel size to scale the video to for processing')
    parser.add_argument('--min-box-scale', '-mbs', type=int, default=50, help='Scale of minimum motion compared to video width (used as 1/min_box_scale')
    parser.add_argument('--threshold', '-t', type=int, default=12, help='Threshold for change in grayscale')
    parser.add_argument('--mintime', '-M', type=float, default=0.5, help='Minimum time for motion, in seconds')
    parser.add_argument('--cachetime', '-C', type=float, default=1.0, help='How long to cache, in seconds')
    parser.add_argument('--avg', '-a', type=float, default=0.1, help='How much to weight the most recent frame in the running average')

    parser.add_argument('--processes', '-J', default=1, type=int, help='Number of processors to use')

    parser.add_argument('--progress', '-p', action='store_true', help='Show progress bar')
    parser.add_argument('--show', '-s', action='store_true', default=False, help='Show main video')
    parser.add_argument('--show-extras', '-e', action='store_true', default=False, help='Show additional processing video')

    parser.add_argument('--cleanup', '-cu', action='store_true', help='Cleanup used frames (do not wait for garbage collection)')
    parser.add_argument('--mem', '-u', action='store_true', help='Run memory usage')
    parser.add_argument('--debug', '-d', action='store_true', help='Debug')
    parser.add_argument('--test', '-T', action='store_true', help='Test which files or camera streams would be processed')
