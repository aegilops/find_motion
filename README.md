# find_motion - Motion and object detection with OpenCV

Video processing Python 3 script to detect motion and objects, with tunable parameters.

Uses brightness, color changes and edge changes to spot movement against a relatively still background.

Caches images for a few frames before and after it detects movement.

With much help from OpenCV, https://www.pyimagesearch.com/ and https://cvlib.net/.

## Data files

This repo no longer directly provides the data files `coco.names`, `yolov4.weights`, `yolov4.cfg`, `yolov4-tiny.weights`, `yolov4-tiny.cfg`.

This is because of excessive LFS bandwidth. You will need to source these files from elsewhere before installing and using the repo.

They are currently available in a [Release of this project on GitHub](https://github.com/aegilops/find_motion/releases).

Place them into the `find_motion/data` directory after cloning, and before installing.

## Installing

First clone the repo and change into the directory.

> NOTE: see above about data files that are required dependencies and are not inside this repo when you clone it

To install as a module do `python3 -mpip install . --user` in the repo directory. You can run it with `find_motion` once it is installed.

Otherwise, use `python3 -mpip install -r requirements.txt --user` to install dependencies. Then call `./find_motion.py` directly.

If you start to get errors to do with progressbar, try reinstalling `progressbar2` (`python3 -mpip install progressbar2 --user`) - something else may have installed `progressbar`, which conflicts with `progressbar2`.

### A note on OpenCV2

OpenCV2 is available as a pre-built package from pypi using pip, but it can lag the latest OpenCV2.

You can build OpenCV2 from source if it has features missing from the pip package, or to possibly improve performance:

* [OpenCV2 GitHub](https://github.com/opencv/opencv/)

### Installing on Windows

Some of the dependencies may be nested so deep that they exceed `MAX_PATH` on Windows.

If you are using Windows 10 or 11, try opting in to the new long file names, which have no length limit:

* [Enable long paths in Windows 10 version 1607 and later](https://docs.microsoft.com/en-us/windows/win32/fileio/naming-a-file#enable-long-paths-in-windows-10-version-1607-and-later)

## Usage

Use `./find_motion.py -h` and `pydoc3 find_motion.py` (or `find_motion -h` and `pydoc3 find_motion` if you installed the module) to get detailed documentation of the command-line options.

### Keyboard controls
* `Ctrl-c`: at the command-line - ends all processing.
* `q`: skip to the next video.
* `spacebar`: on an output frame this will pause processing. On the "unpause" window this will pause/unpause processing.  

### Tuning

To reduce false motion from shadows, wind movement, noise, sun glare and so on, you can control:
* Shade, hue or edge change detection
  Whether to use changes in gray scale shade, color hue or edges to spot motion

* Threshold

  How much difference in gray/hue/edge level between frames counts as a difference
* Minimum time

  How long a difference in the background must be seen for to count as motion
* Cache time

  How long to cache frames before motion is detected and how long to keep showing them after motion stops
* Averaging

  What proportion of the 'background' is provided by the last seen frame (0 to 1)
* Blur scale

  How large the blurring of the image is during processing (larger sizes are slower, but less sensitive to tiny motions)
* Masking

  Areas to ignore. They are defined as tuples of coordinates. They can be provided as literals at the command line or in a JSON file
  * square: `((0, 0), (100, 100))`
  * triangle: `((0, 0), (0, 100), (100, 0))`
  * JSON:
    ```
    [
        [[0, 0], [100, 100]],
        [[0, 0], [0, 100], [100, 0]]
    ]
    ```

### Video sources

Input can come from a list of files, an input folder, or one or more cameras.

### Output

Files are written to an output directory in 2GB sections.

Progress is logged by default in `progress.log` in the output directory.

### Config file

You can provide an INI format file with a single `[settings]` section using the `--config` option.

Use the same names as the optional arguments given in the help for the fields of the settings file.
