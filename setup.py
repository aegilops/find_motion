import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

requirements = [
    "mem_top>=0.1.6",
    "orderedset>=2.0.3",
    "opencv-python==4.5.3.36",
    # https://github.com/opencv/opencv/releases/tag/4.4.0
    "imutils>=0.5.3",
    "numpy>=1.18.5",
    "cvlib>=0.2.5",
    "progressbar2>=3.51.4",
    "tensorflow>=2.6.8",
    "jsonschema>=3.2.0",
    "packaging>=21.0",
    "importlib_resources==5.2.2 ; python_version<'3.7'"
]

setuptools.setup(
    name="find_motion",
    version="1.0.0",
    author="Aegilops",
    author_email="41705651+aegilops@users.noreply.github.com",
    description="Processes video to detect motion and objects, with tunable parameters",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/aegilops/find_motion",
    entry_points={"console_scripts": ["find_motion=find_motion.find_motion:main"], },
    packages=setuptools.find_packages(),
    package_data={"find_motion": ["data/*"]},
    install_requires=requirements,
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Development Status :: 5 - Production/Stable",
        "Environment :: Console",
        "Topic :: Multimedia :: Video",
        "Topic :: Scientific/Engineering :: Image Recognition"
    ],
)
