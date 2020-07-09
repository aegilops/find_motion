import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

requirements = [
    'mem_top',
    'orderedset',
    'opencv-python',
    'imutils',
    'numpy',
    'pynput',
    'cvlib',
    'progressbar2',
    'tensorflow',
    'jsonschema'
]

setuptools.setup(
    name="find_motion",
    version="0.0.3",
    author="Aegilops",
    author_email="41705651+aegilops@users.noreply.github.com",
    description="Processes video to detect motion and objects, with tunable parameters",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/aegilops/find_motion",
    entry_points={"console_scripts": ["find_motion=find_motion.find_motion:main"], },
    packages=setuptools.find_packages(),
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
