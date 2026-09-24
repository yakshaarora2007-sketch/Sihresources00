from setuptools import Extension, find_packages, setup
import numpy


setup(
    name="fovmap-rating-kernel",
    package_dir={"": "src"},
    packages=find_packages("src"),
    ext_modules=[
        Extension(
            "fovmap._rating_kernel",
            ["src/fovmap/_rating_kernel.c"],
            include_dirs=[numpy.get_include()],
        )
    ],
)
