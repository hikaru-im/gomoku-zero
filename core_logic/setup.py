from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy as np

extensions = [
    Extension(
        name="cboard",
        sources=["cboard.pyx", "bitboard.cpp"],
        language="c++",
        extra_compile_args=["-O3", "-std=c++17", "-fopenmp"],
        extra_link_args=["-fopenmp"],
        include_dirs=[np.get_include(), "."],
    )
]

setup(
    name="cboard",
    ext_modules=cythonize(extensions, compiler_directives={"language_level": "3"}),
)
