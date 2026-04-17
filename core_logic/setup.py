from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy as np

extensions = [
    Extension(
        name="core_logic.cboard",
        sources=["core_logic/cboard.pyx", "core_logic/bitboard.cpp"],
        language="c++",
        extra_compile_args=["-O3", "-std=c++17"],
        include_dirs=[np.get_include(), "."],
    )
]

setup(
    name="cboard",
    ext_modules=cythonize(extensions, compiler_directives={"language_level": "3"}),
)
