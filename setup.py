import os
import sys
import subprocess
import multiprocessing
from setuptools.command.install import install
from setuptools.command.build_ext import build_ext
from setuptools import setup, find_packages, Extension

class CMakeExtension(Extension):
    def __init__(self, name, sourcedir=''):
        super().__init__(name, sources=[])
        self.sourcedir = os.path.abspath(sourcedir)

class CMakeBuild(build_ext):
    def run(self):
        for ext in self.extensions:
            self.build_cmake(ext)

    def build_cmake(self, ext):
        extdir = os.path.abspath(os.path.dirname(self.get_ext_fullpath(ext.name)))
        cmake_args = [
            f'-DCMAKE_LIBRARY_OUTPUT_DIRECTORY={extdir}',
            f'-DPYTHON_EXECUTABLE={sys.executable}',
        ]

        cfg = 'Debug' if self.debug else 'Release'
        build_args = ['--config', cfg, '--', f'-j{multiprocessing.cpu_count()}']

        if not os.path.exists(self.build_temp):
            os.makedirs(self.build_temp)

        subprocess.check_call(['cmake', ext.sourcedir] + cmake_args, cwd=self.build_temp)
        subprocess.check_call(['cmake', '--build', '.'] + build_args, cwd=self.build_temp)

        # Ensure the built library is moved to the package directory
        self.copy_extensions_to_source(extdir)

    def copy_extensions_to_source(self, extdir):
        # Define the library name and the target directory
        target_dir = os.path.join(self.build_lib, 'iris')
        os.makedirs(target_dir, exist_ok=True)
        lib_name = 'iris.dll' if sys.platform == 'win32' else 'libiris.so'
        # Copy all built shared libraries to the target directory
        for file_name in os.listdir(extdir):
            if file_name.endswith('.so') or file_name.endswith('.dll'):
                self.copy_file(os.path.join(extdir, file_name), os.path.join(target_dir, file_name))

setup(
    name='iris',
    version='3.0.0',
    author='Narasinga Rao Minisar',
    author_email='miniskarnr@ornl.gov',
    description='IRIS Python package with CMake build',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/ORNL/iris',
    packages=find_packages(where='src/runtime'),
    package_dir={'': 'src/runtime'},
    ext_modules=[CMakeExtension('iris', sourcedir='.')],
    cmdclass={'build_ext': CMakeBuild},
    include_package_data=True,
    zip_safe=False,
)

