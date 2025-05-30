#####################################################
#   Author: Narasinga Rao Miniskar
#   Date: 06/06/2024
#   File: setup.py
#   Contact: miniskarnr@ornl.gov
#   Comment: Files for python pip package
#####################################################
import os
import sys
import subprocess
import multiprocessing
from setuptools.command.install import install
from setuptools.command.build_ext import build_ext
from setuptools import setup, find_packages, Extension

import site
import sys

iris_version="3.0.0"

class CMakeExtension(Extension):
    def __init__(self, name, sourcedir=''):
        super().__init__(name, sources=[])
        self.sourcedir = os.path.abspath(sourcedir)

class CMakeBuild(build_ext):
    def run(self):
        for ext in self.extensions:
            self.build_cmake(ext)

    def build_cmake(self, ext):
        print(dir(self))
        extdir = os.path.abspath(os.path.dirname(self.get_ext_fullpath(ext.name)))
        site_packages_path = site.getsitepackages()[0]
        dist_info_dir = os.path.join(site_packages_path, f'iris')
        print(f"Dist-info directory: {dist_info_dir}")
        os.makedirs(dist_info_dir, exist_ok=True)

        cmake_args = [
            f'-DCMAKE_LIBRARY_OUTPUT_DIRECTORY={extdir}',
            f'-DCMAKE_INSTALL_PREFIX={dist_info_dir}',
            f'-DPYTHON_EXECUTABLE={sys.executable}',
        ]

        cfg = 'Debug' if self.debug else 'Release'
        build_args = ['--config', cfg, '--', f'-j{multiprocessing.cpu_count()}']
        install_args = ['--config', cfg]

        if not os.path.exists(self.build_temp):
            os.makedirs(self.build_temp)

        print('Source dir:', ext.sourcedir)
        self.sourcedir = ext.sourcedir
        print(f'Running CMake configure: {cmake_args}')
        subprocess.check_call(['cmake', ext.sourcedir] + cmake_args, cwd=self.build_temp)
        print(f'Running CMake build: {build_args}')
        subprocess.check_call(['cmake', '--build', '.'] + build_args, cwd=self.build_temp)
        subprocess.check_call(['cmake', '--install', '.'] + install_args, cwd=self.build_temp)

        # Ensure the built library is moved to the package directory
        self.copy_extensions_to_source(extdir)

    def copy_all_files(self, source_dir, dest_dir):
     """
     Copy all files from source_dir to dest_dir.
     :param source_dir: Source directory
     :param dest_dir: Destination directory
     """
     if not os.path.exists(source_dir):
         raise ValueError(f"Source directory '{source_dir}' does not exist.")
     if not os.path.exists(dest_dir):
         os.makedirs(dest_dir)
     for item in os.listdir(source_dir):
         src_path = os.path.join(source_dir, item)
         dst_path = os.path.join(dest_dir, item)
         if os.path.isfile(src_path):
             self.copy_file(src_path, dst_path)  


    def copy_extensions_to_source(self, extdir):
        # Define the library name and the target directory
        print("Ext directory: "+extdir)
        print("Current directory: "+os.getcwd())
        target_dir = os.path.join(self.build_lib, 'iris')
        print(f"Build directory: {target_dir}")
        #os.makedirs(target_dir, exist_ok=True)
        package_dir = os.path.join(os.path.abspath(self.build_lib), 'iris')
        print(f"Package directory: {package_dir}")
        site_packages_path = site.getsitepackages()[0]
        print(f"Site-packages path: {site_packages_path}")
        # Define the dist-info directory
        dist_info_dir = os.path.join(site_packages_path, f'iris')
        print(f"Dist-info directory: {dist_info_dir}")
        os.makedirs(dist_info_dir, exist_ok=True)

        lib_name = 'iris.dll' if sys.platform == 'win32' else 'libiris.so'
        # Copy all built shared libraries to the target directory
        self.copy_file(os.path.join(self.sourcedir, 'src/runtime/iris.py'), os.path.join(dist_info_dir, 'iris.py'))
        self.copy_file(os.path.join(self.sourcedir, 'src/runtime/__init__.py'), os.path.join(dist_info_dir, '__init__.py'))
        #self.copy_all_files(os.path.join(self.sourcedir, 'utils'), os.path.join(dist_info_dir, 'utils'))
        #os.remove(os.path.join(dist_info_dir, 'include/iris/iris.py'))
        #for file_name in os.listdir(extdir):
        #    if file_name.endswith('.so') or file_name.endswith('.dll'):
        #        source_file = os.path.join(extdir, file_name)
        #        target_file = os.path.join(dist_info_dir, file_name)
        #        #print(f"Copying {source_file} to {target_file}")
        #        self.copy_file(source_file, target_file)

setup(
    name='iris',
    version=f'{iris_version}',
    author='Narasinga Rao Minisar',
    author_email='miniskarnr@ornl.gov',
    description='IRIS Python package with CMake build',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/ORNL/iris',
    install_requires=['numpy', 'regex'],
    packages=find_packages(where='src/runtime'),
    package_dir={'': 'src/runtime'},
    ext_modules=[CMakeExtension('iris', sourcedir='.')],
    cmdclass={'build_ext': CMakeBuild},
    include_package_data=True,
    zip_safe=False,
)

