from cx_Freeze import setup, Executable
import sys
import os

# cx_Freeze couldn't detect tcl libraries so I had to manually write their path
os.environ['TCL_LIBRARY'] = r'C:\Users\Aurissan\AppData\Local\Programs\Python\Python36\tcl\tcl8.6'
os.environ['TK_LIBRARY'] = r'C:\Users\Aurissan\AppData\Local\Programs\Python\Python36\tcl\tk8.6'

base = None
if sys.platform == 'win32':
    base = 'Win32GUI'

options = {
    'build_exe': {
        'includes': 'atexit'
    }
}

executables = [
    Executable('C:/users/aurissan/pycharmprojects/diplomaclean/program.py', base=base)
]

#the frozen program was giving errors with those libs so I had to include them manually
additional_mods = ['numpy.core._methods', 'numpy.lib.format', 'scipy.sparse.csgraph._validation',
                   'scipy.spatial.ckdtree']

include_files = ['img',"readme IMPORTANT.txt"]
setup(
    name = "RBFApprox",
    version = "1.0",
    options={'build_exe': {'includes': additional_mods,'include_files': include_files} },
    description = "Function approximation using RBF-network",
    executables = [Executable('C:/users/aurissan/pycharmprojects/diplomaclean/program.py')]
)