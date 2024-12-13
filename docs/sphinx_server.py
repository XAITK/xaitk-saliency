#!/usr/bin/env python

"""This module is designed to used with _livereload to make it a little easier to write Sphinx documentation.

Simply run the command::
    python sphinx_server.py

and browse to http://localhost:5500

livereload_: https://pypi.python.org/pypi/livereload
"""

import os
import sys
from pathlib import Path

from livereload import Server, shell

SCRIPT_DIR = Path(__file__).parent

if sys.platform == "win32":
    print("Using make.bat")
    rebuild_cmd = shell("make.bat html", cwd=".")
else:
    print("Using Makefile")
    rebuild_cmd = shell("make html", cwd=".")

rebuild_root = "_build/html"

# Watch files recursively under these directories.
watch_dirs: list[Path] = [SCRIPT_DIR, SCRIPT_DIR / ".." / "examples"]
# Watch files matching these globs under the above directories.
watch_globs = ["*.rst", "*.ipynb"]

# Code source directory. We want to watch all python files under here.
watch_source_dir = Path("../src/xaitk_saliency")

server = Server()
server.watch("conf.py", rebuild_cmd)
# Cover above configured watch dirs and globs matrix.
for d in watch_dirs:
    for g in watch_globs:
        glob_path = os.path.join(d.resolve(), "**", g)
        print(f"Watching files for glob: {glob_path}")
        server.watch(glob_path, rebuild_cmd)
# Watch source python files.
pkg_src_glob = os.path.join(watch_source_dir.resolve(), "**", "*.py")
print(f"Watching source package python files: {pkg_src_glob}")
server.watch(pkg_src_glob, rebuild_cmd)
# Optionally change to host="0.0.0.0" to make available outside localhost.
server.serve(root=rebuild_root)
