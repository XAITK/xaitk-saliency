#!/usr/bin/env python

"""
This module is designed to used with _livereload to
make it a little easier to write Sphinx documentation.
Simply run the command::
    python sphinx_server.py

and browse to http://localhost:5500

livereload_: https://pypi.python.org/pypi/livereload
"""
import os
import sys

from livereload import Server, shell

if sys.platform == "win32":
    print("Using make.bat")
    rebuild_cmd = shell("make.bat html", cwd=".")
else:
    print("Using Makefile")
    rebuild_cmd = shell("make html", cwd=".")

rebuild_root = "_build/html"

watch_dirs = [
    ".",
    "release_notes",
]

watch_globs = ["*.rst", "*.ipynb"]

watch_source_dir = "../xaitk_saliency"

server = Server()
server.watch("conf.py", rebuild_cmd)
# Cover above configured watch dirs and globs matrix.
for d in watch_dirs:
    for g in watch_globs:
        server.watch(os.path.join(d, g), rebuild_cmd)
# Watch source python files.
for dirpath, dirnames, filenames in os.walk(watch_source_dir):
    server.watch(os.path.join(dirpath, "*.py"), rebuild_cmd)
# Optionally change to host="0.0.0.0" to make available outside localhost.
server.serve(root=rebuild_root)
