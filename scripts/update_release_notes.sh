#!/bin/bash

# Wrapper to pull and invoke the "common" script of the same name from the
# smqtk-core repository. If the environment variable "JUST_DOWNLOAD" is set to
# a non-empty value, we will only download the cache files and not execute
# them.
#
# Script to help with the XAITK-Saliency release process. Performs the following steps:
#   - Poetry version (major, minor, or patch)
#   - Rename release_notes/pending_release file to release_notes/version
#   - Add reference to new release notes file in release_notes.rst
#   - Add new release notes stub file
#
# Two git commits are created. One for the version bump and one for the new
# release notes stub file.
#
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
URL_SCRIPT="https://raw.githubusercontent.com/Kitware/SMQTK-Core/master/scripts/update_release_notes.sh"
URL_STUB="https://raw.githubusercontent.com/Kitware/SMQTK-Core/master/scripts/.pending_notes_stub.rst"
DL_SCRIPT="${SCRIPT_DIR}/.dl_script_cache.sh"
DL_STUB="${SCRIPT_DIR}/.pending_notes_stub.rst"

if [[ ! -f "$DL_SCRIPT" ]]
then
  curl -sSL "$URL_SCRIPT" -o "$DL_SCRIPT"
  curl -sSL "$URL_STUB" -o "$DL_STUB"
fi

if [[ -n "${JUST_DOWNLOAD}" ]]
then
  exit 0
fi

bash "$DL_SCRIPT" "$@"
