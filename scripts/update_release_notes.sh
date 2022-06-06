#!/bin/bash
#
# Wrapper to pull and invoke the "common" script of the same name from the
# smqtk-core repository.
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

bash "$DL_SCRIPT" "$@"
