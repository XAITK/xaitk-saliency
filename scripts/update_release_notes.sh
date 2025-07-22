#!/bin/bash
#
# Script to help with the XAITK-Saliency release process. Performs the following steps:
#   - Poetry version (major, minor, or patch)
#   - Combine release note fragments into one file
#   - Clean pending_release directory
#
# Two git commits are created. One for the version bump and one for the new
# release notes stub file.
#
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="${SCRIPT_DIR}/.."
DOCS_DIR="${PROJECT_DIR}/docs"
RELEASE_NOTES_DIR="${DOCS_DIR}/release_notes"
PENDING_RELEASE_NOTES_DIR="${RELEASE_NOTES_DIR}/pending_release"

# Check args
if [ "$#" != 1 ]
then
  echo "Please enter valid version bump type. Options: major, minor, or patch"
  exit 1
fi

if [ "$1" != 'major' ] && [ "$1" != 'minor' ] && [ "$1" != 'patch' ]
then
  echo "Please enter valid version bump type. Options: major, minor, or patch"
  exit 1
fi

RELEASE_TYPE="$1"
echo "Release type: ${RELEASE_TYPE}"

# Update version
poetry version "${RELEASE_TYPE}"

# Get version
VERSION="$(poetry version -s)"
VERSION_STR="v${VERSION}"
VERSION_SEPERATOR=${VERSION_STR//?/=}

# Combine release notes
bash ${SCRIPT_DIR}/combine_release_notes.sh "${VERSION}"

# Make git commits
git add "${PROJECT_DIR}"/pyproject.toml
git add "${RELEASE_NOTES_DIR}"/v"${VERSION}".rst
git commit -m "Update version number to ${VERSION}"

# Clear pending_release
git rm "${PENDING_RELEASE_NOTES_DIR}/*.rst"
git commit -m "Clear release notes fragments"
