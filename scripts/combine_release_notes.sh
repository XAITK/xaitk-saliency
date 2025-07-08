#!/bin/bash

# Script to generate the DEVEL-JATIC release notes.

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="${SCRIPT_DIR}/.."
DOCS_DIR="${PROJECT_DIR}/docs"
RELEASE_NOTES_DIR="${DOCS_DIR}/release_notes"

usage="Usage: $0 <version-string>"

if [ "$#" -ne 1 ]; then
    echo "$usage" >&2
    exit 1
fi

version="$1"
OUTPUT_FILE="${RELEASE_NOTES_DIR}/v${version}.rst"
PENDING_RELEASE_NOTES_DIR="${RELEASE_NOTES_DIR}/pending_release"

if [ ! -d "$PENDING_RELEASE_NOTES_DIR" ]; then
    echo "Error: Directory '$PENDING_RELEASE_NOTES_DIR' does not exist." >&2
    exit 1
fi

> "$OUTPUT_FILE"

echo "$version" >> "$OUTPUT_FILE"
echo "${version//?/=}" >> "$OUTPUT_FILE"
echo >> "$OUTPUT_FILE"

for file in "$PENDING_RELEASE_NOTES_DIR"/*; do
    grep '^[[:space:]]*[-*]' "$file" | while IFS= read -r line; do
        echo "$line" >> "$OUTPUT_FILE"
        echo >> "$OUTPUT_FILE"
    done
done

truncate -s -1 "$OUTPUT_FILE"

echo "Release notes generated: $OUTPUT_FILE"
