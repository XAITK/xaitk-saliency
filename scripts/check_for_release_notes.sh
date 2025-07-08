#!/bin/bash

# Check if any RST files in the release notes space were modified.
# If any have been modified, then this check criterion has passed.
# Otherwise if nothing has been modified then we exit non-zero to indicate
# that something should have been.

set -e  # trigger non-zero exit if any interstitial command fails.

function usage()
{
  echo "Usage: ${BASH_SOURCE[0]} [OPTIONS] TARGET_BRANCH"
  echo "-h, --help        display help"
  echo ""
  echo "Note: Make sure to run \"git fetch\" in CI to get the list of remote target branches"
}

POSITIONAL=()
while [[ "$#" -gt 0 ]]
do
  key="$1"
  shift
  case "$key" in
    -h|--help)
      usage
      exit 0
      ;;
    *)
      POSITIONAL+=( "$key" )
      ;;
  esac
done

# Use the first positional argument as the target branch name/hash
TARGET_BRANCH_NAME="${POSITIONAL[0]}"

readarray -d $'\0' -t release_notes_mods < <( git diff --name-only -z ${TARGET_BRANCH_NAME}..HEAD -- ./docs/release_notes/pending_release/*.rst )

if [[ "${#release_notes_mods[@]}" -eq 0 ]]
then
  echo "docs/release_notes not updated"
  exit 1
else
  echo "docs/release_notes updated: ${release_notes_mods[@]}"
fi
