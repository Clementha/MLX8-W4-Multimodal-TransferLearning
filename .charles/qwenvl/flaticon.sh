#!/bin/bash

RAW_DIR="../.data/flaticon.com/raw"
TARGET_DIR="../.data/flaticon.com/target"

if [[ "$1" == "--unzip" ]]; then
  # Unzip all .zip files in RAW_DIR if not already extracted
  find "$RAW_DIR" -maxdepth 1 -type f -name "*.zip" | while read -r zipfile; do
    folder="${zipfile%.zip}"
    if [[ ! -d "$folder" ]]; then
      unzip -q "$zipfile" -d "$folder"
      echo "Unzipped: $zipfile"
    else
      echo "Already unzipped: $zipfile"
    fi
  done
  return 0
fi

if [[ "$1" == "--split" ]]; then
  train_dir="$TARGET_DIR/train"
  test_dir="$TARGET_DIR/test"
  mkdir -p "$train_dir" "$test_dir"
  # List only directories (not train/test themselves)
  folders=()
  for d in "$TARGET_DIR"/*; do
    [[ -d "$d" && "$(basename "$d")" != "train" && "$(basename "$d")" != "test" ]] && folders+=("$(basename "$d")")
  done
  total=${#folders[@]}
  if [[ $total -eq 0 ]]; then
    echo "No folders to split."
    return 1
  fi
  split_idx=$((total * 8 / 10))
  for i in "${!folders[@]}"; do
    src="$TARGET_DIR/${folders[$i]}"
    if (( i < split_idx )); then
      mv "$src" "$train_dir/"
      echo "Moved to train: ${folders[$i]}"
    else
      mv "$src" "$test_dir/"
      echo "Moved to test: ${folders[$i]}"
    fi
  done
  return 0
fi

case "$1" in
  --png|--svg|--eps|--psd)
    TYPE="${1#--}"
    shift
    FORCE=0
    if [[ "$1" == "-y" ]]; then
      FORCE=1
    fi
    mkdir -p "$TARGET_DIR/$TYPE"
    # Find all folders named $TYPE under RAW_DIR and copy their parent folder to TARGET_DIR
    find "$RAW_DIR" -type d -name "$TYPE" | while read -r src; do
      parent="$(dirname "$src")"
      base="$(basename "$parent")"
      dest="$TARGET_DIR/$base"
      if [[ -d "$dest/$TYPE" ]]; then
        if [[ $FORCE -eq 1 ]]; then
          mkdir -p "$dest"
          cp -r "$parent/$TYPE" "$dest/"
          echo "Copied: $parent/$TYPE -> $dest/"
        else
          echo "Already exists: $dest/$TYPE"
        fi
      else
        mkdir -p "$dest"
        cp -r "$parent/$TYPE" "$dest/"
        echo "Copied: $parent/$TYPE -> $dest/"
      fi
    done
    ;;
  *)
    echo "Usage: $0 --unzip | --split | --png [-y] | --svg [-y] | --eps [-y] | --psd [-y]"
    return 1
    ;;
esac
