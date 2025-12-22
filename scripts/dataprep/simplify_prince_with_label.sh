#!/usr/bin/env bash
set -euo pipefail
shopt -s nullglob  # globs that don't match expand to nothing (not literal)

# --- USER SETTINGS ---
INPUT_DIR="/mnt/wjdisk/edge_seg_princeton/data/gt_obj"
OUTPUT_DIR="/mnt/wjdisk/edge_seg_princeton/data/gt_simp"
LABEL_ROOT="/mnt/wjdisk/edge_seg_princeton/data/seg_res"  # root that contains 1/, 2/, ...
TARGET_FACE_COUNT=3000
BLENDER_PATH="/opt/blender/blender"
SCRIPT_PATH="/mnt/wjdisk/edge_seg_princeton/blender_process_with_label.py"
# ---------------------

mkdir -p "$OUTPUT_DIR"

for filepath in "$INPUT_DIR"/*.obj; do
  filename=$(basename "$filepath")
  filename_wo_ext="${filename%.*}"

  # --- Find the best numeric token from the filename (prefer rightmost that has labels)
  mapfile -t tokens < <(grep -oE '[0-9]+' <<<"$filename_wo_ext" || true)
  id_num=""   # normalized integer (no leading zeros)

  if ((${#tokens[@]} > 0)); then
    # iterate from rightmost token to leftmost
    for ((i=${#tokens[@]}-1; i>=0; i--)); do
      tok="${tokens[i]}"
      # normalize: strip leading zeros by forcing base-10 arithmetic
      id_dir=$((10#$tok))
      label_dir_try="$LABEL_ROOT/$id_dir"

      # do we have any matching labels in this folder?
      if compgen -G "$label_dir_try/${id_dir}_"'*.seg' > /dev/null || \
         compgen -G "$label_dir_try/${id_dir}_"'*.ser' > /dev/null; then
        id_num="$id_dir"
        break
      fi
    done
  fi

  # fallback if nothing matched
  if [[ -z "$id_num" ]]; then
    id_num=0
  fi

  label_dir="$LABEL_ROOT/$id_num"

  # Collect labels (both .seg and .ser)
  labels=( "$label_dir/${id_num}_"*.seg "$label_dir/${id_num}_"*.ser )

  # Natural sort the labels so _10 comes after _9
  if ((${#labels[@]} > 0)); then
    IFS=$'\n' read -r -d '' -a labels < <(printf '%s\n' "${labels[@]}" | sort -V && printf '\0')
  fi

  # Output path
  output_file="$OUTPUT_DIR/${filename_wo_ext}_${TARGET_FACE_COUNT}.obj"

  echo
  echo "Simplifying $filename  (picked id=$id_num from '$filename_wo_ext')"
  echo "  Label dir: $label_dir"
  echo "  → $output_file"

  if ((${#labels[@]} == 0)); then
    echo "  ! No labels found in $label_dir matching ${id_num}_*.seg|*.ser; proceeding without labels."
    "$BLENDER_PATH" --background --python "$SCRIPT_PATH" -- \
      "$filepath" "$TARGET_FACE_COUNT" "$output_file"
  else
    echo "  + ${#labels[@]} labels (natural sorted):"
    for l in "${labels[@]}"; do echo "    - $(basename "$l")"; done
    "$BLENDER_PATH" --background --python "$SCRIPT_PATH" -- \
      "$filepath" "$TARGET_FACE_COUNT" "$output_file" "${labels[@]}"
  fi
done