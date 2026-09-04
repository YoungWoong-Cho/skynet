#!/usr/bin/env bash
#SBATCH --job-name=sync-dexverse-assets
#SBATCH --partition=rl2-lab
#SBATCH --account=rl2-lab
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=04:00:00
#SBATCH --chdir=/coc/flash7/ycho420
#SBATCH --output=/coc/flash7/ycho420/logs/%x-%j.out
#SBATCH --error=/coc/flash7/ycho420/logs/%x-%j.err

set -euo pipefail
umask 0027

ROOT=/coc/flash7/ycho420
DATASETS_ROOT="$ROOT/datasets"
DEXVERSE_COMMIT=30cc673e27684b9f10186fa6bea731aed246bc9f
RELEASE_REVISION=d4cc36e10e27e481f31ebb7bf5111d37c7b51e51
MANITWIN_REVISION=525bc931c201988806a17fe3b5f96646a00de6ff
MANITWIN_SELECTION=selected-100-medium-diverse-seed42
UV_VERSION=0.12.8
HF_HUB_VERSION=1.29.0

DEXVERSE_RESOURCE="$DATASETS_ROOT/resources/huggingface/dexverse/DexVerse_release/revisions/$RELEASE_REVISION"
MANITWIN_RESOURCE="$DATASETS_ROOT/resources/huggingface/ManiTwin/ManiTwin-100K/revisions/$MANITWIN_REVISION"
BUNDLE_VERSION="release-$RELEASE_REVISION--manitwin-$MANITWIN_REVISION"
BUNDLE="$DATASETS_ROOT/bundles/dexverse/simulation-assets/versions/$BUNDLE_VERSION"

STAGING="$DATASETS_ROOT/.staging/dexverse-assets-$BUNDLE_VERSION"
DEXVERSE_STAGE="$STAGING/resources/dexverse-release"
MANITWIN_STAGE="$STAGING/resources/manitwin-100k"
BUNDLE_STAGE="$STAGING/bundle"
TOOLING="$STAGING/tooling"
STATE="$STAGING/.state"
UV_ROOT="$ROOT/tools/uv/$UV_VERSION"
UV="$UV_ROOT/uv"
PYTHON=/nethome/ycho420/miniconda3/bin/python

mkdir -p "$DATASETS_ROOT/resources" "$DATASETS_ROOT/derivatives" \
  "$DATASETS_ROOT/bundles" "$DATASETS_ROOT/.staging" "$STATE" "$TOOLING" \
  "$ROOT/logs" "$ROOT/tmp/$SLURM_JOB_ID" "$ROOT/.cache/uv" \
  "$ROOT/.cache/huggingface" "$UV_ROOT"
export TMPDIR="$ROOT/tmp/$SLURM_JOB_ID"
export UV_CACHE_DIR="$ROOT/.cache/uv"
export HF_HOME="$ROOT/.cache/huggingface"

exec 9>"$DATASETS_ROOT/.download.lock"
flock 9

stage() {
  local name=$1
  shift
  local marker="$STATE/$name.done"
  if [ -f "$marker" ]; then
    printf 'SKIP %s\n' "$name"
    return 0
  fi
  printf 'START %s\n' "$name"
  "$@"
  date -u +%FT%TZ > "$marker"
  printf 'DONE %s\n' "$name"
}

ready_manifest() {
  local directory=$1
  [ -d "$directory" ] && [ ! -L "$directory" ] && \
    [ -f "$directory/manifest.json" ] && [ -f "$directory/READY" ] && \
    (cd "$directory" && sha256sum --status -c READY)
}

require_unpublished() {
  local target=$1
  if [ -e "$target" ] || [ -L "$target" ]; then
    printf 'Refusing to replace existing non-ready path: %s\n' "$target" >&2
    return 1
  fi
}

fetch_checked() {
  local url=$1
  local destination=$2
  local expected_sha256=$3
  local partial="$destination.part"
  mkdir -p "$(dirname "$destination")"
  if [ -f "$destination" ]; then
    printf '%s  %s\n' "$expected_sha256" "$destination" | sha256sum -c -
    return 0
  fi
  curl --fail --location --retry 8 --retry-delay 5 --retry-all-errors \
    --continue-at - --output "$partial" "$url"
  printf '%s  %s\n' "$expected_sha256" "$partial" | sha256sum -c -
  mv "$partial" "$destination"
}

fetch_pinned() {
  local url=$1
  local destination=$2
  local partial="$destination.part"
  mkdir -p "$(dirname "$destination")"
  if [ -f "$destination" ]; then
    printf 'reuse staged %s\n' "$destination"
    return 0
  fi
  curl --fail --location --retry 8 --retry-delay 5 --retry-all-errors \
    --output "$partial" "$url"
  mv "$partial" "$destination"
}

make_inventory() {
  local data_root=$1
  local output=$2
  find -L "$data_root" -type f -printf '%P\0' |
    LC_ALL=C sort -z |
    while IFS= read -r -d '' relative_path; do
      local hash bytes
      hash=$(sha256sum -- "$data_root/$relative_path" | awk '{print $1}')
      bytes=$(stat -Lc '%s' -- "$data_root/$relative_path")
      printf '%s\t%s\t%s\n' "$hash" "$bytes" "$relative_path"
    done > "$output"
}

inventory_files() {
  wc -l < "$1" | tr -d ' '
}

inventory_bytes() {
  awk -F '\t' '{ total += $2 } END { printf "%.0f", total }' "$1"
}

inventory_hash() {
  sha256sum "$1" | awk '{print $1}'
}

install_uv() {
  local archive="$UV_ROOT/uv-x86_64-unknown-linux-gnu.tar.gz"
  if [ -x "$UV" ]; then
    "$UV" --version | grep -F "uv $UV_VERSION"
    return 0
  fi
  fetch_checked \
    "https://github.com/astral-sh/uv/releases/download/$UV_VERSION/uv-x86_64-unknown-linux-gnu.tar.gz" \
    "$archive" \
    2e2b37e9811e17675a9e70bed5e1a58fc8c0388be63d751d72cc735188c149ff
  tar -xzf "$archive" --strip-components=1 -C "$UV_ROOT" \
    uv-x86_64-unknown-linux-gnu/uv uv-x86_64-unknown-linux-gnu/uvx
  "$UV" --version
}

download_release() {
  local base="https://huggingface.co/datasets/dexverse/DexVerse_release/resolve/$RELEASE_REVISION"
  local archives="$DEXVERSE_STAGE/archives"
  fetch_checked "$base/assets/core_assets.zip?download=true" \
    "$archives/assets/core_assets.zip" \
    762b3985f37b031a97676be464a7a89b9b897d730f12859ec6cda1579a130004
  fetch_checked "$base/assets/long_horizon_extra.zip?download=true" \
    "$archives/assets/long_horizon_extra.zip" \
    efd34ee6817ef262b35c896b8e66216258514c47cbb985af8f13570c9627927f
  fetch_checked "$base/assets/polyhaven_hdris.zip?download=true" \
    "$archives/assets/polyhaven_hdris.zip" \
    83737bc50df13d80cbc96e7139334ec9f5efa7e2c7187b0d391176ec3942c675
  fetch_checked "$base/robot_agents/shadow.zip?download=true" \
    "$archives/robot_agents/shadow.zip" \
    a7bd51f2806c63356e2c88f3a72b8523f9f8172cafc1bf8a168711e6ab8d2f06
  fetch_pinned "$base/assets/MANIFEST.json" \
    "$DEXVERSE_STAGE/source-metadata/assets-manifest.json"
  fetch_pinned "$base/robot_agents/MANIFEST.json" \
    "$DEXVERSE_STAGE/source-metadata/robot-agents-manifest.json"
}

extract_release() {
  local archives="$DEXVERSE_STAGE/archives"
  mkdir -p "$DEXVERSE_STAGE/data/assets" "$DEXVERSE_STAGE/data/robot_agents"
  unzip -q -o "$archives/assets/core_assets.zip" -d "$DEXVERSE_STAGE/data/assets"
  unzip -q -o "$archives/assets/long_horizon_extra.zip" -d "$DEXVERSE_STAGE/data/assets"
  unzip -q -o "$archives/assets/polyhaven_hdris.zip" -d "$DEXVERSE_STAGE/data/assets"
  unzip -q -o "$archives/robot_agents/shadow.zip" -d "$DEXVERSE_STAGE/data/robot_agents"
}

fetch_manitwin_tooling() {
  local base="https://raw.githubusercontent.com/ycyao216/DexVerse/$DEXVERSE_COMMIT/scripts/asset_tools"
  fetch_pinned "$base/download_selected_objects.py" "$TOOLING/download_selected_objects.py"
  fetch_pinned "$base/selected_100_medium_diverse_seed42.csv" "$MANITWIN_STAGE/selection.csv"
}

download_manitwin() {
  mkdir -p "$MANITWIN_STAGE/data/objects"
  "$UV" run --python "$PYTHON" --with "huggingface-hub==$HF_HUB_VERSION" \
    "$TOOLING/download_selected_objects.py" \
    "$MANITWIN_STAGE/data/objects" \
    --csv "$MANITWIN_STAGE/selection.csv" \
    --revision "$MANITWIN_REVISION" \
    --staging-dir "$STAGING/huggingface/manitwin" \
    --max-workers 8
}

write_dexverse_manifest() {
  local created_at files bytes inventory_sha256 assets_manifest_sha256 robot_manifest_sha256
  created_at=$(date -u +%FT%TZ)
  make_inventory "$DEXVERSE_STAGE/data" "$DEXVERSE_STAGE/inventory.sha256"
  files=$(inventory_files "$DEXVERSE_STAGE/inventory.sha256")
  bytes=$(inventory_bytes "$DEXVERSE_STAGE/inventory.sha256")
  inventory_sha256=$(inventory_hash "$DEXVERSE_STAGE/inventory.sha256")
  assets_manifest_sha256=$(sha256sum "$DEXVERSE_STAGE/source-metadata/assets-manifest.json" | awk '{print $1}')
  robot_manifest_sha256=$(sha256sum "$DEXVERSE_STAGE/source-metadata/robot-agents-manifest.json" | awk '{print $1}')

  jq -n \
    --arg created_at "$created_at" \
    --arg revision "$RELEASE_REVISION" \
    --arg code_commit "$DEXVERSE_COMMIT" \
    --arg inventory_sha256 "$inventory_sha256" \
    --arg assets_manifest_sha256 "$assets_manifest_sha256" \
    --arg robot_manifest_sha256 "$robot_manifest_sha256" \
    --argjson file_count "$files" \
    --argjson payload_bytes "$bytes" \
    '{schema_version:"dataset-resource/v1",
      resource_id:("resource:huggingface/dexverse/DexVerse_release@"+$revision),
      status:"ready",kind:"simulation_assets",format:"dexverse-release-assets",created_at:$created_at,
      source:{provider:"huggingface",repository_type:"dataset",namespace:"dexverse",
        name:"DexVerse_release",revision:$revision,
        url:"https://huggingface.co/datasets/dexverse/DexVerse_release"},
      materialization:{completeness:"full_release_bundles",mode:"downloaded",
        data_root:"data",components:["assets/core_assets","assets/long_horizon_extra",
        "assets/polyhaven_hdris","robot_agents/shadow"]},
      integrity:{inventory_path:"inventory.sha256",
        inventory_format:"sha256<TAB>size_bytes<TAB>relative_path",
        inventory_order:"C-locale relative-path order",inventory_sha256:$inventory_sha256,
        file_count:$file_count,payload_bytes:$payload_bytes,
        archives:[
          {path:"archives/assets/core_assets.zip",sha256:"762b3985f37b031a97676be464a7a89b9b897d730f12859ec6cda1579a130004"},
          {path:"archives/assets/long_horizon_extra.zip",sha256:"efd34ee6817ef262b35c896b8e66216258514c47cbb985af8f13570c9627927f"},
          {path:"archives/assets/polyhaven_hdris.zip",sha256:"83737bc50df13d80cbc96e7139334ec9f5efa7e2c7187b0d391176ec3942c675"},
          {path:"archives/robot_agents/shadow.zip",sha256:"a7bd51f2806c63356e2c88f3a72b8523f9f8172cafc1bf8a168711e6ab8d2f06"}],
        source_metadata_sha256:{"source-metadata/assets-manifest.json":$assets_manifest_sha256,
          "source-metadata/robot-agents-manifest.json":$robot_manifest_sha256}},
      compatibility:{source_repository:"https://github.com/ycyao216/DexVerse",source_commit:$code_commit},
      provenance:{publisher:"ops/datasets/download_dexverse_assets.sbatch"},
      license:{status:"not_recorded",action_required:"Review upstream terms before redistribution."}}' \
    > "$DEXVERSE_STAGE/manifest.json.tmp"
  mv "$DEXVERSE_STAGE/manifest.json.tmp" "$DEXVERSE_STAGE/manifest.json"
  (cd "$DEXVERSE_STAGE" && sha256sum manifest.json > READY)
}

write_manitwin_manifest() {
  local created_at files bytes inventory_sha256 selection_sha256 tooling_sha256 object_count
  created_at=$(date -u +%FT%TZ)
  make_inventory "$MANITWIN_STAGE/data" "$MANITWIN_STAGE/inventory.sha256"
  files=$(inventory_files "$MANITWIN_STAGE/inventory.sha256")
  bytes=$(inventory_bytes "$MANITWIN_STAGE/inventory.sha256")
  inventory_sha256=$(inventory_hash "$MANITWIN_STAGE/inventory.sha256")
  selection_sha256=$(sha256sum "$MANITWIN_STAGE/selection.csv" | awk '{print $1}')
  tooling_sha256=$(sha256sum "$TOOLING/download_selected_objects.py" | awk '{print $1}')
  object_count=$(find "$MANITWIN_STAGE/data/objects" -mindepth 2 -maxdepth 2 -type d | wc -l | tr -d ' ')

  jq -n \
    --arg created_at "$created_at" \
    --arg revision "$MANITWIN_REVISION" \
    --arg selection "$MANITWIN_SELECTION" \
    --arg code_commit "$DEXVERSE_COMMIT" \
    --arg inventory_sha256 "$inventory_sha256" \
    --arg selection_sha256 "$selection_sha256" \
    --arg tooling_sha256 "$tooling_sha256" \
    --argjson file_count "$files" \
    --argjson payload_bytes "$bytes" \
    --argjson object_count "$object_count" \
    '{schema_version:"dataset-resource/v1",
      resource_id:("resource:huggingface/ManiTwin/ManiTwin-100K@"+$revision+"#subset="+$selection),
      status:"ready",kind:"simulation_assets",format:"manitwin-object-assets",created_at:$created_at,
      source:{provider:"huggingface",repository_type:"dataset",namespace:"ManiTwin",
        name:"ManiTwin-100K",revision:$revision,
        url:"https://huggingface.co/datasets/ManiTwin/ManiTwin-100K"},
      materialization:{completeness:"selected_subset",mode:"downloaded",data_root:"data/objects",
        selection:{id:$selection,path:"selection.csv",sha256:$selection_sha256,
          selected_object_instances:$object_count}},
      integrity:{inventory_path:"inventory.sha256",
        inventory_format:"sha256<TAB>size_bytes<TAB>relative_path",
        inventory_order:"C-locale relative-path order",inventory_sha256:$inventory_sha256,
        file_count:$file_count,payload_bytes:$payload_bytes},
      provenance:{publisher:"ops/datasets/download_dexverse_assets.sbatch",
        selection_tool_repository:"https://github.com/ycyao216/DexVerse",
        selection_tool_commit:$code_commit,selection_tool_sha256:$tooling_sha256},
      license:{status:"not_recorded",action_required:"Review upstream terms before redistribution."}}' \
    > "$MANITWIN_STAGE/manifest.json.tmp"
  mv "$MANITWIN_STAGE/manifest.json.tmp" "$MANITWIN_STAGE/manifest.json"
  (cd "$MANITWIN_STAGE" && sha256sum manifest.json > READY)
}

publish_revision() {
  local staged=$1
  local target=$2
  [ -f "$staged/manifest.json" ]
  [ -f "$staged/READY" ]
  (cd "$staged" && sha256sum -c READY)
  require_unpublished "$target"
  mkdir -p "$(dirname "$target")"
  chmod a-w "$staged/manifest.json" "$staged/inventory.sha256" "$staged/READY"
  mv -T "$staged" "$target"
  (cd "$target" && sha256sum -c READY)
}

build_bundle() {
  local view="$BUNDLE_STAGE/view/source/dexverse/dexverse"
  local tab files bytes inventory_sha256 created_at
  mkdir -p "$view/assets" "$view/robot_agents"
  ln -s "$DEXVERSE_RESOURCE/data/assets/core_assets" "$view/assets/core_assets"
  ln -s "$DEXVERSE_RESOURCE/data/assets/long_horizon_extra" "$view/assets/long_horizon_extra"
  ln -s "$DEXVERSE_RESOURCE/data/assets/polyhaven_hdris" "$view/assets/polyhaven_hdris"
  ln -s "$MANITWIN_RESOURCE/data/objects" "$view/assets/mani_twin_selected"
  ln -s "$DEXVERSE_RESOURCE/data/robot_agents/shadow" "$view/robot_agents/shadow"

  tab=$(printf '\t')
  {
    awk -F '\t' 'BEGIN {OFS="\t"} {print $1,$2,"source/dexverse/dexverse/" $3}' \
      "$DEXVERSE_RESOURCE/inventory.sha256"
    awk -F '\t' 'BEGIN {OFS="\t"} {path=$3; sub(/^objects\//,"",path);
      print $1,$2,"source/dexverse/dexverse/assets/mani_twin_selected/" path}' \
      "$MANITWIN_RESOURCE/inventory.sha256"
  } | LC_ALL=C sort -t "$tab" -k3,3 > "$BUNDLE_STAGE/inventory.sha256"

  files=$(inventory_files "$BUNDLE_STAGE/inventory.sha256")
  bytes=$(inventory_bytes "$BUNDLE_STAGE/inventory.sha256")
  inventory_sha256=$(inventory_hash "$BUNDLE_STAGE/inventory.sha256")
  created_at=$(date -u +%FT%TZ)

  jq -n \
    --arg created_at "$created_at" \
    --arg version "$BUNDLE_VERSION" \
    --arg release_revision "$RELEASE_REVISION" \
    --arg manitwin_revision "$MANITWIN_REVISION" \
    --arg selection "$MANITWIN_SELECTION" \
    --arg code_commit "$DEXVERSE_COMMIT" \
    --arg dexverse_manifest "$DEXVERSE_RESOURCE/manifest.json" \
    --arg manitwin_manifest "$MANITWIN_RESOURCE/manifest.json" \
    --arg inventory_sha256 "$inventory_sha256" \
    --argjson file_count "$files" \
    --argjson payload_bytes "$bytes" \
    '{schema_version:"dataset-bundle/v1",bundle_id:("bundle:dexverse/simulation-assets@"+$version),
      status:"ready",created_at:$created_at,
      roles:{simulation_assets:[
        {resource:("resource:huggingface/dexverse/DexVerse_release@"+$release_revision),
          manifest:$dexverse_manifest,mounts:[
            {source:"data/assets/core_assets",target:"source/dexverse/dexverse/assets/core_assets"},
            {source:"data/assets/long_horizon_extra",target:"source/dexverse/dexverse/assets/long_horizon_extra"},
            {source:"data/assets/polyhaven_hdris",target:"source/dexverse/dexverse/assets/polyhaven_hdris"},
            {source:"data/robot_agents/shadow",target:"source/dexverse/dexverse/robot_agents/shadow"}]},
        {resource:("resource:huggingface/ManiTwin/ManiTwin-100K@"+$manitwin_revision+"#subset="+$selection),
          manifest:$manitwin_manifest,mounts:[
            {source:"data/objects",target:"source/dexverse/dexverse/assets/mani_twin_selected"}]}]},
      view:{mode:"reference_overlay",root:"view",copied_payload_bytes:0},
      integrity:{inventory_path:"inventory.sha256",
        inventory_format:"sha256<TAB>size_bytes<TAB>relative_path",
        inventory_order:"C-locale relative-path order",inventory_sha256:$inventory_sha256,
        file_count:$file_count,payload_bytes:$payload_bytes},
      compatibility:{repository:"https://github.com/ycyao216/DexVerse",commit:$code_commit,
        expected_package_root:"view/source/dexverse/dexverse"},
      provenance:{publisher:"ops/datasets/download_dexverse_assets.sbatch"}}' \
    > "$BUNDLE_STAGE/manifest.json.tmp"
  mv "$BUNDLE_STAGE/manifest.json.tmp" "$BUNDLE_STAGE/manifest.json"
  (cd "$BUNDLE_STAGE" && sha256sum manifest.json > READY)

  while IFS= read -r -d '' link; do
    [ -e "$link" ] || { printf 'Broken bundle reference: %s\n' "$link" >&2; return 1; }
  done < <(find "$BUNDLE_STAGE/view" -type l -print0)
}

if ready_manifest "$DEXVERSE_RESOURCE"; then
  printf 'READY existing DexVerse resource %s\n' "$DEXVERSE_RESOURCE"
else
  require_unpublished "$DEXVERSE_RESOURCE"
  mkdir -p "$DEXVERSE_STAGE/archives/assets" "$DEXVERSE_STAGE/archives/robot_agents" \
    "$DEXVERSE_STAGE/source-metadata"
  stage dexverse_download download_release
  stage dexverse_extract extract_release
  stage dexverse_manifest write_dexverse_manifest
  stage dexverse_publish publish_revision "$DEXVERSE_STAGE" "$DEXVERSE_RESOURCE"
fi

if ready_manifest "$MANITWIN_RESOURCE"; then
  printf 'READY existing ManiTwin resource %s\n' "$MANITWIN_RESOURCE"
else
  require_unpublished "$MANITWIN_RESOURCE"
  mkdir -p "$MANITWIN_STAGE"
  stage uv_install install_uv
  stage manitwin_tooling fetch_manitwin_tooling
  stage manitwin_download download_manitwin
  stage manitwin_manifest write_manitwin_manifest
  stage manitwin_publish publish_revision "$MANITWIN_STAGE" "$MANITWIN_RESOURCE"
fi

if ready_manifest "$BUNDLE"; then
  printf 'READY existing bundle %s\n' "$BUNDLE"
else
  require_unpublished "$BUNDLE"
  mkdir -p "$BUNDLE_STAGE"
  stage bundle_build build_bundle
  stage bundle_publish publish_revision "$BUNDLE_STAGE" "$BUNDLE"
fi

printf 'DATASET_RESOURCE %s\n' "$DEXVERSE_RESOURCE"
printf 'DATASET_RESOURCE %s\n' "$MANITWIN_RESOURCE"
printf 'DATASET_BUNDLE %s\n' "$BUNDLE"

