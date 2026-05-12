#!/bin/bash
# Author: Yunhao Deng <yunhao.deng@kuleuven.be>
#         Fanchen Kong <fanchen.kong@kuleuven.be>

# This script is meant to be run outside of the Docker container, on the host machine. It removes all the vendor-specific modules and DC scripts that were initialized by 1_git_pull_private_modules.sh. This is useful for cleaning up the environment before starting a new tapeout iteration, or for resetting the environment if something goes wrong.

script_dir="$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

# Repository root (two levels up from this script)
repo_root="$(cd "$script_dir/../../" && pwd)"

# Remove DC scripts
rm -rf "$script_dir/HeMAiAv2_tapeout"

# Remove vendor-specific modules
rm -rf "$repo_root/hw/hemaia/tech_cells_tsmc16"
rm -rf "$repo_root/hw/hemaia/hemaia_d2d_link"
rm -rf "$repo_root/hw/hemaia/hemaia_clk_rst_controller"

# Reset Bender.local entries to empty paths
# Path to Bender.local in the repo root
bender_local_file="$repo_root/Bender.local"

upsert_bender_local_path() {
	local key="$1"
	local path="$2"
	sed -i "/^[[:space:]]*$key:/d" "$bender_local_file"
	printf '  %s: { path: %s }\n' "$key" "$path" >> "$bender_local_file"
}

upsert_bender_local_git() {
	local key="$1"
	local git="$2"
	local rev="$3"
	sed -i "/^[[:space:]]*$key:/d" "$bender_local_file"
	printf '  %s: { git: %s, rev    : %s }\n' "$key" "$git" "$rev" >> "$bender_local_file"
}

if [ -d "$repo_root/local_mirrors/hemaia_d2d_link" ]; then
	upsert_bender_local_path "hemaia_d2d_link" "local_mirrors/hemaia_d2d_link"
else
	sed -i '/^[[:space:]]*hemaia_d2d_link:/d' "$bender_local_file"
fi

if [ -d "$repo_root/local_mirrors/tech_cells_generic" ]; then
	upsert_bender_local_path "tech_cells_generic" "local_mirrors/tech_cells_generic"
else
	upsert_bender_local_git "tech_cells_generic" "https://github.com/KULeuven-MICAS/tech_cells_generic.git" "master"
fi

if [ -d "$repo_root/local_mirrors/hemaia_clk_rst_controller" ]; then
	upsert_bender_local_path "hemaia_clk_rst_controller" "local_mirrors/hemaia_clk_rst_controller"
else
	sed -i '/^[[:space:]]*hemaia_clk_rst_controller:/d' "$bender_local_file"
fi

# Remove the Bender.lock
rm -f "$repo_root/Bender.lock"
