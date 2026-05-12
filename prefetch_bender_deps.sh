#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
MIRROR_DIR="$ROOT_DIR/local_mirrors"
PAUSE_SECS="${PAUSE_SECS:-8}"
RETRY_MAX="${RETRY_MAX:-3}"
SYNC_ONLY="${SYNC_ONLY:-}"

git_safe() {
  env \
    GIT_CONFIG_COUNT=7 \
    GIT_CONFIG_KEY_0=http.version \
    GIT_CONFIG_VALUE_0=HTTP/1.1 \
    GIT_CONFIG_KEY_1=http.maxRequests \
    GIT_CONFIG_VALUE_1=1 \
    GIT_CONFIG_KEY_2=http.lowSpeedLimit \
    GIT_CONFIG_VALUE_2=1 \
    GIT_CONFIG_KEY_3=http.lowSpeedTime \
    GIT_CONFIG_VALUE_3=1800 \
    GIT_CONFIG_KEY_4=fetch.parallel \
    GIT_CONFIG_VALUE_4=1 \
    GIT_CONFIG_KEY_5=submodule.fetchJobs \
    GIT_CONFIG_VALUE_5=1 \
    GIT_CONFIG_KEY_6=init.defaultBranch \
    GIT_CONFIG_VALUE_6=main \
    git "$@"
}

log() {
  echo "[$(date '+%H:%M:%S')] $*"
}

should_sync() {
  local name="$1"

  if [[ -z "$SYNC_ONLY" ]]; then
    return 0
  fi

  [[ ",$SYNC_ONLY," == *",$name,"* ]]
}

pause_between_repos() {
  if [[ "$PAUSE_SECS" -gt 0 ]]; then
    log "pause ${PAUSE_SECS}s before next repository"
    sleep "$PAUSE_SECS"
  fi
}

check_network_failure() {
  local output="$1"

  grep -Eq 'early EOF|invalid index-pack output|RPC failed|SSL_ERROR_SYSCALL|transfer closed|Operation too slow|curl 28|curl 56|curl 92|unexpected disconnect' <<<"$output"
}

fetch_with_retry() {
  local name="$1"
  shift

  local attempt output status

  for ((attempt = 1; attempt <= RETRY_MAX; attempt++)); do
    if output=$(git_safe "$@" 2>&1); then
      printf '%s\n' "$output"
      return 0
    fi

    status=$?
    printf '%s\n' "$output"

    if ! check_network_failure "$output"; then
      return "$status"
    fi

    log "network failure on $name, attempt $attempt/$RETRY_MAX"
    find "${@:2:1}" -type f \( -name 'tmp_pack_*' -o -name 'index.lock' -o -name 'packed-refs.lock' \) -delete 2>/dev/null || true

    if (( attempt == RETRY_MAX )); then
      return "$status"
    fi

    sleep $((PAUSE_SECS * attempt))
  done
}

ensure_repo() {
  local repo_dir="$1"
  local remote_url="$2"

  if [[ ! -d "$repo_dir/.git" ]]; then
    mkdir -p "$repo_dir"
    git_safe -C "$repo_dir" init
  fi

  if git -C "$repo_dir" remote get-url origin >/dev/null 2>&1; then
    git -C "$repo_dir" remote set-url origin "$remote_url"
  else
    git -C "$repo_dir" remote add origin "$remote_url"
  fi
}

checkout_ref() {
  local name="$1"
  local repo_dir="$2"
  local ref="$3"
  local expected_commit

  if [[ "$ref" == refs/heads/* || "$ref" == refs/tags/* ]]; then
    fetch_with_retry "$name" -C "$repo_dir" fetch --force --depth 1 origin "$ref"
    git_safe -C "$repo_dir" checkout --detach FETCH_HEAD
    git_safe -C "$repo_dir" rev-parse FETCH_HEAD
    return
  fi

  if fetch_with_retry "$name" -C "$repo_dir" fetch --force origin "$ref"; then
    git_safe -C "$repo_dir" checkout --detach FETCH_HEAD
    return
  fi

  log "failed to fetch $name @ $ref after $RETRY_MAX attempts"
  return 1
}

sync_submodules() {
  local name="$1"
  local repo_dir="$2"

  if [[ "$name" != "snitch_cluster" ]]; then
    return
  fi

  if [[ -f "$repo_dir/.gitmodules" ]]; then
    git_safe -C "$repo_dir" submodule sync --recursive
    fetch_with_retry "$name submodules" -C "$repo_dir" submodule update --init --recursive --jobs 1
  fi
}

verify_repo() {
  local name="$1"
  local repo_dir="$2"
  local ref="$3"
  local actual_head expected_head

  actual_head=$(git -C "$repo_dir" rev-parse HEAD)

  if [[ "$ref" == refs/heads/* || "$ref" == refs/tags/* ]]; then
    expected_head=$(git -C "$repo_dir" rev-parse FETCH_HEAD)
  else
    expected_head="$ref"
  fi

  if [[ "$actual_head" != "$expected_head" ]]; then
    log "verify failed for $name: expected $expected_head got $actual_head"
    return 1
  fi

  if [[ "$name" == "snitch_cluster" ]]; then
    if git -C "$repo_dir" submodule status --recursive | grep -q '^-'; then
      log "verify failed for snitch_cluster submodules"
      return 1
    fi
  fi
}

sync_repo() {
  local name="$1"
  local url="$2"
  local ref="$3"
  local repo_dir="$MIRROR_DIR/$name"

  should_sync "$name" || return 0

  log "sync $name <- $url @ $ref"
  ensure_repo "$repo_dir" "$url"
  checkout_ref "$name" "$repo_dir" "$ref"
  sync_submodules "$name" "$repo_dir"
  verify_repo "$name" "$repo_dir" "$ref"
  pause_between_repos
}

mkdir -p "$MIRROR_DIR"

sync_repo apb https://github.com/pulp-platform/apb.git 77ddf073f194d44b9119949d2421be59789e69ae
sync_repo apb_timer https://github.com/pulp-platform/apb_timer.git 0cbc6cbc26c94b8e3bf27cc058c48ef89ea3d4c3
sync_repo apb_uart https://github.com/pulp-platform/apb_uart.git b6145341df79137ac584c83e9c081f80a7a40440
sync_repo ara https://github.com/KULeuven-MICAS/ara.git refs/heads/main
sync_repo axi_riscv_atomics https://github.com/pulp-platform/axi_riscv_atomics 97a1dd2ac643c276880420a0cf8eea697f228aa9
sync_repo axi_stream https://github.com/pulp-platform/axi_stream.git 54891ff40455ca94a37641b9da4604647878cc07
sync_repo bingo_hw_manager https://github.com/KULeuven-MICAS/bingo_hw_manager.git refs/heads/main
sync_repo cgra https://github.com/KULeuven-MICAS/snax_cgra.git refs/heads/main
sync_repo clint https://github.com/pulp-platform/clint.git e1357c1d0edddde458aec58363473605f51e539e
sync_repo cluster_icache https://github.com/KULeuven-MICAS/cluster_icache.git refs/heads/main
sync_repo common_cells https://github.com/KULeuven-MICAS/common_cells.git refs/heads/main
sync_repo common_verification https://github.com/pulp-platform/common_verification.git fb1885f48ea46164a10568aeff51884389f67ae3
sync_repo cva6 https://github.com/KULeuven-MICAS/cva6.git refs/heads/kfc/hemaia-integration
sync_repo dimc https://github.com/KULeuven-MICAS/snax-dimc.git refs/heads/main
sync_repo floo_noc https://github.com/KULeuven-MICAS/FlooNoC.git refs/heads/main
sync_repo fpnew https://github.com/pulp-platform/cvfpu.git e5aa6a01b5bbe1675c3aa8872e1203413ded83d1
sync_repo fpu_div_sqrt_mvp https://github.com/pulp-platform/fpu_div_sqrt_mvp.git 86e1f558b3c95e91577c41b2fc452c86b04e85ac
sync_repo hemaia_axi_spi_slave https://github.com/KULeuven-MICAS/hemaia_axi_spi_slave.git refs/heads/main
sync_repo hemaia_clk_rst_controller https://github.com/KULeuven-MICAS/hemaia_clk_rst_controller.git refs/heads/main
sync_repo hemaia_d2d_link https://github.com/KULeuven-MICAS/hemaia_d2d_link.git refs/heads/main
sync_repo hwpe-ctrl https://github.com/pulp-platform/hwpe-ctrl.git 1916c72f024175f1fe351acc3db3c6e9925a117d
sync_repo hwpe-mac-engine https://github.com/KULeuven-MICAS/hwpe-mac-engine.git 5d3b4525b665169fc8321c8a811f3c83ad3c72e8
sync_repo hwpe-stream https://github.com/pulp-platform/hwpe-stream.git 35f6bb74b28acc5922262cf0244e7a7c7fe6a589
sync_repo hypercorex https://github.com/KULeuven-MICAS/hypercorex.git 9e303e1f6454188538347e2f4e60b5a4a1d4bf47
sync_repo idma https://github.com/pulp-platform/iDMA.git 28a36e5e07705549e59fc33db96ab681bc1ca88e
sync_repo obi https://github.com/pulp-platform/obi.git 0155fc34e900c7c884e081c0a1114a247937ff69
sync_repo register_interface https://github.com/pulp-platform/register_interface.git d6e1d4cdaab7870f4faf3f88a1c788eaf5ac129d
sync_repo riscv-dbg https://github.com/pulp-platform/riscv-dbg 358f90110220adf7a083f8b65d157e836d706236
sync_repo scm https://github.com/pulp-platform/scm.git 1976c7efb4979271eee2abe262fde0f9a20e2557
sync_repo snitch_cluster https://github.com/wenbo-li-ee/snax_cluster.git refs/heads/swiglue
sync_repo tech_cells_generic https://github.com/KULeuven-MICAS/tech_cells_generic.git refs/heads/master
sync_repo xdma_axi_adapter https://github.com/KULeuven-MICAS/xdma_axi_adapter.git refs/heads/main

log "done, local mirrors are ready under $MIRROR_DIR"