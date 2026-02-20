#!/bin/bash
# lambda-sync.sh — Sync experiment data with your personal bucket.
#
# Code goes in GitHub. Data (results, reports, etc.) goes in the bucket.
#
# Usage:
#   ./lambda-sync.sh <config-file> <setup|upload|download> [path ...]
#
# Examples:
#   ./lambda-sync.sh enrique.sync.env upload
#   ./lambda-sync.sh enrique.sync.env upload phase0/data/results
#   ./lambda-sync.sh enrique.sync.env download phase0/data
#
# Patterns in .syncignore (repo root) are excluded from every sync.

set -euo pipefail

RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; NC='\033[0m'

CONFIG_FILE="${1:-}"
MODE="${2:-}"
PATHS=("${@:3}")

if [[ -z "$CONFIG_FILE" || -z "$MODE" ]]; then
    echo -e "${RED}Usage: $0 <config-file> <setup|upload|download> [path ...]${NC}"
    exit 1
fi

if [[ ! -f "$CONFIG_FILE" ]]; then
    echo -e "${RED}Error: config file not found: $CONFIG_FILE${NC}"
    exit 1
fi

if [[ "$MODE" != "setup" && "$MODE" != "upload" && "$MODE" != "download" ]]; then
    echo -e "${RED}Error: mode must be 'setup', 'upload', or 'download'${NC}"
    exit 1
fi

# ---------- Load config ----------
# shellcheck disable=SC1090
source "$CONFIG_FILE"

# ---------- Setup mode ----------
if [[ "$MODE" == "setup" ]]; then
    if [[ -n "${GIT_USER_NAME:-}" && -n "${GIT_USER_EMAIL:-}" ]]; then
        git config --global user.name "$GIT_USER_NAME"
        git config --global user.email "$GIT_USER_EMAIL"
        echo -e "${GREEN}Git identity configured: $GIT_USER_NAME <$GIT_USER_EMAIL>${NC}"
    else
        echo -e "${YELLOW}GIT_USER_NAME/GIT_USER_EMAIL not set — skipping git identity.${NC}"
    fi
    if [[ -n "${GITHUB_TOKEN:-}" ]]; then
        git config --global url."https://${GITHUB_TOKEN}@github.com/".insteadOf "https://github.com/"
        echo -e "${GREEN}GitHub credentials configured.${NC}"
    else
        echo -e "${YELLOW}GITHUB_TOKEN not set — skipping GitHub setup.${NC}"
    fi
    if [[ -n "${HF_TOKEN:-}" ]]; then
        echo -e "${YELLOW}HF token present. To load it in this session: source $CONFIG_FILE${NC}"
    fi

    # Install uv if not already present
    if ! command -v uv &>/dev/null && [[ ! -x "$HOME/.local/bin/uv" ]]; then
        echo -e "${YELLOW}Installing uv...${NC}"
        curl -LsSf https://astral.sh/uv/install.sh | sh
    fi
    export PATH="$HOME/.local/bin:$PATH"

    # Set up Python environment from repo root
    # UV_LINK_MODE=copy: uv cache is on local disk, .venv may be on NFS —
    # hardlinks across filesystems aren't supported, so copy upfront.
    export UV_LINK_MODE=copy
    REPO_ROOT=$(git rev-parse --show-toplevel 2>/dev/null || true)
    if [[ -n "$REPO_ROOT" ]]; then
        echo -e "${YELLOW}Setting up Python environment...${NC}"
        uv python install 3.12
        uv sync --project "$REPO_ROOT"
        echo -e "${GREEN}Python environment ready. Activate with: source $REPO_ROOT/.venv/bin/activate${NC}"
    else
        echo -e "${YELLOW}Not in a git repo — skipping uv sync. Run from the repo root after cloning.${NC}"
    fi
    exit 0
fi

# ---------- Validate sync config ----------
for VAR in BUCKET_NAME LAMBDA_ACCESS_KEY_ID LAMBDA_SECRET_ACCESS_KEY LAMBDA_REGION LAMBDA_ENDPOINT_URL; do
    if [[ -z "${!VAR:-}" ]]; then
        echo -e "${RED}Error: $VAR is not set in $CONFIG_FILE${NC}"
        exit 1
    fi
done

# ---------- Detect repo root ----------
REPO_ROOT=$(git rev-parse --show-toplevel 2>/dev/null || true)
if [[ -z "$REPO_ROOT" ]]; then
    echo -e "${RED}Error: not inside a git repository. Run this from within the repo.${NC}"
    exit 1
fi

# ---------- Map Lambda credentials ----------
export AWS_ACCESS_KEY_ID="$LAMBDA_ACCESS_KEY_ID"
export AWS_SECRET_ACCESS_KEY="$LAMBDA_SECRET_ACCESS_KEY"
export AWS_DEFAULT_REGION="$LAMBDA_REGION"

# Lambda AI S3 adapter requires these to avoid NotImplemented checksum errors
export AWS_REQUEST_CHECKSUM_CALCULATION=when_required
export AWS_RESPONSE_CHECKSUM_VALIDATION=when_required

# ---------- Load .syncignore ----------
EXCLUDE_FLAGS=()
SYNCIGNORE="$REPO_ROOT/.syncignore"
if [[ -f "$SYNCIGNORE" ]]; then
    while IFS= read -r line || [[ -n "$line" ]]; do
        [[ "$line" =~ ^[[:space:]]*# ]] && continue       # skip comments
        [[ -z "${line//[[:space:]]/}" ]] && continue       # skip blank lines
        EXCLUDE_FLAGS+=("--exclude" "$line")
    done < "$SYNCIGNORE"
fi

# ---------- Summary ----------
echo -e "${GREEN}Lambda Data Sync${NC}"
echo "  Bucket:   $BUCKET_NAME"
echo "  Endpoint: $LAMBDA_ENDPOINT_URL"
if [[ ${#PATHS[@]} -gt 0 ]]; then
    echo "  Paths:    ${PATHS[*]}"
else
    if [[ "$MODE" == "upload" ]]; then
        echo "  Paths:    all */data/ and */reports/ (auto)"
    else
        echo "  Paths:    full bucket (auto)"
    fi
fi
if [[ ${#EXCLUDE_FLAGS[@]} -gt 0 ]]; then
    echo "  Ignoring: $(( ${#EXCLUDE_FLAGS[@]} / 2 )) pattern(s) from .syncignore"
fi
echo ""

# ---------- Confirmation ----------
if [[ "$MODE" == "upload" ]]; then
    echo -e "${YELLOW}This will overwrite bucket data with local files from the paths above.${NC}"
    read -r -p "Proceed? [y/N] " CONFIRM
elif [[ "$MODE" == "download" ]]; then
    echo -e "${RED}This will overwrite local files with bucket data. Unsynced local changes will be lost.${NC}"
    read -r -p "Proceed? [y/N] " CONFIRM
fi

if [[ "$CONFIRM" != "y" && "$CONFIRM" != "Y" ]]; then
    echo "Aborted."
    exit 0
fi

echo ""

# ---------- Sync ----------
if [[ "$MODE" == "upload" ]]; then
    echo -e "${YELLOW}Uploading → bucket...${NC}"
    if [[ ${#PATHS[@]} -gt 0 ]]; then
        for p in "${PATHS[@]}"; do
            local_dir="$REPO_ROOT/$p"
            if [[ ! -d "$local_dir" ]]; then
                echo -e "${RED}  Warning: $local_dir not found, skipping.${NC}"
                continue
            fi
            echo "  $p/"
            aws s3 sync "$local_dir/" "s3://$BUCKET_NAME/$p/" \
                --endpoint-url "$LAMBDA_ENDPOINT_URL" \
                "${EXCLUDE_FLAGS[@]}"
        done
    else
        FOUND=0
        for dir in "$REPO_ROOT"/*/data "$REPO_ROOT"/*/reports; do
            [[ -d "$dir" ]] || continue
            FOUND=1
            rel="${dir#$REPO_ROOT/}"
            echo "  $rel/"
            aws s3 sync "$dir/" "s3://$BUCKET_NAME/$rel/" \
                --endpoint-url "$LAMBDA_ENDPOINT_URL" \
                "${EXCLUDE_FLAGS[@]}"
        done
        if [[ $FOUND -eq 0 ]]; then
            echo "  No data/ or reports/ directories found."
        fi
    fi
    echo -e "${GREEN}Upload complete.${NC}"

elif [[ "$MODE" == "download" ]]; then
    echo -e "${YELLOW}Downloading bucket → local...${NC}"
    if [[ ${#PATHS[@]} -gt 0 ]]; then
        for p in "${PATHS[@]}"; do
            echo "  $p/"
            mkdir -p "$REPO_ROOT/$p"
            aws s3 sync "s3://$BUCKET_NAME/$p/" "$REPO_ROOT/$p/" \
                --endpoint-url "$LAMBDA_ENDPOINT_URL" \
                "${EXCLUDE_FLAGS[@]}"
        done
    else
        aws s3 sync "s3://$BUCKET_NAME/" "$REPO_ROOT/" \
            --endpoint-url "$LAMBDA_ENDPOINT_URL" \
            "${EXCLUDE_FLAGS[@]}"
    fi
    echo -e "${GREEN}Download complete.${NC}"
fi
