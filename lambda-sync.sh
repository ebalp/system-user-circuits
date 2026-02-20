#!/bin/bash
# lambda-sync.sh — Sync experiment data with your personal bucket.
#
# Code goes in GitHub. Data (results, reports, etc.) goes here.
# Syncs all */data/ and */reports/ directories found in the repo.
#
# Usage:
#   ./lambda-sync.sh <config-file> <setup|upload|download>

set -euo pipefail

RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; NC='\033[0m'

CONFIG_FILE="${1:-}"
MODE="${2:-}"

if [[ -z "$CONFIG_FILE" || -z "$MODE" ]]; then
    echo -e "${RED}Usage: $0 <config-file> <setup|upload|download>${NC}"
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
    if [[ -n "${HF_API_KEY:-}" ]]; then
        echo -e "${YELLOW}To use your HF token in this session: source $CONFIG_FILE${NC}"
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

# ---------- Confirmation ----------
echo -e "${GREEN}Lambda Data Sync${NC}"
echo "  Repo:     $REPO_ROOT"
echo "  Bucket:   $BUCKET_NAME"
echo "  Endpoint: $LAMBDA_ENDPOINT_URL"
echo ""

if [[ "$MODE" == "upload" ]]; then
    echo -e "${YELLOW}This will overwrite bucket data with local data from $REPO_ROOT${NC}"
    read -r -p "Proceed? [y/N] " CONFIRM
elif [[ "$MODE" == "download" ]]; then
    echo -e "${RED}This will overwrite local data in $REPO_ROOT with bucket data. Local changes not uploaded will be lost.${NC}"
    read -r -p "Proceed? [y/N] " CONFIRM
fi

if [[ "$CONFIRM" != "y" && "$CONFIRM" != "Y" ]]; then
    echo "Aborted."
    exit 0
fi

echo ""

# ---------- Sync ----------
if [[ "$MODE" == "upload" ]]; then
    echo -e "${YELLOW}Uploading data → bucket...${NC}"
    FOUND=0
    for dir in "$REPO_ROOT"/*/data "$REPO_ROOT"/*/reports; do
        [[ -d "$dir" ]] || continue
        FOUND=1
        rel="${dir#$REPO_ROOT/}"
        echo "  $rel/"
        aws s3 sync "$dir/" "s3://$BUCKET_NAME/$rel/" \
            --endpoint-url "$LAMBDA_ENDPOINT_URL"
    done
    if [[ $FOUND -eq 0 ]]; then
        echo "  No data/ or reports/ directories found."
    fi
    echo -e "${GREEN}Upload complete.${NC}"

elif [[ "$MODE" == "download" ]]; then
    echo -e "${YELLOW}Downloading bucket → repo...${NC}"
    aws s3 sync "s3://$BUCKET_NAME/" "$REPO_ROOT/" \
        --endpoint-url "$LAMBDA_ENDPOINT_URL"
    echo -e "${GREEN}Download complete.${NC}"
fi
