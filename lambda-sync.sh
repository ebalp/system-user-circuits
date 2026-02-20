#!/bin/bash
# lambda-sync.sh — Sync your Lambda AI filesystem with your personal bucket.
#
# Persists local data (numpy arrays, experiment results, etc.) across instances.
# Code goes in git; data goes here.
#
# Usage:
#   ./lambda-sync.sh <config-file> <upload|download|setup>
#
# First time setup:
#   1. Name your filesystem after yourself (e.g., fs-enrique, fs-alice)
#   2. Clone the repo into the filesystem
#   3. cp sync.env.template <your-name>.sync.env   (fill in your details)
#   4. ./lambda-sync.sh <your-name>.sync.env setup      ← configure git credentials
#   5. ./lambda-sync.sh <your-name>.sync.env upload      ← seed your bucket
#
# Returning to a stale filesystem (worked in another region, synced, came back):
#   1. ./lambda-sync.sh <your-name>.sync.env setup       ← configure git credentials
#   2. ./lambda-sync.sh <your-name>.sync.env download    ← pull latest from bucket
#      ... work ...
#   3. ./lambda-sync.sh <your-name>.sync.env upload      ← save before shutdown

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
        echo -e "${YELLOW}GIT_USER_NAME/GIT_USER_EMAIL not set in $CONFIG_FILE — skipping git identity.${NC}"
    fi
    if [[ -n "${GITHUB_TOKEN:-}" ]]; then
        git config --global url."https://${GITHUB_TOKEN}@github.com/".insteadOf "https://github.com/"
        echo -e "${GREEN}GitHub credentials configured.${NC}"
    else
        echo -e "${YELLOW}GITHUB_TOKEN not set in $CONFIG_FILE — skipping GitHub setup.${NC}"
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

# ---------- Detect filesystem mount from current directory ----------
FILESYSTEM_NAME="$(pwd | cut -d'/' -f4)"
LOCAL_PATH="/lambda/nfs/$FILESYSTEM_NAME"

if [[ ! -d "$LOCAL_PATH" ]]; then
    echo -e "${RED}Error: filesystem not mounted at $LOCAL_PATH${NC}"
    echo "Make sure you are running this from inside a Lambda filesystem."
    exit 1
fi

# ---------- Map Lambda credentials to what the aws CLI expects ----------
export AWS_ACCESS_KEY_ID="$LAMBDA_ACCESS_KEY_ID"
export AWS_SECRET_ACCESS_KEY="$LAMBDA_SECRET_ACCESS_KEY"
export AWS_DEFAULT_REGION="$LAMBDA_REGION"

# Lambda AI S3 adapter requires these to avoid NotImplemented checksum errors
export AWS_REQUEST_CHECKSUM_CALCULATION=when_required
export AWS_RESPONSE_CHECKSUM_VALIDATION=when_required

# ---------- Safety check: show bucket contents before syncing ----------
echo -e "${GREEN}Lambda Filesystem Sync${NC}"
echo "  Filesystem: $FILESYSTEM_NAME"
echo "  Bucket:     $BUCKET_NAME"
echo "  Endpoint:   $LAMBDA_ENDPOINT_URL"
echo "  Mount:      $LOCAL_PATH"
echo "  Mode:       $MODE"
echo ""

BUCKET_CONTENTS="$(aws s3 ls "s3://$BUCKET_NAME/" --endpoint-url "$LAMBDA_ENDPOINT_URL" 2>&1 || true)"

if [[ -z "$BUCKET_CONTENTS" ]]; then
    echo -e "${YELLOW}Bucket is empty (first sync?).${NC}"
else
    echo "Top-level contents of this bucket:"
    echo "$BUCKET_CONTENTS"
fi

echo ""
read -r -p "You are on filesystem '$FILESYSTEM_NAME'. Proceed with $MODE? [y/N] " CONFIRM
if [[ "$CONFIRM" != "y" && "$CONFIRM" != "Y" ]]; then
    echo "Aborted."
    exit 0
fi

echo ""

# ---------- Sync ----------
if [[ "$MODE" == "upload" ]]; then
    echo -e "${YELLOW}Uploading $LOCAL_PATH → bucket/$FILESYSTEM_NAME/...${NC}"
    aws s3 sync "$LOCAL_PATH" "s3://$BUCKET_NAME/$FILESYSTEM_NAME/" \
        --endpoint-url "$LAMBDA_ENDPOINT_URL" \
        --exclude "*/.venv/*" \
        --exclude "*/.ipynb_checkpoints/*" \
        --exclude ".ipynb_checkpoints/*"
    echo -e "${GREEN}Upload complete.${NC}"

elif [[ "$MODE" == "download" ]]; then
    echo -e "${RED}WARNING: This will overwrite local files in $LOCAL_PATH with the bucket contents.${NC}"
    echo -e "${RED}Any local changes that were not uploaded will be lost.${NC}"
    read -r -p "Are you sure? [y/N] " CONFIRM_DL
    if [[ "$CONFIRM_DL" != "y" && "$CONFIRM_DL" != "Y" ]]; then
        echo "Aborted."
        exit 0
    fi
    echo ""
    echo -e "${YELLOW}Downloading bucket/$FILESYSTEM_NAME/ → $LOCAL_PATH...${NC}"
    aws s3 sync "s3://$BUCKET_NAME/$FILESYSTEM_NAME/" "$LOCAL_PATH" \
        --endpoint-url "$LAMBDA_ENDPOINT_URL" \
        --exclude "*/.venv/*" \
        --exclude "*/.ipynb_checkpoints/*" \
        --exclude ".ipynb_checkpoints/*"
    echo -e "${GREEN}Download complete.${NC}"
fi
