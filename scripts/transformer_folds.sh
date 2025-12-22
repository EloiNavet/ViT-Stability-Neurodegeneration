#!/bin/bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TRANSFORMER_SH="${SCRIPT_DIR}/transformer.sh"

if [ ! -x "$TRANSFORMER_SH" ]; then
    echo "Error: transformer.sh not found or not executable at: $TRANSFORMER_SH"
    exit 1
fi

ACTIVE_PID=""

cleanup_active_group() {
    if [ -z "$ACTIVE_PID" ]; then
        return 0
    fi

    # If the leader PID is gone, nothing to clean up.
    if ! kill -0 "$ACTIVE_PID" 2>/dev/null; then
        ACTIVE_PID=""
        return 0
    fi

    PGID=$(ps -o pgid= -p "$ACTIVE_PID" 2>/dev/null | tr -d ' ')
    if [ -n "$PGID" ]; then
        # Only kill the process group created for this fold (avoid global pkill patterns).
        pkill -TERM -g "$PGID" 2>/dev/null || true
        sleep 2
        pkill -KILL -g "$PGID" 2>/dev/null || true
    fi

    ACTIVE_PID=""
}

trap cleanup_active_group INT TERM

# Forward all arguments to transformer.sh, but add --fold 0..9 in a loop.
for i in {0..9}; do
    echo "Running fold $i..."

    # Run each fold in its own session/process-group so we can clean up safely if needed.
    setsid "$TRANSFORMER_SH" "$@" --fold "$i" &
    ACTIVE_PID=$!

    wait "$ACTIVE_PID"
    EXIT_CODE=$?
    ACTIVE_PID=""

    # Leave a short buffer for CUDA/NCCL resource release (matches prior behavior, but without global kills).
    sleep 20

    if [ $EXIT_CODE -ne 0 ]; then
        echo "Fold $i failed with exit code $EXIT_CODE"
        exit $EXIT_CODE
    fi
done

echo "All folds completed successfully."

# Example usage:
# ./scripts/transformer_folds.sh \
#     --training-csv-dir /path/to/10fold_CV/ \
#     --save-dir /path/to/saved_models/ \
#     --intermediate-dir /path/to/intermediate/ \
#     --runname swin-5c-baseline \
#     --eval-csv /path/to/test.csv \
#     --wandb-mode online \
#     --project-name MyProject \
#     --cuda-devices 0,1 \
#     --config configs/swin-5c-no_seed-baseline.yaml
