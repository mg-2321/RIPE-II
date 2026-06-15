#!/bin/bash
# Author: Gayatri Malladi
#
# Run one RIPE-II blackbox poisoning-rate component study.
# Expected env vars:
#   CORPUS, RATE, RETRIEVER
# Optional env vars:
#   RERANKER=none|cross_encoder
#   SWEEP_ROOT=/gscratch/.../rate_sweep_blackbox

#SBATCH --job-name=rate_component
#SBATCH --output=/gscratch/uwb/gayat23/GuardRAG/logs/rate_component_%j.out
#SBATCH --error=/gscratch/uwb/gayat23/GuardRAG/logs/rate_component_%j.err
#SBATCH --time=18:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --account=uwb
#SBATCH --partition=gpu-l40s
#SBATCH --qos=ckpt-gpu
#SBATCH --gres=gpu:1
#SBATCH --no-requeue

set -euo pipefail

PROJ=/mmfs1/home/gayat23/projects/guardrag-thesis
PYTHON=/gscratch/uwb/gayat23/conda/envs/guardrag/bin/python
SWEEP_ROOT=${SWEEP_ROOT:-/gscratch/uwb/gayat23/GuardRAG/IPI_generators/rate_sweep_blackbox}
OUTPUT_DIR=/gscratch/uwb/gayat23/GuardRAG/results/component_study/rate_sweep_blackbox

CORPUS=${CORPUS:?CORPUS is required}
RATE=${RATE:?RATE is required}
RETRIEVER=${RETRIEVER:?RETRIEVER is required}
RERANKER=${RERANKER:-none}

export HF_HOME=/gscratch/uwb/gayat23/hf_cache
export HUGGING_FACE_HUB_TOKEN=$(cat /gscratch/uwb/gayat23/hf_cache/token 2>/dev/null || true)
export HF_TOKEN=${HUGGING_FACE_HUB_TOKEN:-}
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export HF_HUB_OFFLINE=1
export OPENAI_API_KEY=$(cat /gscratch/uwb/gayat23/.openai_api_key 2>/dev/null || true)
export ANTHROPIC_API_KEY=$(cat /gscratch/uwb/gayat23/.anthropic_api_key 2>/dev/null || true)
export PYTORCH_ALLOC_CONF=${PYTORCH_ALLOC_CONF:-expandable_segments:True}
export GUARDRAG_VECTOR_BACKEND=${GUARDRAG_VECTOR_BACKEND:-faiss_hnsw}

RATE_DIR=$SWEEP_ROOT/ipi_${CORPUS}_main_rate${RATE}
MERGED_PATH=$RATE_DIR/${CORPUS}_main_rate${RATE}_attack_merged.jsonl
METADATA_PATH=$RATE_DIR/${CORPUS}_main_rate${RATE}_attack_metadata_v2.jsonl
POISONED_PATH=$RATE_DIR/${CORPUS}_main_rate${RATE}_attack.jsonl
QUERIES_FILE=$RATE_DIR/${CORPUS}_main_rate${RATE}_queries_beir.jsonl
CLEAN_PATH=$PROJ/data/corpus/beir/$CORPUS/corpus.jsonl

mkdir -p "$OUTPUT_DIR" /gscratch/uwb/gayat23/GuardRAG/logs
cd "$PROJ"

suffix=$RETRIEVER
args=(
  --corpus "$CORPUS"
  --tier main_candidate
  --retriever "$RETRIEVER"
  --top-k 10
  --candidate-pool 20
  --queries-file "$QUERIES_FILE"
  --merged-path "$MERGED_PATH"
  --clean-path "$CLEAN_PATH"
  --metadata-path "$METADATA_PATH"
  --poisoned-path "$POISONED_PATH"
)

if [[ "$RERANKER" != "none" && -n "$RERANKER" ]]; then
  suffix="${suffix}_ce"
  args+=(--reranker "$RERANKER" --reranker-model cross-encoder/ms-marco-MiniLM-L-12-v2 --reranker-top-n 10)
fi

output="$OUTPUT_DIR/${CORPUS}_rate${RATE}_${suffix}.jsonl"
args+=(--output "$output")

echo "============================================================"
echo "RIPE-II Rate-Sweep Component Study"
echo "============================================================"
echo "Corpus     : $CORPUS"
echo "Rate       : $RATE"
echo "Retriever  : $RETRIEVER"
echo "Reranker   : $RERANKER"
echo "Node       : $(hostname)"
echo "Output     : $output"
echo "============================================================"

"$PYTHON" -u evaluation/run_component_study.py "${args[@]}"
