#!/bin/bash
# Author: Gayatri Malladi
#
# Unified end-to-end evaluation launcher for RIPE-II.
# Supports multi-corpus, multi-model, multi-retriever sweeps in a single script.
#
# Examples:
#   CORPORA=nfcorpus,scifact,fiqa MODELS=llama-3.1-8b,mistral-7b RETRIEVERS=bm25 sbatch run_full_evaluation.sh
#   CORPORA=nq,msmarco MODELS=llama-3.3-70b-4bit RETRIEVERS=bm25,splade RERANKERS=none,cross_encoder sbatch run_full_evaluation.sh

#SBATCH --job-name=full_eval
#SBATCH --output=/gscratch/uwb/gayat23/GuardRAG/logs/full_eval_%j.out
#SBATCH --error=/gscratch/uwb/gayat23/GuardRAG/logs/full_eval_%j.err
#SBATCH --time=12:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --account=uwb
#SBATCH --partition=gpu-l40s
#SBATCH --qos=ckpt-gpu
#SBATCH --gres=gpu:1
#SBATCH --no-requeue

set -euo pipefail

PROJ=/mmfs1/home/gayat23/projects/guardrag-thesis
PYTHON=/gscratch/uwb/gayat23/conda/envs/guardrag/bin/python
OUTPUT_DIR=/gscratch/uwb/gayat23/GuardRAG/results/live_judge

CORPORA=${CORPORA:-nfcorpus}
TIER=${TIER:-main_candidate}
MODELS=${MODELS:-llama-3.1-8b}
JUDGE=${JUDGE:-gpt-4o-mini}
RETRIEVERS=${RETRIEVERS:-bm25}
RERANKERS=${RERANKERS:-none}
RERANKER_MODEL=${RERANKER_MODEL:-cross-encoder/ms-marco-MiniLM-L-12-v2}
RERANKER_TOP_N=${RERANKER_TOP_N:-10}
SAMPLE=${SAMPLE:-20}
QUERY=${QUERY:-}
NO_CLEAN=${NO_CLEAN:-}
GUARDS=${GUARDS:-}
QUERIES_FILE=${QUERIES_FILE:-}
MERGED_PATH=${MERGED_PATH:-}
CLEAN_PATH=${CLEAN_PATH:-}
METADATA_PATH=${METADATA_PATH:-}
POISONED_PATH=${POISONED_PATH:-}

export HF_HOME=/gscratch/uwb/gayat23/hf_cache
export HUGGING_FACE_HUB_TOKEN=$(cat /gscratch/uwb/gayat23/hf_cache/token 2>/dev/null)
export HF_TOKEN=$HUGGING_FACE_HUB_TOKEN
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export HF_HUB_OFFLINE=1
export OPENAI_API_KEY=$(cat /gscratch/uwb/gayat23/.openai_api_key 2>/dev/null)
export ANTHROPIC_API_KEY=$(cat /gscratch/uwb/gayat23/.anthropic_api_key 2>/dev/null)
export PYTORCH_ALLOC_CONF=${PYTORCH_ALLOC_CONF:-expandable_segments:True}

if [[ -z "${GUARDRAG_VECTOR_BACKEND:-}" ]]; then
    if [[ -n "${GUARDRAG_QDRANT_URL:-${QDRANT_URL:-}}" ]]; then
        export GUARDRAG_VECTOR_BACKEND="qdrant_remote"
    else
        export GUARDRAG_VECTOR_BACKEND="faiss_hnsw"
    fi
fi

split_csv() {
    local raw="$1"
    raw="${raw// /}"
    local IFS=','
    read -r -a items <<< "$raw"
    printf '%s\n' "${items[@]}"
}

mkdir -p "$OUTPUT_DIR" /gscratch/uwb/gayat23/GuardRAG/logs
cd "$PROJ"

echo "============================================================"
echo "RIPE-II Full Evaluation"
echo "============================================================"
echo "Corpora     : $CORPORA"
echo "Tier        : $TIER"
echo "Models      : $MODELS"
echo "Judge       : $JUDGE"
echo "Retrievers  : $RETRIEVERS"
echo "Rerankers   : $RERANKERS"
echo "Node        : $(hostname)"
echo "GPU         : $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A (CPU run)')"
echo "Start       : $(date)"
echo "============================================================"

mapfile -t CORPUS_LIST < <(split_csv "$CORPORA")
mapfile -t MODEL_LIST < <(split_csv "$MODELS")
mapfile -t RETRIEVER_LIST < <(split_csv "$RETRIEVERS")
mapfile -t RERANKER_LIST < <(split_csv "$RERANKERS")

for corpus in "${CORPUS_LIST[@]}"; do
    for model in "${MODEL_LIST[@]}"; do
        for retriever in "${RETRIEVER_LIST[@]}"; do
            for reranker in "${RERANKER_LIST[@]}"; do
                timestamp=$(date +%Y%m%d_%H%M%S)
                ret_suffix="$retriever"
                guard_tag="${GUARDS//+/,}"
                guard_tag="${guard_tag:-raw}"
                guard_tag="${guard_tag//,/--}"
                args=(
                    --corpus "$corpus"
                    --tier "$TIER"
                    --model "$model"
                    --judge "$JUDGE"
                    --retriever "$retriever"
                )

                if [[ "$reranker" != "none" && -n "$reranker" ]]; then
                    ret_suffix="${ret_suffix}+ce"
                    args+=(--reranker "$reranker" --reranker-model "$RERANKER_MODEL" --reranker-top-n "$RERANKER_TOP_N")
                fi

                if [[ -n "$QUERY" ]]; then
                    args+=(--query "$QUERY")
                else
                    args+=(--sample "$SAMPLE")
                fi
                [[ -n "$NO_CLEAN" ]] && args+=(--no-clean-baseline)
                [[ -n "$GUARDS" ]] && args+=(--guards "${GUARDS//+/,}")
                [[ -n "$QUERIES_FILE" ]] && args+=(--queries-file "$QUERIES_FILE")
                [[ -n "$MERGED_PATH" ]] && args+=(--merged-path "$MERGED_PATH")
                [[ -n "$CLEAN_PATH" ]] && args+=(--clean-path "$CLEAN_PATH")
                [[ -n "$METADATA_PATH" ]] && args+=(--metadata-path "$METADATA_PATH")
                [[ -n "$POISONED_PATH" ]] && args+=(--poisoned-path "$POISONED_PATH")

                output="$OUTPUT_DIR/${corpus}_${TIER}_${model}_${ret_suffix}_guards-${guard_tag}_${timestamp}.jsonl"
                args+=(--output "$output")

                echo "------------------------------------------------------------"
                echo "Corpus    : $corpus"
                echo "Model     : $model"
                echo "Retriever : $retriever"
                echo "Reranker  : $reranker"
                echo "Output    : $output"
                echo "------------------------------------------------------------"

                "$PYTHON" -u evaluation/live_judge_eval.py "${args[@]}"
            done
        done
    done
done

echo
echo "Done: $(date)"
