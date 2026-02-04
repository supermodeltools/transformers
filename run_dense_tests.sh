#!/bin/bash

# Script to run tensor parallel (TP) tests for Dense models
# Tests are run in parallel using GPU pairs (each TP test uses 2 GPUs)
# Usage: ./run_dense_tests.sh [/path/to/results]
#        ./run_dense_tests.sh --report /path/to/results

# Define colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
GREY='\033[0;90m'
DIM='\033[0;90m'
NC='\033[0m' # No Color

# Number of GPUs required per TP test
GPUS_PER_TEST=2

# Define models to test (model_name -> test_file)
declare -A MODELS=(
    ["apertus"]="tests/models/apertus/test_modeling_apertus.py"
    ["arcee"]="tests/models/arcee/test_modeling_arcee.py"
    ["bart"]="tests/models/bart/test_modeling_bart.py"
    ["bigbird_pegasus"]="tests/models/bigbird_pegasus/test_modeling_bigbird_pegasus.py"
    ["bitnet"]="tests/models/bitnet/test_modeling_bitnet.py"
    ["blenderbot"]="tests/models/blenderbot/test_modeling_blenderbot.py"
    ["blenderbot_small"]="tests/models/blenderbot_small/test_modeling_blenderbot_small.py"
    ["bloom"]="tests/models/bloom/test_modeling_bloom.py"
    ["blt"]="tests/models/blt/test_modeling_blt.py"
    ["codegen"]="tests/models/codegen/test_modeling_codegen.py"
    ["cohere"]="tests/models/cohere/test_modeling_cohere.py"
    ["cohere2"]="tests/models/cohere2/test_modeling_cohere2.py"
    ["cwm"]="tests/models/cwm/test_modeling_cwm.py"
    ["ernie4_5"]="tests/models/ernie4_5/test_modeling_ernie4_5.py"
    ["exaone4"]="tests/models/exaone4/test_modeling_exaone4.py"
    ["falcon"]="tests/models/falcon/test_modeling_falcon.py"
    ["fsmt"]="tests/models/fsmt/test_modeling_fsmt.py"
    ["gemma"]="tests/models/gemma/test_modeling_gemma.py"
    ["gemma2"]="tests/models/gemma2/test_modeling_gemma2.py"
    ["gemma3"]="tests/models/gemma3/test_modeling_gemma3.py"
    ["gemma3n"]="tests/models/gemma3n/test_modeling_gemma3n.py"
    ["glm"]="tests/models/glm/test_modeling_glm.py"
    ["glm4"]="tests/models/glm4/test_modeling_glm4.py"
    ["gpt2"]="tests/models/gpt2/test_modeling_gpt2.py"
    ["gpt_bigcode"]="tests/models/gpt_bigcode/test_modeling_gpt_bigcode.py"
    ["gpt_neo"]="tests/models/gpt_neo/test_modeling_gpt_neo.py"
    ["gpt_neox"]="tests/models/gpt_neox/test_modeling_gpt_neox.py"
    ["gpt_neox_japanese"]="tests/models/gpt_neox_japanese/test_modeling_gpt_neox_japanese.py"
    ["gptj"]="tests/models/gptj/test_modeling_gptj.py"
    ["helium"]="tests/models/helium/test_modeling_helium.py"
    ["hunyuan_v1_dense"]="tests/models/hunyuan_v1_dense/test_modeling_hunyuan_v1_dense.py"
    ["jais2"]="tests/models/jais2/test_modeling_jais2.py"
    ["led"]="tests/models/led/test_modeling_led.py"
    ["lfm2"]="tests/models/lfm2/test_modeling_lfm2.py"
    ["llama"]="tests/models/llama/test_modeling_llama.py"
    ["longt5"]="tests/models/longt5/test_modeling_longt5.py"
    ["m2m_100"]="tests/models/m2m_100/test_modeling_m2m_100.py"
    ["mamba"]="tests/models/mamba/test_modeling_mamba.py"
    ["mamba2"]="tests/models/mamba2/test_modeling_mamba2.py"
    ["marian"]="tests/models/marian/test_modeling_marian.py"
    ["mbart"]="tests/models/mbart/test_modeling_mbart.py"
    ["ministral"]="tests/models/ministral/test_modeling_ministral.py"
    ["ministral3"]="tests/models/ministral3/test_modeling_ministral3.py"
    ["mistral"]="tests/models/mistral/test_modeling_mistral.py"
    ["mistral3"]="tests/models/mistral3/test_modeling_mistral3.py"
    ["modernbert_decoder"]="tests/models/modernbert_decoder/test_modeling_modernbert_decoder.py"
    ["mpt"]="tests/models/mpt/test_modeling_mpt.py"
    ["mvp"]="tests/models/mvp/test_modeling_mvp.py"
    ["nanochat"]="tests/models/nanochat/test_modeling_nanochat.py"
    ["nemotron"]="tests/models/nemotron/test_modeling_nemotron.py"
    ["olmo"]="tests/models/olmo/test_modeling_olmo.py"
    ["olmo2"]="tests/models/olmo2/test_modeling_olmo2.py"
    ["olmo3"]="tests/models/olmo3/test_modeling_olmo3.py"
    ["opt"]="tests/models/opt/test_modeling_opt.py"
    ["pegasus"]="tests/models/pegasus/test_modeling_pegasus.py"
    ["pegasus_x"]="tests/models/pegasus_x/test_modeling_pegasus_x.py"
    ["persimmon"]="tests/models/persimmon/test_modeling_persimmon.py"
    ["phi"]="tests/models/phi/test_modeling_phi.py"
    ["phi3"]="tests/models/phi3/test_modeling_phi3.py"
    ["plbart"]="tests/models/plbart/test_modeling_plbart.py"
    ["prophetnet"]="tests/models/prophetnet/test_modeling_prophetnet.py"
    ["qwen2"]="tests/models/qwen2/test_modeling_qwen2.py"
    ["qwen3"]="tests/models/qwen3/test_modeling_qwen3.py"
    ["recurrent_gemma"]="tests/models/recurrent_gemma/test_modeling_recurrent_gemma.py"
    ["rwkv"]="tests/models/rwkv/test_modeling_rwkv.py"
    ["seed_oss"]="tests/models/seed_oss/test_modeling_seed_oss.py"
    ["smollm3"]="tests/models/smollm3/test_modeling_smollm3.py"
    ["stablelm"]="tests/models/stablelm/test_modeling_stablelm.py"
    ["starcoder2"]="tests/models/starcoder2/test_modeling_starcoder2.py"
    ["t5"]="tests/models/t5/test_modeling_t5.py"
    ["t5gemma"]="tests/models/t5gemma/test_modeling_t5gemma.py"
    ["t5gemma2"]="tests/models/t5gemma2/test_modeling_t5gemma2.py"
    ["umt5"]="tests/models/umt5/test_modeling_umt5.py"
    ["vaultgemma"]="tests/models/vaultgemma/test_modeling_vaultgemma.py"
    ["xglm"]="tests/models/xglm/test_modeling_xglm.py"
    ["xlstm"]="tests/models/xlstm/test_modeling_xlstm.py"
    ["youtu"]="tests/models/youtu/test_modeling_youtu.py"
)

# Get model names array
MODEL_NAMES=(${!MODELS[@]})

# Report function - print summary from existing results directory
print_report() {
    local results_dir=$1
    
    if [ ! -d "$results_dir" ]; then
        echo "Error: Results directory '$results_dir' does not exist"
        exit 1
    fi
    
    echo "=========================================="
    echo "  Dense Models TP Test Report"
    echo "  Results directory: $results_dir"
    echo "=========================================="
    echo ""
    
    local success_count=0
    local fail_count=0
    local skip_count=0
    local missing_count=0
    
    for model_name in "${MODEL_NAMES[@]}"; do
        local result_file="$results_dir/${model_name}.result"
        if [ -f "$result_file" ]; then
            local result=$(cat "$result_file")
            if [[ "$result" == "SUCCESS" ]]; then
                echo -e "${GREEN}✓ ${model_name}: ${result}${NC}"
                ((success_count++))
            elif [[ "$result" == "SKIPPED" ]]; then
                echo -e "${GREY}○ ${model_name}: ${result}${NC}"
                ((skip_count++))
            else
                echo -e "${RED}✗ ${model_name}: ${result}${NC}"
                # Show last few lines of error
                if [ -f "$results_dir/${model_name}.log" ]; then
                    echo -e "${DIM}  Error snippet:"
                    tail -n 5 "$results_dir/${model_name}.log" | while read -r line; do echo -e "    ${DIM}${line}${NC}"; done
                fi
                ((fail_count++))
            fi
        else
            echo -e "${YELLOW}? ${model_name}: NOT RUN${NC}"
            ((missing_count++))
        fi
    done
    
    echo ""
    echo "-------------------------------------------"
    echo -e "Total: ${GREEN}${success_count} passed${NC}, ${GREY}${skip_count} skipped${NC}, ${RED}${fail_count} failed${NC}, ${YELLOW}${missing_count} not run${NC}"
    echo "=========================================="
    
    if [ $fail_count -gt 0 ]; then
        echo ""
        echo "Failed test logs available in: $results_dir"
        echo "To view: cat $results_dir/<model_name>.log"
        exit 1
    fi
}

# Handle --report argument
if [ "$1" == "--report" ]; then
    if [ -z "$2" ]; then
        echo "Usage: $0 --report /path/to/results"
        exit 1
    fi
    print_report "$2"
    exit 0
fi

# Check available GPUs and calculate parallel slots
AVAILABLE_GPUS=$(nvidia-smi -L 2>/dev/null | wc -l)
if [ "$AVAILABLE_GPUS" -lt "$GPUS_PER_TEST" ]; then
    echo "Need at least $GPUS_PER_TEST GPUs for TP tests, but only $AVAILABLE_GPUS detected!"
    exit 1
fi
NUM_PARALLEL=$((AVAILABLE_GPUS / GPUS_PER_TEST))
echo "Using $AVAILABLE_GPUS GPUs ($NUM_PARALLEL parallel test slots, $GPUS_PER_TEST GPUs each)"

# Handle results directory - use provided path or create temp directory
if [ -n "$1" ]; then
    RESULTS_DIR="$1"
    mkdir -p "$RESULTS_DIR"
    CLEANUP_RESULTS=false
elif [ -n "$RESULTS_DIR" ]; then
    # RESULTS_DIR already set via environment variable
    mkdir -p "$RESULTS_DIR"
    CLEANUP_RESULTS=false
else
    RESULTS_DIR=$(mktemp -d)
    CLEANUP_RESULTS=true
fi

# Only cleanup if we created a temp directory
if [ "$CLEANUP_RESULTS" = true ]; then
    trap "rm -rf $RESULTS_DIR" EXIT
fi

echo "Results directory: $RESULTS_DIR"

echo "=========================================="
echo "  Dense Models TP Test Script"
echo "  (Parallel execution: $NUM_PARALLEL tests at a time)"
echo "=========================================="
echo ""

# Function to run TP pytest tests on a specific GPU pair
run_test() {
    local model_name=$1
    local test_file=$2
    local slot_id=$3
    local result_file="$RESULTS_DIR/${model_name}.result"
    
    # Calculate GPU pair for this slot (slot 0 -> GPUs 0,1; slot 1 -> GPUs 2,3; etc.)
    local gpu_start=$((slot_id * GPUS_PER_TEST))
    local gpu_end=$((gpu_start + GPUS_PER_TEST - 1))
    local gpu_list="${gpu_start},${gpu_end}"
    
    echo -e "${YELLOW}[GPUs ${gpu_list}] Starting: ${model_name}${NC}"
    
    # Run only tensor parallel tests using assigned GPU pair
    CUDA_VISIBLE_DEVICES=$gpu_list \
        python -m pytest -v -rs "$test_file" -k "test_tp_" \
        > "$RESULTS_DIR/${model_name}.log" 2>&1
    
    local exit_code=$?
    local log_file="$RESULTS_DIR/${model_name}.log"
    
    # Check if all tests were skipped (exit code 0 but only skipped tests)
    local skipped_only=false
    if [ $exit_code -eq 0 ]; then
        # Check if there were any passed tests or only skipped
        if grep -q "passed" "$log_file"; then
            skipped_only=false
        elif grep -q "skipped" "$log_file"; then
            skipped_only=true
        fi
    fi
    
    # Write result to file (for collection later)
    if [ "$skipped_only" = true ]; then
        echo "SKIPPED" > "$result_file"
        echo -e "${GREY}○ [GPUs ${gpu_list}] ${model_name}: SKIPPED${NC}"
    elif [ $exit_code -eq 0 ]; then
        echo "SUCCESS" > "$result_file"
        echo -e "${GREEN}✓ [GPUs ${gpu_list}] ${model_name}: SUCCESS${NC}"
    else
        echo "FAILED (exit code: $exit_code)" > "$result_file"
        echo -e "${RED}✗ [GPUs ${gpu_list}] ${model_name}: FAILED (exit code: $exit_code)${NC}"
    fi
}

# Get number of models
NUM_MODELS=${#MODEL_NAMES[@]}

# Track PIDs for waiting
declare -a PIDS=()
declare -a SLOTS=()

# Launch tests in parallel, cycling through available GPU pairs
for i in "${!MODEL_NAMES[@]}"; do
    model_name="${MODEL_NAMES[$i]}"
    test_file="${MODELS[$model_name]}"
    slot_id=$((i % NUM_PARALLEL))
    
    # If we've used all slots, wait for a slot to free up
    if [ ${#PIDS[@]} -ge $NUM_PARALLEL ]; then
        # Wait for any one process to complete
        wait -n 2>/dev/null || wait "${PIDS[0]}"
        # Remove completed PIDs (simplified: just clear and rebuild)
        NEW_PIDS=()
        for pid in "${PIDS[@]}"; do
            if kill -0 "$pid" 2>/dev/null; then
                NEW_PIDS+=("$pid")
            fi
        done
        PIDS=("${NEW_PIDS[@]}")
    fi
    
    run_test "$model_name" "$test_file" "$slot_id" &
    PIDS+=($!)
done

# Wait for all remaining background jobs to complete
echo ""
echo "Waiting for all tests to complete..."
wait

# Print summary
echo ""
echo "=========================================="
echo "  SUMMARY"
echo "=========================================="
echo ""

success_count=0
fail_count=0
skip_count=0

for model_name in "${MODEL_NAMES[@]}"; do
    result_file="$RESULTS_DIR/${model_name}.result"
    if [ -f "$result_file" ]; then
        result=$(cat "$result_file")
        if [[ "$result" == "SUCCESS" ]]; then
            echo -e "${GREEN}✓ ${model_name}: ${result}${NC}"
            ((success_count++))
        elif [[ "$result" == "SKIPPED" ]]; then
            echo -e "${GREY}○ ${model_name}: ${result}${NC}"
            ((skip_count++))
        else
            echo -e "${RED}✗ ${model_name}: ${result}${NC}"
            # Show last few lines of error
            echo -e "${DIM}  Error snippet:"
            tail -n 5 "$RESULTS_DIR/${model_name}.log" | while read -r line; do echo -e "    ${DIM}${line}${NC}"; done
            ((fail_count++))
        fi
    else
        echo -e "${RED}✗ ${model_name}: NO RESULT (test may have crashed)${NC}"
        ((fail_count++))
    fi
done

echo ""
echo "-------------------------------------------"
echo -e "Total: ${GREEN}${success_count} passed${NC}, ${GREY}${skip_count} skipped${NC}, ${RED}${fail_count} failed${NC}"
echo "=========================================="

# Show logs for failed tests
if [ $fail_count -gt 0 ]; then
    echo ""
    echo "Failed test logs available in: $RESULTS_DIR"
    echo "To view: cat $RESULTS_DIR/<model_name>.log"
fi

# Exit with failure if any tests failed
if [ $fail_count -gt 0 ]; then
    exit 1
fi
