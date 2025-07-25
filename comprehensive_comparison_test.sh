#!/bin/bash

# Comprehensive Healthcare Scraper Comparison Test
# Tests: main_hybrid.py vs main_enhanced.py vs main_optimized.py vs main_best_practices.py
# Corporations: LCCA (www.lcca.com) and Ensign Group (https://ensigngroup.net/)

echo "🚀 COMPREHENSIVE HEALTHCARE SCRAPER COMPARISON TEST"
echo "=================================================="
echo "📊 Testing 4 different scraping approaches"
echo "🏥 Testing 2 healthcare corporations"
echo "⏱️  Start time: $(date)"
echo ""

# Configuration
OPENROUTER_API_KEY="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"
TEST_URLS=("https://www.lcca.com" "https://ensigngroup.net")
COMPANY_NAMES=("LCCA" "Ensign_Group")
APPROACHES=("hybrid" "enhanced" "optimized" "best_practices")
SCRIPTS=("main_hybrid.py" "main_enhanced.py" "main_optimized.py" "main_best_practices.py")

# Create master results directory
MASTER_DIR="comparison_test_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$MASTER_DIR"
chmod 777 "$MASTER_DIR"

# Initialize results tracking
RESULTS_FILE="$MASTER_DIR/comparison_results.txt"
CSV_SUMMARY="$MASTER_DIR/comparison_summary.csv"

echo "Test,Company,URL,Script,Facilities_Found,Processing_Time,Success,Output_Files,Notes" > "$CSV_SUMMARY"

echo "📁 Results will be saved in: $MASTER_DIR"
echo ""

# Function to run a single test
run_test() {
    local approach=$1
    local script=$2
    local company=$3
    local url=$4
    local test_num=$5
    local total_tests=$6
    
    echo "🧪 TEST $test_num/$total_tests: $approach approach on $company"
    echo "   Script: $script"
    echo "   URL: $url"
    
    # Create specific output directory
    local output_dir="$MASTER_DIR/${approach}_${company}"
    mkdir -p "$output_dir"
    chmod 777 "$output_dir"
    
    # Record start time
    local start_time=$(date +%s)
    local start_timestamp=$(date)
    
    # Determine Docker image and command based on approach
    local docker_image="healthcare-scraper-llm"
    local docker_cmd=""
    
    case $approach in
        "hybrid")
            docker_cmd="python $script --url $url --model-preference fast --output /app/output --validate"
            ;;
        "enhanced")
            docker_cmd="python $script --url $url --provider openrouter --model meta-llama/llama-3.1-8b-instruct:free --output /app/output"
            ;;
        "optimized")
            docker_cmd="python $script --url $url --output /app/output --llm --validate"
            ;;
        "best_practices")
            docker_cmd="python $script --url $url --output /app/output --validate"
            ;;
    esac
    
    echo "   Command: $docker_cmd"
    echo "   Started: $start_timestamp"
    
    # Run the test
    local success="false"
    local error_msg=""
    local facilities_count=0
    local output_files=""
    
    if docker run --rm --user root \
        -v "$(pwd)/$output_dir:/app/output" \
        -e OPENROUTER_API_KEY="$OPENROUTER_API_KEY" \
        "$docker_image" \
        $docker_cmd > "$output_dir/execution_log.txt" 2>&1; then
        
        success="true"
        echo "   ✅ Execution completed successfully"
        
        # Count facilities found
        if ls "$output_dir"/*.csv >/dev/null 2>&1; then
            facilities_count=$(tail -n +2 "$output_dir"/*.csv 2>/dev/null | wc -l)
            output_files=$(ls "$output_dir"/*.csv "$output_dir"/*.json 2>/dev/null | wc -l)
        fi
        
    else
        success="false"
        error_msg="Docker execution failed"
        echo "   ❌ Execution failed"
    fi
    
    # Calculate processing time
    local end_time=$(date +%s)
    local processing_time=$((end_time - start_time))
    local end_timestamp=$(date)
    
    echo "   ⏱️  Processing time: ${processing_time}s"
    echo "   📊 Facilities found: $facilities_count"
    echo "   📁 Output files: $output_files"
    echo "   Completed: $end_timestamp"
    echo ""
    
    # Log detailed results
    {
        echo "=========================================="
        echo "TEST: $approach approach on $company"
        echo "URL: $url"
        echo "Script: $script"
        echo "Started: $start_timestamp"
        echo "Completed: $end_timestamp"
        echo "Processing Time: ${processing_time}s"
        echo "Success: $success"
        echo "Facilities Found: $facilities_count"
        echo "Output Files: $output_files"
        echo "Output Directory: $output_dir"
        if [ "$success" = "false" ]; then
            echo "Error: $error_msg"
        fi
        echo ""
        echo "Execution Log:"
        cat "$output_dir/execution_log.txt"
        echo ""
        echo "Files Created:"
        ls -la "$output_dir/" 2>/dev/null || echo "No files created"
        echo ""
    } >> "$RESULTS_FILE"
    
    # Add to CSV summary
    local notes=""
    if [ "$success" = "false" ]; then
        notes="$error_msg"
    fi
    
    echo "$approach,$company,$url,$script,$facilities_count,${processing_time}s,$success,$output_files,\"$notes\"" >> "$CSV_SUMMARY"
    
    # Brief pause between tests
    sleep 2
}

# Run all tests
echo "🎯 Starting comprehensive comparison..."
echo ""

test_counter=1
total_tests=$((${#APPROACHES[@]} * ${#TEST_URLS[@]}))

for i in "${!APPROACHES[@]}"; do
    approach="${APPROACHES[$i]}"
    script="${SCRIPTS[$i]}"
    
    for j in "${!TEST_URLS[@]}"; do
        url="${TEST_URLS[$j]}"
        company="${COMPANY_NAMES[$j]}"
        
        run_test "$approach" "$script" "$company" "$url" "$test_counter" "$total_tests"
        test_counter=$((test_counter + 1))
        
        # Longer pause between different companies
        if [ $j -eq 0 ] && [ $i -lt $((${#APPROACHES[@]} - 1)) ]; then
            echo "⏸️  Pausing 10 seconds before next approach..."
            sleep 10
        fi
    done
done

# Generate final summary
echo "🎉 COMPARISON TEST COMPLETE!"
echo "=========================="
echo "⏱️  Total test time: $(date)"
echo "📁 All results saved in: $MASTER_DIR"
echo ""

# Create summary report
SUMMARY_REPORT="$MASTER_DIR/FINAL_SUMMARY.md"

{
    echo "# Healthcare Scraper Comparison Test Results"
    echo ""
    echo "**Test Date:** $(date)"
    echo "**Total Tests:** $total_tests"
    echo "**Corporations Tested:** LCCA, Ensign Group"
    echo "**Approaches Tested:** Hybrid, Enhanced, Optimized, Best Practices"
    echo ""
    echo "## Quick Results Summary"
    echo ""
    echo "| Approach | LCCA Facilities | Ensign Facilities | LCCA Time | Ensign Time | Notes |"
    echo "|----------|----------------|------------------|-----------|-------------|-------|"
    
    # Parse results for summary table
    for approach in "${APPROACHES[@]}"; do
        lcca_facilities=$(grep "^$approach,LCCA," "$CSV_SUMMARY" | cut -d',' -f5)
        ensign_facilities=$(grep "^$approach,Ensign_Group," "$CSV_SUMMARY" | cut -d',' -f5)
        lcca_time=$(grep "^$approach,LCCA," "$CSV_SUMMARY" | cut -d',' -f6)
        ensign_time=$(grep "^$approach,Ensign_Group," "$CSV_SUMMARY" | cut -d',' -f6)
        
        echo "| $approach | $lcca_facilities | $ensign_facilities | $lcca_time | $ensign_time | - |"
    done
    
    echo ""
    echo "## Detailed Results"
    echo ""
    echo "See individual test directories for complete output files:"
    echo ""
    
    for approach in "${APPROACHES[@]}"; do
        for company in "${COMPANY_NAMES[@]}"; do
            echo "- **${approach}_${company}/**: Results for $approach approach on $company"
        done
    done
    
    echo ""
    echo "## Files Generated"
    echo ""
    echo "- \`comparison_results.txt\`: Detailed test logs"
    echo "- \`comparison_summary.csv\`: Machine-readable summary"
    echo "- \`FINAL_SUMMARY.md\`: This summary report"
    echo "- Individual test directories with JSON/CSV output files"
    echo ""
    echo "## Analysis"
    echo ""
    echo "### Best Performing Approach"
    echo "*(Based on total facilities found)*"
    echo ""
    
    # Calculate totals for each approach
    for approach in "${APPROACHES[@]}"; do
        total=$(grep "^$approach," "$CSV_SUMMARY" | cut -d',' -f5 | awk '{sum += $1} END {print sum}')
        echo "- **$approach**: $total total facilities"
    done
    
} > "$SUMMARY_REPORT"

# Display final results
echo "📊 FINAL RESULTS SUMMARY:"
echo "========================"
echo ""

# Show CSV summary
echo "📋 Quick Results Table:"
column -t -s',' "$CSV_SUMMARY"
echo ""

# Show totals by approach
echo "📈 Total Facilities by Approach:"
for approach in "${APPROACHES[@]}"; do
    total=$(grep "^$approach," "$CSV_SUMMARY" | cut -d',' -f5 | awk '{sum += $1} END {print sum}')
    echo "   $approach: $total facilities"
done
echo ""

# Show file structure
echo "📁 Generated Files:"
find "$MASTER_DIR" -type f | head -20
if [ $(find "$MASTER_DIR" -type f | wc -l) -gt 20 ]; then
    echo "   ... and $(( $(find "$MASTER_DIR" -type f | wc -l) - 20 )) more files"
fi
echo ""

echo "✅ Test complete! Check $MASTER_DIR for detailed results."
echo "📖 Read $SUMMARY_REPORT for analysis and recommendations."
echo ""

# Quick validation
echo "🔍 Quick Validation:"
for approach in "${APPROACHES[@]}"; do
    for company in "${COMPANY_NAMES[@]}"; do
        dir="$MASTER_DIR/${approach}_${company}"
        if [ -d "$dir" ]; then
            file_count=$(ls "$dir"/*.csv "$dir"/*.json 2>/dev/null | wc -l)
            if [ $file_count -gt 0 ]; then
                echo "   ✅ ${approach}_${company}: $file_count output files"
            else
                echo "   ❌ ${approach}_${company}: No output files"
            fi
        fi
    done
done

echo ""
echo "🎯 COMPARISON TEST COMPLETED SUCCESSFULLY!"
echo "⏱️  End time: $(date)"

