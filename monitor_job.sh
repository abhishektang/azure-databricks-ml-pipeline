#!/bin/bash

# Monitor Databricks job execution
RUN_ID="743534033370127"
DATABRICKS_CLI="/Users/abhishektanguturi/Projects/env/bin/databricks"
OUTPUT_DIR="/Users/abhishektanguturi/Projects/env/Assignment/INFS7203"

echo "Monitoring job execution (Run ID: $RUN_ID)"
echo "Press Ctrl+C to stop monitoring"
echo ""

while true; do
  STATUS=$($DATABRICKS_CLI runs get --run-id $RUN_ID 2>/dev/null | grep '"life_cycle_state"' | head -1 | cut -d'"' -f4)
  RESULT=$($DATABRICKS_CLI runs get --run-id $RUN_ID 2>/dev/null | grep '"result_state"' | head -1 | cut -d'"' -f4)
  
  echo "[$(date '+%H:%M:%S')] Status: $STATUS $([ -n "$RESULT" ] && echo "| Result: $RESULT")"
  
  if [ "$STATUS" = "TERMINATED" ] || [ "$STATUS" = "INTERNAL_ERROR" ]; then
    echo ""
    echo "================================"
    if [ "$RESULT" = "SUCCESS" ]; then
      echo "✅ JOB COMPLETED SUCCESSFULLY!"
      echo "================================"
      echo ""
      echo "Downloading results..."
      $DATABRICKS_CLI fs cp dbfs:/FileStore/output/s4845110.infs4203 $OUTPUT_DIR/s4845110.infs4203 --overwrite
      echo ""
      echo "✅ Results saved to: $OUTPUT_DIR/s4845110.infs4203"
      echo ""
      echo "First 5 predictions:"
      head -5 $OUTPUT_DIR/s4845110.infs4203
      echo "..."
      echo "Last 3 lines (including scores):"
      tail -3 $OUTPUT_DIR/s4845110.infs4203
    else
      echo "❌ JOB FAILED: $RESULT"
      echo "================================"
      echo "Check logs at: https://adb-1889721579194859.19.azuredatabricks.net/#job/743534033370127/run/1"
    fi
    break
  fi
  
  sleep 30
done
