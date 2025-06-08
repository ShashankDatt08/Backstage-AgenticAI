set -euo pipefail
 
SESSION_ID="${1:-1}"
API_URL="http://localhost:8000/sessions/${SESSION_ID}"
 
echo " Fetching session data for session ${SESSION_ID}..."
curl -s "${API_URL}" > session_data.json
echo " session_data.json written"
 
STATUS=$(jq -r '.status' session_data.json)
STEP=$(jq -r '.current_step' session_data.json)
TICKET=$(jq -r '.ticket_key' session_data.json)
echo " Session Status: $STATUS (step: $STEP)"
echo " Ticket: $TICKET"
echo
 
CODE=$(jq -r '.generated_code // empty' session_data.json)
if [[ -z "$CODE" ]]; then
  echo " No generated_code found in session_data.json"
  exit 1
fi
 
OUTFILE="GeneratedCode_${SESSION_ID}.java"
echo " Writing generated code to $OUTFILE"
printf "%s\n" "$CODE" > "$OUTFILE"
 
echo " Done! Open $OUTFILE to see the full class."