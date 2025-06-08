python3 - << 'EOF'
import json
data = json.load(open('session_data.json'))
code = data.get('generated_code') or ""
with open('BidServiceTest.java','w') as f:
    f.write(code)
print(" Wrote full generated code to BidServiceTest.java")
EOF