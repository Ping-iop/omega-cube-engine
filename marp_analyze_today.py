import sys
sys.path.insert(0, '.')
from marp.router_service import LogAnalyzer
import json
print(json.dumps(LogAnalyzer().analyze_today(), indent=2))
