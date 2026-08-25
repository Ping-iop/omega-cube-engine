import sys
sys.path.insert(0, '.')
from marp.log_analyzer import LogAnalyzer
import json
print(json.dumps(LogAnalyzer().analyze_today(), indent=2))
