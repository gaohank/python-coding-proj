import os
import sys

print(sys.argv[0].split('/')[-1].split('.')[0])
print(sys.argv[0])
print(os.path.dirname(__file__) + "/../")
print(os.path.dirname(sys.argv[0]))
log_path = os.path.join(os.path.dirname(__file__), "../log", "11.txt")
print(log_path)
