import os

current_location = os.getcwd()
print(f"Jupyter is currently running in: {current_location}")
print("-" * 40)
print("Files and folders visible in this exact location:")

# List everything in the folder
for item in os.listdir(current_location):
    print(f"  - {item}")
