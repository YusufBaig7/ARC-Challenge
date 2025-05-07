import os
for file in os.listdir("data/training"):
      if file.endswith(".json"):
            print(file)