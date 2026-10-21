import os

count = 0
types = {}

print("Inside Merged:", os.listdir("Merged"))

for root, dirs, files in os.walk("Merged"):
    for f in files:
        ext = os.path.splitext(f)[1].lower()
        types[ext] = types.get(ext, 0) + 1
        if count < 10:
            print(os.path.join(root, f))
        count += 1

print("Total files:", count)
print("File types:", types)