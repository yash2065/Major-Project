import os

folder = "D:\Major Project\data\\13"
for count in range(0,64):
    for filename in os.listdir(folder):
        dst = f"1-{str(count+1)}.jpg"
        src = f"{folder}/{filename}"
        dst = f"{folder}/{dst}"

    os.rename(src, dst)