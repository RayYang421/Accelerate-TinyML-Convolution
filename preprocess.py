from PIL import Image
import numpy as np


IMG_PATH = "image.png"
TXT_OUT  = "image.txt"
C_OUT    = "image.c"
VAR_NAME = "image"


img = Image.open(IMG_PATH)
img = img.convert("L").resize((32, 32))
arr = np.array(img, dtype=np.uint8)


np.savetxt(TXT_OUT, arr, fmt="%d")


with open(C_OUT, "w") as f:
    f.write(f"uint8_t {VAR_NAME}[32][32] = {{\n")
    for row in arr:
        f.write("    { " + ", ".join(map(str, row)) + " },\n")
    f.write("};\n")

print("Generated:")
print(" - image.txt")
print(" - image.c")
