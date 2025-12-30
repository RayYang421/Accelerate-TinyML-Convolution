import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


Height, Width = 32, 32
Kernel_Height, Kernel_Width = 3, 3

kernel = np.array([
    [1,0,-1],
    [2,0,-2],
    [1,0,-1]
], dtype=np.int32)


image = np.loadtxt("image.txt", dtype=np.int32).reshape(Height, Width)
out_c = np.loadtxt("output.txt", dtype=np.int32)


if out_c.ndim == 1:
    out_c = out_c.reshape(
        Height - Kernel_Height + 1,
        Width  - Kernel_Width  + 1
    )


out_py = np.zeros(
    (Height - Kernel_Height + 1,
     Width  - Kernel_Width  + 1),
    dtype=np.int64
)

for y in range(Height - Kernel_Height + 1):
    for x in range(Width - Kernel_Width + 1):
        out_py[y, x] = np.sum( image[y:y+Kernel_Height, x:x+Kernel_Width] * kernel)

if out_c.shape != out_py.shape:
    print(f" Shape mismatch: C={out_c.shape}, PY={out_py.shape}")

diff = out_c - out_py
absdiff = np.abs(diff)

print("=== Correctness ===")
print("Output shape:", out_c.shape)
print("Max abs diff:", absdiff.max())
print("Mismatch count:", np.count_nonzero(absdiff))


# Visualization
plt.figure(figsize=(16,4))

plt.subplot(1,4,1)
plt.imshow(image, cmap="gray")
plt.title("Input Image (32x32)")
plt.colorbar()
plt.axis("off")

plt.subplot(1,4,2)
plt.imshow(out_c, cmap="gray")
plt.title("C Conv Output")
plt.colorbar()
plt.axis("off")

plt.subplot(1,4,3)
plt.imshow(out_py, cmap="gray")
plt.title("Python Reference Conv")
plt.colorbar()
plt.axis("off")

plt.subplot(1,4,4)
plt.imshow(absdiff, cmap="hot")
plt.title("|C - Python|")
plt.colorbar()
plt.axis("off")

plt.tight_layout()
plt.savefig("visulization.png", dpi=200, bbox_inches="tight")
