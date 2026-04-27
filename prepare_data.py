import os
import shutil

def main():
    os.makedirs("sim/data", exist_ok=True)

    src = "sim/data/embeddings.npy"
    dst = "botify/data/embeddings.npy"
    if os.path.exists(src):
        shutil.copy2(src, dst)
        print(f"Copied {src} -> {dst}")
    else:
        print(f"WARNING: {src} not found, contextual recommender will fail!")

    print("Data preparation complete.")


if __name__ == "__main__":
    main()
