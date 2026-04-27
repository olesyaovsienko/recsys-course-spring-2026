import os
import requests
import shutil


def download_from_gdrive(file_id: str, output_path: str):
    """
    Надёжное скачивание файлов с Google Drive (включая large file confirmation).
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    session = requests.Session()

    base_url = "https://drive.google.com/uc?export=download"

    print("[download] requesting file from Google Drive...")

    response = session.get(base_url, params={"id": file_id}, stream=True)

    # Проверка на confirmation token (для больших файлов)
    token = None
    for k, v in response.cookies.items():
        if k.startswith("download_warning"):
            token = v

    if token:
        print("[download] confirmation token detected, retrying...")
        response = session.get(
            base_url,
            params={"id": file_id, "confirm": token},
            stream=True,
        )

    # Сохраняем файл
    with open(output_path, "wb") as f:
        for chunk in response.iter_content(32768):
            if chunk:
                f.write(chunk)

    print(f"[done] saved to {output_path}")


def main():
    os.makedirs("sim/data", exist_ok=True)
    os.makedirs("botify/data", exist_ok=True)

    # --- Google Drive embeddings ---
    file_id = "1DLumlgn6vyU21U5ObgGNzGxHzq7l-veQ"

    embeddings_path = "sim/data/embeddings.npy"

    download_from_gdrive(file_id, embeddings_path)

    # --- copy into botify for server ---
    dst = "botify/data/embeddings.npy"

    if os.path.exists(embeddings_path):
        shutil.copy2(embeddings_path, dst)
        print(f"[copy] {embeddings_path} -> {dst}")
    else:
        raise RuntimeError("Embeddings download failed")

    print("Data preparation complete.")


if __name__ == "__main__":
    main()