import os
import shutil

release_dir = "STGAT_V2_SOTA_Release"
os.makedirs(release_dir, exist_ok=True)

# 1. Definisi Folder Target
folders_to_copy = {
    "src/stgat_v2": os.path.join(release_dir, "src"),
    "doc": os.path.join(release_dir, "results"),
}

# 2. Salin Direktori Kode & Metrik
for src, dst in folders_to_copy.items():
    if os.path.exists(src):
        # Jika tujuan sudah ada, hapus dulu agar bersih
        if os.path.exists(dst): shutil.rmtree(dst)
        shutil.copytree(src, dst)
        print(f"[+] Menyalin folder: {src} -> {dst}")
        
# 3. Salin Buku Dokumentasi Ilmiah & File Penting
files_to_copy = [
    "STGAT_SOTA_Scientific_Book.html",
    "evaluate_resilient_v2.py",
    "evaluate_stage2_stage3.py"
]

for f in files_to_copy:
    if os.path.exists(f):
        shutil.copy(f, release_dir)
        print(f"[+] Menyalin file: {f}")

# 4. ZIP Direktori
print(f"[*] Mengompresi {release_dir} menjadi ZIP...")
shutil.make_archive(release_dir, 'zip', release_dir)
print(f"[+] SELESAI! File ZIP Rilis Akhir: {release_dir}.zip siap diunduh.")
