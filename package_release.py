import shutil
import os

def archive_release():
    print("[*] Tahap 4: Mengarsipkan seluruh repositori rilis menjadi ZIP...")
    source_dir = "STGAT_Final_Release"
    output_filename = "STGAT_V2_Final_Release"
    
    if not os.path.exists(source_dir):
        print(f"  [!] Kesalahan: Direktori {source_dir} tidak ditemukan!")
        return

    # Create zip archive
    zip_path = shutil.make_archive(output_filename, 'zip', source_dir)
    print(f"  [+] Repositori berhasil dikompresi: {zip_path}")
    print("\n[OK] REPOSITORI FINAL BERHASIL DIKEMAS. Silakan unduh file STGAT_V2_Final_Release.zip")

if __name__ == "__main__":
    archive_release()
