# System Retrieval Analytic

## Cara Menjalankan Project

1. **Clone repository ini** (jika belum):
   ```bash
   git clone <repo url>
   cd retrieval-system
   ```

2. **Buat dan aktifkan virtual environment** (opsional tapi disarankan):
   - **Windows:**
     ```bash
     python -m venv .venv
     .\.venv\Scripts\activate
     ```
   - **Linux/Mac:**
     ```bash
     python3 -m venv .venv
     source .venv/bin/activate
     ```

3. **Install semua dependensi:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Jalankan aplikasi:**
   - **Dengan FastAPI/Uvicorn:**
     ```bash
     uvicorn app.main:app --reload
     ```
   - **Atau (jika hanya ingin test print):**
     ```bash
     python -m app.main
     ```

5. **Akses aplikasi:**
   - Jika menggunakan FastAPI/Uvicorn, buka browser ke: [http://localhost:8000](http://localhost:8000)

---

### Catatan Penting
- Pastikan menjalankan perintah dari folder root project (`retrieval-system`).
- Jika ada error `ModuleNotFoundError: No module named 'routers'`, pastikan menjalankan dengan `python -m app.main` dari root project, bukan dari dalam folder `app`.
- Jika menggunakan virtual environment, pastikan sudah aktif sebelum install dan menjalankan aplikasi.
- Untuk menambah/mengupdate dependensi, edit file `requirements.txt` lalu jalankan kembali `pip install -r requirements.txt`.
