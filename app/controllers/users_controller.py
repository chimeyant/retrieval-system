from fastapi import Request

def fetch(request:Request):
    return {
        "nama":"Ujang Selamat",
        "alamat":"Bandung"
    }