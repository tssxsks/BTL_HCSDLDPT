import os
import psycopg2
from dotenv import load_dotenv

# 1. Load các biến môi trường từ tệp .env
load_dotenv()

def test_db_connection():
    # 2. Lấy thông tin cấu hình từ môi trường
    # Ưu tiên dùng DATABASE_URL nếu có, nếu không thì dùng các biến lẻ
    dsn = os.getenv("DATABASE_URL")
    
    print("--- Đang kiểm tra kết nối Database ---")
    
    try:
        if dsn:
            print(f"Thử kết nối bằng DSN...")
            conn = psycopg2.connect(dsn)
        else:
            print(f"Thử kết nối đến Host: {os.getenv('PGHOST')}, DB: {os.getenv('PGDATABASE')}...")
            conn = psycopg2.connect(
                host=os.getenv("PGHOST", "localhost"),
                port=os.getenv("PGPORT", "5432"),
                database=os.getenv("PGDATABASE", "postgres"),
                user=os.getenv("PGUSER", "postgres"),
                password=os.getenv("PGPASSWORD", "")
            )
        
        # 3. Tạo cursor để thực thi một truy vấn đơn giản
        cur = conn.cursor()
        cur.execute("SELECT version();")
        db_version = cur.fetchone()
        
        print("✅ Kết nối thành công!")
        print(f"Phiên bản PostgreSQL: {db_version[0]}")
        
        # Đóng kết nối
        cur.close()
        conn.close()
        print("---------------------------------------")
        
    except Exception as e:
        print("Kết nối thất bại!")
        print(f"Lỗi chi tiết: {e}")
        print("\nHãy kiểm tra lại:")
        print("1. File .env đã đúng định dạng và mật khẩu chưa?")
        print("2. Dịch vụ PostgreSQL đã được khởi động (Start) chưa?")
        print("3. Tên Database 'postgres' đã tồn tại chưa?")

if __name__ == "__main__":
    test_db_connection()