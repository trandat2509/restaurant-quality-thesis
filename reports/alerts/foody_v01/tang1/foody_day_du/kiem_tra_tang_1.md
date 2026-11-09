# Kết quả cảnh báo tầng 1

- Phản ánh A1–A7 (NEG/MIX): **33,663**.
- A8 đã chốt: **377**; trạng thái: {"CAN_XEM_XET":219,"DA_DUYET":158}.
- Tổng số dòng cảnh báo: **34,040**.
- Số nhận xét có ít nhất một cảnh báo: **20,555**.
- Nguồn nhãn A1–A7: {"model":17922,"manual":15741}.
- Mức độ A1–A7 theo luật: {"unknown":25757,"light":7637,"serious":269}.

## Cách đọc kết quả

- Một nhận xét có thể có nhiều dòng A1–A7 và thêm một dòng A8; không cộng các dòng thành số nhận xét.
- Mức độ A1–A7 là gợi ý từ luật, severity_verified=false; chưa có đánh giá độ chính xác độc lập.
- evidence_text/evidence_spans_json giữ bằng chứng nguồn. severity_evidence_text là ngữ cảnh luật, lưu riêng.
- Đoạn thiếu, cắt từ, quá ngắn, vị trí mơ hồ hoặc MIX máy chưa tách khen/chê: chưa tự xếp mức độ.
- unknown là chưa xác định mức độ, không có nghĩa bình thường hoặc đã loại cảnh báo.
- A8 DA_DUYET/CAN_XEM_XET giữ quyết định gán nhãn; chưa phải trạng thái giải quyết công việc của quản lý.
- Không thêm dự đoán A8 ngoài danh sách system. Nhãn 0 đã duyệt bị loại; phần chưa duyệt không tự đổi thành 0.
- Chưa tạo tầng 2, chưa xây giao diện, không huấn luyện hoặc chạy lại mô hình.

## Các lý do cần xem xét của A1–A7

| Lý do | Số dòng |
|---|---:|
| Bằng chứng tay có nhiều vị trí giống nhau | 23 |
| Bằng chứng cần kiểm tra | 17,945 |
| Bằng chứng chưa đủ điều kiện xếp mức độ | 9,325 |
| Đoạn trích cắt giữa từ | 828 |
| Điểm dự đoán thấp | 1,470 |
| Thiếu bằng chứng | 407 |
| Thiếu bằng chứng hoặc chưa xác định được vị trí | 430 |
| MIX máy chưa tách đoạn khen và chê | 8,259 |
| Ngữ cảnh phủ định hoặc mơ hồ | 139 |
| Chưa khớp luật mức độ của khía cạnh | 15,863 |
| Nhãn cảm xúc chưa được người đọc xác nhận | 17,922 |
| Hai nhãn có điểm gần nhau | 581 |
| Đoạn trích quá ngắn | 2,575 |

Một dòng có thể có nhiều lý do; không cộng cột này thành tổng cảnh báo.
