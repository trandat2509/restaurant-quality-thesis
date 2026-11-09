# Kết quả cảnh báo tầng 1

- Phản ánh A1–A7 (NEG/MIX): **38,182**.
- A8 đã chốt: **377**; trạng thái: {"CAN_XEM_XET":219,"DA_DUYET":158}.
- Tổng số dòng cảnh báo: **38,559**.
- Số nhận xét có ít nhất một cảnh báo: **21,293**.
- Nguồn nhãn A1–A7: {"model":33203,"manual":4979}.
- Mức độ A1–A7 theo luật: {"unknown":31657,"light":6271,"serious":254}.

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
| Bằng chứng tay có nhiều vị trí giống nhau | 10 |
| Bằng chứng cần kiểm tra | 33,213 |
| Bằng chứng chưa đủ điều kiện xếp mức độ | 15,935 |
| Đoạn trích cắt giữa từ | 2,271 |
| Điểm dự đoán thấp | 1,496 |
| Thiếu bằng chứng | 728 |
| Thiếu bằng chứng hoặc chưa xác định được vị trí | 738 |
| MIX máy chưa tách đoạn khen và chê | 12,758 |
| Ngữ cảnh phủ định hoặc mơ hồ | 130 |
| Chưa khớp luật mức độ của khía cạnh | 14,854 |
| Nhãn cảm xúc chưa được người đọc xác nhận | 33,203 |
| Hai nhãn có điểm gần nhau | 651 |
| Đoạn trích quá ngắn | 4,718 |

Một dòng có thể có nhiều lý do; không cộng cột này thành tổng cảnh báo.
