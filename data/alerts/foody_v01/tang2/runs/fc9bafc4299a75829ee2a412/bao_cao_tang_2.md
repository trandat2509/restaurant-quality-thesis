# Kết quả cảnh báo tầng 2

Ngày đánh giá: 2026-10-29T23:59:59.999999+07:00. Mô hình: `phobert_foody_3350_final_v01`.

**Điều kiện tạm: 5 nhận xét, 3 nhóm văn bản mỗi tháng**. Điều kiện tạm: Có.
Mẫu nhỏ làm tỷ lệ biến động mạnh; kết quả chưa xác nhận độ chính xác cảnh báo.

| Nội dung | Kết quả |
|---|---:|
| Nhận xét | 40176 |
| Nhà hàng | 569 |
| Nhà hàng–tháng có nhận xét | 12873 |
| Nhà hàng–tháng đủ điều kiện đánh giá | 127 |
| Nhà hàng có tháng đủ điều kiện | 73 |
| Lượt so sánh khía cạnh đủ điều kiện | 889 |
| Cảnh báo theo khía cạnh | 37 |
| Vàng | 31 |
| Đỏ | 6 |
| Nhà hàng–tháng có cảnh báo | 26 |
| Nhà hàng có cảnh báo | 19 |

Tháng mới nhất có dữ liệu: 2026-07. Tháng đã kết thúc gần nhất:
2026-09, có 0 cảnh báo.
Các cảnh báo ở tháng cũ là kết quả lịch sử, không phải cảnh báo hiện thời.

## Cách tính

- Tháng hiện tại ít nhất 5 nhận xét, 3 nhóm văn bản.
- Từng tháng trong ba tháng lịch trước ít nhất 5 nhận xét,
  3 nhóm. Tháng trống không thay bằng tỷ lệ 0.
- Tỷ lệ chính = số nhãn máy NEG / toàn bộ nhận xét trong tháng, kể cả NONE và MIX trong mẫu số.
- Cơ sở = trung bình cộng ba tỷ lệ tháng trước; mức tăng tính bằng điểm phần trăm.
- Vàng: tỷ lệ ≥ 15%, tăng ≥ 10 điểm phần trăm,
  và ≥ 3 nhận xét NEG. Đỏ: ≥ 25%,
  tăng ≥ 20 điểm phần trăm, và ≥ 5 NEG.
  Mỗi màu phải đồng thời đạt cả ba điều kiện; xét đỏ trước vàng.
- MIX chỉ dùng trong tỷ lệ phụ. Không dùng mức độ tầng 1, A8, nhãn tay hay đoạn trích để quyết định tầng 2.
- Khoảng Wilson 95% mô tả độ bất định của tỷ lệ trong tháng, không phải độ tin cậy của mô hình,
  không hiệu chỉnh nhãn máy sai hoặc sự phụ thuộc giữa các nhận xét và không quyết định màu.
- Nhóm văn bản chỉ kiểm tra độ phủ; đơn vị đếm vẫn là mã nhận xét, không khẳng định các nhận xét độc lập.

## Đầu ra

`trend_coverage.csv`: toàn bộ tháng lịch từ tháng đầu tiên của từng nhà hàng đến tháng đánh giá,
kể cả tháng trống; có lý do và số lượng từng tháng cơ sở.
`trend_metrics.csv`: các tháng có nhận xét × bảy khía cạnh; ô tỷ lệ thiếu để trống.
`trend_alerts.csv`: chỉ các cảnh báo vàng/đỏ. `demo_tang_2.html`: bản xem nhanh tĩnh.

Đổi `active_profile` trong `configs/alerts/tang2/trend_alerts.json` thành `chinh_thuc_20_10` rồi chạy lại để
khôi phục điều kiện chính thức; không sửa `configs/system/foody_v01.json`, không huấn luyện lại.
