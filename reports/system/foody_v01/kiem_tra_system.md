# System chung

Cấu hình duy nhất: configs/system/foody_v01.json.
Đầu vào vận hành nằm cùng corpus ở data/system/foody_v01; báo cáo tại reports/system/foody_v01.
Không cần foody_inputs_v02 hoặc các thư mục alerts cũ.

## Kết quả cập nhật

```json
{
  "reviews": 40176,
  "restaurants": 569,
  "manual_reviews": 3350,
  "model_only_reviews": 36826,
  "aspect_pairs": 281232,
  "manual_aspect_pairs": 23450,
  "model_aspect_pairs": 257782,
  "manual_model_label_disagreements": 933,
  "operational_manual_label_changes_vs_training": 0,
  "a8_alerts": 377,
  "a8_by_status": {
    "CAN_XEM_XET": 219,
    "DA_DUYET": 158
  },
  "a8_explicit_negative_reviews": 98,
  "a8_unlabeled_reviews": 39701,
  "evidence_status_counts": {
    "manual_confirmed": 10742,
    "manual_evidence_needs_review": 17,
    "mixed_evidence_polarity_not_separated": 12758,
    "model_extracted_unverified": 75947,
    "needs_review_no_evidence": 1466,
    "not_mentioned_by_sentiment_model": 167611,
    "not_mentioned_manual": 12691
  },
  "evidence_quality_flag_counts": {
    "ambiguous_manual_position": 17,
    "fragment_inside_word": 4390,
    "manual_annotation_marked_uncertain": 7,
    "missing_evidence": 1466,
    "mixed_polarity_not_separated": 12758,
    "very_short_fragment": 9523
  },
  "base_files_modified": false,
  "model_retrained": false,
  "alerts_rebuilt": false,
  "new_independent_model_evaluation": false,
  "requires_legacy_alerts": false
}
```

## Cách dùng

reviews_master.csv giữ nguyên dữ liệu nền và thông tin quán. train_reviews.csv giữ nguyên tập đã dùng huấn luyện.
aspect_inputs.csv dùng label ưu tiên nhãn tay; model_label và điểm máy giữ riêng. Bằng chứng có nguồn và trạng thái riêng.
a8_inputs.csv chứa cảnh báo A8 đã xử lý. Nhãn A8 chưa duyệt không tự trở thành nhãn 0.
Các cờ bằng chứng chỉ phát hiện lỗi hình thức, không xác nhận mô hình trích đúng ngữ nghĩa.

## Cập nhật lần sau

Sau khi thay kết quả bằng chứng và prediction_manifest.json tương ứng tại nguồn, chạy lại python -m src.pipeline.prepare_system_data.
Chương trình tự ghi đè đúng đầu ra vận hành trong system hiện có. Không tạo thêm phiên bản thư mục, không huấn luyện lại.
Dữ liệu/mô hình nguồn vẫn cần được giữ ở chỗ cũ. Chưa dựng lại các tầng cảnh báo trong bước này.
