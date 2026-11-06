# Báo cáo bài thực hành Ngày 1 – Đọc prediction YOLO11 để hiểu định dạng nhãn

**Ngày chạy:** 6/11/2026

**Runtime Colab:** T4 GPU

**Python / PyTorch / Ultralytics:** Python

**Checkpoint:** `yolo11n-cls.pt`, `yolo11n.pt`, `yolo11n-seg.pt`

**Ngưỡng detection/segmentation dùng khi tạo JSON:** `0.35` (chỉ đổi nếu GV cho phép; nếu đổi hãy ghi rõ)

**Thay đổi so với notebook nguồn:** Không / mô tả rõ thay đổi

> **Cách dùng mẫu này:** copy/điền trên máy, lưu thành `REPORT.md`, rồi **upload** vào panel Files
> của Colab (notebook không tự tạo file này). Ô cuối Colab sẽ đóng ZIP cùng `day1_lab_outputs/`.
> Giải nén và đưa vào `report/` trên GitHub, rồi nộp **link repo** trên LMS.
> Không ghi họ tên, MSSV, email, số điện thoại hoặc dữ liệu cá nhân khác vào báo cáo/output;
> họ tên/MSSV chỉ nằm trong tên repository.

### Từ cần nhớ (đọc nhanh trước khi viết)

| Từ | Nghĩa ngắn |
| --- | --- |
| Prediction | Kết quả model vừa đoán ra; chưa phải nhãn đúng |
| Ground truth | Nhãn do người gán theo quy tắc |
| Taxonomy | Danh sách tên lớp có sẵn kèm model (vd. ImageNet-1K, COCO-80) |
| Checkpoint | File model `.pt` đã huấn luyện sẵn |
| Guideline | Quy tắc gán nhãn do người viết |
| Annotator | Người gán nhãn |
| Reviewer / QC | Người kiểm tra nhãn |
| Escalation | Ca khó → hỏi Lab Coach, không tự bịa |
| Model score | Số “tin” của model; không phải điểm chất lượng nhãn người |

## 1. Phân loại ảnh

**Mở file:** `day1_lab_outputs/classification_predictions.json` và `visuals/classification_top5.png`.  
Chỉ dùng ảnh `sample_id = traffic`, lấy dòng `rank = 1`.

1. Chép record hạng 1: `class_id`=468; `class_name`=cab ; `rank`=1 ; `score`=0.510915; `taxonomy_name`=ImageNet-1K
2. Record này nói về **cả bức ảnh** hay về **một vật thể riêng** trong ảnh? Vì sao bạn nghĩ vậy?
Record nói về cả bức ảnh,không nói về bất cứ một vật thể riêng nào
Vì:
Model sử dụng là yolo11n-cls.pt, đây là mô hình phân loại ảnh (Image Classification).
Mô hình nhận toàn bộ bức ảnh làm đầu vào.
Sau đó, mô hình đưa ra các lớp có khả năng mô tả nội dung của toàn bộ bức ảnh.
Không có thông tin về bounding box (tọa độ khung bao) để xác định vị trí từng vật thể.
3. Danh sách tên lớp (class list) do đâu mà có: model tự nghĩ ra, hay đi kèm taxonomy/checkpoint?
Danh sách lớp được xác định bởi taxonomy và đi kèm với quá trình huấn luyện/checkpoint của mô hình
4. Vì sao cần ghi cả `class_id`, `class_name` và `taxonomy_name`, không chỉ ghi mỗi tên lớp?
vì mỗi trường có ý nghĩa riêng
class_id:Mã số định danh của lớp
class_name:tên lớp,vi dụ cab
taxonomy_name:Bộ danh mục phân loại chứa lớp đó
Lý do cần lưu cả 3:
class_id: Giúp máy tính nhận diện lớp một cách chính xác.
class_name: Giúp con người dễ đọc và hiểu kết quả.
taxonomy_name: Giúp xác định lớp đó thuộc hệ thống phân loại nào.
5. Nếu trong ảnh có nhiều vật / nhiều chủ đề, guideline cần nói rõ điều gì để người gán nhãn chọn **một** lớp?
Guideline cần quy định rõ tiêu chí chọn lớp đại diện cho toàn bộ bức ảnh
Cụ thể:
1.Chọn chủ thể chính xuất hiện trong ảnh
2.Ưu tiên đối tượng nổi bật nhất về kích thước,vị trí hoặc mức độ quan trọng
3.Nếu có nhiều đối tượng,cần xác định rõ đối tượng nào chiếm vai trò chủ đạo
4.Nếu không xác định được chủ thể chính,cần có quy tắc xử lý ảnh mơ hồ hoặc ảnh có nhiều chủ đề
6. Vì sao không được lấy `score` của model làm ground truth?
Vì score chỉ thể hiện mức độ tin cậy của mô hình với dự đoán,chứ không chứng minh dự đoán đó là chính xác


## 2. Phát hiện vật thể

**Mở file:** `day1_lab_outputs/detection_predictions.json` và `visuals/detection_predictions.png`.  
Chỉ dùng ảnh `sample_id = kitchen`. Chọn **một** record.  
Số lượng prediction theo ngưỡng: xem output ô so sánh `0.20` / `0.35` / `0.60` trong notebook.

1. Chép một record: `class_name`=person ; `score`=0.912625 ; `bbox_xyxy`=[385.33, 69.24, 498.92, 348.92] ; `bbox_width`=113.58 ; `bbox_height`=279.68
2. Giải thích box bằng lời thường: góc trên-trái ở đâu, góc dưới-phải ở đâu, rộng/cao bao nhiêu pixel?
Góc trên bên trái có tọa độ(385.33,69.24)pixel
Góc dưới bên phải có tọa độ(498.92,348.92)pixel
Chiều rộng:113.58 pixel
Chiều cao:279.68 pixel
Nói cách khách mô hình phát hiện một người trong ảnh .Khung hình chữ nhật bắt đầu từ vị trí(385.33, 69.24) ở góc trên bên trái và kết thúc tại (498.92, 348.92) ở góc dưới bên phải, bao quanh người đó.
3. Chọn hai ngưỡng (vd. 0.20 và 0.60). Mỗi ngưỡng có bao nhiêu prediction?
Với ngưỡng 0.20, mô hình giữ lại 53 predictions.
Với ngưỡng 0.60, mô hình chỉ giữ lại 25 predictions.
Khi tăng ngưỡng từ 0.20 lên 0.60, số lượng prediction giảm 28 hộp.
4. Khi hạ ngưỡng (giữ nhiều hộp hơn), người **reviewer** phải làm thêm việc gì?
Khi giảm Confidence Threshold, mô hình sẽ giữ lại nhiều dự đoán hơn, bao gồm cả những đối tượng có độ tin cậy thấp. Vì vậy, reviewer cần kiểm tra kỹ:
False Positive: Xác định những bounding box được vẽ nhầm vào vật thể không thuộc đối tượng cần nhận diện.
Duplicate Detection: Kiểm tra các bounding box bị trùng lặp trên cùng một đối tượng.
Sai nhãn (Incorrect Class): Kiểm tra xem mô hình có gán nhầm tên lớp cho đối tượng hay không
Bounding Box: Kiểm tra hộp giới hạn có bao quanh đúng đối tượng và có bị quá rộng hoặc quá hẹp không.
Missing Object: Kiểm tra xem có đối tượng nào xuất hiện trong ảnh nhưng mô hình chưa phát hiện hay không.
Kết luận: Threshold càng thấp thì reviewer càng cần kiểm tra nhiều dự đoán hơn để đảm bảo độ chính xác của dữ liệu.
5. Viết **một quy tắc box** để người gán nhãn làm theo (áp dụng được cho nhiều ảnh, không chỉ ảnh này).
Khi gán nhãn đối tượng,cần tuân thủ các quy tắc sau:
1.Vẽ bounding box nhỏ nhất có thể nhưng phải bao quanh toàn bộ phần đối tượng nhìn thấy được.
2.Không bao gồm các vùng nền hoặc vật thể khác không thuộc đối tượng.
3.Nếu đối tượng bị che khuất bởi vật thể khác, chỉ vẽ phần có thể nhìn thấy.
4.Nếu đối tượng bị cắt bởi cạnh ảnh, chỉ vẽ phần đối tượng nằm trong khung hình.
5.Không tự suy đoán hoặc vẽ thêm những phần đối tượng bị che khuất hoặc nằm ngoài ảnh.
6.Đảm bảo bounding box khớp với hình dạng và vị trí thực tế của đối tượng.
6. Trên ảnh `kitchen`, chỉ ra **một** vật bị che khuất hoặc cắt mép. Bạn sẽ viết thành quy tắc trong guideline, hay đưa lên hỏi Lab Coach (escalation)? Vì sao?
Trên ảnh kitchen,cái kệ chân bồn bị một người che khuất
Quy tắc đề xuất đưa vào guideline:
Khi đối tượng bị cắt bởi cạnh ảnh, reviewer chỉ vẽ bounding box bao quanh phần đối tượng thực sự nhìn thấy bên trong khung hình. Không mở rộng bounding box ra ngoài ảnh và không tự suy đoán phần đối tượng bị khuất hoặc nằm ngoài khung hình.
Lý do: Đây là tình huống thường gặp khi gán nhãn dữ liệu.
 Cách xử lý tương đối rõ ràng, có thể áp dụng thống nhất cho tất cả reviewer.
  Đưa vào guideline giúp giảm sự không nhất quán và hạn chế việc phải hỏi Lab Coach nhiều lần.

## 3. Phân đoạn theo từng đối tượng

**Mở file:** `day1_lab_outputs/segmentation_predictions.json` và `visuals/segmentation_prediction.png`.  
Chỉ dùng ảnh `sample_id = kitchen`. Chọn **một** record.

1. Chép một record: `instance_id`=kitchen-001 ; `class_name`=person ; `score`=0.899318 ; số điểm polygon=`polygon_point_count`=348 ; 3–5 điểm đầu của `polygon_xy`=…[
  [446.0, 70.0],
  [445.0, 71.0],
  [444.0, 71.0],
  [443.0, 72.0],
  [442.0, 72.0]
]
2. So với box, polygon cho thêm thông tin gì về hình dạng vật thể?
Bounding Box chỉ xác định vùng hình chữ nhật bao quanh đối tượng, thông qua 4 tọa độ.

Polygon cung cấp thêm thông tin về đường biên và hình dạng thực tế của đối tượng thông qua nhiều điểm tọa độ nối với nhau.

Cụ thể:
- Thể hiện hình dạng và đường viền của đối tượng chính xác hơn.
- Phân biệt phần nào thuộc đối tượng, phần nào thuộc nền.
- Hỗ trợ xử lý những vật thể có hình dạng phức tạp hoặc không vuông vắn.
- Giúp xác định vùng bị che khuất hoặc các đối tượng nằm sát nhau tốt hơn.
3. `instance_id` dùng để làm gì? Nó **không** phải class ID và **không** phải tracking ID — bạn hiểu câu này thế nào?
instance_id là mã định danh riêng cho từng đối tượng cụ thể được phát hiện trong ảnh.
4. Viết **một quy tắc vẽ biên mask** để người gán nhãn làm theo.
Người gán nhãn phải vẽ đường biên mask sát theo đường viền thực tế của đối tượng, chỉ bao gồm những pixel thuộc đối tượng và không bao gồm phần nền hoặc vật thể khác. Nếu đối tượng bị che khuất, chỉ gán nhãn phần nhìn thấy được, không tự suy đoán phần bị che khuất.
5. Trên ảnh `kitchen`, chỉ ra **một** chỗ biên mờ / hai vật chạm nhau / bị che. Bạn dùng guideline hay escalation? Vì sao?
Tôi chọn khu vực các chiếc bát ở phía dưới bên trái ảnh, nơi một số chiếc bát nằm sát và chồng lấn lên nhau.

Lý do:
- Ranh giới giữa các chiếc bát không hoàn toàn rõ ràng.
- Một phần đường viền của chiếc bát có thể bị chiếc khác che khuất.
- Nếu tự suy đoán đường biên, reviewer có thể vẽ mask lấn sang đối tượng bên cạnh hoặc gộp nhầm hai đối tượng thành một.
Vì vậy, tôi sẽ chuyển trường hợp này lên Lab Coach để xác nhận cách xử lý đường biên khi các vật thể chồng lấn mà không thể xác định rõ ranh giới.

## 4. Vòng đời và kiểm tra chất lượng

Lab hôm nay chỉ **xem prediction**. Chuỗi dưới đây là vòng đời production để bạn nối các khái niệm:

`ảnh thô → guideline → ground truth → huấn luyện → prediction → QC/rework`

Điền bảng (viết ngắn, có dẫn chứng từ JSON/PNG khi nói về lỗi):

| Tác vụ | Ground truth trông như thế nào? (đơn vị nhãn) | Lỗi hoặc chỗ khó quyết định bạn thấy trên ảnh/JSON | Annotator làm gì? | Reviewer xem gì? |
| --- | --- | --- | --- | --- |
| Phân loại ảnh |Mỗi ảnh có một hoặc nhiều nhãn lớp đại diện cho nội dung ảnh, ví dụ: cab. Đơn vị gán nhãn là toàn bộ ảnh  |Ảnh có nhiều đối tượng, khó xác định chủ thể chính; nhãn dự đoán sai hoặc confidence thấp. Ví dụ JSON có class_name, class_id, score.  | Xác định nội dung chính của ảnh và gán đúng nhãn theo taxonomy/guideline | Kiểm tra nhãn có phản ánh đúng nội dung ảnh không, có chọn đúng lớp và tuân thủ guideline không. |
| Phát hiện vật thể | Mỗi đối tượng có một nhãn lớp và một bounding box gồm 4 tọa độ bbox_xyxy. Đơn vị gán nhãn là từng đối tượng. | Box quá rộng/hẹp, sai class, phát hiện trùng, bỏ sót vật thể hoặc đối tượng bị che/cắt cạnh ảnh. Ví dụ JSON có class_name, score, bbox_xyxy. | Vẽ box bao sát đối tượng, gán đúng class; chỉ bao phần nhìn thấy nếu đối tượng bị cắt hoặc che khuất. | Kiểm tra vị trí và kích thước box, class, đối tượng bị bỏ sót, box trùng lặp và false positive. |
| Instance segmentation | Mỗi đối tượng có nhãn lớp, mã định danh riêng instance_id và một vùng mask dạng polygon (polygon_xy). Đơn vị gán nhãn là từng instance và vùng pixel của nó. | Biên mask không rõ, hai vật thể chạm nhau hoặc bị che khuất; polygon có thể lấn sang nền hoặc vật thể khác. Ví dụ kitchen-001 có class_name=person, polygon_point_count=348. | Vẽ mask sát đường biên đối tượng, tách riêng từng instance và chỉ gán phần nhìn thấy được. | Kiểm tra mask có đúng hình dạng, đường biên, class và instance ID không; chú ý vùng chồng lấn, bị che khuất. |

## 5. An toàn dữ liệu

1. Quy tắc bảo vệ dữ liệu của bạn cho bài lab này là gì? (viết 1 câu)
Quy tắc bảo vệ dữ liệu: Chỉ sử dụng dữ liệu được cung cấp và cho phép trong phạm vi bài lab, không tự ý sao chép, chia sẻ hoặc tiết lộ thông tin cá nhân.
2. Nếu gặp ảnh hoặc dữ liệu không đúng phạm vi, bạn sẽ dừng và báo cho ai?
Khi gặp dữ liệu không đúng phạm vi: Tôi sẽ dừng xử lý ngay và báo cho giảng viên hướng dẫn hoặc người phụ trách bài lab để được hướng dẫn.

## 6. Checklist trước khi nộp link repo

Kiểm tra trên GitHub trong thư mục `report/`:

- [x] `REPORT.md` (file này đã điền)
- [x] `day1_lab_outputs/classification_predictions.json`
- [x] `day1_lab_outputs/detection_predictions.json`
- [x] `day1_lab_outputs/segmentation_predictions.json`
- [x] `day1_lab_outputs/IMAGE_ATTRIBUTION.md`
- [x] `day1_lab_outputs/visuals/classification_top5.png`
- [x] `day1_lab_outputs/visuals/detection_predictions.png`
- [x] `day1_lab_outputs/visuals/segmentation_prediction.png`
- [x] Ô validation cuối notebook báo `PASS`
- [x] Không có họ tên, MSSV hoặc dữ liệu nhạy cảm trong báo cáo/output
- [x] Đã nộp **link repository** trên LMS
