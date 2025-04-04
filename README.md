# Image Processing & IoU Calculation Project
## ภาพรวม 
สคริปต์ MATLAB นี้ใช้สำหรับประมวลผลภาพจากชุดข้อมูล โดยทำการแปลงภาพเป็นระดับสีเทา (grayscale thresholding) และประเมินค่ากล่องกรอบ (bounding boxes) ที่คาดการณ์ไว้เทียบกับข้อมูลจริงที่ถูกเก็บไว้ในไฟล์ JSON รูปแบบ COCO (_annotations.coco.json) นอกจากนี้ สคริปต์ยังคำนวณค่า Intersection over Union (IoU) เพื่อวัดความแม่นยำของกล่องกรอบที่คาดการณ์

คุณสมบัติ (Features)

• โหลดภาพจากโฟลเดอร์ที่กำหนด (./dataset)

• อ่านข้อมูลคำอธิบายภาพ (annotations) จากไฟล์ _annotations.coco.json

• แปลงภาพเป็นระดับสีเทา และวิเคราะห์ค่าความเข้มของพิกเซลที่จุดสำคัญ 4 จุด

• กำหนดค่าขีดจำกัด (threshold) แบบไดนามิก ตามระดับความเข้มของสีเทา

• ใช้กระบวนการมอร์โฟโลยี (morphological operations) เพื่อลดสัญญาณรบกวนและแยกวัตถุออกจากพื้นหลัง

• ตรวจจับและระบุตำแหน่งของกล่องกรอบในภาพที่ผ่านการประมวลผล

• เปรียบเทียบกล่องกรอบที่ตรวจจับได้กับกล่องกรอบจริงจากไฟล์คำอธิบายภาพ โดยใช้ค่าความซ้อนทับ IoU

• ประเมินความแม่นยำของการทำนายตำแหน่งกล่องกรอบโดยพิจารณาจากค่า IoU

## ชุดข้อมูลที่ใช้ในการทดลอง 
    รายละเอียดชุดข้อมูลที่ใช้ 
        ชื่อ : Bullet hole object detection Computer Vision Project 
        วัตถุประสงค์ : วิเคราะห์ความแม่นยำในการยิง
        ให้คะแนนการแข่งขัน การสืบสวนนิติเวช
        แหล่งที่มา : Roboflow 
        ขนาดรูปภาพ : 640 x 640 pixels
        จำนวนรูปภาพทั้งหมด : 3099 ภาพ 
            แบ่งเป็น  
                - Train :  2784 ภาพ
                - Valid : 202 ภาพ
                Test : 113 ภาพ 
            จำนวนภาพหลังจากคัด : 156 ภาพ

## เกณฑ์ในการแบ่งค่าสี
```matlab
if gray_avg >= 130 && gray_avg <= 140
     lower_bound = 4;
     upper_bound = 90;
     fprintf('เกณฑ์ที่ 1')
  elseif  gray_avg >= 141 && gray_avg <= 161
     lower_bound = 4;
     upper_bound = 120;
     fprintf('เกณฑ์ที่ 2')
  elseif  gray_avg >=162 && gray_avg <= 179
     lower_bound = 4;
     upper_bound = 130;
     fprintf('เกณฑ์ที่ 3')
  elseif  gray_avg >= 180 && gray_avg <= 190
     lower_bound = 4;
     upper_bound = 150;
     fprintf('เกณฑ์ที่ 4')
  elseif  gray_avg >= 191 && gray_avg <= 195
     lower_bound = 15;
     upper_bound = 150;
     fprintf('เกณฑ์ที่ 5')
  elseif  gray_avg >= 196 && gray_avg <= 212
     lower_bound = 20;
     upper_bound = 160;
     fprintf('เกณฑ์ที่ 6')
  elseif  gray_avg >= 213 && gray_avg <= 218
     lower_bound = 30;
     upper_bound = 170;
     fprintf('เกณฑ์ที่ 7')
  elseif  gray_avg >= 219
     lower_bound = 50;
     upper_bound = 170;
     fprintf('เกณฑ์ที่ 8')
  end
```