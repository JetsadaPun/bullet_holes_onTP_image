clc
clear all
close all
%%
input_folder = './dataset';
jsonFile = '_annotations.coco.json';
image_files = dir(fullfile(input_folder, '*.jpg'));
count_correct_pic = 0;
all_image = 0;
function iou = calculateIoU(bbox1, bbox2)
 x1 = max(bbox1(1), bbox2(1));
 y1 = max(bbox1(2), bbox2(2));
 x2 = min(bbox1(1) + bbox1(3), bbox2(1) + bbox2(3));
 y2 = min(bbox1(2) + bbox1(4), bbox2(2) + bbox2(4));
 if x2 > x1 && y2 > y1
     intersection_area = (x2 - x1) * (y2 - y1);
 else
     intersection_area = 0;
 end
 area_bbox1 = bbox1(3) * bbox1(4);
 area_bbox2 = bbox2(3) * bbox2(4);
 union_area = area_bbox1 + area_bbox2 - intersection_area;
 iou = intersection_area / union_area;
 disp(["intersection_area: ", num2str(intersection_area)]);
 disp(["Union_area: ", num2str(union_area)]);
end
for k = 1:length(image_files)
  image_filename = image_files(k).name;
  ori_image_path = fullfile(input_folder, image_filename);
  ori_image = imread(ori_image_path);
   im = 1;
  gray_image = rgb2gray(ori_image);
  points = [5, 5;
          5, 635;
          635, 5;
          635, 635];
  gray_values = zeros(4,1);
  for i = 1:4
    x = points(i, 1);
    y = points(i, 2);
    gray_values(i) = gray_image(y, x);
    fprintf('ภาพสีเทา - จุด (%d, %d): Grayscale Value = %d\n', x, y, gray_values(i));
   end
  gray_avg = ceil(mean(gray_values));
  fprintf('ค่าเฉลี่ย Grayscale ของ 4 จุด: %.2f\n', gray_avg);
  fprintf('หลังจากปรับค่า: %.2f\n', gray_avg);
  lower_bound = 59;
  upper_bound = 170;
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
  fprintf("\nlower bound: %d\n",lower_bound);
  fprintf("upper bound: %d\n",upper_bound);
  [m, n, p] = size(gray_image);
  threshold_image = gray_image;
  for i = 1:m
      for j = 1:n
          if gray_image(i,j) <= lower_bound || gray_image(i,j) >= upper_bound
              threshold_image(i,j) = 255;
          else
              threshold_image(i,j) = 0;
          end
      end
  end
  % inv_thre_img = ~threshold_image;
  bw_image = imbinarize(threshold_image);
  inv_bw_image = ~bw_image;
  se_erode_lg = strel('disk',1);
  erode_img1 = imerode(inv_bw_image,se_erode_lg);
  erode_img2 = imerode(erode_img1,se_erode_lg);
  se_dilate_lg = strel('disk',3);
  dilate_img1 = imdilate(erode_img2,se_dilate_lg);
  dilate_img2 = imdilate(dilate_img1,se_dilate_lg);
  se_dilate_lg2 = strel('line',5,90);
  dilate_img3 = imdilate(dilate_img2,se_dilate_lg2);

  edges = edge(dilate_img3, 'Canny');
  stats = regionprops(edges, 'BoundingBox', 'Area');
  [~, idx] = max([stats.Area]);
  largest_object = stats(idx);
  rmlg_image = bw_image;
  rmlg_image(round(largest_object.BoundingBox(2)):round(largest_object.BoundingBox(2) + largest_object.BoundingBox(4)), ...
         round(largest_object.BoundingBox(1)):round(largest_object.BoundingBox(1) + largest_object.BoundingBox(3)), :) = 255;
 
  inv_rmlg_image = ~rmlg_image;
  se_erode = strel('disk',1);
  erode1_img = imerode(inv_rmlg_image,se_erode);
  erode2_img = imerode(erode1_img,se_erode);
  erode3_img = imerode(erode2_img,se_erode);

  se_dilate = strel('disk',2);
  dilate1_img = imdilate(erode3_img,se_dilate);
  dilate2_img = imdilate(dilate1_img,se_dilate);
  dilate3_img = imdilate(dilate2_img,se_dilate);
 
  stats = regionprops(dilate3_img, 'BoundingBox');
 
  fid = fopen(jsonFile);
  raw = fread(fid, inf, 'uint8=>char')';
  fclose(fid);
  jsonData = jsondecode(raw);
  image_id = [];
  for i = 1:length(jsonData.images)
     if strcmp(jsonData.images(i).file_name, image_filename)
         image_id = jsonData.images(i).id;
         break;
     end
  end

  disp(['Image ID: ', num2str(image_id)]);
  bounding_boxes = [];
  bbox_count = 0;
  overlapping_count = 0;
  for i = 1:length(jsonData.annotations)
     if jsonData.annotations(i).image_id == image_id
         bbox = jsonData.annotations(i).bbox;
         if length(bbox) == 4 && jsonData.annotations(i).category_id ~= 12 && jsonData.annotations(i).category_id ~= 13
             bbox_count = bbox_count + 1;
             bounding_boxes{bbox_count} = bbox;
         end
     end
  end
  disp([num2str(bbox_count), ' bounding boxes']);
  for i = 1:bbox_count
     bbox = bounding_boxes{i};
     fprintf('%d: x=%.1f, y=%.1f, w=%.1f, h=%.1f\n', i, bbox(1), bbox(2), bbox(3), bbox(4));
  end
  figure;
  imshow(ori_image);
  title(sprintf('Predict Bounding Box: %s', image_filename), 'Interpreter', 'none');
  hold on;
  for k = 1:length(stats)
      bbox = stats(k).BoundingBox;
      rectangle('Position', bbox, 'EdgeColor', 'r', 'LineWidth', 2);
  end
  hold off;
  figure;
  imshow(ori_image);
  title(sprintf('Compare Bounding Box: %s', image_filename), 'Interpreter', 'none');
  hold on;
   for k = 1:length(stats)
      bbox = stats(k).BoundingBox;
      rectangle('Position', bbox, 'EdgeColor', 'r', 'LineWidth', 2);
  end
  for i = 1:bbox_count
      bbox = bounding_boxes{i};
      rectangle('Position', [bbox(1), bbox(2), bbox(3), bbox(4)], 'EdgeColor', 'g', 'LineWidth', 2);
  end
  hold off;
  ious = [];
  for i = 1:length(stats)
     for j = 1:bbox_count
         iou = calculateIoU(stats(i).BoundingBox, bounding_boxes{j});
         disp(["iou: ", num2str(iou)])
         if iou > 0.1
             ious = [ious, iou];
             overlapping_count = overlapping_count + 1;
         end
     end
  end
  if overlapping_count / bbox_count * 100 > 70
      count_correct_pic = count_correct_pic + 1;
      all_image = all_image + 1;
  else
      all_image = all_image + 1;
  end
  disp(['Overlapping Bounding Boxes: ', num2str(overlapping_count),'/',num2str(bbox_count)]);
  disp(['Overlapping Ratio: ', num2str(overlapping_count / bbox_count * 100), '%']);
  disp(['All correct image: ',num2str(count_correct_pic) '/',num2str(all_image)]);
end
