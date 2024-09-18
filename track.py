import cv2
import numpy as np
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator, colors
import os
import json


# tool_dict = {
#         6: "bipolar_forceps",
#         1: "needle_driver",
#         7: "monopolar_curved_scissor",
#         9: "cadiere_forceps",
#         12: "vessel_sealer",
#         13: "force_bipolar",
#         17: "permanent_cautery_hook_spatula",
#         18: "stapler",
#         20: "grasping_retractor",
#         19: "tip_up_fenestrated_grasper",
#         21: "clip_applier",
#         4: "prograsp_forceps"
        
#     }
tool_dict = {
        13: "force_bipolar",
        17: "permanent_cautery_hook_spatula",
        18: "stapler",
        20: "grasping_retractor",
        19: "tip_up_fenestrated_grasper",
        21: "clip_applier",
        4: "prograsp_forceps"
        
    }
x_dict = {
  0: "Needle_driver_1",
  1: "Needle_driver_2",
  2: "Prograsp_forceps_1",
  3: "Needle_driver_3",
  4: "Prograsp_forceps_2",
  5: "Prograsp_forceps_3",
  6: "Bipolar_forceps_2",
  7: "Monopolar_curved_scissors_1",
  8: "Cadiere_forceps_1",
  9: "Cadiere_forceps_2",
  10: "Bipolar_forceps_1",
  11: "Bipolar_forceps_3",
  12: "Vessel_sealer_2",
  13: "Force_bipolar_2",
  14: "Vessel_sealer_1",
  15: "Vessel_sealer_3",
  16: "Force_bipolar_1",
  17: "Permanent_cautery_hook/spatula_2",
  18: "Stapler_1",
  19: "Tip-up_fenestrated_grasper",
  20: "Grasping_retractor_1",
  21: "Clip_applier_2",
  22: "Cadiere_forceps_3",
  23: "Clip_applier_1",
  24: "Force_bipolar_3",
  25: "Permanent_cautery_hook/spatula_1",
  26: "Stapler_2",
  27: "Clip_applier_3",
  28: "Monopolar_curved_scissors_2",
}


model = YOLO("runs/segment/train13/weights/best.pt")  # segmentation model
names = model.model.names
cap = cv2.VideoCapture("original_images/case_145/case_145_video_part_001.mp4")
w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))

out = cv2.VideoWriter("instance-segmentation.avi", cv2.VideoWriter_fourcc(*"MJPG"), fps, (w, h))
i = 0



out.release()
cap.release()
cv2.destroyAllWindows()

import cv2
import os
import numpy as np

case_id = 'case_100'

video_path = f'original_images/{case_id}/{case_id}_video_part_001.mp4'

model = YOLO("/home/ydw-3090/change_code/data/surgtoolloc2022-category-2/best.pt") 
names = model.model.names


out = cv2.VideoWriter("instance-segmentation.avi", cv2.VideoWriter_fourcc(*"MJPG"), fps, (w, h))

cap = cv2.VideoCapture(video_path)

fps = int(cap.get(cv2.CAP_PROP_FPS))
frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

frames_per_second = fps

frames_to_extract = frame_count // frames_per_second
i = 0
extracted_frames = 0
for a in range(frames_to_extract):

    cap.set(cv2.CAP_PROP_POS_FRAMES, a * frames_per_second)
    ret, im0 = cap.read()
    if not ret:
        print("Video frame is empty or video processing has been successfully completed.")
        break
    original_image = im0.copy()
    results = model.predict(im0, conf=0.3, iou = 0.2)
    annotator = Annotator(im0, line_width=2)
    shapes = []
    os.makedirs("track/json", exist_ok=True)
    if results[0].masks is not None:
        clss = results[0].boxes.cls.cpu().tolist()
        confs = results[0].boxes.conf.cpu().tolist()
        masks = results[0].masks.xy
        good_conf = []
        for mask, cls, conf in zip(masks, clss, confs):
            if cls in tool_dict:
                 good_conf.append(conf)
        if len(good_conf) >= 2 :
            if all(good_conf) > 0.8:

                json_path = "track" + "/" + "json" + "/" + case_id + "_track_" + str(i) + ".json"
                for mask, cls in zip(masks, clss):
                    if cls in tool_dict:
                        tool_name = tool_dict[cls]
                        color = colors(int(cls), True)
                        txt_color = annotator.get_txt_color(color)
                        annotator.seg_bbox(mask=mask, mask_color=color, label=names[int(cls)], txt_color=txt_color)
                        contours, _  = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                        os.makedirs("track" + "/" + tool_name, exist_ok=True)
                        os.makedirs("track" + "/" + "imgs", exist_ok=True)
                        cv2.imwrite("track" + "/" + tool_name + "/" + case_id + "_track_" + str(i) + ".png", original_image)

                        
                        tool_name = tool_dict[int(cls)]

                        points = []
                        int_cls = int(cls)
                        for point in mask:
                            points.append([float(point[0]), float(point[1])])
                        shape = {
                            "label": x_dict[int_cls],  # 可以根据实际情况更改标签
                            "points": points,  # 
                            "group_id": None,
                            "shape_type": "polygon",
                            "flags": {}
                        }
                        shapes.append(shape)
                data = {
                    "version": "4.5.6",  # 版本号，可以自定义
                    "flags": {},  # 用户自定义的标志位
                    "shapes": shapes,
                    "imagePath": case_id + "_track_" + str(i) + ".png",
                    "imageData": None,  # 如果需要存储图像的base64编码，则取消注释cv2.imencode()行
                    "imageHeight": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                    "imageWidth": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                }

                # 保存为JSON文件
                with open(json_path, 'w') as f:
                    json.dump(data, f, indent=4)
                cv2.imwrite("track" + "/" + "imgs/" + case_id + "_" +str(i) + "_mask.png", im0)

    i += 1                       
                        




    # out.write(im0)
    # # cv2.imshow("instance-segmentation", im0)

    # if cv2.waitKey(1) & 0xFF == ord("q"):
    #     break

