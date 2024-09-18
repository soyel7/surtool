import SimpleITK
import numpy as np
import cv2
from pandas import DataFrame
from pathlib import Path
from scipy.ndimage import center_of_mass, label
from pathlib import Path
from evalutils import DetectionAlgorithm
from evalutils.validators import (
    UniquePathIndicesValidator,
    DataFrameValidator,
)
from typing import (Tuple)
from evalutils.exceptions import ValidationError
import random
import json
from ultralytics import YOLO

####
# Toggle the variable below to debug locally. The final container would need to have execute_in_docker=True
####
execute_in_docker = False


class VideoLoader():
    def load(self, *, fname):
        path = Path(fname)
        print('File found: ' + str(path))
        if ((str(path)[-3:])) == 'mp4':
            if not path.is_file():
                raise IOError(
                    f"Could not load {fname} using {self.__class__.__qualname__}."
                )
                #cap = cv2.VideoCapture(str(fname))
            #return [{"video": cap, "path": fname}]
            return [{"path": fname}]

# only path valid
    def hash_video(self, input_video):
        pass


class UniqueVideoValidator(DataFrameValidator):
    """
    Validates that each video in the set is unique
    """

    def validate(self, *, df: DataFrame):
        try:
            hashes = df["video"]
        except KeyError:
            raise ValidationError("Column `video` not found in DataFrame.")

        if len(set(hashes)) != len(hashes):
            raise ValidationError(
                "The videos are not unique, please submit a unique video for "
                "each case."
            )

class Surgtoolloc_det(DetectionAlgorithm):
    def __init__(self):
        super().__init__(
            index_key='input_video',
            file_loaders={'input_video': VideoLoader()},
            input_path=Path("/input/") if execute_in_docker else Path("./test/"),
            output_file=Path("/output/surgical-tools.json") if execute_in_docker else Path(
                            "./output/surgical-tools.json"),
            validators=dict(
                input_video=(
                    #UniqueVideoValidator(),
                    UniquePathIndicesValidator(),
                )
            ),
        )
        
        ###                                                                                                     ###
        ###  TODO: adapt the following part for creating your model and loading weights
        ###                                                                                                     ###
        
        
        self.tool_list = ["needle_driver",
                          "monopolar_curved_scissor",
                          "force_bipolar",
                          "clip_applier",
                          "tip_up_fenestrated_grasper",
                          "cadiere_forceps",
                          "bipolar_forceps",
                          "vessel_sealer",
                          "suction_irrigator",
                          "bipolar_dissector",
                          "prograsp_forceps",
                          "stapler",
                          "permanent_cautery_hook_spatula",
                          "grasping_retractor"]

    def process_case(self, *, idx, case):
        # Input video would return the collection of all frames (cap object)
        input_video_file_path = case #VideoLoader.load(case)
        # Detect and score candidates
        scored_candidates = self.predict(case.path) #video file > load evalutils.py
        # Write resulting candidates to result.json for this case
        return dict(type="Multiple 2D bounding boxes", boxes=scored_candidates, version={"major": 1, "minor": 0})

    def save(self):
        with open(str(self._output_file), "w") as f:
            json.dump(self._case_results[0], f, indent=4)

    def generate_bbox(self, frame_id):
        # bbox coordinates are the four corners of a box: [x, y, 0.5]
        # Starting with top left as first corner, then following the clockwise sequence
        # origin is defined as the top left corner of the video frame
        num_predictions = 2
        predictions = []
        for n in range(num_predictions):
            name = f'slice_nr_{frame_id}_' + self.tool_list[n]
            bbox = [[54.7, 95.5, 0.5],
                    [92.6, 95.5, 0.5],
                    [92.6, 136.1, 0.5],
                    [54.7, 136.1, 0.5]]
            score = np.random.rand()
            prediction = {"corners": bbox, "name": name, "probability": score}
            predictions.append(prediction)
        return predictions

    def predict(self, fname) -> DataFrame:
        """
        Inputs:
        fname -> video file path
        
        Output:
        tools -> list of prediction dictionaries (per frame) in the correct format as described in documentation 
        """
        tool_dict = {
                6: "bipolar_forceps",
                1: "needle_driver",
                7: "monopolar_curved_scissor",
                9: "cadiere_forceps",
                12: "vessel_sealer",
                13: "force_bipolar",
                17: "permanent_cautery_hook_spatula",
                18: "stapler",
                20: "grasping_retractor",
                19: "tip_up_fenestrated_grasper",
                21: "clip_applier",
                4: "prograsp_forceps"

            }

        print('Video file to be loaded: ' + str(fname))
        cap = cv2.VideoCapture(str(fname))
        num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        ###                                                                     ###
        ###  TODO: adapt the following part for YOUR submission: make prediction
        ###                                                                     ###
        frame_data = []
        output_data = []
        all_frames_predicted_outputs = []
        for fid in range(num_frames):
            ret, frame = cap.read()
            if not ret:
                continue
            cv2.imwrite(f"./output/frame_{fid}.png", frame)
            model = YOLO("/home/ydw-3090/change_code/data/runs/segment/train9/weights/best.pt")
            results = model(frame, iou = 0.2)
            # Process results list
            for i, result in enumerate(results):
                
                boxes = result.boxes  # Boxes object for bounding box outputs
                probs = boxes.conf  # confidence probabilities
                classes = boxes.cls  # class labels
                xyxy = boxes.xyxy  # bounding box coordinates (x1, y1, x2, y2)
                
                for j in range(len(xyxy)):
                    x1, y1, x2, y2 = xyxy[j].tolist()
                    conf = probs[j].item()
                    class_id = int(classes[j])
                    class_name = None
                    if class_id in tool_dict:
                        if conf > 0:
                            class_name = tool_dict[class_id]

                            # 构建边界框的四个角点
                            corners = [
                                [round(x1, 1), round(y1, 1), 0.5],
                                [round(x2, 1), round(y1, 1), 0.5],
                                [round(x2, 1), round(y2, 1), 0.5],
                                [round(x1, 1), round(y2, 1), 0.5]
                            ]
                            box = {
                                "corners": corners,
                                "name": f"slice_nr_{fid}_{class_name}",
                                "probability": float(conf)
                            }
                            # 将数据添加到 frame_data 中
                            frame_data.append(box)
                        
                            # 将 frame_data 添加到 output_data 列表中
                            output_data.append(frame_data)

            # tool_detections = self.generate_bbox(fid)
            # all_frames_predicted_outputs += tool_detections

        return frame_data


if __name__ == "__main__":
    Surgtoolloc_det().process()
