import torch
from langchain_core.tools import tool
from ultralytics import YOLO
 
model = YOLO("yolo11s.pt")

#@tool
def det_tool(frame: torch.tensor) -> dict:
    results = model.predict(frame, device = "cuda")
    result = results[0]

    data = {}

    for box in result.boxes:
        class_id = int(box.cls[0])
        class_name = model.names[class_id]
        x1, y1, x2, y2 = [round(float(v), 2) for v in box.xyxyn[0]]

        center_x = round((x1 + x2) / 2 , 2)
        center_y = round((y1 + y2) / 2 , 2)

        if not class_name in data.keys():
            data[class_name] = [{center_x, center_y}]
        else:
            data[class_name].append({center_x, center_y})


    return data


if __name__ == "__main__":
    
    image_path = "images/test01.jpg"

    result = det_tool(image_path)

    print(result)
