import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO, YOLOv10

if __name__ == '__main__':
    model = YOLOv10('runs/train/plantdoc_new_yolov10n_PSEMA_D4_1_nopre/weights/best.pt')
    model.val(data='D:\\workspace_py\\ultralytics\\ultralytics\\cfg\\datasets\\plantdoc_new.yaml',
              split='val',
              imgsz=640,
              batch=8,
              # rect=False,
              save_json=True, # if you need to cal coco metrice
              project='runs/val',
              name='plantdoc_new_yolov10n_PSEMA_D4_1_nopre',
              )

