import os
import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO, YOLOv10



if __name__ == '__main__':
    # model = YOLO('runs/train/plantdoc_v10_yolov10n_DCNv33/weights/best.pt')
    # model.val(data='D:\\workspace_py\\ultralytics\\ultralytics\\cfg\\datasets\\plantdoc.yaml',
    #           split='test',
    #           imgsz=640,
    #           batch=8,
    #           # rect=False,
    #           # save_json=True, # if you need to cal coco metrice
    #           project='runs/val',
    #           name='plantdoc_v10_yolov10n_DCNv3',
    #           )

    # model = YOLO(r"D:\workspace_py\ultralytics-20240523\ultralytics-main\runs\train\plantdoc_yolov3\weights\best.pt")
    # model = YOLOv10(r"D:\workspace_py\yolov10\runs\train\plantdoc_v10\weights\best.pt")
    model = YOLO(r"D:\workspace_py\ultralytics-20240713\ultralytics-main\runs\train\plantdoc_new_yolov9t\weights\best.pt")
    # images = [r'D:\病虫害图片\PlantDoc.v1-resize-416x416.yolov8\test\images\earlyblight21__jpg.rf.1d59d01997d3c54faaddc88ce60c2f00.jpg',r'D:\病虫害图片\PlantDoc.v1-resize-416x416.yolov8\test\images\fungus-univ-of-minnesoeta_jpg.rf.f3ee750055c87b145f7fe671c0eb8d7e.jpg',r'D:\病虫害图片\PlantDoc.v1-resize-416x416.yolov8\test\images\apples_apple-scab_02_thm_jpg.rf.7b22987525b900faa0b42c9b1dfa23c1.jpg',r'D:\病虫害图片\PlantDoc.v1-resize-416x416.yolov8\test\images\backus-056-potato-blight_jpg.rf.8b1c7d3597fbc4c1886d70b5880cb8f5.jpg']
    # images = [r'D:\病虫害图片\self\classes\apple\test\images\d7ee6d74-apple_scab_14.jpg']
    # model.predict(r'D:\病虫害图片\self\classes\apple\test\images\d7ee6d74-apple_scab_14.jpg', save=True, conf=0.5)
    images = os.listdir(r'D:\病虫害图片\PlantDoc.v1-resize-416x416.yolov8_bak\test\images')
    images = [os.path.join(r'D:\病虫害图片\PlantDoc.v1-resize-416x416.yolov8_bak\test\images', e) for e in images]
    results = model(images)
    # results.save("result.jpg")
    for i, r in enumerate(results):
    #     boxes = result.boxes  # Boxes object for bounding box outputs
    #     masks = result.masks  # Masks object for segmentation masks outputs
    #     keypoints = result.keypoints  # Keypoints object for pose outputs
    #     probs = result.probs  # Probs object for classification outputs
    #     obb = result.obb  # Oriented boxes object for OBB outputs
    #     result.show()  # display to screen

        r.save(filename=f"./v9_predict/result{i}.jpg")
        # r.save()
