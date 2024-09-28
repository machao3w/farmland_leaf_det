import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO
from ultralytics import YOLOv10

def single_train_v10(data,name):
    model = YOLOv10()
    model.train(data=data,
                cache=False,
                imgsz=640,
                epochs=300,
                batch=16,
                close_mosaic=10,
                workers=4,
                device='0',
                optimizer='SGD',  # using SGD
                # resume='', # last.pt path
                amp=False,  # close amp
                # fraction=0.2,
                project='runs/train',
                name=name,
                exist_ok=False,
                )
def single_train_v10_pre(model, data,name):
    model = YOLOv10(model)
    # model.load('yolov10n.pt')
    model.train(data=data,
                cache=False,
                imgsz=640,
                epochs=300,
                patience=300,
                batch=8,
                close_mosaic=10,
                workers=4,
                device='0',
                optimizer='SGD',  # using SGD
                # resume='', # last.pt path
                amp=False,  # close amp
                # fraction=0.2,
                project='runs/train',
                name=name,
                exist_ok=False,
                )

def single_train_v10_pre_150(model, data,name):
    model = YOLOv10(model)
    # model.load('yolov10n.pt')
    model.train(data=data,
                cache=False,
                imgsz=640,
                epochs=150,
                patience=150,
                batch=8,
                close_mosaic=10,
                workers=4,
                device='0',
                optimizer='SGD',  # using SGD
                # resume='', # last.pt path
                amp=False,  # close amp
                # fraction=0.2,
                project='runs/train',
                name=name,
                exist_ok=False,
                )

def single_train(model,data,name):
    model = YOLO(model)
    # model.load('yolov8n.pt') # loading pretrain weights
    model.train(data=data,
                cache=False,
                imgsz=640,
                epochs=300,
                patience=300,
                batch=8,
                close_mosaic=10,
                workers=4,
                device='0',
                optimizer='SGD', # using SGD
                # resume='', # last.pt path
                amp=False, # close amp
                # fraction=0.2,
                project='runs/train',
                name=name,
                exist_ok=False,
                )

def single_train_v5(model,data,name):
    model = YOLO(model)
    model.load('yolov5nu.pt') # loading pretrain weights
    model.train(data=data,
                cache=False,
                imgsz=640,
                epochs=300,
                batch=8,
                close_mosaic=10,
                workers=4,
                device='0',
                optimizer='SGD', # using SGD
                # resume='', # last.pt path
                amp=False, # close amp
                # fraction=0.2,
                project='runs/train',
                name=name,
                exist_ok=False,
                )

def single_train_v3(model, data, name):
    model = YOLO(model)
    model.load('yolov3.pt')  # loading pretrain weights
    model.train(data=data,
                cache=False,
                imgsz=640,
                epochs=300,
                batch=8,
                close_mosaic=10,
                workers=4,
                device='0',
                optimizer='SGD',  # using SGD
                    # resume='', # last.pt path
                amp=False,  # close amp
                    # fraction=0.2,
                project='runs/train',
                    name=name,
                    exist_ok=False,
                    )

def single_train_v6(model, data, name):
    model = YOLO(model)
    model.load('yolov6-n.pt')  # loading pretrain weights
    model.train(data=data,
                cache=False,
                imgsz=640,
                epochs=300,
                batch=8,
                close_mosaic=10,
                workers=4,
                device='0',
                optimizer='SGD',  # using SGD
                    # resume='', # last.pt path
                amp=False,  # close amp
                    # fraction=0.2,
                project='runs/train',
                    name=name,
                    exist_ok=False,
                    )

def single_train_v6_nopre(model, data, name):
    model = YOLO(model)
    # model.load('yolov6n.yaml')  # loading pretrain weights
    model.train(data=data,
                cache=False,
                imgsz=640,
                epochs=300,
                batch=8,
                close_mosaic=10,
                workers=4,
                device='0',
                optimizer='SGD',  # using SGD
                    # resume='', # last.pt path
                amp=False,  # close amp
                    # fraction=0.2,
                project='runs/train',
                    name=name,
                    exist_ok=False,
                    )

if __name__ == '__main__':
    # single_train_v6_nopre('ultralytics/cfg/models/v6/yolov6.yaml',
    #                 'D:\\workspace_py\\ultralytics\\ultralytics\\cfg\\datasets\\plantdoc_ext712.yaml',
    #                 'plantdoc_ext712_yolov6_nopre')
    # single_train_v6('ultralytics/cfg/models/v6/yolov6.yaml',
    #                 'D:\\workspace_py\\ultralytics\\ultralytics\\cfg\\datasets\\plantdoc_new.yaml',
    #                 'plantdoc_new_yolov6_nopre')
    #
    # single_train_v6('ultralytics/cfg/models/v6/yolov6.yaml',
    #                 'D:\\workspace_py\\ultralytics\\ultralytics\\cfg\\datasets\\plantdoc_ext712.yaml',
    #                 'plantdoc_ext712_yolov6_nopre')

    # single_train_v3('ultralytics/cfg/models/v3/yolov3-tiny.yaml',
    #                 'D:\\workspace_py\\ultralytics\\ultralytics\\cfg\\datasets\\plantdoc_ext712.yaml',
    #                 'plantdoc_ext712_yolov3')
    # single_train_v5('ultralytics/cfg/models/v5/yolov5n.yaml',
    #                 'D:\\workspace_py\\ultralytics\\ultralytics\\cfg\\datasets\\plantdoc_ext712.yaml',
    #                 'plantdoc_ext712_yolov5')
    # single_train('ultralytics/cfg/models/v8/yolov8n.yaml',
    #                 'D:\\workspace_py\\ultralytics\\ultralytics\\cfg\\datasets\\plantdoc_ext712.yaml',
    #                 'plantdoc_ext712_yolov8_nopre')
    #
    # single_train_v10_pre('ultralytics/cfg/models/v10/yolov10n.yaml',
    #                      'D:\\workspace_py\\ultralytics\\ultralytics\\cfg\\datasets\\plantdoc_ext712.yaml',
    #                      'plantdoc_ext712_yolov10n')
    # #
    single_train_v10_pre('ultralytics/cfg/models/v10/yolov10n_PSEMA_down_D4_2.yaml',
                         'D:\\workspace_py\\ultralytics\\ultralytics\\cfg\\datasets\\plantdoc_new.yaml',
                         'plantdoc_new_yolov10n_PSEMA_down_D4_2')

    single_train_v10_pre('ultralytics/cfg/models/v10/yolov10n_PSEMA_down_D4_1.yaml',
                         'D:\\workspace_py\\ultralytics\\ultralytics\\cfg\\datasets\\plantdoc_new.yaml',
                         'plantdoc_new_yolov10n_PSEMA_down_D4_1')
    #
    # single_train_v10_pre('ultralytics/cfg/models/v10/yolov10n_PSEMA_D4_2.yaml',
    #                      'D:\\workspace_py\\ultralytics\\ultralytics\\cfg\\datasets\\plantdoc_ext712.yaml',
    #                      'plantdoc_ext712_yolov10n_PSEMA_D4_2')

    # single_train_v10_pre('ultralytics/cfg/models/v10/yolov10n_D4MergeEMA_2.yaml',
    #                       'D:\\workspace_py\\ultralytics\\ultralytics\\cfg\\datasets\\plantdoc_new.yaml',
    #                       'plantdoc_new_yolov10n_D4MergeEMA_2_nopre')

    # single_train_v10_pre('ultralytics/cfg/models/v10/yolov10n_PSEMA.yaml',
    #                       'D:\\workspace_py\\ultralytics\\ultralytics\\cfg\\datasets\\plantdoc_new.yaml',
    #                       'plantdoc_new_yolov10n_PSEMA_nopre')
    #
    # single_train_v10_pre('ultralytics/cfg/models/v10/yolov10n_PSEMA_D4_1.yaml',
    #                       'D:\\workspace_py\\ultralytics\\ultralytics\\cfg\\datasets\\plantdoc_new.yaml',
    #                       'plantdoc_new_yolov10n_PSEMA_D4_1_nopre')
    #
    # single_train_v10_pre('ultralytics/cfg/models/v10/yolov10n_PSEMA_D4_2.yaml',
    #                       'D:\\workspace_py\\ultralytics\\ultralytics\\cfg\\datasets\\plantdoc_new.yaml',
    #                       'plantdoc_new_yolov10n_PSEMA_D4_2_nopre')
    #
    # single_train_v10_pre('ultralytics/cfg/models/v10/yolov10n_PSEMA_D4_3.yaml',
    #                       'D:\\workspace_py\\ultralytics\\ultralytics\\cfg\\datasets\\plantdoc_new.yaml',
    #                       'plantdoc_new_yolov10n_PSEMA_D4_3_nopre')

    #
    # single_train_v10_pre('ultralytics/cfg/models/v10/yolov10n_DCNv4_EMA_2.yaml',
    #                       'D:\\workspace_py\\ultralytics\\ultralytics\\cfg\\datasets\\plantdoc_new.yaml',
    #                       'plantdoc_new_yolov10n_DCNv4_EMA_2_nopre')
    #
    # single_train_v10_pre('ultralytics/cfg/models/v10/yolov10n_DCNv4_EMA_2.yaml',
    #                       'D:\\workspace_py\\ultralytics\\ultralytics\\cfg\\datasets\\plantdoc.yaml',
    #                       'plantdoc_yolov10n_DCNv4_EMA_2_nopre')
    #
    # single_train_v10_pre('ultralytics/cfg/models/v10/yolov10n.yaml',
    #                       'D:\\workspace_py\\ultralytics\\ultralytics\\cfg\\datasets\\plantdoc_new.yaml',
    #                       'plantdoc_new_yolov10n_nopre')
    #
    # single_train_v10_pre('ultralytics/cfg/models/v10/yolov10n.yaml',
    #                       'D:\\workspace_py\\ultralytics\\ultralytics\\cfg\\datasets\\plantdoc.yaml',
    #                       'plantdoc_yolov10n_nopre')





    # single_train_v10_pre('ultralytics/cfg/models/v10/yolov10n_DCNv4_EMA_2.yaml',
    #                       'D:\\workspace_py\\ultralytics\\ultralytics\\cfg\\datasets\\plantdoc_new.yaml',
    #                       'plantdocnew_yolov10n_DCNv4_EMA_2_W_INNER')
    # single_train_v10_pre('ultralytics/cfg/models/v10/yolov10n.yaml',
    #                       'D:\\workspace_py\\ultralytics\\ultralytics\\cfg\\datasets\\plantdoc_ext.yaml',
    #                       'plantdoc_ext_yolov10n_W_INNER')

    # single_train_v10_pre('ultralytics/cfg/models/v10/yolov10n_DCNv4_EMA_2.yaml',
    #                       'D:\\workspace_py\\ultralytics\\ultralytics\\cfg\\datasets\\plantdoc_ext.yaml',
    #                       'plantdoc_ext_yolov10n_DCNv4_EMA_2')

    # single_train_v10_pre('ultralytics/cfg/models/v10/yolov10n.yaml',
    #                       'D:\\workspace_py\\ultralytics\\ultralytics\\cfg\\datasets\\plantdoc_ext.yaml',
    #                       'plantdoc_ext_yolov10n')

    # single_train_v10_pre('ultralytics/cfg/models/v10/yolov10n_DCNv3_2.yaml',
    #                       'D:\\workspace_py\\ultralytics\\ultralytics\\cfg\\datasets\\plantdoc.yaml',
    #                       'plantdoc_yolov10n_DCNv3_2')
    # single_train_v10_pre('ultralytics/cfg/models/v10/yolov10n_DCNv4_2.yaml',
    #                       'D:\\workspace_py\\ultralytics\\ultralytics\\cfg\\datasets\\plantdoc.yaml',
    #                       'plantdoc_yolov10n_DCNv4_2')
    # single_train_v10_pre('ultralytics/cfg/models/v10/yolov10n_DCNv4_EMA_2.yaml',
    #                       'D:\\workspace_py\\ultralytics\\ultralytics\\cfg\\datasets\\plantdoc.yaml',
    #                       'plantdoc_yolov10n_CA')
    # single_train_v10_pre('ultralytics/cfg/models/v10/yolov10n_CBAM.yaml',
    #                       'D:\\workspace_py\\ultralytics\\ultralytics\\cfg\\datasets\\plantdoc.yaml',
    #                       'plantdoc_yolov10n_CBAM')
    # single_train_v10_pre('ultralytics/cfg/models/v10/yolov10n_EMA.yaml',
    #                       'D:\\workspace_py\\ultralytics\\ultralytics\\cfg\\datasets\\plantdoc.yaml',
    #                       'plantdoc_yolov10n_EMA')

    # single_train_v10_pre('ultralytics/cfg/models/v10/yolov10n_DCNv4_EMA_2.yaml',
    #                       'D:\\workspace_py\\ultralytics\\ultralytics\\cfg\\datasets\\plantdoc_ext.yaml',
    #                       'plantdoc_ext_v10_DCNv4_EMA_2_WISE_INNER')
    # single_train_v10_pre('ultralytics/cfg/models/v10/yolov10n_DCNv4_EMA_2.yaml',
    #                      'D:\\workspace_py\\ultralytics\\ultralytics\\cfg\\datasets\\plantdoc.yaml',
    #                      'plantdoc_v10_DCNv4_EMA_2_WISE_NO_INNER')
    # single_train_v10_pre('ultralytics/cfg/models/v10/yolov10n_DCNv4_EMA_2.yaml','D:\\workspace_py\\ultralytics\\ultralytics\\cfg\\datasets\\plantdoc.yaml','plantdoc_v10_DCNv4_EMA_2')

    # single_train_v10_pre('ultralytics/cfg/models/v10/yolov10n_DCNv4_CBAM_2.yaml','D:\\workspace_py\\ultralytics\\ultralytics\\cfg\\datasets\\plantdoc.yaml','plantdoc_v10_DCNv4_CBAM_2')

    # single_train_v10_pre('ultralytics/cfg/models/v10/yolov10n_DCNv4.yaml','D:\\workspace_py\\ultralytics\\ultralytics\\cfg\\datasets\\plantdoc100x100.yaml','plantdoc100x100_yolov10n_DCNv4')

    # single_train_v10_pre('ultralytics/cfg/models/v10/yolov10n_DCNv4.yaml','D:\\workspace_py\\ultralytics\\ultralytics\\cfg\\datasets\\plantdoc100x100.yaml','plantdoc100x100_yolov10n_DCNv4')

    #
    # single_train_v10_pre('ultralytics/cfg/models/v10/yolov10n_DCNv4.yaml',
    #                      'D:\\workspace_py\\ultralytics\\ultralytics\\cfg\\datasets\\plantdoc100x100.yaml', 'plantdoc100x100_yolov10n_DCNv4')
    #
    # single_train_v10_pre('ultralytics/cfg/models/v10/yolov10n_DCNv3.yaml','D:\\workspace_py\\ultralytics\\ultralytics\\cfg\\datasets\\plantdoc100x100.yaml','plantdoc100x100_v10_yolov10n_DCNv3')

    # single_train_v10_pre('ultralytics/cfg/models/v10/yolov10n_DCNV3_EMA.yaml',
    #                      'D:\\workspace_py\\ultralytics\\ultralytics\\cfg\\datasets\\plantdoc100x100.yaml', 'plantdoc100x100_yolov10n_DCNV3_EMA')
    #
    # single_train_v10_pre('ultralytics/cfg/models/v10/yolov10n_CBAM.yaml',
    #                      'D:\\workspace_py\\ultralytics\\ultralytics\\cfg\\datasets\\plantdoc100x100.yaml', 'plantdoc100x100_yolov10n_CBAM')

    # single_train_v10_pre('ultralytics/cfg/models/v10/yolov10n_DCNv4_EMA.yaml',
    #                      'D:\\workspace_py\\ultralytics\\ultralytics\\cfg\\datasets\\plantdoc100x100.yaml', 'plantdoc100x100_yolov10n_DCNv4_EMA')
    # single_train_v10_pre('ultralytics/cfg/models/v10/yolov10n_DCNv4_CA.yaml',
    #                      'D:\\workspace_py\\ultralytics\\ultralytics\\cfg\\datasets\\plantdoc100x100.yaml', 'plantdoc100x100_yolov10n_DCNv4_CA')
    # single_train_v10_pre('ultralytics/cfg/models/v10/yolov10n_DCNv4_CBAM.yaml',
    #                      'D:\\workspace_py\\ultralytics\\ultralytics\\cfg\\datasets\\plantdoc100x100.yaml', 'plantdoc100x100_yolov10n_DCNv4_CBAM')

    # single_train_v10_pre('ultralytics/cfg/models/v10/yolov10n_ED3.yaml','D:\\workspace_py\\ultralytics\\ultralytics\\cfg\\datasets\\plantdoc.yaml','plantdoc_v10_ED3')
    # single_train_v10('D:\\workspace_py\\ultralytics\\ultralytics\\cfg\\datasets\\plantdoc.yaml','plantdoc_v10')
    # single_train('ultralytics/cfg/models/v8/yolov8-C2f-DCNV3_EMA2_neck2.yaml','D:\\workspace_py\\ultralytics\\ultralytics\\cfg\\datasets\\plantdoc.yaml','plantdoc_ED3_neck2')

    # single_train('ultralytics/cfg/models/v8/yolov8-C2f-DCNV4.yaml','D:\\workspace_py\\ultralytics\\ultralytics\\cfg\\datasets\\plantdoc.yaml','plantdoc_DCNV4')

    # single_train('ultralytics/cfg/models/v8/yolov8-C2f-DCNV3.yaml','D:\\workspace_py\\ultralytics\\ultralytics\\cfg\\datasets\\plantdoc.yaml','plantdoc_DCNV3_test')
    #
    # single_train('ultralytics/cfg/models/v8/yolov8-EMA2.yaml','D:\\workspace_py\\ultralytics\\ultralytics\\cfg\\datasets\\plantdoc.yaml','plantdoc_EMA2')

    # single_train('ultralytics/cfg/models/v8/yolov8-C2f-DCNV3_CBAM2.yaml','D:\\workspace_py\\ultralytics\\ultralytics\\cfg\\datasets\\plantdoc.yaml','plantdoc_DCNV3_CBAM2')

    # single_train('ultralytics/cfg/models/v8/yolov8-C2f-DCNV3_CA2.yaml','D:\\workspace_py\\ultralytics\\ultralytics\\cfg\\datasets\\plantdoc.yaml','plantdoc_DCNV3_CA2')

    # single_train('ultralytics/cfg/models/v8/yolov8-C2f-DCNV3_EMA2.yaml',
    #              'D:\\workspace_py\\ultralytics\\ultralytics\\cfg\\datasets\\plantdoc.yaml', 'plantdoc_ED3_again')
    #
    # single_train('ultralytics/cfg/models/v8/yolov8.yaml',
    #              'D:\\workspace_py\\ultralytics\\ultralytics\\cfg\\datasets\\total2_ext6_82_g.yaml', 'total2_ext6_82_g_CIOU')
    # single_train('ultralytics/cfg/models/v8/yolov8-EMA2.yaml',
    #              'D:\\workspace_py\\ultralytics\\ultralytics\\cfg\\datasets\\total2_ext6_82_g.yaml', 'total2_ext6_82_g_EMA2_CIOU')
    # # #
    # single_train('ultralytics/cfg/models/v8/yolov8-C2f-DCNV3.yaml',
    #              'D:\\workspace_py\\ultralytics\\ultralytics\\cfg\\datasets\\total2_ext6_82_g.yaml', 'total2_ext6_82_g_C2f-DCNV3_CIOU')
    # # # #`
    # single_train('ultralytics/cfg/models/v8/yolov8-C2f-DCNV3_EMA2.yaml',
    #              'D:\\workspace_py\\ultralytics\\ultralytics\\cfg\\datasets\\total2_ext6_82_g.yaml', 'total2_ext6_82_g_C2f-DCNV3_EMA2_CIOU')

    # single_train('ultralytics/cfg/models/v8/yolov8-CA.yaml',
    #              'D:\\workspace_py\\ultralytics\\ultralytics\\cfg\\datasets\\total2_ext6_82.yaml', 'total2_ext6_82_CA')
    #
    # single_train('ultralytics/cfg/models/v8/yolov8-EMA.yaml',
    #              'D:\\workspace_py\\ultralytics\\ultralytics\\cfg\\datasets\\total2_ext6_82.yaml', 'total2_ext6_82_EMA')
    #
    # single_train('ultralytics/cfg/models/v8/yolov8-CBAM.yaml',
    #              'D:\\workspace_py\\ultralytics\\ultralytics\\cfg\\datasets\\total2_ext6_82.yaml', 'total2_ext6_82_CBAM')
    # single_train('ultralytics/cfg/models/v8/yolov8-C2f-DCNV3_2.yaml',
    #              'D:\\workspace_py\\ultralytics\\ultralytics\\cfg\\datasets\\total2_ext6_82.yaml', 'total2_ext6_82_C2f-DCNV3_2')
    # single_train('ultralytics/cfg/models/v8/yolov8-C2f-DCNV3_2_EMA2.yaml',
    #              'D:\\workspace_py\\ultralytics\\ultralytics\\cfg\\datasets\\total2_ext6_82.yaml', 'total2_ext6_82_C2f-DCNV3_2_EMA2')


    # single_train('ultralytics/cfg/models/v8/yolov8-CMA.yaml',
    #              'D:\\workspace_py\\ultralytics\\ultralytics\\cfg\\datasets\\total2_ext5.yaml', 'total2_ext5_CMA')
    # model = YOLO('ultralytics/cfg/models/v8/yolov8-dyhead.yaml')
    # model.load('yolov8n.pt') # loading pretrain weights
    # model.train(data='D:\\workspace_py\\ultralytics\\ultralytics\\cfg\\datasets\\total2_ext2.yaml',
    #             cache=True,
    #             imgsz=640,
    #             epochs=100,
    #             batch=16,
    #             close_mosaic=10,
    #             workers=4,
    #             device='0',
    #             optimizer='SGD', # using SGD
    #             # resume='', # last.pt path
    #             amp=False, # close amp
    #             # fraction=0.2,
    #             project='runs/train',
    #             name='total2_ext2',
    #             exist_ok=False,
    #             )