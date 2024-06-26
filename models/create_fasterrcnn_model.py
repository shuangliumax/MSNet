from models import *

#-------------------------Backbone---------------------------

def return_msnet(
    num_classes, pretrained=True, coco_model=False
):
    model = msnet.create_model(
        num_classes, pretrained=pretrained, coco_model=coco_model
    )
    return model

def return_fasterrcnn_resnet18(
    num_classes, pretrained=True, coco_model=False
):
    model = fasterrcnn_resnet18.create_model(
        num_classes, pretrained, coco_model
    )
    return model


create_model = {
    'fasterrcnn_resnet18': return_fasterrcnn_resnet18,
    'msnet':return_msnet,

    # 'faster_RCNN_ZOOM_case_five': return_faster_RCNN_ZOOM_case_five,

}