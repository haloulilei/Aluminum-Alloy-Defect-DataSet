from ultralytics import YOLO
 

if __name__=='__main__':
    model = YOLO("ultralytics/cfg/models/v8/yolov8-C2f-LSKA.yaml")  # build a new model from scratch
    #print(model.model)


# Train the model

    model.train(data="test.yaml")

