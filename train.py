# Train YOLOv11 from scratch on seat belt dataset


from ultralytics import YOLO

def main():
    
    model = YOLO("yolov11s.yaml")  

    model.train(
        data="data.yaml",   
        epochs=400,         
        imgsz=1024,
        batch=16,
        lr0=0.01,           
        optimizer="SGD",    
        project="runs/train",
        name="SeatBelt_Scratch",
        pretrained=False    
    )

if __name__ == "__main__":
    main()
