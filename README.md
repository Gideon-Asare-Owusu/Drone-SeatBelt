# UAV-based Automatic Seat Belt Compliance Detection

This project uses a **drone-mounted (UAV) camera** and a **customized YOLO11 deep learning model** to automatically detect seat belt use at stop-controlled intersections. It provides a scalable, efficient, and cost-effective alternative to manual compliance surveys.

## Features
- Automated seat belt detection using **YOLO11**
- UAV aerial video capture
- Handles lighting, vehicle types and windshield illumination, and seatbelt-clothing color challenges
- Achieved **94% accuracy** and **mAP = 0.902**
- Outperforms manual surveys and reduces observer bias

## Installation
Clone this repository:
```bash
https://github.com/Gideon-Asare-Owusu/Drone-SeatBelt.git
cd your-repo
pip install -r requirements.txt
```

## Usage
Run detection on test images:
```bash
python detect.py --weights best.pt --source test_images/
```

Train the model:
```bash
python train.py --data data.yaml --weights yolov11.pt --epochs 200
```

## Results
- **mAP:** 0.902  
- **Accuracy:** 94%  
- Best performance at **18 ft UAV elevation with 2.3x zoom**

| Condition                     | Accuracy |
|-------------------------------|----------|
| High seat belt–shirt contrast | 99.5%    |
| Low contrast                  | 91.0%    |
| Sun behind UAV                | 92.5%    |
| UAV facing sun (glare)        | 57.0%    |
| Clear windshield              | 94.0%    |
| Tinted windshield             | 84.5%    |

## Authors
- Gideon Asare Owusu  
- Collaborators: Ashutosh Dumka, Adu-Gyamfi Kojo, Enoch Kwasi Asante, Rishab Jain, Skylar Knickerbocker, Neal Hawkins, Anuj Sharma

## License
This project is licensed under the [MIT License](LICENSE).
