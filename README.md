# YoloV8_Nesne_Tespiti
Modelin eğitimi için iki farklı sınıfa (Elma ve Portakal) ait toplam 200 adet yüksek çözünürlüklü (>1920 piksel) görüntü kullanılmış, bu görüntüler telefon kamerası ile çekilmiş ve model uyumluluğu için 640 piksele yeniden ölçeklendirilmiştir. Görseller, YOLOv8 formatına uygun olacak şekilde bir etiketleme uygulaması kullanılarak etiketlenmiştir. 



 Proje YOLOv8 Tabanlı Nesne Tespiti .ipynb
 Proje YOLOv8 Tabanlı Nesne Tespiti .ipynb_
Proje Raporu: YOLOv8 Tabanlı Nesne Tespiti ve PyQt5 Görsel Arayüz Uygulaması
Adınız: Emmanuel
Soyadınız: HAKIRUWIZERA
Okul Numaranız: 2440631002
GitHub Repo Bağlantısı: https://github.com/Eng-Emmy/YoloV8_Nesne_Tespiti/blob/main/Proje_YOLOv8_Tabanl%C4%B1_Nesne_Tespiti_.ipynb


Proje Açıklaması
Bu projede, YOLOv8 framework’ü kullanılarak nesne tespiti yapan bir model geliştirilmiş ve PyQt5 tabanlı bir grafiksel kullanıcı arayüzüne (GUI) entegre edilmiştir. Modelin eğitimi için iki farklı sınıfa (Elma ve Portakal) ait toplam 200 adet yüksek çözünürlüklü (>1920 piksel) görüntü kullanılmış, bu görüntüler telefon kamerası ile çekilmiş ve model uyumluluğu için 640 piksele yeniden ölçeklendirilmiştir. Görseller, YOLOv8 formatına uygun olacak şekilde bir etiketleme uygulaması kullanılarak etiketlenmiştir. Veri seti hazırlandıktan sonra YOLOv8 modeli konfigüre edilmiş ve eğitilmiş, başarı metrikleri olarak mAP ve kayıp (loss) grafikleri raporlanmıştır. En iyi performans gösteren model ağırlıkları (best.pt) kaydedilmiştir. PyQt5 ile geliştirilen masaüstü arayüz, kullanıcıların bir görüntü yüklemesine, bu görüntü üzerinde tespit edilen nesnelerin bounding box’lar ile gösterilmesine ve tespit edilen sınıfların ve sayılarının listelenmesine olanak sağlamaktadır. Bu çalışma, veri seti oluşturma ve model eğitimi aşamalarından, gerçek zamanlı nesne tespitini kullanıcı dostu bir uygulama üzerinden sunmaya kadar olan süreci kapsamaktadır.

ETİKETLEME
Program: LabelImg

To start labeling, open LabelImg and select the image folder by clicking Open Dir. Then, set the directory where labels will be saved by going to File → Change Save Dir and create a new folder named labels. Enable Auto Save Mode under the View menu to ensure labels are saved automatically. Switch to YOLO format (instead of PascalVOC) by selecting File → PascalVOC and changing it to YOLO. For each image, draw a bounding box around apple and orange and enter the class name in the left panel (use 0 and 1 for two classes).

Screenshot (3).png

MİMARİ
Model: YOLOv8 (Ultralytics) Colab üzerinde kurulumu ve eğitimi kulanılmıştır. Eğitilmiş modellerle transfer öğrenme yapmak basit.

Etiket ve görüntü klasörlerini Google Drive'a atalım
Ensure the YOLOv8 dataset structure is correct before uploading to Google Drive:
dataset (images: train and valid; Labels:train and valid)
YOLOV8 kurulumu
Ultralytics, YOLOv8’in geliştiricisi ve resmi bakım sağlayıcısıdır; kodları, modelleri ve araçları sunarak YOLOv8’in eğitilmesini ve dağıtılmasını kolay ve verimli hale getirir.

[ ]
# YOLOv8 kurulumu
!pip install ultralytics

Collecting ultralytics
  Downloading ultralytics-8.3.239-py3-none-any.whl.metadata (37 kB)
Requirement already satisfied: numpy>=1.23.0 in /usr/local/lib/python3.12/dist-packages (from ultralytics) (2.0.2)
Requirement already satisfied: matplotlib>=3.3.0 in /usr/local/lib/python3.12/dist-packages (from ultralytics) (3.10.0)
Requirement already satisfied: opencv-python>=4.6.0 in /usr/local/lib/python3.12/dist-packages (from ultralytics) (4.12.0.88)
Requirement already satisfied: pillow>=7.1.2 in /usr/local/lib/python3.12/dist-packages (from ultralytics) (11.3.0)
Requirement already satisfied: pyyaml>=5.3.1 in /usr/local/lib/python3.12/dist-packages (from ultralytics) (6.0.3)
Requirement already satisfied: requests>=2.23.0 in /usr/local/lib/python3.12/dist-packages (from ultralytics) (2.32.4)
Requirement already satisfied: scipy>=1.4.1 in /usr/local/lib/python3.12/dist-packages (from ultralytics) (1.16.3)
Requirement already satisfied: torch>=1.8.0 in /usr/local/lib/python3.12/dist-packages (from ultralytics) (2.9.0+cu126)
Requirement already satisfied: torchvision>=0.9.0 in /usr/local/lib/python3.12/dist-packages (from ultralytics) (0.24.0+cu126)
Requirement already satisfied: psutil>=5.8.0 in /usr/local/lib/python3.12/dist-packages (from ultralytics) (5.9.5)
Requirement already satisfied: polars>=0.20.0 in /usr/local/lib/python3.12/dist-packages (from ultralytics) (1.31.0)
Collecting ultralytics-thop>=2.0.18 (from ultralytics)
  Downloading ultralytics_thop-2.0.18-py3-none-any.whl.metadata (14 kB)
Requirement already satisfied: contourpy>=1.0.1 in /usr/local/lib/python3.12/dist-packages (from matplotlib>=3.3.0->ultralytics) (1.3.3)
Requirement already satisfied: cycler>=0.10 in /usr/local/lib/python3.12/dist-packages (from matplotlib>=3.3.0->ultralytics) (0.12.1)
Requirement already satisfied: fonttools>=4.22.0 in /usr/local/lib/python3.12/dist-packages (from matplotlib>=3.3.0->ultralytics) (4.61.0)
Requirement already satisfied: kiwisolver>=1.3.1 in /usr/local/lib/python3.12/dist-packages (from matplotlib>=3.3.0->ultralytics) (1.4.9)
Requirement already satisfied: packaging>=20.0 in /usr/local/lib/python3.12/dist-packages (from matplotlib>=3.3.0->ultralytics) (25.0)
Requirement already satisfied: pyparsing>=2.3.1 in /usr/local/lib/python3.12/dist-packages (from matplotlib>=3.3.0->ultralytics) (3.2.5)
Requirement already satisfied: python-dateutil>=2.7 in /usr/local/lib/python3.12/dist-packages (from matplotlib>=3.3.0->ultralytics) (2.9.0.post0)
Requirement already satisfied: charset_normalizer<4,>=2 in /usr/local/lib/python3.12/dist-packages (from requests>=2.23.0->ultralytics) (3.4.4)
Requirement already satisfied: idna<4,>=2.5 in /usr/local/lib/python3.12/dist-packages (from requests>=2.23.0->ultralytics) (3.11)
Requirement already satisfied: urllib3<3,>=1.21.1 in /usr/local/lib/python3.12/dist-packages (from requests>=2.23.0->ultralytics) (2.5.0)
Requirement already satisfied: certifi>=2017.4.17 in /usr/local/lib/python3.12/dist-packages (from requests>=2.23.0->ultralytics) (2025.11.12)
Requirement already satisfied: filelock in /usr/local/lib/python3.12/dist-packages (from torch>=1.8.0->ultralytics) (3.20.0)
Requirement already satisfied: typing-extensions>=4.10.0 in /usr/local/lib/python3.12/dist-packages (from torch>=1.8.0->ultralytics) (4.15.0)
Requirement already satisfied: setuptools in /usr/local/lib/python3.12/dist-packages (from torch>=1.8.0->ultralytics) (75.2.0)
Requirement already satisfied: sympy>=1.13.3 in /usr/local/lib/python3.12/dist-packages (from torch>=1.8.0->ultralytics) (1.14.0)
Requirement already satisfied: networkx>=2.5.1 in /usr/local/lib/python3.12/dist-packages (from torch>=1.8.0->ultralytics) (3.6.1)
Requirement already satisfied: jinja2 in /usr/local/lib/python3.12/dist-packages (from torch>=1.8.0->ultralytics) (3.1.6)
Requirement already satisfied: fsspec>=0.8.5 in /usr/local/lib/python3.12/dist-packages (from torch>=1.8.0->ultralytics) (2025.3.0)
Requirement already satisfied: nvidia-cuda-nvrtc-cu12==12.6.77 in /usr/local/lib/python3.12/dist-packages (from torch>=1.8.0->ultralytics) (12.6.77)
Requirement already satisfied: nvidia-cuda-runtime-cu12==12.6.77 in /usr/local/lib/python3.12/dist-packages (from torch>=1.8.0->ultralytics) (12.6.77)
Requirement already satisfied: nvidia-cuda-cupti-cu12==12.6.80 in /usr/local/lib/python3.12/dist-packages (from torch>=1.8.0->ultralytics) (12.6.80)
Requirement already satisfied: nvidia-cudnn-cu12==9.10.2.21 in /usr/local/lib/python3.12/dist-packages (from torch>=1.8.0->ultralytics) (9.10.2.21)
Requirement already satisfied: nvidia-cublas-cu12==12.6.4.1 in /usr/local/lib/python3.12/dist-packages (from torch>=1.8.0->ultralytics) (12.6.4.1)
Requirement already satisfied: nvidia-cufft-cu12==11.3.0.4 in /usr/local/lib/python3.12/dist-packages (from torch>=1.8.0->ultralytics) (11.3.0.4)
Requirement already satisfied: nvidia-curand-cu12==10.3.7.77 in /usr/local/lib/python3.12/dist-packages (from torch>=1.8.0->ultralytics) (10.3.7.77)
Requirement already satisfied: nvidia-cusolver-cu12==11.7.1.2 in /usr/local/lib/python3.12/dist-packages (from torch>=1.8.0->ultralytics) (11.7.1.2)
Requirement already satisfied: nvidia-cusparse-cu12==12.5.4.2 in /usr/local/lib/python3.12/dist-packages (from torch>=1.8.0->ultralytics) (12.5.4.2)
Requirement already satisfied: nvidia-cusparselt-cu12==0.7.1 in /usr/local/lib/python3.12/dist-packages (from torch>=1.8.0->ultralytics) (0.7.1)
Requirement already satisfied: nvidia-nccl-cu12==2.27.5 in /usr/local/lib/python3.12/dist-packages (from torch>=1.8.0->ultralytics) (2.27.5)
Requirement already satisfied: nvidia-nvshmem-cu12==3.3.20 in /usr/local/lib/python3.12/dist-packages (from torch>=1.8.0->ultralytics) (3.3.20)
Requirement already satisfied: nvidia-nvtx-cu12==12.6.77 in /usr/local/lib/python3.12/dist-packages (from torch>=1.8.0->ultralytics) (12.6.77)
Requirement already satisfied: nvidia-nvjitlink-cu12==12.6.85 in /usr/local/lib/python3.12/dist-packages (from torch>=1.8.0->ultralytics) (12.6.85)
Requirement already satisfied: nvidia-cufile-cu12==1.11.1.6 in /usr/local/lib/python3.12/dist-packages (from torch>=1.8.0->ultralytics) (1.11.1.6)
Requirement already satisfied: triton==3.5.0 in /usr/local/lib/python3.12/dist-packages (from torch>=1.8.0->ultralytics) (3.5.0)
Requirement already satisfied: six>=1.5 in /usr/local/lib/python3.12/dist-packages (from python-dateutil>=2.7->matplotlib>=3.3.0->ultralytics) (1.17.0)
Requirement already satisfied: mpmath<1.4,>=1.1.0 in /usr/local/lib/python3.12/dist-packages (from sympy>=1.13.3->torch>=1.8.0->ultralytics) (1.3.0)
Requirement already satisfied: MarkupSafe>=2.0 in /usr/local/lib/python3.12/dist-packages (from jinja2->torch>=1.8.0->ultralytics) (3.0.3)
Downloading ultralytics-8.3.239-py3-none-any.whl (1.1 MB)
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 1.1/1.1 MB 31.5 MB/s eta 0:00:00
Downloading ultralytics_thop-2.0.18-py3-none-any.whl (28 kB)
Installing collected packages: ultralytics-thop, ultralytics
Successfully installed ultralytics-8.3.239 ultralytics-thop-2.0.18
Drive Bağlantısı ve Dataset Yolunu Tanımlama

[ ]
# Google Drive'ı bağla
from google.colab import drive
drive.mount('/content/drive')

# Dataset yolu
dataset_path = '/content/drive/Mydrive/Colab_yolo/fruits'
Mounted at /content/drive
data.yaml dosyasını oluştur
YOLOv8 eğitim sürecinde data.yaml dosyası, veri kümesinin yapılandırılmasında kritik bir rol üstlenmektedir. Bu dosya, modelin eğitim ve doğrulama aşamalarında kullanılacak görüntülerin ve etiketlerin konumlarını tanımlamakla birlikte, sınıf isimlerini de belirtir. Başka bir deyişle, data.yaml dosyası, YOLOv8’in veri kümesine erişimini ve sınıfların doğru şekilde yorumlanmasını sağlayan temel yapılandırma bileşenidir.


[ ]
# Writing YAML content
import yaml
import os # Import the os module

data = {
    'path': '/content/drive/MyDrive/Colab_yolo/fruits',
    'train': 'images/train',
    'val': 'images/val',
    'names': ['Apple', 'Orange'],
    'nc': 2,  # Number of classes
    'class_names': ['Apple', 'Orange']
}

yaml_path = '/content/drive/MyDrive/Colab_yolo/fruits/data.yaml'

# Ensure the directory exists before writing the file
os.makedirs(os.path.dirname(yaml_path), exist_ok=True)

with open(yaml_path, 'w') as file:
    yaml.dump(data, file)

EĞİTİM
Eğitim ilerledikçe en iyi ağırlıkları (best.pt) kaydedecek.


[ ]
!yolo task=detect mode=train \
    model=yolov8s.pt \
    data='/content/drive/MyDrive/Colab_yolo/fruits/data.yaml' \
    epochs=10 \
    imgsz=640 \
    batch=16 \
    optimizer=AdamW \
    lr0=0.001 \
    lrf=0.1 \
    momentum=0.937 \
    weight_decay=0.0001 \
    warmup_epochs=3.0 \
    patience=30 \
    dropout=0.1 \
    hsv_h=0.015 \
    hsv_s=0.7 \
    hsv_v=0.4 \
    degrees=0.0 \
    scale=0.5 \
    shear=0.0 \
    perspective=0.0 \
    flipud=0.0 \


Creating new Ultralytics Settings v0.0.6 file ✅ 
View Ultralytics Settings with 'yolo settings' or at '/root/.config/Ultralytics/settings.json'
Update Settings with 'yolo settings key=value', i.e. 'yolo settings runs_dir=path/to/dir'. For help see https://docs.ultralytics.com/quickstart/#ultralytics-settings.
Downloading https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8s.pt to 'yolov8s.pt': 100% ━━━━━━━━━━━━ 21.5MB 105.4MB/s 0.2s
Ultralytics 8.3.239 🚀 Python-3.12.12 torch-2.9.0+cu126 CUDA:0 (Tesla T4, 15095MiB)
engine/trainer: agnostic_nms=False, amp=True, augment=False, auto_augment=randaugment, batch=16, bgr=0.0, box=7.5, cache=False, cfg=None, classes=None, close_mosaic=10, cls=0.5, compile=False, conf=None, copy_paste=0.0, copy_paste_mode=flip, cos_lr=False, cutmix=0.0, data=/content/drive/MyDrive/Colab_yolo/fruits/data.yaml, degrees=0.0, deterministic=True, device=None, dfl=1.5, dnn=False, dropout=0.1, dynamic=False, embed=None, epochs=10, erasing=0.4, exist_ok=False, fliplr=0.5, flipud=0.0, format=torchscript, fraction=1.0, freeze=None, half=False, hsv_h=0.015, hsv_s=0.7, hsv_v=0.4, imgsz=640, int8=False, iou=0.7, keras=False, kobj=1.0, line_width=None, lr0=0.001, lrf=0.1, mask_ratio=4, max_det=300, mixup=0.0, mode=train, model=yolov8s.pt, momentum=0.937, mosaic=1.0, multi_scale=False, name=train, nbs=64, nms=False, opset=None, optimize=False, optimizer=AdamW, overlap_mask=True, patience=30, perspective=0.0, plots=True, pose=12.0, pretrained=True, profile=False, project=None, rect=False, resume=False, retina_masks=False, save=True, save_conf=False, save_crop=False, save_dir=/content/runs/detect/train, save_frames=False, save_json=False, save_period=-1, save_txt=False, scale=0.5, seed=0, shear=0.0, show=False, show_boxes=True, show_conf=True, show_labels=True, simplify=True, single_cls=False, source=None, split=val, stream_buffer=False, task=detect, time=None, tracker=botsort.yaml, translate=0.1, val=True, verbose=True, vid_stride=1, visualize=False, warmup_bias_lr=0.1, warmup_epochs=3.0, warmup_momentum=0.8, weight_decay=0.0001, workers=8, workspace=None
Downloading https://ultralytics.com/assets/Arial.ttf to '/root/.config/Ultralytics/Arial.ttf': 100% ━━━━━━━━━━━━ 755.1KB 28.8MB/s 0.0s
Overriding model.yaml nc=80 with nc=2

                   from  n    params  module                                       arguments                     
  0                  -1  1       928  ultralytics.nn.modules.conv.Conv             [3, 32, 3, 2]                 
  1                  -1  1     18560  ultralytics.nn.modules.conv.Conv             [32, 64, 3, 2]                
  2                  -1  1     29056  ultralytics.nn.modules.block.C2f             [64, 64, 1, True]             
  3                  -1  1     73984  ultralytics.nn.modules.conv.Conv             [64, 128, 3, 2]               
  4                  -1  2    197632  ultralytics.nn.modules.block.C2f             [128, 128, 2, True]           
  5                  -1  1    295424  ultralytics.nn.modules.conv.Conv             [128, 256, 3, 2]              
  6                  -1  2    788480  ultralytics.nn.modules.block.C2f             [256, 256, 2, True]           
  7                  -1  1   1180672  ultralytics.nn.modules.conv.Conv             [256, 512, 3, 2]              
  8                  -1  1   1838080  ultralytics.nn.modules.block.C2f             [512, 512, 1, True]           
  9                  -1  1    656896  ultralytics.nn.modules.block.SPPF            [512, 512, 5]                 
 10                  -1  1         0  torch.nn.modules.upsampling.Upsample         [None, 2, 'nearest']          
 11             [-1, 6]  1         0  ultralytics.nn.modules.conv.Concat           [1]                           
 12                  -1  1    591360  ultralytics.nn.modules.block.C2f             [768, 256, 1]                 
 13                  -1  1         0  torch.nn.modules.upsampling.Upsample         [None, 2, 'nearest']          
 14             [-1, 4]  1         0  ultralytics.nn.modules.conv.Concat           [1]                           
 15                  -1  1    148224  ultralytics.nn.modules.block.C2f             [384, 128, 1]                 
 16                  -1  1    147712  ultralytics.nn.modules.conv.Conv             [128, 128, 3, 2]              
 17            [-1, 12]  1         0  ultralytics.nn.modules.conv.Concat           [1]                           
 18                  -1  1    493056  ultralytics.nn.modules.block.C2f             [384, 256, 1]                 
 19                  -1  1    590336  ultralytics.nn.modules.conv.Conv             [256, 256, 3, 2]              
 20             [-1, 9]  1         0  ultralytics.nn.modules.conv.Concat           [1]                           
 21                  -1  1   1969152  ultralytics.nn.modules.block.C2f             [768, 512, 1]                 
 22        [15, 18, 21]  1   2116822  ultralytics.nn.modules.head.Detect           [2, [128, 256, 512]]          
Model summary: 129 layers, 11,136,374 parameters, 11,136,358 gradients, 28.6 GFLOPs

Transferred 349/355 items from pretrained weights
Freezing layer 'model.22.dfl.conv.weight'
AMP: running Automatic Mixed Precision (AMP) checks...
Downloading https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n.pt to 'yolo11n.pt': 100% ━━━━━━━━━━━━ 5.4MB 93.1MB/s 0.1s
AMP: checks passed ✅
train: Fast image access ✅ (ping: 0.6±0.2 ms, read: 0.8±0.3 MB/s, size: 308.0 KB)
train: Scanning /content/drive/MyDrive/Colab_yolo/fruits/labels/train/Apple... 160 images, 0 backgrounds, 0 corrupt: 100% ━━━━━━━━━━━━ 160/160 1.2it/s 2:09
train: New cache created: /content/drive/MyDrive/Colab_yolo/fruits/labels/train/Apple.cache
albumentations: Blur(p=0.01, blur_limit=(3, 7)), MedianBlur(p=0.01, blur_limit=(3, 7)), ToGray(p=0.01, method='weighted_average', num_output_channels=3), CLAHE(p=0.01, clip_limit=(1.0, 4.0), tile_grid_size=(8, 8))
val: Fast image access ✅ (ping: 1.5±1.0 ms, read: 0.5±0.3 MB/s, size: 170.8 KB)
val: Scanning /content/drive/MyDrive/Colab_yolo/fruits/labels/val/Apple... 40 images, 0 backgrounds, 0 corrupt: 100% ━━━━━━━━━━━━ 40/40 1.3it/s 30.9s
val: New cache created: /content/drive/MyDrive/Colab_yolo/fruits/labels/val/Apple.cache
Plotting labels to /content/runs/detect/train/labels.jpg... 
optimizer: AdamW(lr=0.001, momentum=0.937) with parameter groups 57 weight(decay=0.0), 64 weight(decay=0.0001), 63 bias(decay=0.0)
Image sizes 640 train, 640 val
Using 2 dataloader workers
Logging results to /content/runs/detect/train
Starting training for 10 epochs...
Closing dataloader mosaic
albumentations: Blur(p=0.01, blur_limit=(3, 7)), MedianBlur(p=0.01, blur_limit=(3, 7)), ToGray(p=0.01, method='weighted_average', num_output_channels=3), CLAHE(p=0.01, clip_limit=(1.0, 4.0), tile_grid_size=(8, 8))

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
       1/10      3.73G      2.983      4.533      3.396         67        640: 100% ━━━━━━━━━━━━ 10/10 1.6it/s 6.1s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 2/2 1.0it/s 1.9s
                   all         40        120      0.238      0.184      0.231      0.153

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
       2/10      4.54G      2.479       3.06      2.566         40        640: 100% ━━━━━━━━━━━━ 10/10 2.8it/s 3.5s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 2/2 2.3it/s 0.9s
                   all         40        120     0.0601      0.441     0.0418     0.0103

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
       3/10      4.57G      2.356      2.636        2.4         57        640: 100% ━━━━━━━━━━━━ 10/10 3.5it/s 2.9s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 2/2 5.1it/s 0.4s
                   all         40        120      0.045       0.17     0.0365     0.0118

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
       4/10      4.61G      2.244      2.512       2.22         73        640: 100% ━━━━━━━━━━━━ 10/10 3.6it/s 2.8s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 2/2 5.6it/s 0.4s
                   all         40        120      0.262      0.297      0.209      0.103

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
       5/10      4.65G      2.198      2.487      2.228         72        640: 100% ━━━━━━━━━━━━ 10/10 3.6it/s 2.8s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 2/2 2.8it/s 0.7s
                   all         40        120      0.227      0.369      0.245     0.0728

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
       6/10      4.68G      2.115      2.437      2.248         81        640: 100% ━━━━━━━━━━━━ 10/10 3.2it/s 3.2s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 2/2 4.4it/s 0.4s
                   all         40        120       0.56      0.391      0.429      0.205

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
       7/10      4.72G      2.133      2.323        2.2         68        640: 100% ━━━━━━━━━━━━ 10/10 3.6it/s 2.7s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 2/2 4.4it/s 0.5s
                   all         40        120      0.496      0.457      0.485      0.255

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
       8/10      4.76G      2.092      2.309      2.198         64        640: 100% ━━━━━━━━━━━━ 10/10 3.6it/s 2.8s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 2/2 4.8it/s 0.4s
                   all         40        120      0.608      0.477      0.464      0.268

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
       9/10      4.79G      2.036      2.201      2.105         59        640: 100% ━━━━━━━━━━━━ 10/10 3.0it/s 3.3s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 2/2 2.8it/s 0.7s
                   all         40        120      0.615      0.478      0.514      0.302

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
      10/10      4.83G      1.896       2.14      2.076         38        640: 100% ━━━━━━━━━━━━ 10/10 3.6it/s 2.8s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 2/2 5.5it/s 0.4s
                   all         40        120      0.697       0.46      0.525      0.334

10 epochs completed in 0.013 hours.
Optimizer stripped from /content/runs/detect/train/weights/last.pt, 22.5MB
Optimizer stripped from /content/runs/detect/train/weights/best.pt, 22.5MB

Validating /content/runs/detect/train/weights/best.pt...
Ultralytics 8.3.239 🚀 Python-3.12.12 torch-2.9.0+cu126 CUDA:0 (Tesla T4, 15095MiB)
Model summary (fused): 72 layers, 11,126,358 parameters, 0 gradients, 28.4 GFLOPs
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 2/2 4.0it/s 0.5s
                   all         40        120      0.726      0.459      0.525      0.333
                 Apple         21         71      0.498     0.0563       0.13      0.052
                Orange         19         49      0.955      0.862       0.92      0.614
Speed: 0.2ms preprocess, 4.2ms inference, 0.0ms loss, 1.6ms postprocess per image
Results saved to /content/runs/detect/train
💡 Learn more at https://docs.ultralytics.com/modes/train

[ ]
import os

val_images_path = '/content/drive/MyDrive/Colab_yolo/fruits/images/val'
if os.path.exists(val_images_path):
    print(f"Directory exists: {val_images_path}")
    # Optionally, list some contents
    # print(os.listdir(val_images_path)[:5])
else:
    print(f"Directory does NOT exist: {val_images_path}")
    print("Please ensure your dataset is properly structured in Google Drive with validation images in this path.")
Directory exists: /content/drive/MyDrive/Colab_yolo/fruits/images/val
PERFORMANS
Drive'ın soldaki dosyalar bölümüne baktığımızda üretilen performans metriklerini indirdi.


[ ]
from google.colab import drive
drive.mount('/content/drive')
Drive already mounted at /content/drive; to attempt to forcibly remount, call drive.mount("/content/drive", force_remount=True).
Üretilen Best.pt dosyaları hangileri?


[ ]
import os

root_path = '/content/runs/detect/'

best_models = []
for folder in os.listdir(root_path):
    if folder.startswith('train'):
        best_path = os.path.join(root_path, folder, 'weights', 'best.pt')
        if os.path.exists(best_path):
            best_models.append(best_path)

print("Best.pt dosyaları:")
for model in best_models:
    print(model)

Best.pt dosyaları:
/content/runs/detect/train/weights/best.pt
Gerçek görüntü üzerinde test et

[ ]
from ultralytics import YOLO

# Eğitilmiş modeli yükle
model = YOLO('runs/detect/train/weights/best.pt')

# Örnek görüntü üzerinde tahmin yap, sonuçlar runs/detect/predict içine kaydedildi.
results = model.predict('/content/drive/MyDrive/Colab_yolo/fruits/Orange test1.jpeg', save=True)
results[0].show()



[ ]
results = model.predict('/content/drive/MyDrive/Colab_yolo/fruits/Orange test2.jpeg', save=True)
results[0].show()

REFERENCE

Wang, H., Liu, C., Cai, Y., Chen, L., & Li, Y. (2024). YOLOv8-QSD: An improved small object detection algorithm for autonomous vehicles based on YOLOv8. IEEE Transactions on Instrumentation and Measurement, 73, 1-16.

Büyükgökoğlan, E., & Uğuz, S. (2025). Development of a Performance Evaluation System in Turkish Folk Dance Using Deep Learning-Based Pose Estimation. Tehnički vjesnik, 32(5), 1817-1824.

Sai, J. M., Priyadarshini, G. I., Goud, G. P., Varma, D. H. V. S., Pitchai, R., & Jyothirmai, D. (2024, June). Smart and sustainable framework for Maize Leaf Disease Prediction using Deep Learning Techniques. In 2024 First International Conference on Technological Innovations and Advance Computing (TIACOMP) (pp. 467-471). IEEE.

Paewboontra, W., & Nimsuk, N. (2024). Detecting Multi-scale Rose Apple Skin and Defects Using Instance Segmentation with Anchors Optimization. IEEE Access.

Colab paid products - Cancel contracts here
