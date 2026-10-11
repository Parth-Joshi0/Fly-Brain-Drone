# BananaModel UML Diagrams

UML diagrams for the `BananaModel/` folder. See [README.md](README.md)
for what the model does and how well it performs.

The folder has two separate parts:

- **Training** (`config.py`, `dataset.py`, `model.py`, `train.py`): run
  offline to produce `checkpoints/best_banana_model.pth`.
- **Inference** (`banana_detector.py`): loads that checkpoint and is
  imported by the rest of the project. It does **not** import any of the
  training modules. It builds its own copy of the MobileNetV2 head and
  its own resize/normalise transform.

## 1. Class diagram

Python modules that only contain functions and constants are shown as
`<<module>>` classes.

```mermaid
classDiagram
    direction LR

    class Detection {
        +tuple box
        +float det_conf
        +str label
        +float cls_conf
        +Detection(box, det_conf, label, cls_conf)
        +__repr__() str
    }

    class BananaDetector {
        +torch.device device
        +int img_size
        +float detector_conf_threshold
        +int detector_img_size
        +float classifier_conf_threshold
        +float padding_ratio
        +list~str~ default_class_names
        +list~str~ class_names
        +MobileNetV2 classifier
        +Compose classify_transform
        +YOLO detector
        +BananaDetector(checkpoint_path, class_names_path, ...)
        -_load_class_names(class_names_path) list~str~
        -_build_classifier(num_classes) MobileNetV2
        +detect(frame) list~Detection~
        +annotate(frame, detections) ndarray
        +detect_and_annotate(frame) tuple
    }

    class banana_detector {
        <<module>>
        +str CHECKPOINT_DIR
        +int COCO_BANANA_CLASS_ID = 46
    }

    class YOLO {
        <<ultralytics>>
        +__call__(frame, conf, imgsz, classes, verbose) Results
    }

    class MobileNetV2 {
        <<torchvision>>
        +features
        +Sequential classifier
        +int last_channel = 1280
    }

    banana_detector *-- BananaDetector : defines
    banana_detector *-- Detection : defines
    BananaDetector *-- "1" YOLO : detector
    BananaDetector *-- "1" MobileNetV2 : classifier
    BananaDetector ..> Detection : creates
```

```mermaid
classDiagram
    direction LR

    class config {
        <<module>>
        +torch.device DEVICE
        +str TRAIN_DIR = "dataset/train"
        +str VAL_DIR = "dataset/valid"
        +str TEST_DIR = "dataset/test"
        +str CHECKPOINT_PATH
        +int IMG_SIZE = 224
        +int BATCH_SIZE = 32
        +float LR = 1e-3
        +int EPOCHS = 50
        +int PATIENCE = 10
        +int NUM_WORKERS = 4
    }

    class dataset {
        <<module>>
        +Compose train_transform
        +Compose val_transform
        +get_loaders() tuple
    }

    class model {
        <<module>>
        +build_model(num_classes) MobileNetV2
    }

    class train {
        <<module>>
        +train() None
        +evaluate(model, loader, criterion) tuple
    }

    class MobileNetV2 {
        <<torchvision>>
    }

    class ImageFolder {
        <<torchvision>>
    }

    class DataLoader {
        <<torch>>
    }

    train ..> config : reads
    train ..> dataset : get_loaders()
    train ..> model : build_model()
    dataset ..> config : reads
    dataset ..> ImageFolder : creates
    dataset ..> DataLoader : creates
    model ..> MobileNetV2 : creates
```

`get_loaders()` returns `(train_loader, val_loader, class_names)`, and
`evaluate()` returns `(loss, accuracy)`.

## 2. Activity diagram: training (`python train.py`)

```mermaid
flowchart TD
    start([start]) --> loaders["get_loaders()<br/>ImageFolder train + valid<br/>train gets augmentation"]
    loaders --> build["build_model(6)<br/>MobileNetV2 ImageNet weights<br/>freeze features, new head"]
    build --> setup["CrossEntropyLoss<br/>Adam on classifier params, lr 1e-3<br/>best_val_acc = 0, patience = 0"]
    setup --> epoch{"epoch < EPOCHS (50)?"}
    epoch -- no --> done([end])
    epoch -- yes --> trainloop["model.train()<br/>for each batch: forward, loss,<br/>backward, optimizer.step()"]
    trainloop --> evalstep["evaluate(model, val_loader)<br/>gives val_loss, val_acc"]
    evalstep --> log["print epoch losses and accuracies"]
    log --> better{"val_acc > best_val_acc?"}
    better -- yes --> save["best_val_acc = val_acc<br/>patience = 0<br/>save state_dict to CHECKPOINT_PATH"]
    save --> epoch
    better -- no --> inc["patience += 1"]
    inc --> stop{"patience >= PATIENCE (10)?"}
    stop -- no --> epoch
    stop -- yes --> early["print early stopping"] --> done
```
