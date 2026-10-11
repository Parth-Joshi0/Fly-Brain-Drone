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

## 2. Package / dependency diagram

How the files depend on each other and on the rest of the repo. Solid
arrows are Python imports; dashed arrows are files read or written at
runtime.

```mermaid
flowchart LR
    subgraph Callers["Rest of the repo"]
        main["main.py"]
        fly["Drone/fly_tello.py"]
        brain["NeuralPathways/ScaredEating/<br/>scared_eating_brain.py"]
        simtest["Simulator/Tests/<br/>test_banana_sim.py"]
        seek["Controllers/<br/>banana_seek_controller.py"]
    end

    subgraph BananaModel
        subgraph Inference
            bd["banana_detector.py<br/>BananaDetector, Detection"]
        end
        subgraph Training
            tr["train.py"]
            ds["dataset.py"]
            md["model.py"]
            cf["config.py"]
        end
        ckpt[("checkpoints/<br/>best_banana_model.pth")]
        cn[("checkpoints/<br/>class_names.json<br/>(missing, optional)")]
        yw[("yolov8n.pt")]
        data[("dataset/train, valid<br/>image folders")]
    end

    main --> bd
    fly --> bd
    simtest --> bd
    brain -- "Detection only" --> bd
    main -. "passes BananaDetector into" .-> seek

    tr --> ds
    tr --> md
    tr --> cf
    ds --> cf

    ds -. reads .-> data
    tr -. writes .-> ckpt
    bd -. reads .-> ckpt
    bd -. "reads if present" .-> cn
    bd -. "reads (auto-downloads)" .-> yw
```

## 3. Sequence diagram: `BananaDetector.detect(frame)`

What happens on each camera frame (for example from `Drone/fly_tello.py`).

```mermaid
sequenceDiagram
    autonumber
    actor Caller as Caller<br/>(fly_tello / controller)
    participant BD as BananaDetector
    participant Y as YOLO (yolov8n)
    participant CV as OpenCV / PIL
    participant T as classify_transform
    participant C as MobileNetV2 classifier

    Caller->>BD: detect(frame BGR)
    BD->>Y: detector(frame, conf=0.15, imgsz, classes=[46])
    Y-->>BD: results[0].boxes

    loop for each box
        BD->>BD: pad box by padding_ratio (8%), clamp to frame
        BD->>BD: crop = frame[cy1:cy2, cx1:cx2]
        alt crop is empty
            BD->>BD: skip box
        else crop has pixels
            BD->>CV: cvtColor BGR to RGB, Image.fromarray
            CV-->>BD: PIL image
            BD->>T: resize 224x224, ToTensor, Normalize
            T-->>BD: input tensor (1x3x224x224)
            BD->>C: forward (no_grad)
            C-->>BD: logits (1x6)
            BD->>BD: softmax, max gives cls_conf, pred_idx
            alt cls_conf < classifier_conf_threshold
                BD->>BD: skip box
            else
                BD->>BD: label = class_names[pred_idx]
                BD->>BD: append Detection(box, det_conf, label, cls_conf)
            end
        end
    end

    BD-->>Caller: list of Detection
```

`detect_and_annotate(frame)` calls `detect(frame)` and then
`annotate(frame, detections)`, which draws a green box and a
`label cls% (det det%)` caption on the frame in place.

## 4. Sequence diagram: `BananaDetector` construction

```mermaid
sequenceDiagram
    autonumber
    actor Caller
    participant BD as BananaDetector
    participant FS as checkpoints/
    participant TV as torchvision
    participant U as ultralytics

    Caller->>BD: BananaDetector(...)
    BD->>BD: pick device (CUDA if available, else CPU)
    BD->>FS: open class_names.json
    alt file exists
        FS-->>BD: class names
    else FileNotFoundError
        BD->>BD: print warning, use default_class_names (6 classes)
    end
    BD->>TV: mobilenet_v2(weights=None)
    BD->>BD: replace head with Dropout(0.2), Linear(1280, 6)
    BD->>FS: torch.load(best_banana_model.pth)
    FS-->>BD: state_dict
    BD->>BD: load_state_dict, eval()
    BD->>BD: build classify_transform
    BD->>U: YOLO(yolov8n.pt)
    U-->>BD: detector (downloads weights on first run)
    BD-->>Caller: ready detector
```

## 5. Activity diagram: training (`python train.py`)

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
