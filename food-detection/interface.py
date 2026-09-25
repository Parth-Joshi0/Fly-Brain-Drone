import cv2
import torch
from PIL import Image
import config
from model import build_model
from dataset import val_transform

# You need class_names saved somewhere reference-able — see note below
class_names = ['freshripe', 'freshunripe', 'overripe', 'ripe', 'rotten', 'unripe']

model = build_model(len(class_names)).to(config.DEVICE)
model.load_state_dict(torch.load(config.CHECKPOINT_PATH, map_location=config.DEVICE))
model.eval()

cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    input_tensor = val_transform(img).unsqueeze(0).to(config.DEVICE)

    with torch.no_grad():
        output = model(input_tensor)
        probs = torch.softmax(output, dim=1)
        conf, pred = torch.max(probs, 1)
        label = f"{class_names[pred.item()]} ({conf.item():.2f})"

    cv2.putText(frame, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow('Banana Ripeness', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()