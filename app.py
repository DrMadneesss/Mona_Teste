from fastapi import FastAPI, File, UploadFile
from fastapi.responses import StreamingResponse
import torch
import io
from PIL import Image

app = FastAPI()

# Carregar o modelo
model = torch.hub.load('ultralytics/yolov5', 'custom', path='yolov5_model.pt', source='local')

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    image_bytes = await file.read()
    img = Image.open(io.BytesIO(image_bytes))

    results = model(img, size=640)

    # Salvar o resultado em buffer
    buffer = io.BytesIO()
    results.render()  # renderiza a máscara na imagem
    result_img = Image.fromarray(results.ims[0])
    result_img.save(buffer, format="JPEG")
    buffer.seek(0)

    return StreamingResponse(buffer, media_type="image/jpeg")
