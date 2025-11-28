# pipeline.py
import os
import json
from io import BytesIO

import torch
from PIL import Image
from torchvision import models, transforms as T
from pydantic import ValidationError
from openai import OpenAI
import requests

from ClassificationLitModuleDenseNet import ClassificationLitModuleDenseNet
from PlantCareCard import PlantCareCard
from rag_utils import PlantCareRAG


class PlantInferencePipeline:
    def __init__(self, model_path: str, api_key: str):
        self.client = OpenAI(api_key=api_key or os.getenv("OPENAI_API_KEY"))
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.labels = [
            "aloevera","banana","bilimbi","cantaloupe","cassava","coconut","corn","cucumber","curcuma",
            "eggplant","galangal","ginger","guava","kale","longbeans","mango","melon","orange","paddy",
            "papaya","peper chili","pineapple","pomelo","shallot","soybeans","spinach","sweet potatoes",
            "tobacco","waterapple","watermelon"
        ]
        num_classes = len(self.labels)

        try:
            backbone = models.densenet169(weights=models.DenseNet169_Weights.DEFAULT)
        except Exception:
            backbone = models.densenet169(pretrained=True)

        for param in backbone.features.parameters():
            param.requires_grad = False

        in_features = backbone.classifier.in_features
        backbone.classifier = torch.nn.Linear(in_features, num_classes)

        self.model = ClassificationLitModuleDenseNet(backbone, num_classes=num_classes)

        checkpoint = torch.load(model_path, map_location='cpu')
        loaded = False
        if isinstance(checkpoint, dict):
            if 'state_dict' in checkpoint:
                try:
                    self.model.load_state_dict(checkpoint['state_dict'], strict=False)
                    loaded = True
                except Exception:
                    try:
                        sd = checkpoint['state_dict']
                        new_sd = {k.replace('model.', '').replace('_inner.', ''): v for k, v in sd.items()}
                        self.model.load_state_dict(new_sd, strict=False)
                        loaded = True
                    except Exception:
                        loaded = False
            if not loaded:
                try:
                    self.model.load_state_dict(checkpoint, strict=False)
                    loaded = True
                except Exception:
                    loaded = False
        else:
            try:
                self.model.load_state_dict(checkpoint, strict=False)
                loaded = True
            except Exception:
                loaded = False

        if not loaded:
            print(f"Warning: could not load checkpoint from {model_path} into ClassificationLitModuleDenseNet; continuing without pretrained weights for head.")

        self.model.eval()

        try:
            weights = getattr(models, 'DenseNet169_Weights', None)
            if weights is not None:
                try:
                    self.transform = models.DenseNet169_Weights.DEFAULT.transforms()
                except Exception:
                    self.transform = T.Compose([
                        T.Resize(256),
                        T.CenterCrop(224),
                        T.ToTensor(),
                        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                    ])
            else:
                self.transform = T.Compose([
                    T.Resize(256),
                    T.CenterCrop(224),
                    T.ToTensor(),
                    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                ])
        except Exception:
            self.transform = T.Compose([
                T.Resize(256),
                T.CenterCrop(224),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])

        self.plant_prompt_template = """
You are a world-class botanist and horticulturist.
You generate expert Plant Care Cards for any plant species.

Use the following retrieved context (which may include care guides) to inform your answer.
If information conflicts, prefer precise, actionable guidance and note uncertainties briefly.

Retrieved context:
{retrieved_context}

Return your answer strictly as a JSON object with the EXACT schema:

{{
  "common_name": "string",
  "latin_name": "string",
  "care_difficulty": "string",
  "watering_frequency": "string",
  "sunlight": "string",
  "soil_type": "string",
  "fertilizer": "string",
  "outdoors": true,
  "notes": "string"
}}

All fields are REQUIRED.

Plant name: "{plant_name}"
        """

        # Initialize RAG
        try:
            self.rag = PlantCareRAG(api_key=self.api_key)
        except Exception as e:
            print(f"Warning: RAG initialization failed: {e}")
            self.rag = None

    def load_image(self, image_input):
        if isinstance(image_input, str):  # URL
            response = requests.get(image_input)
            response.raise_for_status()
            return Image.open(BytesIO(response.content)).convert("RGB")

        if hasattr(image_input, "read"):
            return Image.open(image_input).convert("RGB")

        return Image.open(BytesIO(image_input)).convert("RGB")

    def classify_image(self, img: Image.Image) -> str:
        img_tensor = self.transform(img).unsqueeze(0)
        with torch.no_grad():
            try:
                device = next(self.model.parameters()).device
                img_tensor = img_tensor.to(device)
            except Exception:
                pass
            logits = self.model(img_tensor)
            if isinstance(logits, dict) and 'preds' in logits:
                pred_idx = int(logits['preds'].cpu().numpy().item()) if hasattr(logits['preds'], 'cpu') else int(logits['preds'])
            else:
                pred_idx = torch.argmax(logits, dim=1).item()
        return self.labels[pred_idx]

    def generate_plant_card(self, plant_name: str, retrieved_context: str = "") -> PlantCareCard:
        prompt = self.plant_prompt_template.format(
            plant_name=plant_name,
            retrieved_context=retrieved_context or "No additional context available."
        )

        response = self.client.chat.completions.create(
            model="gpt-4.1",
            messages=[
                {"role": "system", "content": "You generate structured plant care data."},
                {"role": "user", "content": prompt}
            ],
            response_format={"type": "json_object"}
        )

        raw_json = response.choices[0].message.content
        try:
            card = PlantCareCard(**json.loads(raw_json))
        except ValidationError as e:
            print("Error validating PlantCareCard:", e)
            card = None

        return card

    def inference(self, image_input):
        img = self.load_image(image_input)

        plant_label = self.classify_image(img)

        # Retrieve context from LanceDB
        retrieved_context = ""
        if self.rag is not None:
            try:
                records = self.rag.query_by_label(plant_label, k=5)
                retrieved_context = self.rag.format_context(records)
            except Exception as e:
                print(f"Warning: RAG query failed for label '{plant_label}': {e}")

        card = self.generate_plant_card(plant_label, retrieved_context=retrieved_context)

        return {
            "predicted_plant": plant_label,
            "plant_care_card": card.model_dump() if card else None,
            "rag_context_available": bool(retrieved_context)
        }
