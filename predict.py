# Prediction interface for Cog ⚙️
# https://github.com/replicate/cog/blob/main/docs/python.md
import os
cache = '/src/weights/'
os.environ['TORCH_HOME'] = '/src/weights/'
os.environ['HF_HOME'] = '/src/weights/'
os.environ['HUGGINGFACE_HUB_CACHE'] = '/src/weights/'
if not os.path.exists(cache):
    os.makedirs(cache)

import torch
from PIL import Image
from lavis.models import load_model_and_preprocess
from cog import BasePredictor, Input, Path


class Predictor(BasePredictor):
    def setup(self):
        """Load the model into memory to make running multiple predictions efficient"""
        self.device = torch.device("cuda") if torch.cuda.is_available() else "cpu"
        self.model, self.vis_processors, _ = load_model_and_preprocess(
            name="blip2_t5", model_type="pretrain_flant5xl", is_eval=True, device=self.device,
        )
        self.model.to(self.device)


    def predict(
        self,
        image: Path = Input(description="Input image to query"),
        question: str = Input(
            description="Question to ask about this image", 
            default="What is this a picture of?"),
        use_nucleus_sampling: bool = Input(
            default="Whether to use nucleus sampling to respond to questions",
            default=False
        ),
        context: str = Input(
            description="Optional - previous questions and answers to be used as context for answering current question",
            default=None
        )
    ) -> str:
        """Run a single prediction on the model"""
        raw_image = Image.open(image).convert('RGB')
        image = self.vis_processors["eval"](raw_image).unsqueeze(0).to(self.device)
        q =  f"Question: {question} Answer:"
        if context:
            q = " ".join([context, q])
        print(f"prompt for model: {q}")
        response = self.model.generate({"image": image, "prompt": q}, use_nucleus_sampling=use_nucleus_sampling)
        
        return response[0]
