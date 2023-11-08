import os
import torch
from PIL import Image
from cog import BasePredictor, Input, Path
from lavis.models import load_model_and_preprocess
from download_utils import download_weights


WEIGHTS_CACHE_DIR = "/src/weights/"
WEIGHTS_URL = "https://weights.replicate.delivery/default/blip-2/weights.tar"

if not os.path.exists(WEIGHTS_CACHE_DIR):
    download_weights(url=WEIGHTS_URL,dest=WEIGHTS_CACHE_DIR)
os.environ["TORCH_HOME"] = os.environ["HF_HOME"] = os.environ["HUGGINGFACE_HUB_CACHE"] = WEIGHTS_CACHE_DIR


class Predictor(BasePredictor):
    def setup(self):
        """Load the model into memory to make running multiple predictions efficient"""
        self.device = torch.device("cuda") if torch.cuda.is_available() else "cpu"
        self.model, self.vis_processors, _ = load_model_and_preprocess(
            name="blip2_t5",
            model_type="pretrain_flant5xl",
            is_eval=True,
            device=self.device,
        )
        self.model.to(self.device)

    def predict(
        self,
        image: Path = Input(description="Input image to query or caption"),
        caption: bool = Input(
            description="Select if you want to generate image captions instead of asking questions",
            default=False,
        ),
        question: str = Input(
            description="Question to ask about this image. Leave blank for captioning",
            default="What is this a picture of?",
        ),
        context: str = Input(
            description="Optional - previous questions and answers to be used as context for answering current question",
            default=None,
        ),
        use_nucleus_sampling: bool = Input(
            description="Toggles the model using nucleus sampling to generate responses",
            default=False,
        ),
        temperature: float = Input(
            description="Temperature for use with nucleus sampling",
            ge=0.5,
            le=1.0,
            default=1.0,
        ),
    ) -> str:
        """Run a single prediction on the model"""
        raw_image = Image.open(image).convert("RGB")
        image = self.vis_processors["eval"](raw_image).unsqueeze(0).to(self.device)

        if caption or question == "":
            print("captioning")
            response = self.model.generate({"image": image})
            return response[0]

        q = f"Question: {question} Answer:"
        if context:
            q = " ".join([context, q])
        print(f"input for question answering: {q}")
        if use_nucleus_sampling:
            response = self.model.generate(
                {"image": image, "prompt": q},
                use_nucleus_sampling=use_nucleus_sampling,
                temperature=temperature,
            )
        else:
            response = self.model.generate({"image": image, "prompt": q})

        return response[0]
