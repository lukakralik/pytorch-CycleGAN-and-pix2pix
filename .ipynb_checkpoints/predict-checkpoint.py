import os
import torch
from PIL import Image
import numpy as np
from options.test_options import TestOptions
from models import create_model
from cuda_selector import auto_cuda
import os
from pathlib import Path
from tqdm.auto import tqdm

class Pix2PixPredictor:
    def __init__(self, checkpoint_dir, model_name, direction="AtoB", device=auto_cuda()):
        opt = TestOptions().parse()
        opt.isTrain = False
        opt.checkpoints_dir = checkpoint_dir
        opt.name = model_name
        opt.model = "pix2pix"
        opt.norm = 'batch'
        opt.dataset_mode = 'aligned' 
        opt.direction = direction
        opt.device = torch.device(device if torch.cuda.is_available() else "cpu")
        opt.num_threads = 0
        opt.batch_size = 1
        opt.serial_batches = True
        opt.no_flip = True
        opt.eval = True
        opt.preprocess = "none"
        opt.load_size = 256
        opt.crop_size = 256
        opt.dataset_mode = "single"
        opt.input_nc = 3
        opt.output_nc = 3
        opt.netG = "unet_256"

        # epoch to load
        opt.epoch = "latest"

        self.opt = opt
        self.model = create_model(opt)

        self.model.load_networks(opt.epoch)
        self.model.netG.eval()

    def predict(self, image_list, output_dir=None):
        generated_images = []
        transform = torch.nn.Identity()  # you may need the repo’s transform (ToTensor+Norm)

        for i, img in enumerate(tqdm(image_list)): # TODO: accept loaded images directly
            if isinstance(img, str):
                img = Image.open(img).convert("RGB")
            # img = img.resize((256, 256), Image.BICUBIC)
            img_tensor = torch.from_numpy(
                (np.array(img).astype(np.float32) / 255.0 - 0.5) / 0.5
            ).permute(2, 0, 1).unsqueeze(0)

            with torch.no_grad():
                fake_B = self.model.netG(img_tensor)

            fake_image = fake_B.detach().cpu().numpy()
            fake_image = (np.transpose(fake_image, (0, 2, 3, 1)) + 1) / 2.0 * 255.0
            fake_image = fake_image[0][:, :fake_image[0].shape[1] // 2]
            
            pil_image = Image.fromarray(fake_image.astype(np.uint8))
            generated_images.append(pil_image)

            if output_dir:
                os.makedirs(output_dir, exist_ok=True)
                base_name = f"image_{i}.png"
                pil_image.save(os.path.join(output_dir, base_name))

        return generated_images


if __name__ == "__main__":
    run_dir = "src/shape_generation"
    predictor = Pix2PixPredictor(
        checkpoint_dir=f"{run_dir}/pix2pix/checkpoints",
        model_name="oxides",
        direction="AtoB"
    )

    input_dir = Path(f"{run_dir}/artificial")
    output_dir = Path(f"{run_dir}/outputs")
    print(output_dir)

    image_paths = sorted(input_dir.glob("*.png"))

    outputs = predictor.predict([str(p) for p in image_paths], output_dir=output_dir)