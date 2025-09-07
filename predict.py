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
        self.model.netG.to(opt.device)
        self.model.netG.eval()


    def predict(self, image_list)
        generated_images = []

        for i, img in enumerate(tqdm(image_list)):
            if isinstance(img, np.ndarray):
                img_tensor = torch.from_numpy(
                    (img.astype(np.float32) / 255.0 - 0.5) / 0.5
                )
                if img_tensor.dim() == 2:
                    img_tensor = img_tensor.unsqueeze(-1).repeat(1, 1, 3)
                img_tensor = img_tensor.permute(2, 0, 1)
            else:
                img_tensor = torch.from_numpy(
                    (np.array(img).astype(np.float32) / 255.0 - 0.5) / 0.5
                ).permute(2, 0, 1)
            
            img_tensor = img_tensor.unsqueeze(0).to(self.opt.device)

            with torch.no_grad():
                fake_B = self.model.netG(img_tensor)

            fake_image = fake_B.detach().cpu().numpy()
            fake_image = (np.transpose(fake_image, (0, 2, 3, 1)) + 1) / 2.0 * 255.0
            
            if fake_image.shape[3] == 6:
                fake_image = fake_image[0, :, :, 3:]
            else:
                fake_image = fake_image[0]
            
            pil_image = Image.fromarray(fake_image.astype(np.uint8))
            generated_images.append(pil_image)

        return generated_images