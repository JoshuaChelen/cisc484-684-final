import os
import importlib.util
import matplotlib.pyplot as plt
import torchvision

#load dataloader
SCRIPT_PATH=os.path.join(os.path.dirname(__file__), "09_pytorch_dataloader.py")
spec=importlib.util.spec_from_file_location("pytorch_dataloader", SCRIPT_PATH)
module=importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

build_dataloaders=module.build_dataloaders

#build train loader
train_loader, _, _=build_dataloaders(batch_size=16, num_workers=0)
images, labels=next(iter(train_loader))

#make grid
grid=torchvision.utils.make_grid(images, nrow=4, padding=2)

plt.figure(figsize=(10, 10))
plt.imshow(grid.permute(1, 2, 0).cpu().numpy())
plt.title("16 Augmented Training Images")
plt.axis("off")
plt.tight_layout()

OUT_PATH=os.path.join(os.path.dirname(__file__), "grid.png")
plt.savefig(OUT_PATH, dpi=200, bbox_inches="tight")
plt.show

print(f"Saved augmented grid to: {OUT_PATH}")