from thingsvision.utils.storing import save_features
from thingsvision import Extractor
from thingsvision.utils.data import ImageDataset, DataLoader
import torch
import torchvision


behavior = False

if behavior:
    base_path = "../visualization/reference_images"
    out_path = "/home/florian/THINGS/vgg_bn_features_behavior"

else:
    n_samples_per_class = 12
    base_path = '/home/florian/THINGS/{}'
    root = base_path.format(f'image_data/images{n_samples_per_class}')
    out_path = base_path.format(f'vgg_bn_features_{n_samples_per_class}')


batch_size = 1
backend = 'pt'
model_name = 'vgg16_bn'
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)
module_name = 'classifier.3'


# model = Model(model_name, pretrained=True, model_path=None, device=device, backend=backend)
model = Extractor(model_name, pretrained=True, model_path=None, device=device, source='torchvision')

# transforms = 
trans = torchvision.transforms.Compose([
    torchvision.transforms.Resize(size=256),
    torchvision.transforms.CenterCrop(size=(224, 224)),
    torchvision.transforms.ToTensor()])

dataset = ImageDataset(
        root=root,
        out_path=out_path,
        backend=model.backend,
        transforms=model.get_transformations(),
        class_names=None,
        file_names=None,
)

batches = DataLoader(dataset=dataset, batch_size=batch_size, backend=model.backend)
features = model.extract_features(
				batches=batches,
				module_name=module_name,
				flatten_acts=False,
				clip=False,
)
save_features(features, out_path, file_format='npy')
