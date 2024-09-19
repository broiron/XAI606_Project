from .llff import LLFFDataset
# from .llff import LLFFDataset_latent_rgb
from .llff import LLFFDataset_latent
from .llff import LLFFDataset_latent_mask
from .llff import Facedataset

from .blender import BlenderDataset
from .nsvf import NSVF
from .tankstemple import TanksTempleDataset
from .your_own_data import YourOwnDataset



dataset_dict = {'blender': BlenderDataset,
               'llff':LLFFDataset,
            #    'llff_latent_rgb':LLFFDataset_latent_rgb,
               'face':Facedataset,
               'llff_latent':LLFFDataset_latent,
               'llff_latent_mask':LLFFDataset_latent_mask,
               'tankstemple':TanksTempleDataset,
               'nsvf':NSVF,
               'own_data':YourOwnDataset}