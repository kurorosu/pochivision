from .blur import BlurProcessor
from .grayscale import GrayscaleProcessor

PROCESSOR_REGISTRY = {
    "grayscale": GrayscaleProcessor,
    "blur": BlurProcessor,
}
