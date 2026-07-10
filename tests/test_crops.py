"""Crop-math characterization: guards the GLM smart_resize numerics and bbox handling."""

from PIL import Image

from bumblebee.crops import crop_region, smart_resize
from bumblebee.layout.labels import normalize_bbox
from bumblebee.models import Page
from tests.conftest import make_region


def test_smart_resize_fixed_points():
    kwargs = {"t": 2, "t_factor": 2, "h_factor": 28, "w_factor": 28}
    assert smart_resize(h=100, w=100, **kwargs) == (112, 112)  # below min_pixels -> upscaled
    assert smart_resize(h=1400, w=1000, **kwargs) == (812, 588)  # above max_pixels -> downscaled
    assert smart_resize(h=3000, w=2400, **kwargs) == (784, 616)
    assert smart_resize(h=30, w=900, **kwargs) == (28, 896)  # thin strip clamps to one factor
    assert smart_resize(h=799, w=601, **kwargs) == (812, 588)  # rounding to factor multiples


def test_normalize_bbox_scaling_and_clamping():
    assert normalize_bbox((10.0, 20.0, 500.0, 700.0), 1000, 800) == (10, 25, 500, 875)
    assert normalize_bbox((-5.0, 0.0, 1200.0, 900.0), 1000, 800) == (0, 0, 1000, 1000)
    assert normalize_bbox((0.4, 0.6, 999.5, 799.4), 1000, 800) == (0, 1, 1000, 999)


def _page(width=200, height=100):
    return Page(page_index=0, width=width, height=height, image=Image.new("RGB", (width, height)))


def _region(bbox):
    return make_region(0, 0, "text", "text", bbox)


def test_crop_region_maps_normalized_bbox_to_pixels():
    cropped = crop_region(_page(), _region([100, 200, 600, 800]))
    assert cropped.size == (100, 60)  # (600-100)/1000*200, (800-200)/1000*100


def test_crop_region_degenerate_box_yields_min_one_pixel():
    cropped = crop_region(_page(), _region([500, 500, 500, 500]))
    assert cropped.size == (1, 1)


def test_crop_region_full_page():
    cropped = crop_region(_page(), _region([0, 0, 1000, 1000]))
    assert cropped.size == (200, 100)
