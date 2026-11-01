import numpy as np
import SimpleITK as sitk

from segagent.data.preprocessing import preprocess_case, same_geometry


def test_preprocess_preserves_label_values_and_matches_geometry() -> None:
    image_array = np.asarray(
        [[[-1200.0, -1000.0], [400.0, 800.0]], [[-300.0, 0.0], [100.0, 200.0]]],
        dtype=np.float32,
    )
    label_array = np.zeros_like(image_array, dtype=np.uint8)
    label_array[1, 1, 1] = 1
    image = sitk.GetImageFromArray(image_array)
    label = sitk.GetImageFromArray(label_array)
    image.SetSpacing((2.0, 2.0, 2.0))
    label.CopyInformation(image)

    output_image, output_label, _ = preprocess_case(
        image=image,
        label=label,
        instance=None,
        orientation="RAS",
        target_spacing=(1.0, 1.0, 1.0),
        image_interpolation="linear",
        label_interpolation="nearest",
        hu_window=(-1000.0, 400.0),
        output_range=(0.0, 1.0),
        image_padding_hu=-1000.0,
    )

    values = sitk.GetArrayFromImage(output_image)
    labels = sitk.GetArrayFromImage(output_label)
    assert output_image.GetSpacing() == (1.0, 1.0, 1.0)
    assert same_geometry(output_image, output_label)
    assert float(values.min()) >= 0.0
    assert float(values.max()) <= 1.0
    assert set(np.unique(labels)).issubset({0, 1})
