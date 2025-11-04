import torch

from projects.adaptive_qat.utils.quantization import BitRange, LayerBitController


def test_bit_controller_init_and_clamp():
    controller = LayerBitController(
        layer_names=["layer1", "layer2"],
        init_bits={"layer1": 6.0, "layer2": 4.0},
        bit_range=BitRange(minimum=2.0, maximum=8.0),
    )
    bits = controller.bits()
    assert torch.isclose(bits["layer1"], torch.tensor(6.0))
    assert torch.isclose(bits["layer2"], torch.tensor(4.0))

    controller.parameter("layer2").data.fill_(10.0)
    controller.clamp_()
    bits = controller.bits()
    assert torch.isclose(bits["layer2"], torch.tensor(8.0))
