from __future__ import annotations

from pathlib import Path
import struct
import sys


def _varint(value: int) -> bytes:
    out = bytearray()
    value = int(value)
    while value >= 0x80:
        out.append((value & 0x7F) | 0x80)
        value >>= 7
    out.append(value)
    return bytes(out)


def _key(field: int, wire: int) -> bytes:
    return _varint((field << 3) | wire)


def _int(field: int, value: int) -> bytes:
    return _key(field, 0) + _varint(value)


def _str(field: int, value: str) -> bytes:
    data = value.encode("utf-8")
    return _key(field, 2) + _varint(len(data)) + data


def _bytes(field: int, value: bytes) -> bytes:
    return _key(field, 2) + _varint(len(value)) + value


def _msg(field: int, value: bytes) -> bytes:
    return _key(field, 2) + _varint(len(value)) + value


def _tensor(name: str, dims: list[int], values: list[float]) -> bytes:
    raw = struct.pack("<" + "f" * len(values), *values)
    body = bytearray()
    for dim in dims:
        body += _int(1, dim)
    body += _int(2, 1)
    body += _str(8, name)
    body += _bytes(9, raw)
    return bytes(body)


def _dimension(value: int | str) -> bytes:
    if isinstance(value, int):
        return _int(1, value)
    return _str(2, value)


def _shape(dims: list[int | str]) -> bytes:
    body = bytearray()
    for dim in dims:
        body += _msg(1, _dimension(dim))
    return bytes(body)


def _tensor_type(elem_type: int, dims: list[int | str]) -> bytes:
    body = _int(1, elem_type) + _msg(2, _shape(dims))
    return _msg(1, body)


def _value_info(name: str, dims: list[int | str]) -> bytes:
    return _str(1, name) + _msg(2, _tensor_type(1, dims))


def _ints_attribute(name: str, values: list[int]) -> bytes:
    body = bytearray()
    body += _str(1, name)
    for value in values:
        body += _int(8, value)
    body += _int(20, 7)
    return bytes(body)


def _node(op_type: str, inputs: list[str], outputs: list[str], name: str, attrs: list[bytes] | None = None) -> bytes:
    body = bytearray()
    for value in inputs:
        body += _str(1, value)
    for value in outputs:
        body += _str(2, value)
    body += _str(3, name)
    body += _str(4, op_type)
    for attr in attrs or []:
        body += _msg(5, attr)
    return bytes(body)


def build_model() -> bytes:
    gray_weights = [0.299, 0.587, 0.114]
    sobel_x = [-1.0, 0.0, 1.0, -2.0, 0.0, 2.0, -1.0, 0.0, 1.0]
    sobel_y = [-1.0, -2.0, -1.0, 0.0, 0.0, 0.0, 1.0, 2.0, 1.0]
    pads = _ints_attribute("pads", [1, 1, 1, 1])
    graph = bytearray()
    for node in [
        _node("Conv", ["input", "gray_w"], ["luma"], "rgb_to_luma"),
        _node("Conv", ["luma", "sobel_x"], ["gx"], "sobel_x", [pads]),
        _node("Conv", ["luma", "sobel_y"], ["gy"], "sobel_y", [pads]),
        _node("Abs", ["gx"], ["abs_gx"], "abs_gx"),
        _node("Abs", ["gy"], ["abs_gy"], "abs_gy"),
        _node("Add", ["abs_gx", "abs_gy"], ["line"], "edge_strength"),
    ]:
        graph += _msg(1, node)
    graph += _str(2, "fd6_sobel_line_guide")
    graph += _msg(5, _tensor("gray_w", [1, 3, 1, 1], gray_weights))
    graph += _msg(5, _tensor("sobel_x", [1, 1, 3, 3], sobel_x))
    graph += _msg(5, _tensor("sobel_y", [1, 1, 3, 3], sobel_y))
    graph += _msg(11, _value_info("input", [1, 3, "height", "width"]))
    graph += _msg(12, _value_info("line", [1, 1, "height", "width"]))
    opset = _str(1, "") + _int(2, 13)
    model = bytearray()
    model += _int(1, 8)
    model += _str(2, "fd6")
    model += _str(3, "sobel-line-guide")
    model += _msg(7, bytes(graph))
    model += _msg(8, opset)
    return bytes(model)


def main(argv: list[str]) -> int:
    output = Path(argv[1]) if len(argv) > 1 else Path("models") / "line_guide.onnx"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(build_model())
    print(f"Wrote {output} ({output.stat().st_size} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
