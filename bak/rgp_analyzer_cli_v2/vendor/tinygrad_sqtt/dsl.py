from __future__ import annotations


class _Bits:
    def __getitem__(self, key):
        if isinstance(key, slice):
            return BitField(key.start, key.stop)
        return BitField(key, key)


bits = _Bits()


class BitField:
    name: str | None

    def __init__(self, hi: int, lo: int, default=0):
        self.hi = hi
        self.lo = lo
        self.default = default
        self.name = None
        self.mask = (1 << (hi - lo + 1)) - 1

    def __set_name__(self, owner, name: str):
        self.name = name

    def __eq__(self, other):
        if isinstance(other, int):
            return FixedBitField(self.hi, self.lo, other)
        raise TypeError(f"BitField.__eq__ expects int, got {type(other).__name__}")

    def encode(self, value) -> int:
        return int(value)

    def decode(self, value):
        return value

    def set(self, raw: int, value) -> int:
        encoded = self.encode(self.default if value is None else value)
        if encoded < 0:
            encoded &= self.mask
        if encoded < 0 or encoded > self.mask:
            raise RuntimeError(f"field '{self.name}': value {encoded} does not fit in {self.hi - self.lo + 1} bits")
        return (raw & ~(self.mask << self.lo)) | (encoded << self.lo)

    def __get__(self, obj, objtype=None):
        if obj is None:
            return self
        return self.decode((obj._raw >> self.lo) & self.mask)

    def __set__(self, obj, value):
        obj._raw = self.set(obj._raw, value)

    def enum(self, enum_cls):
        return EnumBitField(self.hi, self.lo, enum_cls)


class FixedBitField(BitField):
    def set(self, raw: int, value=None) -> int:
        if value is not None:
            raise AssertionError(f"FixedBitField does not accept values, got {value}")
        return super().set(raw, self.default)


class EnumBitField(BitField):
    def __init__(self, hi: int, lo: int, enum_cls):
        super().__init__(hi, lo)
        self._enum = enum_cls

    def encode(self, value) -> int:
        return value.value

    def decode(self, value):
        return self._enum(value)


class Inst:
    def size(self) -> int:
        return 4
