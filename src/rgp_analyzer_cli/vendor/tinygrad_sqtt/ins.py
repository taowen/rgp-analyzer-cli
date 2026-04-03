from __future__ import annotations

from .dsl import Inst


class _SEndPgm(Inst):
    op_name = "S_ENDPGM"

    def __repr__(self) -> str:
        return "s_endpgm"


def s_endpgm() -> Inst:
    return _SEndPgm()
