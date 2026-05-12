from __future__ import annotations

import json
import os
import sys
import traceback
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    __package__ = "hallway_avatar_gen.core"

from .qremeshify_runtime.lib import Quadwild, QWException


def _run(payload: dict) -> str:
    qw = Quadwild(payload["mesh_path"])
    if not payload["useCache"]:
        qw.remeshAndField(
            remesh=payload["enableRemesh"],
            enableSharp=payload["enableSharp"],
            sharpAngle=payload["sharpAngle"],
        )
        qw.trace()

    qw.quadrangulate(
        payload["enableSmoothing"],
        payload["scaleFact"],
        payload["fixedChartClusters"],
        payload["alpha"],
        payload["ilpMethod"],
        payload["timeLimit"],
        payload["gapLimit"],
        payload["minimumGap"],
        payload["isometry"],
        payload["regularityQuadrilaterals"],
        payload["regularityNonQuadrilaterals"],
        payload["regularityNonQuadrilateralsWeight"],
        payload["alignSingularities"],
        payload["alignSingularitiesWeight"],
        payload["repeatLosingConstraintsIterations"],
        payload["repeatLosingConstraintsQuads"],
        payload["repeatLosingConstraintsNonQuads"],
        payload["repeatLosingConstraintsAlign"],
        payload["hardParityConstraint"],
        payload["flowConfig"],
        payload["satsumaConfig"],
        payload["callbackTimeLimit"],
        payload["callbackGapLimit"],
    )

    final_mesh_path = qw.output_smoothed_path if payload["enableSmoothing"] else qw.output_path
    if not os.path.isfile(final_mesh_path):
        raise QWException(f"missing output mesh: {final_mesh_path}")
    return final_mesh_path


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if len(argv) != 1:
        print("usage: python -m hallway_avatar_gen.core.qremeshify_worker <payload.json>", file=sys.stderr)
        return 2
    payload_path = Path(argv[0])
    try:
        payload = json.loads(payload_path.read_text(encoding="utf-8"))
        final_mesh_path = _run(payload)
        print(json.dumps({"ok": True, "output": final_mesh_path}))
        return 0
    except BaseException as exc:
        print(f"QRemeshify worker error: {exc}", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
