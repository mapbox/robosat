import re
import os
import sys
import json
import matplotlib
from pathlib import Path
from mercantile import feature
from robosat.tiles import pixel_to_location

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def plot(out, history):
    plt.figure()

    n = max(map(len, history.values()))
    plt.xticks(range(n), [v + 1 for v in range(n)])

    plt.grid()

    for values in history.values():
        plt.plot(values)

    plt.xlabel("epoch")
    plt.legend(list(history))

    plt.savefig(out, format="png")
    plt.close()


def web_ui(out, base_url, coverage_tiles, selected_tiles, ext, template):

    try:
        if os.path.isfile(template):
            web_ui = open(template, "r").read()
        else:
            web_ui = open(os.path.join(Path(__file__).parent, "tools", "templates", template), "r").read()
    except:
        sys.exit("Unable to open Web UI template {}".format(template))

    web_ui = re.sub("{{base_url}}", base_url, web_ui)
    web_ui = re.sub("{{ext}}", ext, web_ui)
    web_ui = re.sub("{{tiles}}", "tiles.json" if selected_tiles else "''", web_ui)

    if coverage_tiles:
        # Could surely be improve, but for now, took the first tile to center on
        tile = list(coverage_tiles)[0]
        x, y, z = map(int, [tile.x, tile.y, tile.z])
        web_ui = re.sub("{{zoom}}", str(z), web_ui)
        web_ui = re.sub("{{center}}", str(list(pixel_to_location(tile, 0.5, 0.5))[::-1]), web_ui)

    with open(os.path.join(out, "index.html"), "w", encoding="utf-8") as fp:
        fp.write(web_ui)

    if selected_tiles:
        with open(os.path.join(out, "tiles.json"), "w", encoding="utf-8") as fp:
            fp.write('{"type":"FeatureCollection","features":[')
            first = True
            for tile in selected_tiles:
                prop = '"properties":{{"x":{},"y":{},"z":{}}}'.format(int(tile.x), int(tile.y), int(tile.z))
                geom = '"geometry":{}'.format(json.dumps(feature(tile, precision=6)["geometry"]))
                fp.write('{}{{"type":"Feature",{},{}}}'.format("," if not first else "", geom, prop))
                first = False
            fp.write("]}")

