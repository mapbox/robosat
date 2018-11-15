import re
import os
from robosat.tiles import pixel_to_location
import mercantile
import json
import matplotlib

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


def leaflet(out, base_url, tiles, grid, ext):

    grid = json.dumps([mercantile.feature(tile) for tile in grid]) if grid else "''"

    leaflet = open("./robosat/tools/templates/leaflet.html", "r").read()
    leaflet = re.sub("{{base_url}}", base_url, leaflet)
    leaflet = re.sub("{{ext}}", ext, leaflet)
    leaflet = re.sub("{{grid}}", grid, leaflet)

    # Could surely be improve, but for now, took the first tile to center on
    tile = list(tiles)[0]
    x, y, z = map(int, [tile.x, tile.y, tile.z])
    leaflet = re.sub("{{zoom}}", str(z), leaflet)
    leaflet = re.sub("{{center}}", str(list(pixel_to_location(tile, 0.5, 0.5))[::-1]), leaflet)

    f = open(os.path.join(out, "index.html"), "w", encoding="utf-8")
    f.write(leaflet)
    f.close()
