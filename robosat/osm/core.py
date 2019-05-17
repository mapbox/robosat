import os
import uuid

import geojson


class FeatureStorage:
    """Stores features on disk and handles batching.

       Note: you have to call flush at the end to flush the last partial batch.
    """

    def __init__(self, out, batch):
        assert batch > 0

        self.out = out
        self.batch = batch

        self.features = []

    def add(self, feature):
        if len(self.features) >= self.batch:
            self.flush()

        self.features.append(feature)

    def flush(self):
        if not self.features:
            return

        collection = geojson.FeatureCollection(self.features)

        base, ext = os.path.splitext(self.out)
        suffix = uuid.uuid4().hex

        out = "{}-{}{}".format(base, suffix, ext)

        with open(out, "w") as fp:
            geojson.dump(collection, fp)

        self.features.clear()


def is_polygon(way):
    """Checks if the way is a polygon.

    Args
      way: the osmium.osm.Way to check.

    Returns:
      True if the way is a polygon, False otherwise.

    Note: The geometry shape can still be invalid (e.g. self-intersecting).
    """

    if not way.is_closed():
        return False

    if len(way.nodes) < 4:
        return False

    return True
