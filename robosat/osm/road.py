import sys

import math
import osmium
import geojson
import shapely.geometry

from robosat.osm.core import FeatureStorage


class RoadHandler(osmium.SimpleHandler):
    """Extracts road polygon features (visible in satellite imagery) from the map.
    """

    highway_attributes = {
        "motorway": {
            "lanes": 4,
            "lane_width": 3.75,
            "left_hard_shoulder_width": 0.75,
            "right_hard_shoulder_width": 3.0,
        },
        "trunk": {"lanes": 3, "lane_width": 3.75, "left_hard_shoulder_width": 0.75, "right_hard_shoulder_width": 3.0},
        "primary": {
            "lanes": 2,
            "lane_width": 3.75,
            "left_hard_shoulder_width": 0.50,
            "right_hard_shoulder_width": 1.50,
        },
        "secondary": {
            "lanes": 1,
            "lane_width": 3.50,
            "left_hard_shoulder_width": 0.00,
            "right_hard_shoulder_width": 0.75,
        },
        "tertiary": {
            "lanes": 1,
            "lane_width": 3.50,
            "left_hard_shoulder_width": 0.00,
            "right_hard_shoulder_width": 0.75,
        },
        "unclassified": {
            "lanes": 1,
            "lane_width": 3.50,
            "left_hard_shoulder_width": 0.00,
            "right_hard_shoulder_width": 0.00,
        },
        "residential": {
            "lanes": 1,
            "lane_width": 3.50,
            "left_hard_shoulder_width": 0.00,
            "right_hard_shoulder_width": 0.75,
        },
        "service": {
            "lanes": 1,
            "lane_width": 3.00,
            "left_hard_shoulder_width": 0.00,
            "right_hard_shoulder_width": 0.00,
        },
        "motorway_link": {
            "lanes": 2,
            "lane_width": 3.75,
            "left_hard_shoulder_width": 0.75,
            "right_hard_shoulder_width": 3.00,
        },
        "trunk_link": {
            "lanes": 2,
            "lane_width": 3.75,
            "left_hard_shoulder_width": 0.50,
            "right_hard_shoulder_width": 1.50,
        },
        "primary_link": {
            "lanes": 1,
            "lane_width": 3.50,
            "left_hard_shoulder_width": 0.00,
            "right_hard_shoulder_width": 0.75,
        },
        "secondary_link": {
            "lanes": 1,
            "lane_width": 3.50,
            "left_hard_shoulder_width": 0.00,
            "right_hard_shoulder_width": 0.75,
        },
        "tertiary_link": {
            "lanes": 1,
            "lane_width": 3.50,
            "left_hard_shoulder_width": 0.00,
            "right_hard_shoulder_width": 0.00,
        },
    }

    road_filter = set(highway_attributes.keys())

    EARTH_MEAN_RADIUS = 6371004.0

    def __init__(self, out, batch):
        super().__init__()
        self.storage = FeatureStorage(out, batch)

    def way(self, w):
        if "highway" not in w.tags:
            return

        if w.tags["highway"] not in self.road_filter:
            return

        left_hard_shoulder_width = self.highway_attributes[w.tags["highway"]]["left_hard_shoulder_width"]
        lane_width = self.highway_attributes[w.tags["highway"]]["lane_width"]
        lanes = self.highway_attributes[w.tags["highway"]]["lanes"]
        right_hard_shoulder_width = self.highway_attributes[w.tags["highway"]]["right_hard_shoulder_width"]

        if "oneway" not in w.tags:
            lanes = lanes * 2
        elif w.tags["oneway"] == "no":
            lanes = lanes * 2

        if "lanes" in w.tags:
            try:
                # Roads have at least one lane; guard against data issues.
                lanes = max(int(w.tags["lanes"]), 1)

                # Todo: take into account related lane tags
                # https://wiki.openstreetmap.org/wiki/Tag:busway%3Dlane
                # https://wiki.openstreetmap.org/wiki/Tag:cycleway%3Dlane
                # https://wiki.openstreetmap.org/wiki/Key:parking:lane
            except ValueError:
                print("Warning: invalid feature: https://www.openstreetmap.org/way/{}".format(w.id), file=sys.stderr)

        road_width = left_hard_shoulder_width + lane_width * lanes + right_hard_shoulder_width

        if "width" in w.tags:
            try:
                # At least one meter wide, for road classes specified above
                road_width = max(float(w.tags["width"]), 1.0)

                # Todo: handle optional units such as "2 m"
                # https://wiki.openstreetmap.org/wiki/Key:width
            except ValueError:
                print("Warning: invalid feature: https://www.openstreetmap.org/way/{}".format(w.id), file=sys.stderr)

        geometry = geojson.LineString([(n.lon, n.lat) for n in w.nodes])
        shape = shapely.geometry.shape(geometry)
        geometry_buffer = shape.buffer(math.degrees(road_width / 2.0 / self.EARTH_MEAN_RADIUS))

        if shape.is_valid:
            feature = geojson.Feature(geometry=shapely.geometry.mapping(geometry_buffer))
            self.storage.add(feature)
        else:
            print("Warning: invalid feature: https://www.openstreetmap.org/way/{}".format(w.id), file=sys.stderr)

    def flush(self):
        self.storage.flush()
