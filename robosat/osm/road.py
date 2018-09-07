import sys

import math
import osmium
import geojson
import shapely.geometry

from robosat.osm.core import is_polygon


class RoadHandler(osmium.SimpleHandler):
    """Extracts road polygon features (visible in satellite imagery) from the map.
    """

    #  highway=* to discard because these features are not vislible in satellite imagery
    @staticmethod
    def road_filter():
        return {
            "motorway",
            "trunk",
            "primary",
            "secondary",
            "tertiary",
            "unclassified",
            "residential",
            "service",
            "motorway_link",
            "trunk_link",
            "primary_link",
            "secondary_link",
            "tertiary_link"
        }

    def __init__(self):
        super().__init__()
        self.features = []

    def way(self, w):
        if "highway" not in w.tags:
            return

        if w.tags["highway"] not in self.road_filter():
            return

        left_hard_shoulder_width = 0.0
        lane_width = 3.0
        lanes = 1
        right_hard_shoulder_width = 0.0

        if w.tags["highway"] == "motorway":
            left_hard_shoulder_width = 0.75
            lane_width = 3.75
            lanes = 4
            right_hard_shoulder_width = 3.0
        elif w.tags["highway"] == "trunk":
            left_hard_shoulder_width = 0.75
            lane_width = 3.75
            lanes = 3
            right_hard_shoulder_width = 3.0
        elif w.tags["highway"] == "primary":
            left_hard_shoulder_width = 0.50
            lane_width = 3.75
            lanes = 2
            right_hard_shoulder_width = 1.50
        elif w.tags["highway"] == "secondary":
            left_hard_shoulder_width = 0.00
            lane_width = 3.50
            lanes = 1
            right_hard_shoulder_width = 0.75
        elif w.tags["highway"] == "tertiary":
            left_hard_shoulder_width = 0.00
            lane_width = 3.50
            lanes = 1
            right_hard_shoulder_width = 0.75
        elif w.tags["highway"] == "unclassified":
            left_hard_shoulder_width = 0.00
            lane_width = 3.50
            lanes = 1
            right_hard_shoulder_width = 0.00
        elif w.tags["highway"] == "residential":
            left_hard_shoulder_width = 0.00
            lane_width = 3.50
            lanes = 1
            right_hard_shoulder_width = 0.75
        elif w.tags["highway"] == "service":
            left_hard_shoulder_width = 0.00
            lane_width = 3.00
            lanes = 1
            right_hard_shoulder_width = 0.00
        elif w.tags["highway"] == "motorway_link":
            left_hard_shoulder_width = 0.75
            lane_width = 3.75
            lanes = 2
            right_hard_shoulder_width = 3.00
        elif w.tags["highway"] == "trunk_link":
            left_hard_shoulder_width = 0.75
            lane_width = 3.75
            lanes = 2
            right_hard_shoulder_width = 1.50
        elif w.tags["highway"] == "primary_link":
            left_hard_shoulder_width = 0.75
            lane_width = 3.50
            lanes = 1
            right_hard_shoulder_width = 0.75
        elif w.tags["highway"] == "secondary_link":
            left_hard_shoulder_width = 0.75
            lane_width = 3.50
            lanes = 1
            right_hard_shoulder_width = 0.75
        elif w.tags["highway"] == "tertiary_link":
            left_hard_shoulder_width = 0.75
            lane_width = 3.50
            lanes = 1
            right_hard_shoulder_width = 0.00

        if "oneway" not in w.tags:
            lanes = lanes * 2
        elif w.tags["oneway"] == "no":
            lanes = lanes * 2

        if "lanes" in w.tags:
            lanes = int(w.tags["lanes"])

        road_width = left_hard_shoulder_width + lane_width * lanes + right_hard_shoulder_width

        if "width" in w.tags:
            road_width = float(w.tags["width"])

        geometry = geojson.LineString([(n.lon, n.lat) for n in w.nodes])
        shape = shapely.geometry.shape(geometry)
        geometry_buffer = shape.buffer(math.degrees(road_width / 2.0 / 6371004.0))

        if shape.is_valid:
            feature = geojson.Feature(geometry=shapely.geometry.mapping(geometry_buffer))
            self.features.append(feature)
        else:
            print("Warning: invalid feature: https://www.openstreetmap.org/way/{}".format(w.id), file=sys.stderr)

    def save(self, out):
        collection = geojson.FeatureCollection(self.features)

        with open(out, "w") as fp:
            geojson.dump(collection, fp)
