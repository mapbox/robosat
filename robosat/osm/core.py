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
