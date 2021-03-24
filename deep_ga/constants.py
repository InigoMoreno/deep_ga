def get_gps_references():
    """Get gps references

    Returns:
        tuple: containing easting northing and elevation references
    """
    cloudEastingOffset = 344178.0
    cloudNorthingOffset = 3127579.0
    cloudElevationOffset = 2442.0
    robotEastingOffset = 344043.8
    robotNorthingOffset = 3127408.3
    robotElevationOffset = 2543.0

    eastingReference = robotEastingOffset - cloudEastingOffset
    northingReference = robotNorthingOffset - cloudNorthingOffset
    elevationReference = robotElevationOffset - cloudElevationOffset

    return (eastingReference, northingReference, elevationReference)
