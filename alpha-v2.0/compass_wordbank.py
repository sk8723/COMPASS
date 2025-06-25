# ------------------------------ Category Definitions ------------------------------

categories_dict = {
    'Water Surface': ['Water Surface', 'River Surface', 'Lake Surface', 
                      'Reflection Water', 'Lagoon Surface', 'Estuary Surface', 
                      'Harbor Surface', 'Bay Surface'],
    'Tree': ['Tree', 'Palm Tree', 'Oak Tree', 'Cypress'],
    'Boat': ['Boat', 'Ship', 'Dinghy'],
    'Boat parts': ['Boat parts', 'Motor', 'Mast', 'Hull', 'Sail'],
    'Rock': ['Rock', 'Boulder', 'Stones', 'Pebbles', 'Gravel'],
    'Sand': ['Sand', 'Beach'],
    'Grass': ['Grass', 'Field'],
    'Dock': ['Dock', 'Pier', 'Jettie', 'Wharf'],
    'Street Light Pole': ['Street Light Pole', 'Piling'],
    'Tire': ['Tire', 'Wheel'],
    'Street': ['Street', 'Sidewalk', 'Trail'],
    'Boat Ramp': ['Boat Ramp', 'Railings'],
    'Shoreline Barrier': ['Shoreline Barrier', 'Sea Wall', 'Wall', 'Barrier', 'Fence', 'Retaining Wall', 'Mossy Wall'],
    'Bridge': ['Bridge', 'Bridge Support'],
    'Buoy': ['Buoy', 'Water Buoy'],
    'Building': ['Building', 'House', 'Shed', 'Cabin'],
    'Lighthouse': ['Lighthouse'],
    'Person': ['Person'],
    'Car': ['Car', 'Truck', 'Van', 'Motorcycle'],
    'Sky': ['Sky', 'Clouds', 'Clear Sky', 'Overcast']
}

# Flattened list of all categories
categories = [item for sublist in categories_dict.values() for item in sublist]

# ------------------------------ Category Colors ------------------------------

category_colors = {
    'Water Surface': (255, 0, 0),          # Blue
    'Tree': (0, 128, 0),             # Green
    'Boat': (0, 0, 255),                   # Red
    'Boat parts': (0, 255, 255),           # Yellow
    'Rock': (128, 128, 128),               # Gray
    'Sand': (0, 204, 255),                 # Light Orange
    'Grass': (0, 255, 0),                  # Bright Green
    'Dock': (128, 0, 128),                 # Purple
    'Street Light Pole': (255, 255, 0),    # Cyan
    'Tire': (147, 20, 255),                # Pink
    'Street': (19, 69, 139),               # Dark Brown
    'Boat Ramp': (180, 130, 70),           # Steel Blue
    'Shoreline Barrier': (105, 105, 105),  # Dark Gray
    'Bridge': (192, 192, 192),             # Silver
    'Buoy': (180, 105, 255),               # Hot Pink
    'Building': (255, 0, 255),             # Magenta
    'Lighthouse': (230, 216, 173),         # Light Blue
    'Person': (0, 69, 255),                # Orange Red
    'Car': (128, 0, 0),                    # Dark Navy
    'Sky': (255, 255, 200)                 # light blue
}