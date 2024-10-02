import osmnx as ox
import folium
import math
import warnings

# Suppress specific FutureWarnings
warnings.filterwarnings("ignore", category=FutureWarning)

class VehicleRouting:
    def __init__(self, data):
        self.data = data
    
    def haversine(self, lat1, lon1, lat2, lon2):
    # Calculate the great circle distance in kilometers between two points
        R = 6371.0  # Radius of the Earth in kilometers
        dlat = math.radians(lat2 - lat1)
        dlon = math.radians(lon2 - lon1)
        a = (math.sin(dlat / 2) ** 2 +
            math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon / 2) ** 2)
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
        distance = R * c  # Distance in kilometers
        return distance

    def find_map(self, predicted_time):
        restaurant_lat = self.data['Restaurant_latitude']
        restaurant_lon = self.data['Restaurant_longitude']
        delivery_lat = self.data['Delivery_location_latitude']
        delivery_lon = self.data['Delivery_location_longitude']
        time_taken_minutes = self.data.get('Time_taken(min)', 1)  # Default to 1 to avoid division by zero

        graph = ox.graph_from_point((restaurant_lat, restaurant_lon), dist=1000, network_type='drive')

        # Find the nearest nodes
        start_node = ox.distance.nearest_nodes(graph, restaurant_lon, restaurant_lat)
        end_node = ox.distance.nearest_nodes(graph, delivery_lon, delivery_lat)

        # Calculate the shortest route
        route = ox.shortest_path(graph, start_node, end_node)

        # Get the route as a GeoDataFrame
        route_gdf = ox.routing.route_to_gdf(graph, route)

        # Get the route length in meters
        route_length = route_gdf['length'].sum()

        # Calculate delivery distance using haversine
        distance_km = self.haversine(restaurant_lat, restaurant_lon, delivery_lat, delivery_lon)

        # Convert time taken from data
        time_taken_hours = time_taken_minutes / 60

        # Calculate average speed
        average_speed = distance_km / time_taken_hours if time_taken_hours > 0 else 0

        # Calculate estimated delivery time in seconds
        delivery_time_seconds = route_length / (average_speed * 1000 / 3600)  # Convert speed to m/s
        delivery_time_minutes = delivery_time_seconds / 60  # Convert to minutes

        # Explore the route with GeoPandas
        route_gdf.explore().save("output/route_map.html")

        # Print estimated delivery time
        print("Step 5: Prediction:")
        print(f"    Estimated delivery time: {delivery_time_minutes:.2f} minutes")
        print(f"    Predicted Time by model: {predicted_time} minutes")


# data = {
# 'Delivery_person_Age': 37,
# 'Delivery_person_Ratings': 4.9,
# 'Restaurant_latitude': 22.745049,
# 'Restaurant_longitude': 75.892471,
# 'Delivery_location_latitude': 22.765049,
# 'Delivery_location_longitude': 75.912471,
# 'Time_taken(min)': 24
# }
# vrp = VehicleRouting(data)
# vrp.find_map(22.745049,	75.892471,	22.765049,	75.912471, 24.274398636271936 )